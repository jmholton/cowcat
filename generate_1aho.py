#!/usr/bin/env python3
"""
generate_1aho.py — Training data pipeline for the 1AHO 48-conformer system.

Each sample:
  1. Jiggle refmacout_minRfree.pdb (all 48 protein chains) with Gaussian noise
     scaled by B factors.
  2. Place N_flood random water positions, compute their SF contribution.
  3. Build fobs.mtz: FP = |F_jig_prot + F_flood + F_bulk_solv|.
  4. Build starthere.pdb from jiggled conf using S8 fusion strategy.
  5. Run refmac (NCYC 10) against fobs.mtz.
  6. Write CCP4 .map files: truth, 2fofc, fofc, fc.
  7. Write metadata.json.

Run with:
    ccp4-python generate_1aho.py --nsamples 5 --outdir data_1aho_n5 [--submit]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

# Unbuffered stdout so SLURM captures output even if the task is preempted
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)   # line-buffered
import tempfile
import time
from pathlib import Path

import numpy as np
import gemmi

# ── import shared functions from explore_1aho_fusion ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from explore_1aho_fusion import (
    parse_conformers,
    build_reduced_pdb,
    generate_occ_groups,
    parse_rfactors,
    load_density_map,
    MAINCHAIN_ATOMS,
)

SCRIPT_DIR = Path(__file__).parent
REFMAC5    = Path('/programs/ccp4-8.0/bin/refmac5')
DMIN       = 0.965
SAMPLE_RATE = 3.0

# Default 1AHO source files (relative to SCRIPT_DIR or absolute)
DEFAULT_PDB = SCRIPT_DIR / '1aho' / 'refmacout_minRfree.pdb'
DEFAULT_MTZ = SCRIPT_DIR / '1aho' / 'refme_minRfree.mtz'

# Strategy S8: best from exploration
S8_STRATEGY = ('cluster_k3', 'cluster_k8', 'cluster_k3', 'adaptive')
S8_BOUQ_THR = 1.5
S8_MC_THR   = 0.5

# Flood water calibration: n_flood × occ → 8% rms noise on FP
DEFAULT_N_FLOOD   = 1764
DEFAULT_FLOOD_OCC = 0.05

# shift_scale: Gaussian B-based displacement giving ~8% ΔF/F on the 48-conformer model.
# Calibration (Python Gaussian, B-based σ per atom):
#   ss=0.10 → ΔF/F=1.45%,  ss=0.20 → 2.94%,  ss=0.30 → 4.81%
# Linear extrapolation to 8%: ss ≈ 0.50.  The 48 conformers average out noise by ~√48.
DEFAULT_SHIFT_SCALE = 0.50

# Refmac cycles for training data (shorter than exploration's NCYC 50)
DEFAULT_NCYC = 50


# ─────────────────────────────────────────────────────────────────────────────
# Jiggling
# ─────────────────────────────────────────────────────────────────────────────

def jiggle_structure(st, shift_scale, seed):
    """Return cloned gemmi.Structure with all protein atom positions perturbed.

    Displacement is drawn from N(0, sigma) where sigma = shift_scale * sqrt(B/8pi²).
    Water residues are kept unchanged (they are in the Fpart / flood model).
    """
    rng = np.random.default_rng(seed)
    st2 = st.clone()
    for chain in st2[0]:
        for res in chain:
            if res.name in ('HOH', 'WAT', 'H2O'):
                continue
            for atom in res:
                sigma = shift_scale * np.sqrt(max(atom.b_iso, 1.0) / (8.0 * np.pi ** 2))
                atom.pos.x += float(rng.normal(0.0, sigma))
                atom.pos.y += float(rng.normal(0.0, sigma))
                atom.pos.z += float(rng.normal(0.0, sigma))
    return st2


def write_protein_only_pdb(st, out_path):
    """Write PDB with all residues (protein + structural waters) for sfcalc.
    Returns list of protein chain names (HOH-only chains excluded from list)."""
    st2 = gemmi.Structure()
    st2.cell = st.cell
    st2.spacegroup_hm = st.spacegroup_hm
    mdl = gemmi.Model('1')
    chain_names = []
    for chain in st[0]:
        ch2 = gemmi.Chain(chain.name)
        for res in chain:
            ch2.add_residue(res.clone())
        mdl.add_chain(ch2)
        if any(res.name not in ('HOH', 'WAT', 'H2O') for res in chain):
            chain_names.append(chain.name)
    st2.add_model(mdl)
    st2.write_pdb(str(out_path))
    return chain_names


# ─────────────────────────────────────────────────────────────────────────────
# Flood waters
# ─────────────────────────────────────────────────────────────────────────────

def place_flood_waters(cell, existing_xyzs, n_flood, occ, seed, min_dist=2.0):
    """Return (n_flood, 3) array of Cartesian positions for flood waters.

    Positions are random in the unit cell, avoiding existing atoms by min_dist Å.
    """
    rng = np.random.default_rng(seed)
    a, b, c = cell.a, cell.b, cell.c
    existing = np.array(existing_xyzs) if existing_xyzs else np.zeros((0, 3))

    positions = []
    max_attempts = n_flood * 50
    for _ in range(max_attempts):
        if len(positions) >= n_flood:
            break
        # Random fractional coord → Cartesian (P1, orthogonal)
        frac = rng.random(3)
        pos  = gemmi.Fractional(*frac)
        xyz  = cell.orthogonalize(pos)
        pt   = np.array([xyz.x, xyz.y, xyz.z])
        if existing.shape[0] > 0:
            dists = np.linalg.norm(existing - pt, axis=1)
            if dists.min() < min_dist:
                continue
        positions.append(pt)
        existing = np.vstack([existing, pt[None, :]]) if existing.shape[0] > 0 else pt[None, :]
    return np.array(positions[:n_flood])


def flood_sf_dict(positions, occ, cell, spacegroup_hm, tmpdir):
    """Compute SFs for a set of water positions; return {(h,k,l): complex_F}."""
    if len(positions) == 0:
        return {}

    st_w = gemmi.Structure()
    st_w.cell = cell
    st_w.spacegroup_hm = spacegroup_hm
    mdl = gemmi.Model('1')
    ch  = gemmi.Chain('W')
    for i, (x, y, z) in enumerate(positions):
        res = gemmi.Residue()
        res.name  = 'HOH'
        res.seqid = gemmi.SeqId(i + 1, ' ')
        a = gemmi.Atom()
        a.name    = 'O'
        a.element = gemmi.Element('O')
        a.pos     = gemmi.Position(x, y, z)
        a.occ     = float(occ)
        a.b_iso   = 20.0
        res.add_atom(a)
        ch.add_residue(res)
    mdl.add_chain(ch)
    st_w.add_model(mdl)

    flood_pdb = tmpdir / '_flood.pdb'
    flood_mtz = tmpdir / '_flood.mtz'
    st_w.write_pdb(str(flood_pdb))

    subprocess.run(
        ['gemmi', 'sfcalc', f'--dmin={DMIN}', f'--to-mtz={flood_mtz}', str(flood_pdb)],
        capture_output=True, check=True,
    )

    m  = gemmi.read_mtz_file(str(flood_mtz))
    h  = np.array(m.column_with_label('H'),    dtype=np.int32)
    k  = np.array(m.column_with_label('K'),    dtype=np.int32)
    l  = np.array(m.column_with_label('L'),    dtype=np.int32)
    fc = np.array(m.column_with_label('FC'),   dtype=np.float64)
    ph = np.array(m.column_with_label('PHIC'), dtype=np.float64)
    F  = fc * np.exp(1j * np.radians(ph))
    return {(int(h[i]), int(k[i]), int(l[i])): F[i] for i in range(len(h))}


# ─────────────────────────────────────────────────────────────────────────────
# SF / MTZ construction
# ─────────────────────────────────────────────────────────────────────────────

def build_sample_mtz(prot_only_pdb, refme_path, flood_dict, tmpdir):
    """Build fobs.mtz and truth.mtz for one training sample.

    fobs.mtz: FP, SIGFP, FreeR_flag, Fpart, PHIpart
    truth.mtz: FC (= FP), PHIC (= phase of total F)

    Returns (fobs_mtz_path, truth_mtz_path).
    """
    # Protein SFs
    prot_mtz = tmpdir / '_prot_sf.mtz'
    subprocess.run(
        ['gemmi', 'sfcalc', f'--dmin={DMIN}', f'--to-mtz={prot_mtz}', str(prot_only_pdb)],
        capture_output=True, check=True,
    )

    prot = gemmi.read_mtz_file(str(prot_mtz))
    refme = gemmi.read_mtz_file(str(refme_path))

    h_p  = np.array(prot.column_with_label('H'),    dtype=np.int32)
    k_p  = np.array(prot.column_with_label('K'),    dtype=np.int32)
    l_p  = np.array(prot.column_with_label('L'),    dtype=np.int32)
    fc_p = np.array(prot.column_with_label('FC'),   dtype=np.float64)
    ph_p = np.array(prot.column_with_label('PHIC'), dtype=np.float64)
    F_prot = fc_p * np.exp(1j * np.radians(ph_p))

    h_r  = np.array(refme.column_with_label('H'),        dtype=np.int32)
    k_r  = np.array(refme.column_with_label('K'),        dtype=np.int32)
    l_r  = np.array(refme.column_with_label('L'),        dtype=np.int32)
    fp_r = np.array(refme.column_with_label('Fpart'),    dtype=np.float64)
    pp_r = np.array(refme.column_with_label('PHIpart'),  dtype=np.float64)
    fr_r = np.array(refme.column_with_label('FreeR_flag'), dtype=np.float32)

    refme_dict = {
        (int(h_r[i]), int(k_r[i]), int(l_r[i])): (fp_r[i], pp_r[i], fr_r[i])
        for i in range(len(h_r))
    }

    fp_out   = np.zeros(len(h_p), dtype=np.float32)
    sp_out   = np.zeros(len(h_p), dtype=np.float32)
    fr_out   = np.zeros(len(h_p), dtype=np.float32)
    fpart_out = np.zeros(len(h_p), dtype=np.float32)
    ppart_out = np.zeros(len(h_p), dtype=np.float32)
    # truth complex SFs (same HKL set as prot)
    fc_t  = np.zeros(len(h_p), dtype=np.float32)
    ph_t  = np.zeros(len(h_p), dtype=np.float32)

    for i in range(len(h_p)):
        hkl = (int(h_p[i]), int(k_p[i]), int(l_p[i]))
        fpa, ppa, fra = refme_dict.get(hkl, (0.0, 0.0, 0.0))
        F_bulk  = fpa * np.exp(1j * np.radians(ppa))
        F_flood = flood_dict.get(hkl, 0.0)
        F_solv  = F_bulk + F_flood
        F_total = F_prot[i] + F_solv
        amp     = float(np.abs(F_total))
        fp_out[i]    = amp
        sp_out[i]    = max(0.01, 0.02 * amp)
        fr_out[i]    = float(fra)
        fpart_out[i] = float(np.abs(F_solv))
        ppart_out[i] = float(np.degrees(np.angle(F_solv))) if abs(F_solv) > 0 else 0.0
        fc_t[i]      = amp
        ph_t[i]      = float(np.degrees(np.angle(F_total))) if amp > 0 else 0.0

    # Write fobs.mtz
    out = gemmi.Mtz()
    out.cell       = prot.cell
    out.spacegroup = prot.spacegroup
    out.add_dataset('HKL_base')
    for lbl in ('H', 'K', 'L'):
        out.add_column(lbl, 'H')
    out.add_dataset('data')
    out.add_column('FP',         'F')
    out.add_column('SIGFP',      'Q')
    out.add_column('FreeR_flag', 'I')
    out.add_column('Fpart',      'F')
    out.add_column('PHIpart',    'P')
    data = np.column_stack([h_p, k_p, l_p, fp_out, sp_out, fr_out, fpart_out, ppart_out])
    out.set_data(data.astype(np.float32))
    fobs_mtz = tmpdir / 'fobs.mtz'
    out.write_to_file(str(fobs_mtz))

    # Write truth.mtz (FC/PHIC = total F from jiggled model)
    truth_out = gemmi.Mtz()
    truth_out.cell       = prot.cell
    truth_out.spacegroup = prot.spacegroup
    truth_out.add_dataset('HKL_base')
    for lbl in ('H', 'K', 'L'):
        truth_out.add_column(lbl, 'H')
    truth_out.add_dataset('data')
    truth_out.add_column('FC',   'F')
    truth_out.add_column('PHIC', 'P')
    tdata = np.column_stack([h_p, k_p, l_p, fc_t, ph_t])
    truth_out.set_data(tdata.astype(np.float32))
    truth_mtz = tmpdir / 'truth.mtz'
    truth_out.write_to_file(str(truth_mtz))

    return fobs_mtz, truth_mtz


# ─────────────────────────────────────────────────────────────────────────────
# Refmac
# ─────────────────────────────────────────────────────────────────────────────

def run_refmac_sample(starthere_pdb, fobs_mtz, ncyc, tmpdir):
    """Run refmac; return (R, Rfree, log_text, out_mtz_path or None)."""
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

    kw  = b'LABIN FP=FP SIGFP=SIGFP FPART1=Fpart PHIP1=PHIpart FREE=FreeR_flag\n'
    kw += b'LABOUT FC=FC PHIC=PHIC FWT=FWT PHWT=PHWT'
    kw += b' DELFWT=DELFWT PHDELWT=PHDELWT\n'
    kw += b'solvent no\n'
    kw += b'weight matrix 5\n'
    kw += b'scpart 1\n'
    kw += b'damp 0.5 0.5\n'
    kw += b'make hout Y\n'
    kw += b'make hydr Y\n'
    kw += f'NCYC {ncyc}\n'.encode()
    kw += generate_occ_groups(starthere_pdb)
    kw += b'END\n'

    out_mtz  = tmpdir / 'refmacout.mtz'
    out_pdb  = tmpdir / 'refmacout.pdb'

    try:
        r = subprocess.run(
            [str(REFMAC5),
             'XYZIN',  str(starthere_pdb),
             'XYZOUT', str(out_pdb),
             'HKLIN',  str(fobs_mtz),
             'HKLOUT', str(out_mtz),
             'LIBOUT', str(tmpdir / '_refmac.lib')],
            input=kw,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(tmpdir),
            timeout=1800,   # 30 min hard limit
        )
        log = r.stdout.decode(errors='replace')
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or b'').decode(errors='replace') + '\nTIMEOUT after 1800s\n'
        (tmpdir / 'refmac.log').write_text(log)
        return None, None, log, None
    (tmpdir / 'refmac.log').write_text(log)
    rwork, rfree = parse_rfactors(log)
    return rwork, rfree, log, (out_mtz if out_mtz.exists() else None)


# ─────────────────────────────────────────────────────────────────────────────
# Map writing
# ─────────────────────────────────────────────────────────────────────────────

def mtz_to_ccp4(mtz_path, f_col, phi_col, out_path):
    mtz  = gemmi.read_mtz_file(str(mtz_path))
    grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=SAMPLE_RATE)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(out_path))


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample(
    sample_idx, outdir, pdb_path, mtz_path, density_grid,
    shift_scale=DEFAULT_SHIFT_SCALE,
    n_flood=DEFAULT_N_FLOOD,
    flood_occ=DEFAULT_FLOOD_OCC,
    ncyc=DEFAULT_NCYC,
    seed=None,
    debug=False,
):
    """Run the full pipeline for one training sample.

    Returns (sample_idx, ok, info_string).
    """
    t0 = time.time()
    outdir     = Path(outdir).resolve()
    sample_dir = outdir / f'sample_{sample_idx:05d}'

    if sample_dir.exists() and (sample_dir / 'metadata.json').exists():
        return sample_idx, True, 'already done'

    rng_seed = sample_idx if seed is None else seed + sample_idx
    tmpdir   = Path(tempfile.mkdtemp(prefix=f'1aho_{sample_idx:05d}_'))
    timings  = {}

    def _t(label, t_prev):
        now = time.time()
        timings[label] = round(now - t_prev, 1)
        return now

    try:
        t = time.time()

        # 1. Load original 48-conformer structure, apply jiggle
        st_orig = gemmi.read_structure(str(pdb_path))
        st_jig  = jiggle_structure(st_orig, shift_scale, rng_seed)
        jig_pdb = tmpdir / 'jiggled.pdb'
        st_jig.write_pdb(str(jig_pdb))
        t = _t('jiggle', t)

        # 2. Parse jiggled conformers
        st_jig_r, chain_names, conf_data = parse_conformers(jig_pdb)
        t = _t('parse', t)

        # 3. Write protein-only PDB (no HOH, all 48 chains) for sfcalc
        prot_only_pdb = tmpdir / 'prot_only.pdb'
        write_protein_only_pdb(st_jig_r, prot_only_pdb)

        # 4. Flood waters: existing atom positions for avoidance check
        existing_xyzs = []
        for chain in st_jig_r[0]:
            for res in chain:
                for atom in res:
                    existing_xyzs.append([atom.pos.x, atom.pos.y, atom.pos.z])
        flood_pos = place_flood_waters(
            st_jig_r.cell, existing_xyzs, n_flood, flood_occ,
            seed=rng_seed + 1,
        )
        f_dict = flood_sf_dict(flood_pos, flood_occ, st_jig_r.cell,
                               st_jig_r.spacegroup_hm, tmpdir)
        t = _t('flood_waters', t)

        # 5. Build fobs.mtz and truth.mtz
        fobs_mtz, truth_mtz = build_sample_mtz(
            prot_only_pdb, mtz_path, f_dict, tmpdir
        )
        t = _t('build_mtz', t)

        # 6. Build partial model using S8 strategy
        starthere_pdb = tmpdir / 'starthere.pdb'
        _, n_bouq, n_alt = build_reduced_pdb(
            st_jig_r, chain_names, conf_data,
            strategy=S8_STRATEGY,
            density_grid=density_grid,
            bouquet_threshold=S8_BOUQ_THR,
            mc_bouq_threshold=S8_MC_THR,
            out_pdb=starthere_pdb, tmpdir=tmpdir,
        )
        t = _t('build_partial', t)

        # 7. Refmac
        rwork, rfree, log, out_mtz = run_refmac_sample(
            starthere_pdb, fobs_mtz, ncyc, tmpdir
        )
        t = _t('refmac', t)

        # 8. Write maps
        sample_dir.mkdir(parents=True, exist_ok=True)
        mtz_to_ccp4(truth_mtz,  'FC',     'PHIC',     sample_dir / 'truth.map')
        if out_mtz:
            mtz_to_ccp4(out_mtz, 'FWT',    'PHWT',     sample_dir / '2fofc.map')
            mtz_to_ccp4(out_mtz, 'DELFWT', 'PHDELWT',  sample_dir / 'fofc.map')
            mtz_to_ccp4(out_mtz, 'FC',     'PHIC',     sample_dir / 'fc.map')
        t = _t('maps', t)

        # 9. Copy useful files
        shutil.copy2(jig_pdb,       sample_dir / 'truth_full.pdb')
        shutil.copy2(starthere_pdb, sample_dir / 'partial.pdb')
        shutil.copy2(tmpdir / 'refmac.log', sample_dir / 'refmac.log')
        if out_mtz:
            shutil.copy2(out_mtz, sample_dir / 'refmacout.mtz')
        out_pdb = tmpdir / 'refmacout.pdb'
        if out_pdb.exists():
            shutil.copy2(out_pdb, sample_dir / 'refmacout.pdb')

        if debug:
            dbg = sample_dir / 'debug'
            if dbg.exists():
                shutil.rmtree(str(dbg))
            shutil.copytree(str(tmpdir), str(dbg))

        t = _t('copy', t)

        # 10. metadata
        meta = dict(
            sample_idx=int(sample_idx),
            pdb_source=str(pdb_path),
            shift_scale=shift_scale,
            n_flood=n_flood,
            flood_occ=flood_occ,
            ncyc=ncyc,
            n_bouquet_residues=n_bouq,
            n_altloc_residues=n_alt,
            rwork=rwork,
            rfree=rfree,
            dmin=DMIN,
            step_timings=timings,
        )
        (sample_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))

        elapsed = time.time() - t0
        timing_str = '  '.join(f'{k}={v}s' for k, v in timings.items())
        return sample_idx, True, f'ok {elapsed:.0f}s  R={rwork}  Rf={rfree}\n  {timing_str}'

    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / 'error.log').write_text(msg)
        if debug:
            dbg = sample_dir / 'debug'
            if dbg.exists():
                shutil.rmtree(str(dbg))
            try:
                shutil.copytree(str(tmpdir), str(dbg))
            except Exception:
                pass
        elapsed = time.time() - t0
        return sample_idx, False, f'FAILED {elapsed:.0f}s: {e}'

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# SLURM submission
# ─────────────────────────────────────────────────────────────────────────────

def submit_slurm_array(nsamples, outdir, pdb, mtz, shift_scale, n_flood,
                       flood_occ, ncyc, max_array, seed, partition):
    script = SCRIPT_DIR / f'_slurm_{outdir.name}.sh'
    me     = Path(__file__).resolve()

    lines = [
        '#!/bin/bash',
        f'#SBATCH --job-name=gen1aho_{outdir.name}',
        f'#SBATCH --array=0-{nsamples - 1}%{max_array}',
        '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=1',
        '#SBATCH --mem=4G',
        f'#SBATCH --partition={partition}',
        '#SBATCH --export=ALL',
        '',
        'IDX=$SLURM_ARRAY_TASK_ID',
        f'ccp4-python {me} \\',
        f'  --sample-id $IDX \\',
        f'  --outdir {outdir} \\',
        f'  --pdb {pdb} \\',
        f'  --mtz {mtz} \\',
        f'  --shift-scale {shift_scale} \\',
        f'  --n-flood {n_flood} \\',
        f'  --flood-occ {flood_occ} \\',
        f'  --ncyc {ncyc} \\',
        f'  --seed {seed}',
        '',
    ]
    script.write_text('\n'.join(lines))
    script.chmod(0o755)
    r = subprocess.run(['sbatch', str(script)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    print(f'Script: {script}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb',         default=str(DEFAULT_PDB))
    ap.add_argument('--mtz',         default=str(DEFAULT_MTZ))
    ap.add_argument('--outdir',      default='data_1aho')
    ap.add_argument('--nsamples',    type=int, default=1000)
    ap.add_argument('--sample-id',   type=int, default=None,
                    help='Run a single sample (SLURM array mode)')
    ap.add_argument('--shift-scale', type=float, default=DEFAULT_SHIFT_SCALE)
    ap.add_argument('--n-flood',     type=int,   default=DEFAULT_N_FLOOD)
    ap.add_argument('--flood-occ',   type=float, default=DEFAULT_FLOOD_OCC)
    ap.add_argument('--ncyc',        type=int,   default=DEFAULT_NCYC)
    ap.add_argument('--seed',        type=int,   default=42)
    ap.add_argument('--workers',     type=int,   default=1)
    ap.add_argument('--max-array',   type=int,   default=300)
    ap.add_argument('--partition',   default='debug')
    ap.add_argument('--submit',      action='store_true')
    ap.add_argument('--debug',       action='store_true')
    args = ap.parse_args()

    pdb_path = Path(args.pdb).resolve()
    mtz_path = Path(args.mtz).resolve()
    outdir   = Path(args.outdir).resolve()

    if args.submit:
        outdir.mkdir(parents=True, exist_ok=True)
        submit_slurm_array(
            args.nsamples, outdir, pdb_path, mtz_path,
            args.shift_scale, args.n_flood, args.flood_occ,
            args.ncyc, args.max_array, args.seed, args.partition,
        )
        return

    # Pre-load shared density grid (fixed from original refmacout_minRfree.mtz)
    refmac_mtz = pdb_path.parent / 'refmacout_minRfree.mtz'
    print(f'Loading density grid from {refmac_mtz}...')
    density_grid = load_density_map(str(refmac_mtz))

    common_kw = dict(
        pdb_path=pdb_path, mtz_path=mtz_path, density_grid=density_grid,
        shift_scale=args.shift_scale, n_flood=args.n_flood,
        flood_occ=args.flood_occ, ncyc=args.ncyc, seed=args.seed,
        debug=args.debug,
    )

    if args.sample_id is not None:
        # Single sample (SLURM array worker)
        idx, ok, msg = generate_sample(args.sample_id, outdir, **common_kw)
        print(f'sample_{idx:05d}: {msg}')
        return

    # Local multi-worker run
    if args.workers > 1:
        from multiprocessing import Pool
        import functools
        fn = functools.partial(generate_sample, outdir=outdir, **common_kw)
        with Pool(args.workers) as pool:
            for idx, ok, msg in pool.imap_unordered(fn, range(args.nsamples)):
                status = 'OK ' if ok else 'ERR'
                print(f'[{status}] sample_{idx:05d}: {msg}')
    else:
        for idx in range(args.nsamples):
            _, ok, msg = generate_sample(idx, outdir, **common_kw)
            status = 'OK ' if ok else 'ERR'
            print(f'[{status}] sample_{idx:05d}: {msg}')


if __name__ == '__main__':
    main()
