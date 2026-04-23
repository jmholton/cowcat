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
DEFAULT_PDB     = SCRIPT_DIR / '1aho' / 'refmacout_minRfree.pdb'
DEFAULT_MTZ     = SCRIPT_DIR / '1aho' / 'refme_minRfree.mtz'
DEFAULT_OBS_MTZ = SCRIPT_DIR / '1aho' / '1aho.mtz'   # real data: valid HKL mask + FreeR_flag

# Strategy S8: best from exploration
S8_STRATEGY = ('cluster_k3', 'cluster_k8', 'cluster_k3', 'adaptive')
S8_BOUQ_THR = 1.5
S8_MC_THR   = 0.5

# Flood water calibration: occ * sqrt(n_flood) = FLOOD_LINE_K gives Rfree ~11%
# Grid fit: Rfree = 0.0129 * occ*sqrt(nf) + 0.0417  (R²=0.989)
FLOOD_LINE_K    = 5.27   # occ * sqrt(n_flood) for Rfree ~11%
FLOOD_NF_MIN    = 700    # log-uniform sampling range
FLOOD_NF_MAX    = 4000
DEFAULT_N_FLOOD   = 1764   # used only when --vary-flood not set
DEFAULT_FLOOD_OCC = 0.13   # used only when --vary-flood not set

# shift_scale: Gaussian B-based displacement giving ~8% ΔF/F on the 48-conformer model.
# Calibration (Python Gaussian, B-based σ per atom):
#   ss=0.10 → ΔF/F=1.45%,  ss=0.20 → 2.94%,  ss=0.30 → 4.81%
# Linear extrapolation to 8%: ss ≈ 0.50.  The 48 conformers average out noise by ~√48.
DEFAULT_SHIFT_SCALE = 0.50

# Refmac cycles for training data
DEFAULT_NCYC = 10


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



# ─────────────────────────────────────────────────────────────────────────────
# Flood waters
# ─────────────────────────────────────────────────────────────────────────────

def place_flood_waters(cell, existing_xyzs, n_flood, occ, seed, min_dist=2.0):
    """Return (n_flood, 3) array of Cartesian positions for flood waters.

    Positions are random in the unit cell, avoiding existing atoms by min_dist Å.
    """
    rng = np.random.default_rng(seed)
    existing = np.array(existing_xyzs) if existing_xyzs else np.zeros((0, 3))

    positions = []
    max_attempts = n_flood * 50
    for _ in range(max_attempts):
        if len(positions) >= n_flood:
            break
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


def add_flood_chain(st, positions, occ, chain_name='W'):
    """Add flood water atoms to a cloned structure; return the new structure."""
    st2 = st.clone()
    ch  = gemmi.Chain(chain_name)
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
    st2[0].add_chain(ch)
    return st2


def set_sulfur_aniso(st, rng):
    """Set random anisotropic U_ij for all S atoms in place.

    Eigenvalues are distributed around U_iso with ~40% spread; trace preserved.
    """
    S_elem = gemmi.Element('S')
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element == S_elem:
                    u_iso = atom.b_iso / (8.0 * np.pi ** 2)
                    v = rng.normal(0.0, 0.4 * u_iso, 3)
                    v -= v.mean()
                    evals = np.maximum(u_iso + v, 0.01 * u_iso)
                    M = rng.normal(0.0, 1.0, (3, 3))
                    Q, _ = np.linalg.qr(M)
                    U = Q @ np.diag(evals) @ Q.T
                    atom.aniso = gemmi.SMat33f(
                        float(U[0, 0]), float(U[1, 1]), float(U[2, 2]),
                        float(U[0, 1]), float(U[0, 2]), float(U[1, 2]),
                    )


def add_hoh_chains_to_pdb(st_source, pdb_path):
    """Append all HOH-only chains from st_source into the PDB at pdb_path (in place)."""
    st = gemmi.read_structure(str(pdb_path))
    for chain in st_source[0]:
        if all(res.name in ('HOH', 'WAT', 'H2O') for res in chain):
            st[0].add_chain(chain.clone())
    st.write_pdb(str(pdb_path))


# ─────────────────────────────────────────────────────────────────────────────
# SF / MTZ construction
# ─────────────────────────────────────────────────────────────────────────────

def build_sample_mtz(truth_full_pdb, refme_path, obs_mtz_path, tmpdir, suffix=''):
    """Build fobs.mtz and truth.mtz for one training sample.

    truth_full_pdb: jiggled protein + structural waters + flood waters.
    truth.mtz: FC/PHIC = sfcalc(truth_full_pdb)  (no bulk solvent)
    fobs.mtz:  FP = |F_truth + F_bulk|, using obs_mtz_path for valid HKL mask
               and FreeR_flag; refme_path for Fpart/PHIpart.

    Returns (fobs_mtz_path, truth_mtz_path).
    """
    truth_sf_mtz = tmpdir / f'_truth_sf{suffix}.mtz'
    subprocess.run(
        ['gemmi', 'sfcalc', f'--dmin={DMIN}', f'--to-mtz={truth_sf_mtz}', str(truth_full_pdb)],
        capture_output=True, check=True,
    )

    prot  = gemmi.read_mtz_file(str(truth_sf_mtz))
    refme = gemmi.read_mtz_file(str(refme_path))
    obs   = gemmi.read_mtz_file(str(obs_mtz_path))

    h_p  = np.array(prot.column_with_label('H'),    dtype=np.int32)
    k_p  = np.array(prot.column_with_label('K'),    dtype=np.int32)
    l_p  = np.array(prot.column_with_label('L'),    dtype=np.int32)
    fc_p = np.array(prot.column_with_label('FC'),   dtype=np.float64)
    ph_p = np.array(prot.column_with_label('PHIC'), dtype=np.float64)
    F_truth = fc_p * np.exp(1j * np.radians(ph_p))
    truth_dict = {
        (int(h_p[i]), int(k_p[i]), int(l_p[i])): F_truth[i]
        for i in range(len(h_p))
    }

    # obs (1aho.mtz): valid HKL mask only (NaN FP = missing observation)
    h_o  = np.array(obs.column_with_label('H'),  dtype=np.int32)
    k_o  = np.array(obs.column_with_label('K'),  dtype=np.int32)
    l_o  = np.array(obs.column_with_label('L'),  dtype=np.int32)
    fp_o = np.array(obs.column_with_label('FP'), dtype=np.float32)
    valid_hkls = frozenset(
        (int(h_o[i]), int(k_o[i]), int(l_o[i]))
        for i in range(len(h_o)) if not np.isnan(fp_o[i])
    )

    # Drive output from refme: 100% complete FreeR_flag, Fpart/PHIpart for all HKLs
    h_r  = np.array(refme.column_with_label('H'),          dtype=np.int32)
    k_r  = np.array(refme.column_with_label('K'),          dtype=np.int32)
    l_r  = np.array(refme.column_with_label('L'),          dtype=np.int32)
    fp_r = np.array(refme.column_with_label('Fpart'),      dtype=np.float64)
    pp_r = np.array(refme.column_with_label('PHIpart'),    dtype=np.float64)
    fr_r = np.array(refme.column_with_label('FreeR_flag'), dtype=np.float32)

    nan32 = float('nan')
    n = len(h_r)
    fp_out    = np.full(n, nan32, dtype=np.float32)
    sp_out    = np.full(n, nan32, dtype=np.float32)
    fr_out    = fr_r.copy()                             # complete FreeR_flag from refme
    fpart_out = fp_r.astype(np.float32)
    ppart_out = pp_r.astype(np.float32)

    for i in range(n):
        hkl = (int(h_r[i]), int(k_r[i]), int(l_r[i]))
        if hkl not in valid_hkls:
            continue                                    # leave FP/SIGFP as NaN
        F_t    = truth_dict.get(hkl, 0.0)
        F_bulk = fp_r[i] * np.exp(1j * np.radians(pp_r[i]))
        amp    = float(np.abs(F_t + F_bulk))
        fp_out[i] = amp
        sp_out[i] = max(0.01, 0.02 * amp)

    data = np.column_stack([h_r, k_r, l_r, fp_out, sp_out, fr_out, fpart_out, ppart_out])
    out = gemmi.Mtz()
    out.cell       = refme.cell
    out.spacegroup = refme.spacegroup
    out.add_dataset('HKL_base')
    for lbl in ('H', 'K', 'L'):
        out.add_column(lbl, 'H')
    out.add_dataset('data')
    out.add_column('FP',         'F')
    out.add_column('SIGFP',      'Q')
    out.add_column('FreeR_flag', 'I')
    out.add_column('Fpart',      'F')
    out.add_column('PHIpart',    'P')
    out.set_data(data)
    fobs_mtz = tmpdir / f'fobs{suffix}.mtz'
    out.write_to_file(str(fobs_mtz))

    return fobs_mtz, truth_sf_mtz


# ─────────────────────────────────────────────────────────────────────────────
# Refmac
# ─────────────────────────────────────────────────────────────────────────────

def run_refmac_sample(starthere_pdb, fobs_mtz, ncyc, tmpdir, suffix=''):
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

    out_mtz  = tmpdir / f'refmacout{suffix}.mtz'
    out_pdb  = tmpdir / f'refmacout{suffix}.pdb'

    try:
        r = subprocess.run(
            [str(REFMAC5),
             'XYZIN',  str(starthere_pdb),
             'XYZOUT', str(out_pdb),
             'HKLIN',  str(fobs_mtz),
             'HKLOUT', str(out_mtz),
             'LIBOUT', str(tmpdir / f'_refmac{suffix}.lib')],
            input=kw,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(tmpdir),
            timeout=1800,   # 30 min hard limit
        )
        log = r.stdout.decode(errors='replace')
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or b'').decode(errors='replace') + '\nTIMEOUT after 1800s\n'
        (tmpdir / f'refmac{suffix}.log').write_text(log)
        return None, None, log, None
    (tmpdir / f'refmac{suffix}.log').write_text(log)
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
    sample_idx, outdir, pdb_path, mtz_path, obs_mtz_path, density_grid,
    shift_scale=DEFAULT_SHIFT_SCALE,
    n_flood=DEFAULT_N_FLOOD,
    flood_occ=DEFAULT_FLOOD_OCC,
    vary_flood=False,
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
    if vary_flood:
        rng_flood = np.random.default_rng(rng_seed + 4)
        log_nf = rng_flood.uniform(np.log(FLOOD_NF_MIN), np.log(FLOOD_NF_MAX))
        n_flood = int(np.round(np.exp(log_nf)))
        flood_occ = float(FLOOD_LINE_K / np.sqrt(n_flood))
    ccp4_scr = Path(os.environ.get('CCP4_SCR', '/tmp'))
    for stale in ccp4_scr.glob(f'1aho_{sample_idx:05d}_*'):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
    tmpdir   = Path(tempfile.mkdtemp(prefix=f'1aho_{sample_idx:05d}_', dir=ccp4_scr))
    timings  = {}

    def _t(label, t_prev):
        now = time.time()
        timings[label] = round(now - t_prev, 1)
        return now

    try:
        t = time.time()

        # 1. Load original 48-conformer structure, apply jiggle (waters unchanged)
        st_orig = gemmi.read_structure(str(pdb_path))
        st_jig  = jiggle_structure(st_orig, shift_scale, rng_seed)
        set_sulfur_aniso(st_jig, np.random.default_rng(rng_seed + 2))
        jig_pdb = tmpdir / 'jiggled.pdb'
        st_jig.write_pdb(str(jig_pdb))
        t = _t('jiggle', t)

        # 2. Parse jiggled conformers (protein chains only)
        st_jig_r, chain_names, conf_data = parse_conformers(jig_pdb)
        t = _t('parse', t)

        # 3. Flood waters: place random HOH positions avoiding existing atoms
        existing_xyzs = []
        for chain in st_jig_r[0]:
            for res in chain:
                for atom in res:
                    existing_xyzs.append([atom.pos.x, atom.pos.y, atom.pos.z])
        flood_pos = place_flood_waters(
            st_jig_r.cell, existing_xyzs, n_flood, flood_occ,
            seed=rng_seed + 1,
        )

        # 4. Build truth_full.pdb = jiggled protein + 258 structural waters + flood waters
        st_truth = add_flood_chain(st_jig, flood_pos, flood_occ, chain_name='W')
        truth_pdb = tmpdir / 'truth_full.pdb'
        st_truth.write_pdb(str(truth_pdb))
        t = _t('flood_waters', t)

        # 5. Build fobs.mtz (FP = |F_truth + F_bulk|) and truth.mtz (sfcalc of truth_full.pdb)
        fobs_mtz, truth_mtz = build_sample_mtz(truth_pdb, mtz_path, obs_mtz_path, tmpdir)
        t = _t('build_mtz', t)

        # 6. Build partial model using S8 strategy + append 258 structural waters
        starthere_pdb = tmpdir / 'starthere.pdb'
        _, n_bouq, n_alt = build_reduced_pdb(
            st_jig_r, chain_names, conf_data,
            strategy=S8_STRATEGY,
            density_grid=density_grid,
            bouquet_threshold=S8_BOUQ_THR,
            mc_bouq_threshold=S8_MC_THR,
            out_pdb=starthere_pdb, tmpdir=tmpdir,
            rng=np.random.default_rng(rng_seed + 3),
        )
        # Add structural waters (from jiggled model — identical to original since HOH not jiggled)
        add_hoh_chains_to_pdb(st_jig_r, starthere_pdb)
        t = _t('build_partial', t)

        # 7a. Refmac with flood waters
        rwork, rfree, log, out_mtz = run_refmac_sample(
            starthere_pdb, fobs_mtz, ncyc, tmpdir
        )
        t = _t('refmac', t)

        # 7b. Second refmac run with no flood waters (same partial model, bulk-only Fobs)
        noflood_pdb   = tmpdir / 'truth_noflood.pdb'
        st_jig.write_pdb(str(noflood_pdb))   # jiggled protein + 258 waters, no flood chain
        fobs_noflood, _ = build_sample_mtz(noflood_pdb, mtz_path, obs_mtz_path, tmpdir,
                                           suffix='_noflood')
        _, _, log_nf, out_mtz_nf = run_refmac_sample(
            starthere_pdb, fobs_noflood, ncyc, tmpdir, suffix='_noflood'
        )
        t = _t('refmac_noflood', t)

        # 8. Write maps
        sample_dir.mkdir(parents=True, exist_ok=True)
        mtz_to_ccp4(truth_mtz,  'FC',     'PHIC',     sample_dir / 'truth.map')
        if out_mtz:
            mtz_to_ccp4(out_mtz, 'FWT',    'PHWT',     sample_dir / '2fofc.map')
            mtz_to_ccp4(out_mtz, 'DELFWT', 'PHDELWT',  sample_dir / 'fofc.map')
            mtz_to_ccp4(out_mtz, 'FC',     'PHIC',     sample_dir / 'fc.map')
        if out_mtz_nf:
            mtz_to_ccp4(out_mtz_nf, 'FWT',    'PHWT',    sample_dir / '2fofc_nf.map')
            mtz_to_ccp4(out_mtz_nf, 'DELFWT', 'PHDELWT', sample_dir / 'fofc_nf.map')
            mtz_to_ccp4(out_mtz_nf, 'FC',     'PHIC',    sample_dir / 'fc_nf.map')
        t = _t('maps', t)

        # 9. Copy useful files
        shutil.copy2(truth_pdb,     sample_dir / 'truth_full.pdb')
        shutil.copy2(starthere_pdb, sample_dir / 'partial.pdb')
        shutil.copy2(tmpdir / 'refmac.log', sample_dir / 'refmac.log')
        if out_mtz:
            shutil.copy2(out_mtz, sample_dir / 'refmacout.mtz')
        out_pdb = tmpdir / 'refmacout.pdb'
        if out_pdb.exists():
            shutil.copy2(out_pdb, sample_dir / 'refmacout.pdb')
        if log_nf:
            (sample_dir / 'refmac_nf.log').write_text(log_nf)
        if out_mtz_nf:
            shutil.copy2(out_mtz_nf, sample_dir / 'refmacout_nf.mtz')

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

def submit_slurm_array(nsamples, outdir, pdb, mtz, obs_mtz, shift_scale, n_flood,
                       flood_occ, vary_flood, ncyc, max_array, seed, partition,
                       account=None, qos=None, time='00:20:00'):
    script = SCRIPT_DIR / f'_slurm_{outdir.name}.sh'
    me     = Path(__file__).resolve()

    flood_args = ['  --vary-flood \\'] if vary_flood else [
        f'  --n-flood {n_flood} \\',
        f'  --flood-occ {flood_occ} \\',
    ]
    account_line = f'#SBATCH --account={account}\n' if account else ''
    qos_line     = f'#SBATCH --qos={qos}\n'         if qos     else ''

    lines = [
        '#!/bin/bash',
        f'#SBATCH --job-name=gen1aho_{outdir.name}',
        f'#SBATCH --array=0-{nsamples - 1}%{max_array}',
        '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=1',
        '#SBATCH --mem=4G',
        f'#SBATCH --partition={partition}',
        f'#SBATCH --time={time}',
    ] + ([f'#SBATCH --account={account}'] if account else []) \
      + ([f'#SBATCH --qos={qos}']         if qos     else []) + [
        '#SBATCH --export=ALL',
        '',
        'mkdir -p "${CCP4_SCR:-/tmp}"',
        'IDX=$SLURM_ARRAY_TASK_ID',
        f'ccp4-python {me} \\',
        f'  --sample-id $IDX \\',
        f'  --outdir {outdir} \\',
        f'  --pdb {pdb} \\',
        f'  --mtz {mtz} \\',
        f'  --obs-mtz {obs_mtz} \\',
        f'  --shift-scale {shift_scale} \\',
    ] + flood_args + [
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
    ap.add_argument('--obs-mtz',     default=str(DEFAULT_OBS_MTZ))
    ap.add_argument('--outdir',      default='data_1aho')
    ap.add_argument('--nsamples',    type=int, default=1000)
    ap.add_argument('--sample-id',   type=int, default=None,
                    help='Run a single sample (SLURM array mode)')
    ap.add_argument('--shift-scale', type=float, default=DEFAULT_SHIFT_SCALE)
    ap.add_argument('--n-flood',     type=int,   default=DEFAULT_N_FLOOD)
    ap.add_argument('--flood-occ',   type=float, default=DEFAULT_FLOOD_OCC)
    ap.add_argument('--vary-flood',  action='store_true',
                    help='Sample n_flood log-uniformly [%d,%d] per sample; occ=%.2f/sqrt(nf)'
                         % (FLOOD_NF_MIN, FLOOD_NF_MAX, FLOOD_LINE_K))
    ap.add_argument('--ncyc',        type=int,   default=DEFAULT_NCYC)
    ap.add_argument('--seed',        type=int,   default=42)
    ap.add_argument('--workers',     type=int,   default=1)
    ap.add_argument('--max-array',   type=int,   default=300)
    ap.add_argument('--partition',   default='debug')
    ap.add_argument('--account',     default=None,
                    help='SLURM account (e.g. pc_als831)')
    ap.add_argument('--qos',         default=None,
                    help='SLURM QOS (e.g. lr_normal)')
    ap.add_argument('--time',        default='00:20:00',
                    help='SLURM walltime per task (default: 00:20:00)')
    ap.add_argument('--submit',      action='store_true')
    ap.add_argument('--debug',       action='store_true')
    args = ap.parse_args()

    pdb_path     = Path(args.pdb).resolve()
    mtz_path     = Path(args.mtz).resolve()
    obs_mtz_path = Path(args.obs_mtz).resolve()
    outdir       = Path(args.outdir).resolve()

    if args.submit:
        outdir.mkdir(parents=True, exist_ok=True)
        submit_slurm_array(
            args.nsamples, outdir, pdb_path, mtz_path, obs_mtz_path,
            args.shift_scale, args.n_flood, args.flood_occ, args.vary_flood,
            args.ncyc, args.max_array, args.seed, args.partition,
            account=args.account, qos=args.qos, time=args.time,
        )
        return

    # Pre-load shared density grid (fixed from original refmacout_minRfree.mtz)
    refmac_mtz = pdb_path.parent / 'refmacout_minRfree.mtz'
    print(f'Loading density grid from {refmac_mtz}...')
    density_grid = load_density_map(str(refmac_mtz))

    common_kw = dict(
        pdb_path=pdb_path, mtz_path=mtz_path, obs_mtz_path=obs_mtz_path,
        density_grid=density_grid,
        shift_scale=args.shift_scale, n_flood=args.n_flood,
        flood_occ=args.flood_occ, vary_flood=args.vary_flood,
        ncyc=args.ncyc, seed=args.seed,
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
