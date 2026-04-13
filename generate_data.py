#!/usr/bin/env python3
"""
generate_data.py  –  Training data generation for CNN electron density reconstruction.

For each sample:
  1. randompdb.com places random O atoms in a P1 40×40×40 Å cell
  2. B factors are randomised (log-normal, 5–120 Å²)
  3. gemmi sfcalc computes structure factors for the FULL model  →  truth.mtz
  4. refme.mtz is built with F=|FC_truth|, SIGF=2%*|FC_truth|
  5. Binary search on deletion fraction → refmac 5 cycles → Rwork ≈ 20%
  6. Maps exported as CCP4 .map files:
       truth.map   – ground-truth Fo density (FC/PHIC of full model)
       2fofc.map   – 2Fo-Fc from refmac    (FWT/PHWT)
       fofc.map    – Fo-Fc difference map  (DELFWT/PHDELWT)
       fc.map      – Fc density            (FC/PHIC of partial model)
       metadata.json

Usage:
    python generate_data.py --nsamples 500 --outdir ./data --workers 4
"""

import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import gemmi
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
RANDOMPDB  = SCRIPT_DIR / 'randompdb.com'
CONVERGE   = SCRIPT_DIR / 'converge_refmac.com'

# ── fixed crystallographic parameters ─────────────────────────────────────────
# CELL is set from --cell-size argument; default 40 Å cubic P1
SG          = 'P1'
DMIN        = 2.0    # resolution cutoff (Å)
MIND        = 0      # minimum inter-atom distance; 0 = no constraint (fast)
VM          = 2.4    # Matthews coefficient
SAMPLE_RATE = 3.0    # map oversampling; at 2 Å → ~0.67 Å/voxel

# ── B factor distribution (log-normal, matches protein atom statistics) ────────
BFAC_MU    = np.log(20.0)   # mean of ln(B)
BFAC_SIGMA = 0.7
BFAC_MIN   = 5.0
BFAC_MAX   = 120.0

# ── deletion fraction ──────────────────────────────────────────────────────────
DELETE_FRAC = 0.20   # fraction of atoms to remove before refinement

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def run(cmd, cwd):
    """Run a subprocess, raising RuntimeError on failure."""
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stdout = result.stdout.decode(errors='replace') if result.stdout else ''
        stderr = result.stderr.decode(errors='replace') if result.stderr else ''
        raise RuntimeError(
            "Command failed: {}\n--- stdout ---\n{}\n--- stderr ---\n{}".format(
                ' '.join(str(c) for c in cmd), stdout, stderr)
        )
    return result


def parse_rwork(pdb_path):
    """Extract Rwork from REMARK records written by refmac in refmacout.pdb."""
    with open(pdb_path) as f:
        for line in f:
            if 'R VALUE' in line and 'WORKING' in line:
                m = re.search(r'(\d+\.\d+)\s*$', line)
                if m:
                    return float(m.group(1))
    return None


def col_array(mtz, label):
    """Return a named MTZ column as a numpy float32 array."""
    return np.asarray(mtz.column_with_label(label), dtype=np.float32)


def find_fc_phi_labels(mtz):
    """Find the (amplitude, phase) label pair for calculated structure factors."""
    existing = {col.label for col in mtz.columns}
    for f_lbl, phi_lbl in (
        ('FC_ALL', 'PHIC_ALL'),
        ('FC',     'PHIC'),
        ('F_calc', 'PHI_calc'),
    ):
        if f_lbl in existing and phi_lbl in existing:
            return f_lbl, phi_lbl
    raise RuntimeError(
        f'No FC/PHIC pair found in MTZ. Available columns: {sorted(existing)}'
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ══════════════════════════════════════════════════════════════════════════════

def step1_random_atoms(tmpdir, natoms=None, cell=None):
    """Run randompdb.com → random.pdb (all B=20, to be randomised next).
    If natoms is given, use -N to fix the atom count; otherwise use -Vm."""
    if cell is None:
        cell = ('40', '40', '40', '90', '90', '90')
    if natoms is not None:
        extra = ['-N', str(natoms)]
    else:
        extra = ['-Vm', str(VM)]
    run([RANDOMPDB] + list(cell) + [SG, '-minD', str(MIND)] + extra, tmpdir)
    pdb = tmpdir / 'random.pdb'
    if not pdb.exists():
        raise RuntimeError('randompdb.com did not produce random.pdb')


def step2_randomise_bfac(tmpdir):
    """Replace uniform B=20 with per-atom log-normal B factors; write truth_full.pdb."""
    st = gemmi.read_structure(str(tmpdir / 'random.pdb'))
    rng = np.random.default_rng()
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    b = rng.lognormal(BFAC_MU, BFAC_SIGMA)
                    atom.b_iso = float(np.clip(b, BFAC_MIN, BFAC_MAX))
    st.write_pdb(str(tmpdir / 'truth_full.pdb'))


def step3_sfcalc(tmpdir):
    """
    gemmi sfcalc → truth.mtz with FC/PHIC columns.
    These are the 'observed' |Fo| in this simulation (we computed them from
    the full model, so we also know the true phases).
    """
    run(['gemmi', 'sfcalc', f'--dmin={DMIN}', '--to-mtz=truth.mtz', 'truth_full.pdb'],
        tmpdir)
    if not (tmpdir / 'truth.mtz').exists():
        raise RuntimeError('gemmi sfcalc did not produce truth.mtz')


def step4_build_refme_mtz(tmpdir):
    """
    Build refme.mtz for refmac input:
      F    = |FC_truth|   (the simulated 'observed' amplitudes)
      SIGF = 0.02 * |FC_truth|  (flat 2% uncertainty)
    """
    mtz = gemmi.read_mtz_file(str(tmpdir / 'truth.mtz'))
    f_lbl, _ = find_fc_phi_labels(mtz)

    H    = col_array(mtz, 'H').astype(np.float32)
    K    = col_array(mtz, 'K').astype(np.float32)
    L    = col_array(mtz, 'L').astype(np.float32)
    FC   = col_array(mtz, f_lbl)
    SIGF = np.maximum(0.02 * FC, 0.001)

    mtz_out = gemmi.Mtz()
    mtz_out.cell       = mtz.cell
    mtz_out.spacegroup = mtz.spacegroup

    ds0 = mtz_out.add_dataset('HKL_base')
    ds0.wavelength = 0.0
    ds1 = mtz_out.add_dataset('data')
    ds1.wavelength = 1.0

    mtz_out.add_column('H',    'H', dataset_id=0)
    mtz_out.add_column('K',    'H', dataset_id=0)
    mtz_out.add_column('L',    'H', dataset_id=0)
    mtz_out.add_column('F',    'F', dataset_id=1)
    mtz_out.add_column('SIGF', 'Q', dataset_id=1)

    data = np.column_stack([H, K, L, FC, SIGF]).astype(np.float32)
    mtz_out.set_data(data)
    mtz_out.write_to_file(str(tmpdir / 'refme.mtz'))


def _write_partial(st_full, tmpdir, n_remove, rng):
    """
    Delete n_remove random residues (WAT atoms) from a clone of st_full.
    Write starthere.pdb. Returns the list of removed residue indices (sorted).
    """
    st = st_full.clone()
    chain = st[0][0]
    n_total = len(chain)
    n_remove = min(n_remove, n_total - 1)   # keep at least 1 atom
    removed = sorted(
        int(i) for i in rng.choice(n_total, size=n_remove, replace=False)
    )
    for idx in reversed(removed):           # delete high-index first
        del chain[idx]
    st.write_pdb(str(tmpdir / 'starthere.pdb'))
    return removed


def step5_6_delete_and_refine(tmpdir, nmissing=None):
    """
    Delete nmissing atoms at random (or DELETE_FRAC of total if nmissing is None),
    then run 5 cycles of refmac.
    Returns (removed_indices, rwork, n_atoms_total).
    """
    st_full = gemmi.read_structure(str(tmpdir / 'truth_full.pdb'))
    n_atoms = sum(1 for chain in st_full[0] for res in chain for _ in res)
    if nmissing is not None:
        n_remove = min(int(nmissing), n_atoms - 1)
    else:
        n_remove = max(1, round(n_atoms * DELETE_FRAC))

    rng     = np.random.default_rng()
    removed = _write_partial(st_full, tmpdir, n_remove, rng)

    run([CONVERGE, 'starthere.pdb', 'refme.mtz', 'NCYC=5', 'noconverge'], tmpdir)

    rwork = parse_rwork(str(tmpdir / 'refmacout.pdb'))
    if rwork is None:
        raise RuntimeError('Could not parse Rwork from refmacout.pdb')

    return removed, rwork, n_atoms


def step7_maps_to_ccp4(tmpdir, outdir):
    """
    Write four CCP4 map files from refmacout.mtz and truth.mtz.
    Returns the grid shape (tuple) for the metadata.
    """
    def write_map(mtz_path, f_lbl, phi_lbl, out_path):
        mtz  = gemmi.read_mtz_file(str(mtz_path))
        grid = mtz.transform_f_phi_to_map(f_lbl, phi_lbl, sample_rate=SAMPLE_RATE)
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header()
        ccp4.write_ccp4_map(str(out_path))
        return grid.shape

    refmac_mtz = tmpdir / 'refmacout.mtz'
    truth_mtz  = tmpdir / 'truth.mtz'
    mtz_truth  = gemmi.read_mtz_file(str(truth_mtz))
    fc_lbl, phi_lbl = find_fc_phi_labels(mtz_truth)

    shape = write_map(refmac_mtz, 'FWT',    'PHWT',    outdir / '2fofc.map')
    write_map(refmac_mtz, 'DELFWT', 'PHDELWT', outdir / 'fofc.map')
    write_map(refmac_mtz, 'FC',     'PHIC',    outdir / 'fc.map')
    write_map(truth_mtz,  fc_lbl,   phi_lbl,   outdir / 'truth.map')
    return shape


# ══════════════════════════════════════════════════════════════════════════════
# Full sample pipeline
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FILES = {'truth.map', '2fofc.map', 'fofc.map', 'fc.map', 'metadata.json'}

def generate_sample(sample_idx, outdir_root, natoms=None, cell=None, nmissing=None):
    outdir = Path(outdir_root) / f'sample_{sample_idx:05d}'
    if outdir.exists() and REQUIRED_FILES.issubset({f.name for f in outdir.iterdir()}):
        log.info('[%05d] already complete, skipping', sample_idx)
        return str(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f'cnn_{sample_idx:05d}_') as tmp:
        tmpdir = Path(tmp)

        log.info('[%05d] step 1/7 generating random atoms ...', sample_idx)
        step1_random_atoms(tmpdir, natoms=natoms, cell=cell)

        log.info('[%05d] step 2/7 randomising B factors ...', sample_idx)
        step2_randomise_bfac(tmpdir)

        log.info('[%05d] step 3/7 computing truth structure factors ...', sample_idx)
        step3_sfcalc(tmpdir)

        log.info('[%05d] step 4/7 building refme.mtz ...', sample_idx)
        step4_build_refme_mtz(tmpdir)

        if nmissing is not None:
            log.info('[%05d] steps 5+6 deleting %d atom(s), running refmac ...', sample_idx, nmissing)
        else:
            log.info('[%05d] steps 5+6 deleting %.0f%% of atoms, running refmac ...',
                     sample_idx, DELETE_FRAC * 100)
        removed, rwork, n_atoms = step5_6_delete_and_refine(tmpdir, nmissing=nmissing)
        n_partial = n_atoms - len(removed)

        log.info('[%05d] Rwork=%.3f  atoms full=%d partial=%d',
                 sample_idx, rwork, n_atoms, n_partial)

        log.info('[%05d] step 7/7 writing CCP4 maps ...', sample_idx)
        grid_shape = step7_maps_to_ccp4(tmpdir, outdir)

    meta = {
        'n_atoms_full':         n_atoms,
        'n_atoms_partial':      n_partial,
        'deletion_fraction':    round(len(removed) / n_atoms, 4),
        'rwork':                round(rwork, 4),
        'removed_atom_indices': removed,
        'cell':                 [float(v) for v in (cell or ('40','40','40','90','90','90'))],
        'dmin':                 DMIN,
        'grid_shape':           list(grid_shape),
    }
    (outdir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    log.info('[%05d] done → %s', sample_idx, outdir)
    return str(outdir)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate CNN training data: 4 CCP4 map files per sample.'
    )
    parser.add_argument('--nsamples', type=int, default=10,
                        help='Number of training samples to generate (default: 10)')
    parser.add_argument('--outdir',   default='./data',
                        help='Root output directory (default: ./data)')
    parser.add_argument('--workers',   type=int, default=1,
                        help='Max concurrent srun jobs (default: 1 = run locally without srun)')
    parser.add_argument('--partition', default=None,
                        help='SLURM partition name for srun jobs (default: none)')
    parser.add_argument('--start',    type=int, default=0,
                        help='Starting sample index, useful for resuming (default: 0)')
    parser.add_argument('--natoms',   type=int, default=None,
                        help='Fix number of atoms via -N (default: use -Vm)')
    parser.add_argument('--nmissing', type=int, default=None,
                        help='Fix number of deleted atoms (default: DELETE_FRAC * n_atoms)')
    parser.add_argument('--cell-size', type=float, default=40.0,
                        help='Cubic cell edge in Å (default: 40)')
    parser.add_argument('--verbose',  action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    cs = str(int(args.cell_size)) if args.cell_size == int(args.cell_size) else str(args.cell_size)
    cell = (cs, cs, cs, '90', '90', '90')

    for script in (RANDOMPDB, CONVERGE):
        if not script.exists():
            sys.exit(f'ERROR: required script not found: {script}')

    # Smoke-test gemmi and key CCP4 programs
    try:
        import gemmi as _g
        log.debug('gemmi version: %s', _g.__version__)
    except ImportError:
        sys.exit('ERROR: gemmi Python package not found (pip install gemmi)')

    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    indices = range(args.start, args.start + args.nsamples)

    if args.workers > 1:
        srun_extra = ['--partition=' + args.partition] if args.partition else []

        def submit_one(i):
            """Submit a single-sample job via srun and wait for it to finish."""
            cmd = (
                ['srun', '--ntasks=1', '--export=ALL'] + srun_extra
                + ['ccp4-python', str(Path(__file__).resolve()),
                   '--nsamples', '1',
                   '--start',    str(i),
                   '--outdir',   str(outdir)]
                + (['--natoms', str(args.natoms)] if args.natoms is not None else [])
                + (['--nmissing', str(args.nmissing)] if args.nmissing is not None else [])
                + (['--cell-size', str(args.cell_size)] if args.cell_size != 40.0 else [])
                + (['--verbose'] if args.verbose else [])
            )
            log.info('Submitting sample %05d via srun ...', i)
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                err = (result.stderr or result.stdout or b'').decode(errors='replace')
                log.error('Sample %05d FAILED (srun exit %d):\n%s',
                          i, result.returncode, err)
            else:
                # Echo the worker's log output at debug level
                if args.verbose and result.stdout:
                    for line in result.stdout.decode(errors='replace').splitlines():
                        log.debug('[srun %05d] %s', i, line)
                log.info('Sample %05d done', i)

        # ThreadPoolExecutor: each thread blocks on one srun call.
        # SLURM queues any excess submissions, so --workers controls how many
        # srun processes are alive at once (not just submitted).
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(submit_one, i): i for i in indices}
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    log.error('Sample %05d FAILED: %s', i, exc)
    else:
        for i in indices:
            try:
                generate_sample(i, outdir, natoms=args.natoms, cell=cell, nmissing=args.nmissing)
            except Exception as exc:
                log.error('Sample %05d FAILED: %s', i, exc)


if __name__ == '__main__':
    main()
