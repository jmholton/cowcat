#!/usr/bin/env python3
"""
generate_data.py  –  Training data generation for CNN electron density reconstruction.

For each sample:
  1. randompdb.com places random O atoms in a P1 40×40×40 Å cell
  2. B factors are randomised (log-normal, 5–120 Å²)
  3. gemmi sfcalc computes structure factors for the FULL model  →  truth.mtz
  4. A random subset of atoms is deleted                         →  partial.pdb
  5. gemmi sfcalc computes structure factors for partial model   →  partial.mtz
  6. scaleit scales FC_truth to FC_partial → Fobs_scaled, scale_k, scale_B
  7. Unweighted map coefficients computed directly (no refmac):
       FWT    = 2|Fo| - |Fc|   (Fo = Fobs_scaled)
       DELFWT = |Fo| - |Fc|
       phases = PHIC_partial
  8. Maps exported as CCP4 .map files:
       truth.map   – ground-truth Fo density (FC/PHIC of full model)
       2fofc.map   – unweighted 2Fo-Fc  (FWT/PHWT)
       fofc.map    – Fo-Fc difference   (DELFWT/PHDELWT)
       fc.map      – Fc density         (FC/PHIC of partial model)
       metadata.json

Usage:
    python generate_data.py --nsamples 500 --outdir ./data --workers 4
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gemmi
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
RANDOMPDB  = SCRIPT_DIR / 'randompdb.com'

# ── fixed crystallographic parameters ─────────────────────────────────────────
SG          = 'P1'
DMIN        = 2.0    # resolution cutoff (Å)
MIND        = 0      # minimum inter-atom distance; 0 = no constraint (fast)
VM          = 2.4    # Matthews coefficient
SAMPLE_RATE = 3.0    # map oversampling; at 2 Å → ~0.67 Å/voxel

# ── B factor distribution (log-normal, matches protein atom statistics) ────────
BFAC_MU    = np.log(20.0)
BFAC_SIGMA = 0.7
BFAC_MIN   = 5.0
BFAC_MAX   = 120.0

# ── deletion fraction ──────────────────────────────────────────────────────────
DELETE_FRAC = 0.20   # fraction of atoms to remove

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
    """Run randompdb.com → random.pdb (all B=20, to be randomised next)."""
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


def step3_sfcalc_full(tmpdir):
    """gemmi sfcalc on full model → truth.mtz with FC/PHIC columns."""
    run(['gemmi', 'sfcalc', f'--dmin={DMIN}', '--to-mtz=truth.mtz', 'truth_full.pdb'],
        tmpdir)
    if not (tmpdir / 'truth.mtz').exists():
        raise RuntimeError('gemmi sfcalc did not produce truth.mtz')


def step4_delete_atoms(tmpdir, nmissing=None):
    """
    Delete nmissing atoms at random from truth_full.pdb → partial.pdb.
    Returns (removed_indices, n_atoms_total).
    """
    st_full = gemmi.read_structure(str(tmpdir / 'truth_full.pdb'))
    chain = st_full[0][0]
    n_atoms = len(chain)

    if nmissing is not None:
        n_remove = min(int(nmissing), n_atoms - 1)
    else:
        n_remove = max(1, round(n_atoms * DELETE_FRAC))

    rng = np.random.default_rng()
    removed = sorted(int(i) for i in rng.choice(n_atoms, size=n_remove, replace=False))
    for idx in reversed(removed):
        del chain[idx]
    st_full.write_pdb(str(tmpdir / 'partial.pdb'))
    return removed, n_atoms


def step4_partial_occupancy(tmpdir, nmissing=None):
    """
    Set nmissing atoms to random occupancies drawn from Uniform(0, 1); write partial.pdb.
    All atoms are retained; the selected atoms have reduced occupancy so their
    contribution to Fc is fractional.  Returns (selected_indices, occ_values, n_atoms_total).
    """
    st_full = gemmi.read_structure(str(tmpdir / 'truth_full.pdb'))
    chain = st_full[0][0]
    n_atoms = len(chain)

    if nmissing is not None:
        n_partial = min(int(nmissing), n_atoms)
    else:
        n_partial = max(1, round(n_atoms * DELETE_FRAC))

    rng = np.random.default_rng()
    selected = sorted(int(i) for i in rng.choice(n_atoms, size=n_partial, replace=False))
    occs = rng.uniform(0.0, 1.0, size=n_partial).tolist()

    occ_map = {idx: occ for idx, occ in zip(selected, occs)}
    for res_idx, residue in enumerate(chain):
        if res_idx in occ_map:
            for atom in residue:
                atom.occ = float(occ_map[res_idx])

    st_full.write_pdb(str(tmpdir / 'partial.pdb'))
    return selected, [round(o, 4) for o in occs], n_atoms


def step5_sfcalc_partial(tmpdir):
    """gemmi sfcalc on partial model → partial.mtz with FC/PHIC columns."""
    run(['gemmi', 'sfcalc', f'--dmin={DMIN}', '--to-mtz=partial.mtz', 'partial.pdb'],
        tmpdir)
    if not (tmpdir / 'partial.mtz').exists():
        raise RuntimeError('gemmi sfcalc did not produce partial.mtz')


def step5b_scale_ftrue(tmpdir):
    """
    Scale FC_truth to FC_partial using scaleit (isotropic B refinement).

    In real crystallography the overall scale and temperature factor of the
    observed data relative to Fcalc are unknown.  Here we simulate this by
    finding k and B such that:

        Fobs_scaled(hkl) = k * exp(-B / (4 * d(hkl)^2)) * FC_truth(hkl)

    best matches FC_partial in a least-squares sense.  The scaled amplitudes
    are returned as the simulated Fobs, and k / B are stored as metadata so
    the CNN can be trained to predict them.

    Writes scaleit_input.mtz and scaleit_output.mtz to tmpdir.
    Returns (Fobs_scaled_array, scale_k, scale_B).
    """
    mtz_t = gemmi.read_mtz_file(str(tmpdir / 'truth.mtz'))
    mtz_p = gemmi.read_mtz_file(str(tmpdir / 'partial.mtz'))
    fc_lbl_t, _ = find_fc_phi_labels(mtz_t)
    fc_lbl_p, _ = find_fc_phi_labels(mtz_p)

    H  = col_array(mtz_p, 'H').astype(np.int32)
    K  = col_array(mtz_p, 'K').astype(np.int32)
    L  = col_array(mtz_p, 'L').astype(np.int32)
    Fc = col_array(mtz_p, fc_lbl_p)   # FC_partial  → FP  (reference, not scaled)
    Ft = col_array(mtz_t, fc_lbl_t)   # FC_truth    → FPH1 (scaled to match Fc)

    # Build combined MTZ for scaleit: FP=Fcalc, FPH1=Ftrue, synthetic SIGF=F/30
    mtz_cad = gemmi.Mtz()
    mtz_cad.cell       = mtz_p.cell
    mtz_cad.spacegroup = mtz_p.spacegroup
    ds0 = mtz_cad.add_dataset('HKL_base'); ds0.wavelength = 0.0
    ds1 = mtz_cad.add_dataset('data');     ds1.wavelength = 1.0
    for lbl in ('H', 'K', 'L'):
        mtz_cad.add_column(lbl, 'H', dataset_id=0)
    mtz_cad.add_column('Fcalc',    'F', dataset_id=1)
    mtz_cad.add_column('SIGFcalc', 'Q', dataset_id=1)
    mtz_cad.add_column('Ftrue',    'F', dataset_id=1)
    mtz_cad.add_column('SIGFtrue', 'Q', dataset_id=1)
    mtz_cad.set_data(np.column_stack([
        H, K, L,
        Fc, np.maximum(Fc / 30.0, 1e-6),
        Ft, np.maximum(Ft / 30.0, 1e-6),
    ]).astype(np.float32))

    cad_path = tmpdir / 'scaleit_input.mtz'
    out_path  = tmpdir / 'scaleit_output.mtz'
    mtz_cad.write_to_file(str(cad_path))

    scaleit_stdin = (
        'TITLE Scale Ftrue to Fcalc\n'
        f'RESO {DMIN}\n'
        'NOWT\n'
        'refine isotropic\n'
        'LABIN FP=Fcalc SIGFP=SIGFcalc FPH1=Ftrue SIGFPH1=SIGFtrue\n'
        'CONV ABS 0.0001 TOLR 0.000000001 NCYC 4\n'
        'END\n'
    )
    result = subprocess.run(
        ['scaleit', 'HKLIN', str(cad_path), 'HKLOUT', str(out_path)],
        input=scaleit_stdin.encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(tmpdir),
    )
    log_text = result.stdout.decode(errors='replace')
    if result.returncode != 0:
        raise RuntimeError(
            f'scaleit failed:\n{log_text}\n'
            + result.stderr.decode(errors='replace')
        )

    # Parse scale (k) and isotropic B from scaleit log — mirrors diff.com awk patterns:
    #   scale: awk '$1=="Derivative" && !/itle/{print $3}' → field 3 of last Derivative line
    #   B:     awk '/equivalent iso/{print $NF}'           → last field of equiv-iso line
    scale_k_raw = None
    scale_B_raw = None
    for line in log_text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'Derivative' and 'itle' not in line and len(fields) >= 3:
            scale_k_raw = fields[2]           # $3 in awk (0-indexed: fields[2])
        if 'equivalent iso' in line and fields:
            scale_B_raw = fields[-1]          # $NF
    scale_m = scale_k_raw is not None
    b_m     = scale_B_raw is not None
    if not scale_m or not b_m:
        # log the output so we can diagnose parsing mismatch later
        (tmpdir / 'scaleit.log').write_text(log_text)
        raise RuntimeError(
            f'Could not parse scale/B from scaleit output '
            f'(written to {tmpdir}/scaleit.log):\n{log_text[:3000]}'
        )
    scale_k = float(scale_k_raw)
    scale_B = float(scale_B_raw)

    # Apply the fitted scale to Ftrue: Fobs_scaled = k * exp(-B / (4*d^2)) * Ftrue
    # scaleit convention: derivative_scaled = k * exp(-B * (sin θ/λ)^2) * derivative
    #                     sin θ/λ = 1/(2d)  →  (sin θ/λ)^2 = 1/(4d^2)
    cell = mtz_p.cell
    d_vals = np.array([cell.calculate_d([int(h), int(k), int(l)]) for h, k, l in zip(H, K, L)],
                      dtype=np.float32)
    Fobs_scaled = (scale_k * np.exp(-scale_B / (4.0 * d_vals**2)) * Ft).astype(np.float32)

    return Fobs_scaled, scale_k, scale_B


def step6_build_maps(tmpdir, outdir, Fo_scaled=None):
    """
    Compute unweighted map coefficients and cross-Patterson directly (no refmac).

    Fo     = Fo_scaled if provided, else |FC_truth| (raw, unscaled)
    Fc     = |FC_partial|
    PHIc   = PHIC_partial

    2FoFc = 2*Fo - Fc   (negative F = phase flip, not clamped)
    FoFc  = Fo - Fc
    all three use PHIc (phases of the partial model)

    Cross-Patterson = IFFT[ FFT(FoFc_map) * conj(FFT(Fc_map)) ]
    Saved as crossp.npy alongside the CCP4 maps.

    Returns grid shape tuple.
    """
    mtz_t = gemmi.read_mtz_file(str(tmpdir / 'truth.mtz'))
    mtz_p = gemmi.read_mtz_file(str(tmpdir / 'partial.mtz'))

    fc_lbl_t, phi_lbl_t = find_fc_phi_labels(mtz_t)
    fc_lbl_p, phi_lbl_p = find_fc_phi_labels(mtz_p)

    def hkl_index(mtz):
        H = col_array(mtz, 'H').astype(np.int32)
        K = col_array(mtz, 'K').astype(np.int32)
        L = col_array(mtz, 'L').astype(np.int32)
        return H, K, L

    H_t, K_t, L_t = hkl_index(mtz_t)
    H_p, K_p, L_p = hkl_index(mtz_p)

    Fo   = Fo_scaled if Fo_scaled is not None else col_array(mtz_t, fc_lbl_t)
    Fc   = col_array(mtz_p, fc_lbl_p)
    PHIc = col_array(mtz_p, phi_lbl_p)

    # In P1 with the same cell and dmin, both sfcalc runs yield the same HKL
    # set in the same order. Assert this rather than silently misaligning.
    if not (np.array_equal(H_t, H_p) and
            np.array_equal(K_t, K_p) and
            np.array_equal(L_t, L_p)):
        raise RuntimeError(
            'HKL mismatch between truth.mtz and partial.mtz — '
            f'truth has {len(H_t)} reflections, partial has {len(H_p)}'
        )

    # Build combined MTZ
    mtz_out = gemmi.Mtz()
    mtz_out.cell       = mtz_p.cell
    mtz_out.spacegroup = mtz_p.spacegroup

    ds0 = mtz_out.add_dataset('HKL_base')
    ds0.wavelength = 0.0
    ds1 = mtz_out.add_dataset('data')
    ds1.wavelength = 1.0

    for lbl in ('H', 'K', 'L'):
        mtz_out.add_column(lbl, 'H', dataset_id=0)
    mtz_out.add_column('2FoFc', 'F', dataset_id=1)
    mtz_out.add_column('FoFc',  'F', dataset_id=1)
    mtz_out.add_column('FC',    'F', dataset_id=1)
    mtz_out.add_column('PHIc',  'P', dataset_id=1)

    data = np.column_stack([
        H_p, K_p, L_p,
        2.0 * Fo - Fc, Fo - Fc, Fc, PHIc,
    ]).astype(np.float32)
    mtz_out.set_data(data)

    # Compute all grids in memory
    grid_2fofc = mtz_out.transform_f_phi_to_map('2FoFc', 'PHIc', sample_rate=SAMPLE_RATE)
    grid_fofc  = mtz_out.transform_f_phi_to_map('FoFc',  'PHIc', sample_rate=SAMPLE_RATE)
    grid_fc    = mtz_out.transform_f_phi_to_map('FC',    'PHIc', sample_rate=SAMPLE_RATE)
    grid_truth = mtz_t.transform_f_phi_to_map(fc_lbl_t, phi_lbl_t, sample_rate=SAMPLE_RATE)

    arr_fofc = np.array(grid_fofc, copy=False)
    arr_fc   = np.array(grid_fc,   copy=False)
    F_fofc   = np.fft.rfftn(arr_fofc)
    F_fc     = np.fft.rfftn(arr_fc)

    # Cross-Patterson: IFFT[ FFT(FoFc) * conj(FFT(Fc)) ]
    crossp = np.fft.irfftn(
        F_fofc * np.conj(F_fc), s=arr_fofc.shape,
    ).real.astype(np.float32)
    np.save(str(outdir / 'crossp.npy'), crossp)

    # Difference Patterson: IFFT[ |FFT(FoFc)|^2 ] — autocorrelation of FoFc map
    diffp = np.fft.irfftn(
        np.abs(F_fofc) ** 2, s=arr_fofc.shape,
    ).real.astype(np.float32)
    np.save(str(outdir / 'diffp.npy'), diffp)

    def write_map(grid, out_path):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header()
        ccp4.write_ccp4_map(str(out_path))

    write_map(grid_2fofc, outdir / '2fofc.map')
    write_map(grid_fofc,  outdir / 'fofc.map')
    write_map(grid_fc,    outdir / 'fc.map')
    write_map(grid_truth, outdir / 'truth.map')
    return grid_2fofc.shape


# ══════════════════════════════════════════════════════════════════════════════
# Full sample pipeline
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FILES = {'truth.map', '2fofc.map', 'fofc.map', 'fc.map', 'crossp.npy', 'diffp.npy', 'metadata.json'}

def generate_sample(sample_idx, outdir_root, natoms=None, cell=None, nmissing=None, partial_occ=False):
    outdir = Path(outdir_root) / f'sample_{sample_idx:05d}'
    if outdir.exists() and REQUIRED_FILES.issubset({f.name for f in outdir.iterdir()}):
        log.info('[%05d] already complete, skipping', sample_idx)
        return str(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f'cnn_{sample_idx:05d}_') as tmp:
        tmpdir = Path(tmp)

        log.info('[%05d] step 1/6 generating random atoms ...', sample_idx)
        step1_random_atoms(tmpdir, natoms=natoms, cell=cell)

        log.info('[%05d] step 2/6 randomising B factors ...', sample_idx)
        step2_randomise_bfac(tmpdir)

        log.info('[%05d] step 3/6 sfcalc full model ...', sample_idx)
        step3_sfcalc_full(tmpdir)

        if partial_occ:
            log.info('[%05d] step 4/6 setting partial occupancies ...', sample_idx)
            selected, occs, n_atoms = step4_partial_occupancy(tmpdir, nmissing=nmissing)
            n_partial = n_atoms
            log.info('[%05d] atoms full=%d  partial-occ=%d', sample_idx, n_atoms, len(selected))
        else:
            log.info('[%05d] step 4/6 deleting atoms ...', sample_idx)
            removed, n_atoms = step4_delete_atoms(tmpdir, nmissing=nmissing)
            n_partial = n_atoms - len(removed)
            log.info('[%05d] atoms full=%d partial=%d', sample_idx, n_atoms, n_partial)

        log.info('[%05d] step 5/6 sfcalc partial model ...', sample_idx)
        step5_sfcalc_partial(tmpdir)

        log.info('[%05d] step 5b/6 scaling Ftrue to Fcalc with scaleit ...', sample_idx)
        Fobs_scaled, scale_k, scale_B = step5b_scale_ftrue(tmpdir)
        log.info('[%05d] scale_k=%.6f  scale_B=%.4f', sample_idx, scale_k, scale_B)

        log.info('[%05d] step 6/6 building maps ...', sample_idx)
        grid_shape = step6_build_maps(tmpdir, outdir, Fo_scaled=Fobs_scaled)

    cell_list = [float(v) for v in (cell or ('40','40','40','90','90','90'))]
    if partial_occ:
        meta = {
            'n_atoms_full':             n_atoms,
            'n_atoms_partial':          n_partial,
            'partial_occ_mode':         True,
            'partial_occ_atom_indices': selected,
            'partial_occ_values':       occs,
            'scale_k':                  round(float(scale_k), 6),
            'scale_B':                  round(float(scale_B), 4),
            'cell':                     cell_list,
            'dmin':                     DMIN,
            'grid_shape':               list(grid_shape),
        }
    else:
        meta = {
            'n_atoms_full':         n_atoms,
            'n_atoms_partial':      n_partial,
            'deletion_fraction':    round(len(removed) / n_atoms, 4),
            'removed_atom_indices': removed,
            'scale_k':              round(float(scale_k), 6),
            'scale_B':              round(float(scale_B), 4),
            'cell':                 cell_list,
            'dmin':                 DMIN,
            'grid_shape':           list(grid_shape),
        }
    (outdir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    log.info('[%05d] done → %s', sample_idx, outdir)
    return str(outdir)


# ══════════════════════════════════════════════════════════════════════════════
# SLURM helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_array_spec(indices):
    """Convert a list of ints to a compact SLURM array spec, e.g. '0-9,15,20-29'."""
    if not indices:
        return ''
    s = sorted(set(indices))
    parts = []
    start = end = s[0]
    for n in s[1:]:
        if n == end + 1:
            end = n
        else:
            parts.append(f'{start}-{end}' if end > start else str(start))
            start = end = n
    parts.append(f'{start}-{end}' if end > start else str(start))
    return ','.join(parts)


def _pending_indices(outdir, all_indices):
    """Return indices whose output directory is not yet complete."""
    pending = []
    for i in all_indices:
        d = Path(outdir) / f'sample_{i:05d}'
        try:
            done = d.exists() and REQUIRED_FILES.issubset({f.name for f in d.iterdir()})
        except OSError:
            done = False
        if not done:
            pending.append(i)
    return pending


def _sbatch_array(script_path, outdir_abs, pending, partition, max_array,
                  natoms, nmissing, cell_size, verbose, partial_occ=False):
    """Submit a sbatch array job; return the SLURM job-id string."""
    array_spec = _make_array_spec(pending)
    if max_array and max_array > 0:
        array_spec += f'%{max_array}'

    extra = []
    if natoms   is not None: extra += [f'--natoms {natoms}']
    if nmissing is not None: extra += [f'--nmissing {nmissing}']
    if cell_size != 40.0:    extra += [f'--cell-size {cell_size}']
    if verbose:              extra += ['--verbose']
    if partial_occ:          extra += ['--partial-occ']
    extra_str = ' '.join(extra)

    lines = [
        '#!/bin/bash',
        '#SBATCH --job-name=cnn_gen',
        '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=1',
        '#SBATCH --export=ALL',
    ]
    if partition:
        lines.append(f'#SBATCH --partition={partition}')
    lines += [
        '',
        f'exec ccp4-python {script_path} \\',
        f'    --nsamples 1 --start $SLURM_ARRAY_TASK_ID \\',
        f'    --outdir {outdir_abs} {extra_str}',
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False,
                                     prefix='cnn_gen_', dir='/tmp') as f:
        f.write('\n'.join(lines) + '\n')
        script_file = f.name

    try:
        result = subprocess.run(
            ['sbatch', f'--array={array_spec}', script_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            sys.exit('sbatch failed:\n' + result.stderr.decode(errors='replace'))
        return result.stdout.decode().strip().split()[-1]
    finally:
        os.unlink(script_file)


def _wait_for_job(job_id, total, log_interval=30):
    """Poll squeue until the array job is gone; log progress periodically."""
    last_log = 0.0
    while True:
        sq = subprocess.run(
            ['squeue', '-j', job_id, '-h', '-o', '%T'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        states = sq.stdout.decode().split()
        if not states:
            break
        now = time.time()
        if now - last_log >= log_interval:
            n_run  = states.count('RUNNING')
            n_pend = states.count('PENDING')
            done   = total - len(states)
            log.info('Job %s: %d/%d done  running=%d  pending=%d',
                     job_id, done, total, n_run, n_pend)
            last_log = now
        time.sleep(10)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate CNN training data: 4 CCP4 map files per sample.'
    )
    parser.add_argument('--nsamples',    type=int, default=10,
                        help='Number of training samples to generate (default: 10)')
    parser.add_argument('--outdir',      default='./data',
                        help='Root output directory (default: ./data)')
    parser.add_argument('--partition',   default=None,
                        help='SLURM partition for sbatch array jobs')
    parser.add_argument('--max-array',   type=int, default=0,
                        help='Max simultaneous array tasks (0 = unlimited, default: 0)')
    parser.add_argument('--start',       type=int, default=0,
                        help='Starting sample index (default: 0)')
    parser.add_argument('--natoms',      type=int, default=None,
                        help='Fix number of atoms via -N (default: use -Vm)')
    parser.add_argument('--nmissing',    type=int, default=None,
                        help='Fix number of deleted atoms (default: DELETE_FRAC * n_atoms)')
    parser.add_argument('--cell-size',   type=float, default=40.0,
                        help='Cubic cell edge in Å (default: 40)')
    parser.add_argument('--verbose',     action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--partial-occ', action='store_true',
                        help='Partial occupancy mode: give nmissing atoms random occ in [0,1) '
                             'instead of deleting them')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    cs   = str(int(args.cell_size)) if args.cell_size == int(args.cell_size) else str(args.cell_size)
    cell = (cs, cs, cs, '90', '90', '90')

    if not RANDOMPDB.exists():
        sys.exit(f'ERROR: required script not found: {RANDOMPDB}')

    try:
        import gemmi as _g
        log.debug('gemmi version: %s', _g.__version__)
    except ImportError:
        sys.exit('ERROR: gemmi Python package not found (pip install gemmi)')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Single-sample mode: run directly (called by sbatch array tasks) ───────
    if args.nsamples == 1:
        i = args.start
        try:
            generate_sample(i, outdir_root=outdir, natoms=args.natoms,
                            cell=cell, nmissing=args.nmissing,
                            partial_occ=args.partial_occ)
            log.info('Done. ok=1  errors=0')
        except Exception as exc:
            log.error('Sample %05d FAILED: %s', i, exc)
            sys.exit(1)
        return

    # ── Multi-sample mode: submit sbatch array ────────────────────────────────
    all_indices = list(range(args.start, args.start + args.nsamples))
    pending     = _pending_indices(outdir, all_indices)
    skipped     = len(all_indices) - len(pending)

    if skipped:
        log.info('Skipping %d already-complete samples; %d to generate', skipped, len(pending))

    if not pending:
        log.info('All %d samples already complete.', len(all_indices))
        return

    script_path = str(Path(__file__).resolve())
    outdir_abs  = str(outdir.resolve())

    job_id = _sbatch_array(
        script_path, outdir_abs, pending, args.partition, args.max_array,
        args.natoms, args.nmissing, args.cell_size, args.verbose, args.partial_occ,
    )
    log.info('Submitted SLURM array job %s  (%d tasks)', job_id, len(pending))

    _wait_for_job(job_id, len(pending))

    ok     = sum(1 for i in pending
                 if (outdir / f'sample_{i:05d}' / 'metadata.json').exists())
    errors = len(pending) - ok
    log.info('Done. ok=%d  skipped=%d  errors=%d', ok, skipped, errors)


if __name__ == '__main__':
    main()
