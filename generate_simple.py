#!/usr/bin/env ccp4-python
"""
generate_simple.py — Simple N-O-atom training samples in a configurable unit cell.

Truth: N random O atoms in a unit cell (default: 45.9×40.7×30.1 Å P 21 21 21,
d_min=0.965 Å → 96×128×144 grid).
Partial: N_MISS atoms deleted (or modified via occupancy/B/xyz perturbations) →
starthere.pdb. Refmac refines the partial model; maps written as sample_NNNNN/.

Supports alternate-conformer clusters, partial occupancy, B-factor/xyz shifts,
and the --no-refmac path (sfcalc+scaleit).

Usage:
    ccp4-python generate_simple.py --nsamples 500 --outdir data/data_simple_n500
    ccp4-python generate_simple.py --submit --nsamples 1000 --seed 0 \\
        --outdir data/data_simple_s0 \\
        --partition lr6 --account pc_als831 --qos lr_normal
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gemmi
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

CELL        = (45.9, 40.7, 30.1, 90.0, 90.0, 90.0)   # matches 1AHO training data → 96×128×144
DMIN        = 0.965
N_ATOMS     = 20
N_MISS      = 1
SPACEGROUP  = 'P 21 21 21'
NCYC        = 20
SAMPLE_RATE = 3.0
MAX_ARRAY   = 1000

# B factor distribution constants (used by altconf and bfac_shift features)
BFAC_MU    = np.log(20.0)
BFAC_SIGMA = 0.7
BFAC_MIN   = 5.0
BFAC_MAX   = 120.0

REFMAC5 = Path(shutil.which('refmac5') or '/programs/ccp4-8.0/bin/refmac5')

_ALTLOC_LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# ── Atom placement ────────────────────────────────────────────────────────────

def place_atoms(n, rng, min_dist=2.5, b_range=(10.0, 30.0)):
    """Return list of (x, y, z, b_iso) for n O atoms in the unit cell."""
    a, b, c = CELL[:3]
    b_lo, b_hi = b_range
    positions = []
    for _ in range(n * 300):
        if len(positions) >= n:
            break
        x, y, z = rng.random() * a, rng.random() * b, rng.random() * c
        if positions:
            dists = [((x-px)**2 + (y-py)**2 + (z-pz)**2)**0.5
                     for px, py, pz, _ in positions]
            if min(dists) < min_dist:
                continue
        b_iso = b_lo if b_hi == b_lo else float(rng.uniform(b_lo, b_hi))
        positions.append((x, y, z, b_iso))
    return positions[:n]


def write_pdb(positions, out_path, occs=None):
    """Write PDB for a list of (x, y, z, b_iso) atoms.

    occs: optional list of per-atom occupancies (default 1.0 for all).
    Negative occupancies are valid — gemmi sfcalc subtracts the atomic
    scattering contribution, producing negative density in truth.map.
    """
    a, b, c, al, be, ga = CELL
    _sg = gemmi.find_spacegroup_by_name(SPACEGROUP)
    _z  = len(list(_sg.operations())) if _sg else 1
    _hm = _sg.hm if _sg else SPACEGROUP
    lines = [
        f'CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} {_hm:<11s}{_z:3d}\n'
    ]
    for i, (x, y, z, b_iso) in enumerate(positions):
        occ = occs[i] if occs is not None else 1.0
        lines.append(
            f'HETATM{i+1:5d}  O   HOH A{i+1:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}'
            f'{occ:6.2f}{b_iso:6.2f}          '
            f'  O\n'
        )
    lines.append('END\n')
    Path(out_path).write_text(''.join(lines))


# ── MTZ construction ──────────────────────────────────────────────────────────

def build_fobs_mtz(truth_sf_mtz, out_path, rng, freer_fraction=0.05):
    """FP = |FC_truth|, SIGFP = 0.02·FP, random FreeR_flag."""
    mtz  = gemmi.read_mtz_file(str(truth_sf_mtz))
    fc   = np.array(mtz.column_with_label('FC'),   dtype=np.float32)
    h    = np.array(mtz.column_with_label('H'),    dtype=np.int32)
    k    = np.array(mtz.column_with_label('K'),    dtype=np.int32)
    l    = np.array(mtz.column_with_label('L'),    dtype=np.int32)
    sigf = np.maximum(0.01, 0.02 * fc)
    free = (rng.random(len(h)) < freer_fraction).astype(np.float32)

    out = gemmi.Mtz()
    out.cell       = mtz.cell
    out.spacegroup = mtz.spacegroup
    out.add_dataset('HKL_base')
    for lbl in ('H', 'K', 'L'):
        out.add_column(lbl, 'H')
    out.add_dataset('data')
    out.add_column('FP',         'F')
    out.add_column('SIGFP',      'Q')
    out.add_column('FreeR_flag', 'I')
    out.set_data(np.column_stack([h, k, l, fc, sigf, free]))
    out.write_to_file(str(out_path))


def mtz_to_ccp4(mtz_path, f_col, phi_col, out_path):
    mtz  = gemmi.read_mtz_file(str(mtz_path))
    grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=SAMPLE_RATE)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(out_path))


def _find_fc_phi_labels(mtz):
    """Find (amplitude, phase) label pair for calculated structure factors."""
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


def _col_array(mtz, label):
    """Return a named MTZ column as a numpy float32 array."""
    return np.asarray(mtz.column_with_label(label), dtype=np.float32)


# ── Refmac ────────────────────────────────────────────────────────────────────

def run_refmac(starthere_pdb, fobs_mtz, tmpdir, verbose=False):
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)
    kw  = b'LABIN FP=FP SIGFP=SIGFP FREE=FreeR_flag\n'
    kw += b'LABOUT FC=FC PHIC=PHIC FWT=FWT PHWT=PHWT'
    kw += b' DELFWT=DELFWT PHDELWT=PHDELWT\n'
    kw += f'NCYC {NCYC}\n'.encode()
    kw += b'REFI TYPE REST RESI MLKF\n'
    kw += b'SCALE LSSC ANISO BULK\n'
    kw += b'SOLVENT YES\n'
    kw += b'MAKE HYDR NO\n'

    out_mtz = tmpdir / 'refmacout.mtz'
    out_pdb = tmpdir / 'refmacout.pdb'
    r = subprocess.run(
        [str(REFMAC5),
         'XYZIN',  str(starthere_pdb.resolve()),
         'XYZOUT', str(out_pdb),
         'HKLIN',  str(fobs_mtz.resolve()),
         'HKLOUT', str(out_mtz),
         'LIBOUT', str(tmpdir / '_refmac.lib')],
        input=kw, capture_output=True, cwd=str(tmpdir),
    )
    log = r.stdout.decode(errors='replace')
    (tmpdir / 'refmac.log').write_text(log)
    if verbose:
        print(log[-2000:])
    rwork = rfree = None
    for line in reversed(log.splitlines()):
        if 'R factor' in line and 'Rfree' in line:
            try:
                parts = line.split()
                rwork, rfree = float(parts[-2]), float(parts[-1])
            except Exception:
                pass
            break
    return rwork, rfree, out_mtz if out_mtz.exists() else None


# ── Scaleit (no-refmac path) ──────────────────────────────────────────────────

def _sfcalc_partial(tmpdir, partial_pdb):
    """Run gemmi sfcalc on partial_pdb → partial.mtz."""
    r = subprocess.run(
        ['gemmi', 'sfcalc', f'--dmin={DMIN}', '--to-mtz=partial.mtz', str(partial_pdb.name)],
        capture_output=True, cwd=str(tmpdir),
    )
    if r.returncode != 0:
        raise RuntimeError(f'sfcalc (partial) failed: {r.stderr.decode()[-300:]}')
    if not (tmpdir / 'partial.mtz').exists():
        raise RuntimeError('gemmi sfcalc did not produce partial.mtz')


def _scale_ftrue(tmpdir, truth_sf_mtz):
    """
    Scale FC_truth to FC_partial with scaleit → (Fobs_scaled, scale_k, scale_B, R_factor).
    Uses the same isotropic-B protocol as generate_data.py:step5b_scale_ftrue.
    """
    mtz_t = gemmi.read_mtz_file(str(truth_sf_mtz))
    mtz_p = gemmi.read_mtz_file(str(tmpdir / 'partial.mtz'))
    fc_lbl_t, _ = _find_fc_phi_labels(mtz_t)
    fc_lbl_p, _ = _find_fc_phi_labels(mtz_p)

    H  = _col_array(mtz_p, 'H').astype(np.int32)
    K  = _col_array(mtz_p, 'K').astype(np.int32)
    L  = _col_array(mtz_p, 'L').astype(np.int32)
    Fc = _col_array(mtz_p, fc_lbl_p)
    Ft = _col_array(mtz_t, fc_lbl_t)

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

    scale_k_raw = scale_B_raw = None
    for line in log_text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'Derivative' and 'itle' not in line and len(fields) >= 3:
            scale_k_raw = fields[2]
        if 'equivalent iso' in line and fields:
            scale_B_raw = fields[-1]
    if scale_k_raw is None or scale_B_raw is None:
        (tmpdir / 'scaleit.log').write_text(log_text)
        raise RuntimeError(
            f'Could not parse scale/B from scaleit output '
            f'(written to {tmpdir}/scaleit.log):\n{log_text[:3000]}'
        )
    scale_k = float(scale_k_raw)
    scale_B = float(scale_B_raw)

    cell = mtz_p.cell
    d_vals = np.array([cell.calculate_d([int(h), int(k), int(l)]) for h, k, l in zip(H, K, L)],
                      dtype=np.float32)
    Fobs_scaled = (scale_k * np.exp(-scale_B / (4.0 * d_vals**2)) * Ft).astype(np.float32)
    R_factor    = float(np.sum(np.abs(Fc - Fobs_scaled)) / np.sum(np.abs(Fc)))
    return Fobs_scaled, scale_k, scale_B, R_factor


def _build_maps_no_refmac(tmpdir, truth_sf_mtz, partial_pdb, sample_dir):
    """
    No-refmac path: sfcalc partial → scaleit → unweighted 2Fo-Fc/Fo-Fc maps.
    Returns (scale_k, scale_B, R_factor, grid_shape).
    """
    _sfcalc_partial(tmpdir, partial_pdb)
    Fobs_scaled, scale_k, scale_B, R_factor = _scale_ftrue(tmpdir, truth_sf_mtz)

    mtz_t = gemmi.read_mtz_file(str(truth_sf_mtz))
    mtz_p = gemmi.read_mtz_file(str(tmpdir / 'partial.mtz'))
    fc_lbl_t, phi_lbl_t = _find_fc_phi_labels(mtz_t)
    fc_lbl_p, phi_lbl_p = _find_fc_phi_labels(mtz_p)

    H_t = _col_array(mtz_t, 'H').astype(np.int32)
    K_t = _col_array(mtz_t, 'K').astype(np.int32)
    L_t = _col_array(mtz_t, 'L').astype(np.int32)
    H_p = _col_array(mtz_p, 'H').astype(np.int32)
    K_p = _col_array(mtz_p, 'K').astype(np.int32)
    L_p = _col_array(mtz_p, 'L').astype(np.int32)
    Fc   = _col_array(mtz_p, fc_lbl_p)
    PHIc = _col_array(mtz_p, phi_lbl_p)

    if not (np.array_equal(H_t, H_p) and np.array_equal(K_t, K_p) and np.array_equal(L_t, L_p)):
        raise RuntimeError('HKL mismatch between truth.mtz and partial.mtz')

    mtz_out = gemmi.Mtz()
    mtz_out.cell       = mtz_p.cell
    mtz_out.spacegroup = mtz_p.spacegroup
    ds0 = mtz_out.add_dataset('HKL_base'); ds0.wavelength = 0.0
    ds1 = mtz_out.add_dataset('data');     ds1.wavelength = 1.0
    for lbl in ('H', 'K', 'L'):
        mtz_out.add_column(lbl, 'H', dataset_id=0)
    mtz_out.add_column('FWT',      'F', dataset_id=1)
    mtz_out.add_column('DELFWT',   'F', dataset_id=1)
    mtz_out.add_column('FC',       'F', dataset_id=1)
    mtz_out.add_column('PHIc',     'P', dataset_id=1)
    mtz_out.set_data(np.column_stack([
        H_p, K_p, L_p,
        2.0 * Fobs_scaled - Fc,
        Fobs_scaled - Fc,
        Fc, PHIc,
    ]).astype(np.float32))

    grid_2fofc = mtz_out.transform_f_phi_to_map('FWT',    'PHIc', sample_rate=SAMPLE_RATE)
    grid_fofc  = mtz_out.transform_f_phi_to_map('DELFWT', 'PHIc', sample_rate=SAMPLE_RATE)
    grid_fc    = mtz_out.transform_f_phi_to_map('FC',     'PHIc', sample_rate=SAMPLE_RATE)
    grid_truth = mtz_t.transform_f_phi_to_map(fc_lbl_t,  phi_lbl_t, sample_rate=SAMPLE_RATE)

    def write_map(grid, out):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header()
        ccp4.write_ccp4_map(str(out))

    write_map(grid_2fofc, sample_dir / '2fofc.map')
    write_map(grid_fofc,  sample_dir / 'fofc.map')
    write_map(grid_fc,    sample_dir / 'fc.map')
    write_map(grid_truth, sample_dir / 'truth.map')
    return scale_k, scale_B, R_factor, grid_2fofc.shape


# ── Alternate-conformer cluster insertion ──────────────────────────────────────

def _insert_altconf_clusters(positions, rng, n_altconfs, altconf_rms, n_clusters, all_clusters):
    """
    Given list of (x, y, z, b_iso) positions, replace n_clusters of them with
    alt-conf clusters.

    Returns:
      truth_positions  — updated list of (x, y, z, b_iso, occ, altloc) tuples
                         (non-cluster atoms have occ=1.0, altloc='')
      partial_positions — list of (x, y, z, b_iso) for the partial model
                         (cluster atoms replaced by centroid)
      cluster_meta     — list of dicts (one per cluster)
    """
    n = len(positions)
    if all_clusters:
        n_clusters = n
    n_clusters = min(n_clusters, n)

    target_idx = sorted(rng.choice(n, size=n_clusters, replace=False).tolist())
    target_set = set(target_idx)

    sigma = altconf_rms / np.sqrt(3.0)

    # Build truth positions: non-cluster atoms as-is, cluster atoms expanded
    truth_positions = []   # (x, y, z, b, occ, altloc)
    partial_positions = []
    cluster_meta = []

    for i, (x, y, z, b) in enumerate(positions):
        if i not in target_set:
            truth_positions.append((x, y, z, b, 1.0, ''))
            partial_positions.append((x, y, z, b))
        else:
            center = np.array([x, y, z])
            displacements = rng.normal(0.0, sigma, size=(n_altconfs, 3))
            pos_arr  = center + displacements
            b_arr    = np.clip(
                np.exp(rng.normal(BFAC_MU, BFAC_SIGMA, size=n_altconfs)), BFAC_MIN, BFAC_MAX
            )
            total_occ = float(rng.uniform(0.5, 1.5))
            props     = rng.dirichlet(np.ones(n_altconfs))
            occs      = (total_occ * props).tolist()
            centroid  = pos_arr.mean(axis=0)
            part_occ  = float(np.clip(total_occ * np.exp(rng.normal(0.0, 0.2)), 0.05, 3.0))
            part_b    = float(np.clip(np.exp(rng.normal(BFAC_MU, BFAC_SIGMA)), BFAC_MIN, BFAC_MAX))

            for k in range(n_altconfs):
                truth_positions.append((
                    float(pos_arr[k, 0]), float(pos_arr[k, 1]), float(pos_arr[k, 2]),
                    float(b_arr[k]), float(occs[k]), _ALTLOC_LABELS[k % len(_ALTLOC_LABELS)]
                ))
            partial_positions.append((float(centroid[0]), float(centroid[1]),
                                      float(centroid[2]), part_b))
            cluster_meta.append({
                'atom_idx':    i,
                'n_altconfs':  n_altconfs,
                'altconf_rms': round(float(altconf_rms), 4),
                'positions':   [[round(float(v), 4) for v in p] for p in pos_arr],
                'b_factors':   [round(float(bv), 2) for bv in b_arr],
                'total_occ':   round(total_occ, 4),
                'occs':        [round(float(o), 4) for o in occs],
                'centroid':    [round(float(v), 4) for v in centroid],
                'partial_occ': round(part_occ, 4),
                'partial_b':   round(part_b, 2),
            })

    return truth_positions, partial_positions, cluster_meta


def write_pdb_with_altloc(truth_positions, out_path):
    """
    Write PDB for truth model with optional altloc atoms.
    truth_positions: list of (x, y, z, b_iso, occ, altloc) tuples.
    Atoms with altloc='' are written as regular HETATM; others use HETATM with altloc column.
    Each atom gets a unique residue number (sequential, like write_pdb does).
    """
    a, b, c, al, be, ga = CELL
    _sg = gemmi.find_spacegroup_by_name(SPACEGROUP)
    _z  = len(list(_sg.operations())) if _sg else 1
    _hm = _sg.hm if _sg else SPACEGROUP
    lines = [
        f'CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} {_hm:<11s}{_z:3d}\n'
    ]
    # assign residue numbers: non-cluster atoms get their original index+1,
    # cluster atoms (same original atom, multiple conformers) share a residue number.
    # We track residue number by scanning through; cluster conformers share seq_idx.
    seq_idx = 0
    prev_altloc = None
    current_res = 0
    for i, (x, y, z, b_iso, occ, altloc) in enumerate(truth_positions):
        # Start new residue when: non-altloc atom, or first altloc of a cluster
        if altloc == '' or prev_altloc == '':
            seq_idx += 1
            current_res = seq_idx
        elif altloc == _ALTLOC_LABELS[0]:
            seq_idx += 1
            current_res = seq_idx
        altloc_col = altloc if altloc else ' '
        lines.append(
            f'HETATM{i+1:5d}  O   HOH {altloc_col}A{current_res:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}'
            f'{occ:6.2f}{b_iso:6.2f}          '
            f'  O\n'
        )
        prev_altloc = altloc
    lines.append('END\n')
    Path(out_path).write_text(''.join(lines))


# ── Partial model modifications ────────────────────────────────────────────────

def apply_partial_occ(positions, rng, n_modify):
    """
    Return modified positions list and metadata.
    Selected atoms get occ set to Uniform(0.8, 1.0) in metadata; positions list
    is augmented with (occ, altloc) so write_pdb_with_altloc can write them.
    Since write_pdb writes occ=1.0 for all atoms, we handle this differently:
    we return the (positions, selected, occs) and caller decides what to do.
    """
    n = len(positions)
    n_modify = min(n_modify, n)
    selected = sorted(rng.choice(n, size=n_modify, replace=False).tolist())
    occs = rng.uniform(0.8, 1.0, size=n_modify).tolist()
    return selected, [round(o, 4) for o in occs]


def write_pdb_partial_occ(positions, selected_occs, out_path):
    """Write partial model PDB with per-atom occupancy (for partial_occ mode)."""
    a, b, c, al, be, ga = CELL
    _sg = gemmi.find_spacegroup_by_name(SPACEGROUP)
    _z  = len(list(_sg.operations())) if _sg else 1
    _hm = _sg.hm if _sg else SPACEGROUP
    lines = [
        f'CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} {_hm:<11s}{_z:3d}\n'
    ]
    for i, (x, y, z, b_iso) in enumerate(positions):
        occ = selected_occs.get(i, 1.0)
        lines.append(
            f'HETATM{i+1:5d}  O   HOH A{i+1:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}'
            f'{occ:6.2f}{b_iso:6.2f}          '
            f'  O\n'
        )
    lines.append('END\n')
    Path(out_path).write_text(''.join(lines))


def apply_bfac_shift(positions, rng, sigma, n_modify=None, selected=None):
    """
    Return new positions list with B factors shifted by N(0, sigma).
    If selected is given, shift only those atom indices.
    If n_modify is given and selected is None, pick n_modify random atoms.
    Otherwise shift all atoms.
    Returns (new_positions, n_modified).
    """
    positions = list(positions)
    n = len(positions)
    if selected is not None:
        indices = selected
    elif n_modify is not None:
        n_modify = min(n_modify, n)
        indices = rng.choice(n, size=n_modify, replace=False).tolist()
    else:
        indices = list(range(n))
    idx_set = set(int(i) for i in indices)
    new_pos = []
    for i, (x, y, z, b) in enumerate(positions):
        if i in idx_set:
            shift = float(rng.normal(0.0, sigma))
            b = float(np.clip(b + shift, BFAC_MIN, BFAC_MAX))
        new_pos.append((x, y, z, b))
    return new_pos, len(idx_set)


def apply_xyz_shift(positions, rng, sigma, n_modify=None, selected=None):
    """
    Return new positions list with xyz shifted by N(0, sigma) Å.
    If selected is given, shift only those atom indices.
    If n_modify is given and selected is None, pick n_modify random atoms.
    Otherwise shift all atoms.
    Returns (new_positions, n_modified).
    """
    positions = list(positions)
    n = len(positions)
    if selected is not None:
        indices = selected
    elif n_modify is not None:
        n_modify = min(n_modify, n)
        indices = rng.choice(n, size=n_modify, replace=False).tolist()
    else:
        indices = list(range(n))
    idx_set = set(int(i) for i in indices)
    new_pos = []
    for i, (x, y, z, b) in enumerate(positions):
        if i in idx_set:
            dx, dy, dz = rng.normal(0.0, sigma, size=3)
            x, y, z = x + dx, y + dy, z + dz
        new_pos.append((x, y, z, b))
    return new_pos, len(idx_set)


# ── Per-sample pipeline ───────────────────────────────────────────────────────

def generate_sample(sample_idx, outdir, seed=None,
                    n_atoms=None, n_miss=None,
                    natoms_range=None, modified_range=None,
                    partial_occ=False,
                    xyz_shift=None, xyz_natoms=None,
                    bfac_shift=None, bfac_natoms=None,
                    n_altconfs=1, altconf_rms=0.5,
                    n_clusters=1, all_clusters=False,
                    b_range=(10.0, 30.0),
                    truth_occ_range=(1.0, 1.0),
                    no_refmac=False, verbose=False):
    outdir     = Path(outdir).resolve()
    sample_dir = outdir / f'sample_{sample_idx:05d}'

    if sample_dir.exists() and (sample_dir / 'metadata.json').exists():
        return sample_idx, True, 'already done'

    rng_seed = sample_idx if seed is None else seed + sample_idx
    rng      = np.random.default_rng(rng_seed)

    # Resolve n_atoms / n_miss from ranges (per-sample draw)
    if natoms_range is not None:
        lo, hi = natoms_range
        n_atoms_eff = int(rng.integers(lo, hi + 1))
    else:
        n_atoms_eff = n_atoms if n_atoms is not None else N_ATOMS

    if modified_range is not None:
        lo, hi = modified_range
        n_miss_eff = int(rng.integers(lo, hi + 1))
    elif n_miss is not None:
        n_miss_eff = n_miss
    elif natoms_range is not None:
        n_miss_eff = max(1, n_atoms_eff // 4)
    else:
        n_miss_eff = N_MISS

    ccp4_scr = Path(os.environ.get('CCP4_SCR', '/tmp')) / outdir.name
    ccp4_scr.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix=f'simple_{sample_idx:05d}_', dir=ccp4_scr))

    try:
        # ── Place atoms ──────────────────────────────────────────────────────
        positions = place_atoms(n_atoms_eff, rng, b_range=b_range)
        if len(positions) < n_atoms_eff:
            return sample_idx, False, f'only placed {len(positions)}/{n_atoms_eff} atoms'

        occ_lo, occ_hi = truth_occ_range

        # Initialise partial-model metadata variables (altconf path skips the else block)
        missing_idx  = []
        sel_occ_idx  = []
        sel_occs     = []
        bfac_n_mod   = None
        xyz_n_mod    = None
        link_indices = None

        # ── Alternate-conformer clusters ──────────────────────────────────────
        cluster_meta = None
        if n_altconfs >= 2:
            # truth_occ_range not applied to altconf path (clusters have their own occ logic)
            truth_pos, partial_positions, cluster_meta = _insert_altconf_clusters(
                positions, rng, n_altconfs, altconf_rms, n_clusters, all_clusters
            )
            truth_pdb    = tmpdir / 'truth.pdb'
            starthere_pdb = tmpdir / 'starthere.pdb'
            write_pdb_with_altloc(truth_pos, truth_pdb)
            write_pdb(partial_positions, starthere_pdb)
        else:
            truth_pdb = tmpdir / 'truth.pdb'
            # truth_occs resolved in deletion path below after missing_idx is known
            truth_occs = None

            # ── Build partial model (standard path) ───────────────────────────
            # Order of modifications: deletion → partial_occ → bfac_shift → xyz_shift
            # Only one "deletion" mode is active at a time; the shifts can stack.

            partial_positions = list(positions)

            if partial_occ:
                # Partial occupancy mode: keep atoms, lower their occ
                sel_occ_idx, sel_occs = apply_partial_occ(partial_positions, rng, n_miss_eff)
                # Bfac / xyz shifts applied to same atoms if linked
                link_indices = sel_occ_idx
            else:
                # Atom deletion mode
                n_del = min(n_miss_eff, len(partial_positions) - 1)
                missing_idx = sorted(rng.choice(len(partial_positions), size=n_del,
                                                replace=False).tolist())
                missing_set = set(missing_idx)
                partial_positions = [p for i, p in enumerate(partial_positions)
                                     if i not in missing_set]
                link_indices = None
                # Only the missing atom(s) get non-unity occ in truth; present atoms stay 1.0
                if occ_lo != 1.0 or occ_hi != 1.0:
                    truth_occs = [1.0] * len(positions)
                    for idx in missing_idx:
                        truth_occs[idx] = float(rng.uniform(occ_lo, occ_hi))
            # write truth PDB now that truth_occs is finalised
            write_pdb(positions, truth_pdb, occs=truth_occs)

            if bfac_shift is not None:
                partial_positions, bfac_n_mod = apply_bfac_shift(
                    partial_positions, rng, bfac_shift,
                    n_modify=bfac_natoms,
                    selected=link_indices,
                )

            if xyz_shift is not None:
                partial_positions, xyz_n_mod = apply_xyz_shift(
                    partial_positions, rng, xyz_shift,
                    n_modify=xyz_natoms,
                    selected=link_indices,
                )

            starthere_pdb = tmpdir / 'starthere.pdb'
            if partial_occ and (bfac_shift is None and xyz_shift is None):
                # Write with custom occupancies (no shifts)
                occ_map = {idx: occ for idx, occ in zip(sel_occ_idx, sel_occs)}
                write_pdb_partial_occ(list(positions), occ_map, starthere_pdb)
            elif partial_occ:
                # After bfac/xyz shifts, partial_positions has same atom count as truth;
                # we still need per-atom occs. positions were reindexed by the shifts
                # (which preserve ordering). Build occ map on original indices.
                occ_map = {idx: occ for idx, occ in zip(sel_occ_idx, sel_occs)}
                write_pdb_partial_occ(partial_positions, occ_map, starthere_pdb)
            else:
                write_pdb(partial_positions, starthere_pdb)

        # ── Structure factors for truth ───────────────────────────────────────
        truth_sf_mtz = tmpdir / 'truth_sf.mtz'
        r = subprocess.run(
            ['gemmi', 'sfcalc', f'--dmin={DMIN}', f'--to-mtz={truth_sf_mtz}', str(truth_pdb)],
            capture_output=True,
        )
        if r.returncode != 0:
            return sample_idx, False, f'sfcalc failed: {r.stderr.decode()[-300:]}'

        # ── Maps ──────────────────────────────────────────────────────────────
        sample_dir.mkdir(parents=True, exist_ok=True)

        scale_k = scale_B = R_factor = None
        rwork = rfree = None

        if no_refmac:
            scale_k, scale_B, R_factor, _shape = _build_maps_no_refmac(
                tmpdir, truth_sf_mtz, starthere_pdb, sample_dir
            )
            r_str = f'scale_k={scale_k:.4f} R={R_factor:.4f}' if scale_k is not None else 'sfcalc'
        else:
            fobs_mtz = tmpdir / 'fobs.mtz'
            build_fobs_mtz(truth_sf_mtz, fobs_mtz, rng)
            rwork, rfree, out_mtz = run_refmac(starthere_pdb, fobs_mtz, tmpdir, verbose=verbose)
            if out_mtz is None:
                return sample_idx, False, 'refmac produced no output MTZ'
            refmac_log = tmpdir / 'refmac.log'
            if refmac_log.exists():
                shutil.copy2(refmac_log, sample_dir / 'refmac.log')
            out_pdb = tmpdir / 'refmacout.pdb'
            if out_pdb.exists():
                shutil.copy2(out_pdb, sample_dir / 'refmacout.pdb')

            mtz_to_ccp4(truth_sf_mtz, 'FC',     'PHIC',    sample_dir / 'truth.map')
            mtz_to_ccp4(out_mtz,      'FWT',    'PHWT',    sample_dir / '2fofc.map')
            mtz_to_ccp4(out_mtz,      'DELFWT', 'PHDELWT', sample_dir / 'fofc.map')
            mtz_to_ccp4(out_mtz,      'FC',     'PHIC',    sample_dir / 'fc.map')
            r_str = f'R={rwork:.4f} Rf={rfree:.4f}' if rwork is not None else 'R=n/a'

        # ── Metadata ──────────────────────────────────────────────────────────
        meta = {
            'sample_idx': sample_idx,
            'n_atoms':    n_atoms_eff,
            'n_miss':     n_miss_eff,
            'cell':       list(CELL),
            'dmin':       DMIN,
            'spacegroup': SPACEGROUP,
        }
        if rwork is not None:
            meta['r']  = rwork
            meta['rf'] = rfree
        if scale_k is not None:
            meta['scale_k']  = round(float(scale_k), 6)
            meta['scale_B']  = round(float(scale_B), 4)
            meta['R_factor'] = round(R_factor, 4)
        if missing_idx:
            meta['missing_idx'] = missing_idx
        if sel_occ_idx:
            meta['partial_occ_mode']   = True
            meta['partial_occ_idx']    = sel_occ_idx
            meta['partial_occ_values'] = sel_occs
        if bfac_shift is not None:
            meta['bfac_shift_sigma']     = round(float(bfac_shift), 4)
            meta['bfac_shift_natoms']    = bfac_natoms
            meta['bfac_shift_n_modified'] = bfac_n_mod
        if xyz_shift is not None:
            meta['xyz_sigma']      = round(float(xyz_shift), 6)
            meta['xyz_natoms']     = xyz_natoms
            meta['xyz_n_modified'] = xyz_n_mod
        if cluster_meta is not None:
            meta['n_altconfs']  = n_altconfs
            meta['altconf_rms'] = round(float(altconf_rms), 4)
            meta['n_clusters']  = len(cluster_meta)
            meta['all_clusters'] = all_clusters
            meta['clusters']    = cluster_meta
        if occ_lo != 1.0 or occ_hi != 1.0:
            meta['truth_occ_range'] = [round(occ_lo, 4), round(occ_hi, 4)]
        if truth_occs is not None:
            meta['truth_occs'] = [round(o, 4) for o in truth_occs]
        meta['no_refmac'] = no_refmac

        (sample_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
        return sample_idx, True, r_str

    except Exception:
        import traceback
        return sample_idx, False, traceback.format_exc()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── SLURM submission ──────────────────────────────────────────────────────────

def _cli_flags_for_task(args):
    """Build the CLI flags to pass to each SLURM array task."""
    flags = []
    flags += [f'--natoms {args.natoms}'] if args.natoms is not None else []
    if args.natoms_range is not None:
        flags += [f'--natoms-range {args.natoms_range[0]} {args.natoms_range[1]}']
    flags += [f'--modified {args.modified}'] if args.modified is not None else []
    if args.modified_range is not None:
        flags += [f'--modified-range {args.modified_range[0]} {args.modified_range[1]}']
    flags += [f'--n-atoms {args.n_atoms}'] if args.n_atoms is not None else []   # alias
    flags += [f'--spacegroup "{SPACEGROUP}"']
    flags += [f'--cell {CELL[0]} {CELL[1]} {CELL[2]}']
    flags += [f'--dmin {DMIN}']
    if args.partial_occ:
        flags += ['--partial-occ']
    if args.xyz_shift is not None:
        flags += [f'--xyz-shift {args.xyz_shift}']
    if args.xyz_natoms is not None:
        flags += [f'--xyz-natoms {args.xyz_natoms}']
    if args.bfac_shift is not None:
        flags += [f'--bfac-shift {args.bfac_shift}']
    if args.bfac_natoms is not None:
        flags += [f'--bfac-natoms {args.bfac_natoms}']
    if args.n_altconfs >= 2:
        flags += [f'--n-altconfs {args.n_altconfs}']
        flags += [f'--altconf-rms {args.altconf_rms}']
    if args.n_clusters > 1:
        flags += [f'--n-clusters {args.n_clusters}']
    if args.all_clusters:
        flags += ['--all-clusters']
    if args.b_range != [10.0, 30.0]:
        flags += [f'--b-range {args.b_range[0]} {args.b_range[1]}']
    if args.truth_occ_range != [1.0, 1.0]:
        flags += [f'--truth-occ-range {args.truth_occ_range[0]} {args.truth_occ_range[1]}']
    if args.no_refmac:
        flags += ['--no-refmac']
    if args.verbose:
        flags += ['--verbose']
    return ' '.join(flags)


def submit(args):
    outdir  = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    script  = Path(__file__).resolve()
    max_arr = args.max_array
    nsamples = args.nsamples
    start    = args.start
    seed     = args.seed
    task_flags = _cli_flags_for_task(args)

    n_batches = (nsamples + max_arr - 1) // max_arr
    for b in range(n_batches):
        batch_start = start + b * max_arr
        batch_end   = start + min((b + 1) * max_arr, nsamples) - 1
        arr_size    = batch_end - batch_start   # 0-indexed: 0..(arr_size)

        sh  = outdir / f'_batch{b}.sh'
        log = outdir / f'slurm_b{b}_%a.out'
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name=simple_{outdir.name}',
            f'#SBATCH --partition={args.partition}',
            '#SBATCH --ntasks=1',
            f'#SBATCH --array=0-{arr_size}',
            f'#SBATCH --output={log}',
            '#SBATCH --export=ALL',
        ]
        if args.account:
            lines.append(f'#SBATCH --account={args.account}')
        if args.qos:
            lines.append(f'#SBATCH --qos={args.qos}')
        lines += [
            'mkdir -p "${CCP4_SCR:-/tmp}"',
            f'cd {SCRIPT_DIR}',
            f'ccp4-python {script} --task $(( {batch_start} + $SLURM_ARRAY_TASK_ID ))'
            f' --outdir {outdir} --seed {seed} {task_flags}',
        ]
        sh.write_text('\n'.join(lines) + '\n')
        r = subprocess.run(['sbatch', str(sh)], capture_output=True, text=True)
        print(f'  Batch {b} (samples {batch_start}–{batch_end}): '
              f'{r.stdout.strip() or r.stderr.strip()}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global N_ATOMS, N_MISS, SPACEGROUP, CELL, DMIN

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nsamples',    type=int, default=100)
    ap.add_argument('--seed',        type=int, default=0)
    ap.add_argument('--outdir',      required=True)
    ap.add_argument('--submit',      action='store_true')
    ap.add_argument('--task',        type=int, default=None,
                    help='SLURM array task: generate sample_idx=TASK')
    ap.add_argument('--start',       type=int, default=0,
                    help='Starting sample index for sequential / batch submission (default: 0)')
    ap.add_argument('--partition',   default='lr6')
    ap.add_argument('--account',     default='pc_als831')
    ap.add_argument('--qos',         default='lr_normal')
    ap.add_argument('--max-array',   type=int, default=MAX_ARRAY,
                    help=f'Max SLURM array size per batch (default: {MAX_ARRAY})')

    # ── Atom count ────────────────────────────────────────────────────────────
    ap.add_argument('--natoms',          type=int, default=None,
                    help='Fixed number of O atoms per sample (default: 20)')
    ap.add_argument('--n-atoms',         type=int, default=None, dest='n_atoms',
                    help='Alias for --natoms (backward compat)')
    ap.add_argument('--natoms-range',    type=int, nargs=2, metavar=('MIN', 'MAX'),
                    help='Draw natoms uniformly from [MIN, MAX] per sample')

    # ── Partial model ─────────────────────────────────────────────────────────
    ap.add_argument('--modified',        type=int, default=None,
                    help='Number of atoms deleted/modified to make partial model (default: 1)')
    ap.add_argument('--modified-range',  type=int, nargs=2, metavar=('MIN', 'MAX'),
                    help='Draw n_modified uniformly from [MIN, MAX] per sample')
    ap.add_argument('--partial-occ',     action='store_true',
                    help='Give --modified atoms random occ in [0.8,1.0] instead of deleting them')

    # ── Positional / B-factor perturbations ───────────────────────────────────
    ap.add_argument('--xyz-shift',       type=float, default=None, metavar='SIGMA',
                    help='Add Gaussian N(0,SIGMA) Å noise to atom xyz in partial model')
    ap.add_argument('--xyz-natoms',      type=int, default=None, metavar='N',
                    help='Number of atoms to apply --xyz-shift to (default: all)')
    ap.add_argument('--bfac-shift',      type=float, default=None, metavar='SIGMA',
                    help='Add Gaussian N(0,SIGMA) Å² noise to B factors in partial model')
    ap.add_argument('--bfac-natoms',     type=int, default=None, metavar='N',
                    help='Number of atoms to apply --bfac-shift to (default: all)')

    # ── Alternate conformers ──────────────────────────────────────────────────
    ap.add_argument('--n-altconfs',      type=int, default=1, metavar='N',
                    help='Number of alt conformers per cluster in truth (default: 1 = disabled)')
    ap.add_argument('--altconf-rms',     type=float, default=0.5, metavar='SIGMA',
                    help='RMS 3D displacement of each alt conf from centroid (Å, default: 0.5)')
    ap.add_argument('--n-clusters',      type=int, default=1, metavar='N',
                    help='Number of atoms to split into alt-conf clusters (default: 1)')
    ap.add_argument('--all-clusters',    action='store_true',
                    help='Split every atom into an alt-conf cluster (overrides --n-clusters)')

    # ── Truth occupancy ───────────────────────────────────────────────────────
    ap.add_argument('--truth-occ-range', type=float, nargs=2, default=[1.0, 1.0],
                    metavar=('MIN', 'MAX'),
                    help='Draw per-atom truth occupancy from Uniform[MIN, MAX] '
                         '(default: 1.0 1.0 = fixed at 1). Negative values produce '
                         'negative-density atoms in truth.map, yielding negative '
                         'Fo-Fc peaks where the partial model has spurious density.')

    # ── Pipeline control ──────────────────────────────────────────────────────
    ap.add_argument('--b-range',         type=float, nargs=2, default=[10.0, 30.0],
                    metavar=('MIN', 'MAX'),
                    help='B-factor range for placed atoms; MIN==MAX → fixed B '
                         '(default: 10 30)')
    ap.add_argument('--no-refmac',       action='store_true',
                    help='Skip refmac; use sfcalc+scaleit map coefficients instead')
    ap.add_argument('--verbose',         action='store_true')

    # ── Crystallographic parameters ───────────────────────────────────────────
    ap.add_argument('--spacegroup',      default=None,
                    help='Space group HM symbol (default: P 21 21 21)')
    ap.add_argument('--cell',            nargs=3, type=float, default=None,
                    metavar=('A', 'B', 'C'),
                    help='Unit cell a b c in Å (default: 45.9 40.7 30.1)')
    ap.add_argument('--dmin',            type=float, default=None,
                    help='Resolution cutoff in Å (default: 0.965)')

    args = ap.parse_args()

    # Apply global overrides from CLI
    # natoms: --natoms takes priority; --n-atoms is the alias
    if args.natoms is not None:
        N_ATOMS = args.natoms
    elif args.n_atoms is not None:
        N_ATOMS = args.n_atoms
    if args.modified is not None:
        N_MISS = args.modified
    if args.spacegroup is not None:
        SPACEGROUP = args.spacegroup
    if args.cell is not None:
        CELL = tuple(args.cell) + (90.0, 90.0, 90.0)
    if args.dmin is not None:
        DMIN = args.dmin

    # Convenience: resolve natoms/modified for the task-level call
    natoms_arg   = args.natoms if args.natoms is not None else (args.n_atoms if args.n_atoms is not None else None)
    modified_arg = args.modified

    if args.submit:
        submit(args)
        return

    if args.task is not None:
        idx, ok, msg = generate_sample(
            args.task, args.outdir, seed=args.seed,
            n_atoms=natoms_arg, n_miss=modified_arg,
            natoms_range=args.natoms_range,
            modified_range=args.modified_range,
            partial_occ=args.partial_occ,
            xyz_shift=args.xyz_shift, xyz_natoms=args.xyz_natoms,
            bfac_shift=args.bfac_shift, bfac_natoms=args.bfac_natoms,
            n_altconfs=args.n_altconfs, altconf_rms=args.altconf_rms,
            n_clusters=args.n_clusters, all_clusters=args.all_clusters,
            b_range=tuple(args.b_range),
            truth_occ_range=tuple(args.truth_occ_range),
            no_refmac=args.no_refmac, verbose=args.verbose,
        )
        print(f'sample {idx}: {"ok" if ok else "FAILED"}  {msg}')
        return

    # Local sequential run
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for i in range(args.start, args.start + args.nsamples):
        idx, success, msg = generate_sample(
            i, outdir, seed=args.seed,
            n_atoms=natoms_arg, n_miss=modified_arg,
            natoms_range=args.natoms_range,
            modified_range=args.modified_range,
            partial_occ=args.partial_occ,
            xyz_shift=args.xyz_shift, xyz_natoms=args.xyz_natoms,
            bfac_shift=args.bfac_shift, bfac_natoms=args.bfac_natoms,
            n_altconfs=args.n_altconfs, altconf_rms=args.altconf_rms,
            n_clusters=args.n_clusters, all_clusters=args.all_clusters,
            b_range=tuple(args.b_range),
            truth_occ_range=tuple(args.truth_occ_range),
            no_refmac=args.no_refmac, verbose=args.verbose,
        )
        if success:
            ok += 1
        else:
            fail += 1
            print(f'  FAILED sample {idx}: {msg}')
        if (i - args.start + 1) % 10 == 0:
            print(f'  {i - args.start + 1}/{args.nsamples}  ok={ok}  fail={fail}')
    print(f'Done. ok={ok}  fail={fail}')


if __name__ == '__main__':
    main()
