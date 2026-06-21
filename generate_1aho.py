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
    build_starthere_pdb,
    write_full_conf_pdb,
    generate_occ_groups,
    parse_rfactors,
    MAINCHAIN_ATOMS,
)

SCRIPT_DIR = Path(__file__).parent
REFMAC5    = Path(shutil.which('refmac5') or '/programs/ccp4-8.0/bin/refmac5')
DMIN       = 0.965
SAMPLE_RATE = 3.0

# Default 1AHO source files (relative to SCRIPT_DIR or absolute)
DEFAULT_PDB     = SCRIPT_DIR / '1aho' / 'gt48.pdb'
DEFAULT_MTZ     = SCRIPT_DIR / '1aho' / 'gt48.mtz'
DEFAULT_OBS_MTZ = SCRIPT_DIR / '1aho' / '1aho.mtz'   # real data: valid HKL mask + FreeR_flag

# ── Wilson B correction ───────────────────────────────────────────────────────
# Jiggling inflates effective B by ~(1 + 2·shift_scale²) per atom.  Applying
# exp(−ΔB·s²/4) to F_truth after sfcalc brings the resolution-dependent
# intensity falloff back in line with the real 1AHO data, so the CNN sees a
# realistic high-resolution tail.  Same logic as generate_protein.py.
_REF_WILSON_B = None   # computed once from DEFAULT_OBS_MTZ; cached here


def _wilson_b(F, s2, n_bins=20, min_per_bin=10):
    """Wilson B from arrays of |F| and s²=1/d².  Returns 0.0 if insufficient data."""
    valid = (F > 0) & np.isfinite(F) & np.isfinite(s2)
    F, s2 = F[valid].astype(np.float64), s2[valid].astype(np.float64)
    if s2.size < min_per_bin * 3:
        return 0.0
    edges = np.linspace(s2.min(), s2.max(), n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (s2 >= edges[i]) & (s2 < edges[i + 1])
        if m.sum() < min_per_bin:
            continue
        xs.append(float(s2[m].mean()))
        ys.append(float(np.log((F[m] ** 2).mean())))
    if len(xs) < 3:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(-2.0 * slope)


def _reference_wilson_b(obs_mtz_path=None):
    """Return Wilson B of the real 1AHO data (cached after first call)."""
    global _REF_WILSON_B
    if _REF_WILSON_B is not None:
        return _REF_WILSON_B
    path = Path(obs_mtz_path) if obs_mtz_path else DEFAULT_OBS_MTZ
    if not path.exists():
        return None
    mtz = gemmi.read_mtz_file(str(path))
    h = np.asarray(mtz.column_with_label('H'),  dtype=np.int32)
    k = np.asarray(mtz.column_with_label('K'),  dtype=np.int32)
    l = np.asarray(mtz.column_with_label('L'),  dtype=np.int32)
    F = np.asarray(mtz.column_with_label('FP'), dtype=np.float32)
    cell = mtz.cell
    s2 = np.array([cell.calculate_1_d2([int(h_), int(k_), int(l_)])
                   for h_, k_, l_ in zip(h, k, l)], dtype=np.float64)
    _REF_WILSON_B = _wilson_b(F, s2)
    return _REF_WILSON_B

DEFAULT_K_CONFORMERS = 32

# Flood water calibration for --vary-flood (occ_max = FLOOD_LINE_K_SYM/sqrt(N), B log-uniform):
#   Vary_flood flood line (derived analytically, verified against ft17):
#     Rfree = _VARY_FLOOD_SLOPE(b_lo,b_hi) * FLOOD_LINE_K + floor
#     slope = _C_RF * Z * sqrt(E_B[ano_ff(B)^2] / 3)
#     E_B[ano_ff^2] = (4pi)^3 * (b_lo^-3 - b_hi^-3) / (3*ln(b_hi/b_lo))
#   _C_RF = 0.003710 from random_flood ft7-11; floor = 0.0374 (shift_scale=0 intercept).
#   Use --flood-rfree-target to set the target Rfree; FLOOD_LINE_K is computed at runtime.
# For Uniform[-occ_max,+occ_max]: occ_max * sqrt(N) = FLOOD_LINE_K * sqrt(3) = FLOOD_LINE_K_SYM
_C_RF        = 0.003710  # Rfree per unit sigma_ΔF; from random_flood ft7-11 calibration
FLOOD_FLOOR  = 0.0374    # Rfree floor (shift_scale=0, k_conformers=32)
FLOOD_LINE_K     = 3.07             # → Rfree ~11% (B∈[5,80], shift_scale=0, k=32)
FLOOD_LINE_K_SYM = 3.07 * 3**0.5  # = 5.31; overridden at runtime by --flood-rfree-target


def _vary_flood_k(b_lo, b_hi, rfree_target, floor=None):
    """Compute FLOOD_LINE_K for vary_flood to hit rfree_target.
    Rfree = slope * K + floor  where slope = _C_RF * Z * sqrt(E_B[ano_ff^2]/3)."""
    if floor is None:
        floor = FLOOD_FLOOR
    E_inv_B3  = (b_lo**-3 - b_hi**-3) / (3.0 * np.log(b_hi / b_lo))
    E_ano_ff2 = (4.0 * np.pi)**3 * E_inv_B3
    slope     = _C_RF * 8.0 * np.sqrt(E_ano_ff2 / 3.0)
    return (rfree_target - floor) / slope
FLOOD_NF_MIN    = 700    # log-uniform sampling range for --vary-flood
FLOOD_NF_MAX    = 4000
DEFAULT_N_FLOOD   = 1764   # used only when neither --vary-flood nor --random-flood is set
DEFAULT_FLOOD_OCC = 0.083  # used only when neither --vary-flood nor --random-flood is set

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

    When min_dist <= 0, positions are drawn uniformly over the unit cell with no
    avoidance check (fast path). Otherwise positions avoid existing atoms by min_dist Å.
    `occ` is ignored here; call sample_flood_occs separately for per-water occupancies.
    """
    rng = np.random.default_rng(seed)

    if min_dist <= 0:
        fracs = rng.random((n_flood, 3))
        positions = np.array([
            [*(cell.orthogonalize(gemmi.Fractional(*f)),)]
            for f in fracs
        ])
        # gemmi.Position objects → plain floats
        positions = np.array([
            [cell.orthogonalize(gemmi.Fractional(*f)).x,
             cell.orthogonalize(gemmi.Fractional(*f)).y,
             cell.orthogonalize(gemmi.Fractional(*f)).z]
            for f in fracs
        ])
        return positions

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


def sample_flood_occs(n_flood, occ_lo, occ_hi, seed):
    """Draw per-water occupancies uniformly from [occ_lo, occ_hi].

    Both values may be negative. When occ_lo == occ_hi the result is a constant array.
    """
    if occ_lo == occ_hi:
        return np.full(n_flood, occ_lo, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return rng.uniform(occ_lo, occ_hi, size=n_flood).astype(np.float32)


def sample_flood_bfactors(n_flood, b_lo, b_hi, seed):
    """Draw per-water B factors log-uniformly from [b_lo, b_hi].

    Log-uniform spacing gives equal density to each octave of B, so sharp
    isolated bumps (small B) and diffuse blobs (large B) are sampled equally
    per decade — good for covering disorder on a variety of length scales.
    When b_lo == b_hi all waters get the same B factor.
    """
    if b_lo == b_hi:
        return np.full(n_flood, b_lo, dtype=np.float32)
    rng = np.random.default_rng(seed)
    log_bs = rng.uniform(np.log(b_lo), np.log(b_hi), size=n_flood)
    return np.exp(log_bs).astype(np.float32)


# ── Peak-height-controlled flood water sampling ───────────────────────────────
# Peak density of a 1-electron pure-Gaussian atom at its centre (x=0):
#   ano_ff(B) = (4π/B)^1.5   [e/Å³ per electron]
# This gives the maximum contribution of that atom to the electron density map.
# For Z electrons at occupancy occ: peak = Z × occ × ano_ff(B)  [e/Å³]
#
# Target normalisation: σ_fofc_real — RMS of the real 1AHO Fo-Fc map (~0.1 e/Å³).
# Each flood atom's Z×occ is drawn from Uniform[0, peak_sigma × σ_fofc_real / ano_ff(B)],
# so its individual peak is at most peak_sigma σ_fofc_real in the (unabsorbed) map.
#
# The auto-clip FLOOD_LINE_K_SYM/√N is applied after Z×occ→occ conversion to keep
# the total SF contribution (and hence Rfree) bounded independently of N.

_REAL_FOFC_SIGMA = None   # RMS of real 1AHO Fo-Fc map, cached


def _real_fofc_sigma(fofc_map_path=None):
    """RMS of the real 1AHO Fo-Fc map (e/Å³), cached.

    Default source: 1aho_test/fofc.map (from the real 1AHO diffraction data).
    This is the natural noise floor for flood water peaks — peaks drawn to be
    at most peak_sigma × σ_fofc are at most peak_sigma σ above real-data noise.
    """
    global _REAL_FOFC_SIGMA
    if _REAL_FOFC_SIGMA is not None:
        return _REAL_FOFC_SIGMA
    path = Path(fofc_map_path) if fofc_map_path else SCRIPT_DIR / '1aho_test' / 'fofc.map'
    if path.exists():
        m = gemmi.read_ccp4_map(str(path))
        arr = np.array(m.grid, copy=False).ravel()
        _REAL_FOFC_SIGMA = float(arr.std())
    else:
        _REAL_FOFC_SIGMA = 0.10   # fallback: typical 1AHO Fo-Fc sigma (e/Å³)
    return _REAL_FOFC_SIGMA


def _ano_ff_peak(B_arr):
    """Peak density (e/Å³) of a 1-electron pure-Gaussian atom, from B factor.

    ano_ff(B) = (4π/B)^1.5
    For Z electrons at occupancy occ: peak = Z × occ × ano_ff(B).
    """
    B = np.asarray(B_arr, dtype=np.float64)
    return (4.0 * np.pi / np.maximum(B, 1e-6)) ** 1.5


def sample_flood_waters_by_peak(n_flood, b_lo, b_hi, peak_sigma,
                                seed, occ_max_clip=None,
                                obs_mtz_path=None, gt48_mtz_path=None,
                                fofc_sigma_path=None, z_atom=8):
    """Draw B factors then solve Z×occ so each water's peak ≤ peak_sigma × σ_fofc.

    For each flood water i:
      B_i        ~ LogUniform[b_lo, b_hi]
      ano_ff_i   = (4π/B_i)^1.5          [e/Å³ per electron at occ=1]
      z_occ_max  = peak_sigma × σ_fofc / ano_ff_i
      z_occ_i    ~ Uniform[0, z_occ_max]  [total electrons at this site]
      sign_i     ~ ±1
      occ_i      = sign_i × z_occ_i / z_atom

    occ_max_clip (default: FLOOD_LINE_K_SYM/√N) limits |occ| to bound Rfree.
    Peaks are expressed in units of σ_fofc ≈ 0.1 e/Å³ (real 1AHO Fo-Fc noise).

    z_atom: electron count of the flood atom element (default 8 for oxygen).
    """
    rng         = np.random.default_rng(seed)
    sigma_fofc  = _real_fofc_sigma(fofc_sigma_path)

    log_bs      = rng.uniform(np.log(b_lo), np.log(b_hi), size=n_flood)
    B_vals      = np.exp(log_bs).astype(np.float32)
    ano_ff      = _ano_ff_peak(B_vals).astype(np.float32)    # e/Å³ per electron

    z_occ_max   = (peak_sigma * sigma_fofc) / np.maximum(ano_ff, 1e-9)
    z_occs      = rng.uniform(0.0, 1.0, size=n_flood).astype(np.float32) * z_occ_max
    signs       = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_flood)
    occs        = signs * z_occs / float(z_atom)

    if occ_max_clip is not None:
        occs = np.clip(occs, -float(occ_max_clip), float(occ_max_clip))
    return occs.astype(np.float32), B_vals


def add_flood_chain(st, positions, occ, b_iso=20.0, chain_name='W'):
    """Add flood water atoms to a cloned structure; return the new structure.

    occ and b_iso may each be a scalar or a 1-D array (one value per water).
    occ may be negative.
    """
    st2 = st.clone()
    if len(positions) == 0:
        return st2
    n = len(positions)
    occ_arr = occ   if hasattr(occ,   '__len__') else np.full(n, float(occ))
    biso_arr = b_iso if hasattr(b_iso, '__len__') else np.full(n, float(b_iso))
    ch  = gemmi.Chain(chain_name)
    for i, (x, y, z) in enumerate(positions):
        res = gemmi.Residue()
        res.name  = 'HOH'
        res.seqid = gemmi.SeqId(i + 1, ' ')
        a = gemmi.Atom()
        a.name    = 'O'
        a.element = gemmi.Element('O')
        a.pos     = gemmi.Position(x, y, z)
        a.occ     = float(occ_arr[i])
        a.b_iso   = float(biso_arr[i])
        res.add_atom(a)
        ch.add_residue(res)
    st2[0].add_chain(ch)
    return st2


def swap_conformer_assignments(conf_data, chain_names, residue_keys, swaps_per_residue, rng):
    """Apply pairwise conformer swaps to the partial model's conf_data.

    For each residue, draws Poisson(swaps_per_residue) pairwise swaps. Each swap
    exchanges the atom data for two randomly chosen conformer chains at that residue,
    breaking backbone connectivity between adjacent residues. Truth structure is unaffected.
    """
    if swaps_per_residue <= 0.0:
        return conf_data
    nc = len(chain_names)
    conf_data2 = {cn: dict(rd) for cn, rd in conf_data.items()}
    for rk in residue_keys:
        n_swaps = int(rng.poisson(swaps_per_residue))
        for _ in range(n_swaps):
            i, j = rng.choice(nc, size=2, replace=False)
            cn_i, cn_j = chain_names[i], chain_names[j]
            ri = conf_data2[cn_i].get(rk)
            rj = conf_data2[cn_j].get(rk)
            if ri is not None and rj is not None:
                conf_data2[cn_i][rk] = rj
                conf_data2[cn_j][rk] = ri
    return conf_data2


def _swap_two(conf_data, chain_names, residue_keys, swaps_per_residue, seed):
    """Convenience wrapper: swap with given seed, return new conf_data."""
    return swap_conformer_assignments(
        conf_data, chain_names, residue_keys, swaps_per_residue,
        np.random.default_rng(seed),
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
    r = subprocess.run(
        ['gemmi', 'sfcalc', f'--dmin={DMIN}', f'--to-mtz={truth_sf_mtz}', str(truth_full_pdb)],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f'gemmi sfcalc failed:\n{r.stderr.decode()[-2000:]}')

    prot  = gemmi.read_mtz_file(str(truth_sf_mtz))
    refme = gemmi.read_mtz_file(str(refme_path))
    obs   = gemmi.read_mtz_file(str(obs_mtz_path))

    h_p  = np.array(prot.column_with_label('H'),    dtype=np.int32)
    k_p  = np.array(prot.column_with_label('K'),    dtype=np.int32)
    l_p  = np.array(prot.column_with_label('L'),    dtype=np.int32)
    fc_p = np.array(prot.column_with_label('FC'),   dtype=np.float64)
    ph_p = np.array(prot.column_with_label('PHIC'), dtype=np.float64)

    # ── Wilson B correction ───────────────────────────────────────────────
    # Jiggling inflates the effective B factor; rescale F_truth to match the
    # real 1AHO Wilson B before assembling fobs and writing truth.map.
    cell_p = prot.cell
    s2_p   = np.array([cell_p.calculate_1_d2([int(h_p[i]), int(k_p[i]), int(l_p[i])])
                        for i in range(len(h_p))], dtype=np.float64)
    ref_B  = _reference_wilson_b(obs_mtz_path)
    gen_B  = _wilson_b(fc_p, s2_p)
    if ref_B and gen_B:
        delta_B = ref_B - gen_B
        wilson_scale = np.exp(-delta_B * s2_p / 4.0)
        fc_p = fc_p * wilson_scale
        print(f'    Wilson B: ref={ref_B:.2f} gen={gen_B:.2f} ΔB={delta_B:+.2f} Å² applied')
        # Rewrite truth_sf_mtz with corrected FC so truth.map has correct falloff
        out_corr = gemmi.Mtz()
        out_corr.cell       = prot.cell
        out_corr.spacegroup = prot.spacegroup
        out_corr.add_dataset('HKL_base')
        for lbl in ('H', 'K', 'L'):
            out_corr.add_column(lbl, 'H')
        out_corr.add_dataset('data')
        out_corr.add_column('FC',   'F')
        out_corr.add_column('PHIC', 'P')
        out_corr.set_data(
            np.column_stack([h_p, k_p, l_p,
                             fc_p.astype(np.float32),
                             ph_p.astype(np.float32)])
        )
        out_corr.write_to_file(str(truth_sf_mtz))

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
    kw += b'weight matrix 10\n'
    kw += b'scpart 1\n'
    kw += b'damp 0.5 0.5\n'
    kw += b'make hout Y\n'
    kw += b'make hydr Y\n'
    kw += f'NCYC {ncyc}\n'.encode()
    kw += generate_occ_groups(starthere_pdb)
    kw += b'END\n'

    out_mtz  = tmpdir / f'refmacout{suffix}.mtz'
    out_pdb  = tmpdir / f'refmacout{suffix}.pdb'
    tmpdir.mkdir(parents=True, exist_ok=True)  # guard against transient cleanup

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
    sample_idx, outdir, pdb_path, mtz_path, obs_mtz_path,
    k_conformers=DEFAULT_K_CONFORMERS,
    shift_scale=DEFAULT_SHIFT_SCALE,
    n_flood=DEFAULT_N_FLOOD,
    flood_occ=DEFAULT_FLOOD_OCC,
    flood_occ_lo=None,
    flood_occ_hi=None,
    flood_b_lo=1.0,
    flood_b_hi=15.0,
    flood_peak_sigma=3.0,
    flood_occ_max_clip=None,
    flood_min_dist=0.0,
    vary_flood=False,
    random_flood=False,
    flood_rfree_target=None,
    flood_floor=FLOOD_FLOOR,
    ncyc=DEFAULT_NCYC,
    swaps_per_residue=0.0,
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
    ccp4_scr = Path(os.environ.get('CCP4_SCR', '/tmp')) / outdir.name
    ccp4_scr.mkdir(parents=True, exist_ok=True)
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
        jig_pdb = tmpdir / 'jiggled.pdb'
        st_jig.write_pdb(str(jig_pdb))
        t = _t('jiggle', t)

        # 2. Parse jiggled conformers; apply swaps.
        #    swap A (seed+5) → conf_data_truth: the correct conformer arrangement.
        #    swap B (seed+6) → conf_data_model: derived from truth, so swaps_per_residue
        #    controls only the truth↔model delta, not absolute disorder from original.
        #    When swaps_per_residue=0 both equal the original conf_data.
        st_jig_r, chain_names, conf_data = parse_conformers(jig_pdb)
        residue_keys = list(conf_data[chain_names[0]].keys())
        conf_data_truth = _swap_two(conf_data, chain_names, residue_keys,
                                    swaps_per_residue, rng_seed + 5)
        conf_data_model = _swap_two(conf_data_truth, chain_names, residue_keys,
                                    swaps_per_residue, rng_seed + 6)
        t = _t('parse', t)

        # 3. Flood waters: place random HOH positions; per-water occ from [occ_lo, occ_hi]
        _occ_lo = flood_occ_lo if flood_occ_lo is not None else flood_occ
        _occ_hi = flood_occ_hi if flood_occ_hi is not None else flood_occ
        rng_flood = np.random.default_rng(rng_seed + 4)
        if random_flood:
            # N log-uniform; occ solved per-water to hit a target peak amplitude,
            # then clipped to maintain Rfree~11% across all N values.
            # Auto-clip: occ_rms × √N = FLOOD_LINE_K → clip = FLOOD_LINE_K_SYM/√N
            # (same calibration as vary_flood but occ distribution shaped by peak_sigma).
            # --flood-occ-max overrides the auto-clip.
            log_nf  = rng_flood.uniform(np.log(max(FLOOD_NF_MIN, 1)),
                                        np.log(max(FLOOD_NF_MAX, 1)))
            n_flood = int(np.round(np.exp(log_nf)))
            _clip   = flood_occ_max_clip   # None = no clip (default)
        elif vary_flood:
            # Controlled Rfree ~11%: N random, occ scaled to compensate.
            log_nf  = rng_flood.uniform(np.log(FLOOD_NF_MIN), np.log(FLOOD_NF_MAX))
            n_flood = int(np.round(np.exp(log_nf)))
            if flood_rfree_target is not None:
                k_sym = _vary_flood_k(flood_b_lo, flood_b_hi,
                                      flood_rfree_target, flood_floor) * 3**0.5
            else:
                k_sym = FLOOD_LINE_K_SYM
            mid = k_sym / np.sqrt(n_flood)
            _occ_lo = -mid
            _occ_hi =  mid

        existing_xyzs = [] if flood_min_dist <= 0 else [
            [atom.pos.x, atom.pos.y, atom.pos.z]
            for chain in st_jig_r[0]
            for res in chain
            for atom in res
        ]
        flood_pos = place_flood_waters(
            st_jig_r.cell, existing_xyzs, n_flood, None,
            seed=rng_seed + 1, min_dist=flood_min_dist,
        )
        if random_flood:
            flood_occs, flood_bisos = sample_flood_waters_by_peak(
                n_flood, flood_b_lo, flood_b_hi, flood_peak_sigma,
                seed=rng_seed + 7,
                occ_max_clip=_clip,
                obs_mtz_path=obs_mtz_path, gt48_mtz_path=mtz_path,
            )
            _occ_lo = float(flood_occs.min())
            _occ_hi = float(flood_occs.max())
        else:
            flood_occs  = sample_flood_occs(n_flood, _occ_lo, _occ_hi, seed=rng_seed + 7)
            flood_bisos = sample_flood_bfactors(n_flood, flood_b_lo, flood_b_hi, seed=rng_seed + 8)

        # 4. Build truth_full.pdb from conf_data_truth + structural HOH + flood waters.
        #    Truth reflects swap A's conformer assignment hypothesis.
        truth_pdb = tmpdir / 'truth_full.pdb'
        write_full_conf_pdb(conf_data_truth, chain_names, st_jig_r,
                            flood_pos, flood_occs,
                            st_jig_r.cell, st_jig_r.spacegroup_hm, truth_pdb,
                            flood_biso=flood_bisos)
        t = _t('flood_waters', t)

        # 5. Build fobs.mtz (FP = |F_truth + F_bulk|) and truth.mtz (sfcalc of truth_full.pdb)
        fobs_mtz, truth_mtz = build_sample_mtz(truth_pdb, mtz_path, obs_mtz_path, tmpdir)
        t = _t('build_mtz', t)

        # 6. Build partial model: select k_conformers chains by maximin, combine via ref atom order
        starthere_pdb = tmpdir / 'starthere.pdb'
        n_alt = build_starthere_pdb(
            chain_names, conf_data_model, st_jig_r,
            k=k_conformers, ref_pdb=pdb_path,
            out_pdb=starthere_pdb, workdir=tmpdir,
        )
        # Add structural waters (from jiggled model — HOH not jiggled)
        add_hoh_chains_to_pdb(st_jig_r, starthere_pdb)
        t = _t('build_partial', t)

        # 7a. Refmac with flood waters
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
        shutil.copy2(truth_pdb,     sample_dir / 'truth_full.pdb')
        shutil.copy2(starthere_pdb, sample_dir / 'partial.pdb')
        shutil.copy2(tmpdir / 'refmac.log', sample_dir / 'refmac.log')
        shutil.copy2(fobs_mtz,      sample_dir / 'refme.mtz')
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
            swaps_per_residue=swaps_per_residue,
            n_flood=int(n_flood),
            flood_mode=('random' if random_flood else 'vary' if vary_flood else 'fixed'),
            flood_occ_lo=float(_occ_lo),
            flood_occ_hi=float(_occ_hi),
            flood_peak_sigma=float(flood_peak_sigma) if random_flood else None,
            flood_occ_max_clip=(float(_clip) if _clip is not None else None) if random_flood else None,
            flood_sigma_fofc=float(_real_fofc_sigma()) if random_flood else None,
            flood_min_dist=flood_min_dist,
            flood_b_lo=float(flood_b_lo),
            flood_b_hi=float(flood_b_hi),
            ncyc=ncyc,
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
                       flood_occ, flood_occ_lo, flood_occ_hi,
                       flood_b_lo, flood_b_hi, flood_peak_sigma, flood_occ_max_clip,
                       flood_min_dist, flood_nf_range,
                       vary_flood, random_flood, ncyc, swaps_per_residue,
                       max_array, seed, partition,
                       k_conformers=DEFAULT_K_CONFORMERS,
                       flood_rfree_target=None, flood_floor=FLOOD_FLOOR,
                       account=None, qos=None, time=None):
    script = SCRIPT_DIR / f'_slurm_{outdir.name}.sh'
    me     = Path(__file__).resolve()

    if random_flood:
        flood_args = ['  --random-flood \\',
                      f'  --flood-peak-sigma {flood_peak_sigma} \\']
        if flood_occ_max_clip is not None:
            flood_args += [f'  --flood-occ-max {flood_occ_max_clip} \\']
        if flood_nf_range:
            flood_args += [f'  --flood-nf-range {flood_nf_range[0]} {flood_nf_range[1]} \\']
    elif vary_flood:
        flood_args = ['  --vary-flood \\']
        if flood_nf_range:
            flood_args += [f'  --flood-nf-range {flood_nf_range[0]} {flood_nf_range[1]} \\']
        if flood_rfree_target is not None:
            flood_args += [f'  --flood-rfree-target {flood_rfree_target} \\']
            if flood_floor != FLOOD_FLOOR:
                flood_args += [f'  --flood-floor {flood_floor} \\']
    elif flood_occ_lo is not None or flood_occ_hi is not None:
        lo = flood_occ_lo if flood_occ_lo is not None else flood_occ
        hi = flood_occ_hi if flood_occ_hi is not None else flood_occ
        flood_args = [
            f'  --n-flood {n_flood} \\',
            f'  --flood-occ-range {lo} {hi} \\',
        ]
    else:
        flood_args = [
            f'  --n-flood {n_flood} \\',
            f'  --flood-occ {flood_occ} \\',
        ]
    flood_args += [f'  --flood-b-range {flood_b_lo} {flood_b_hi} \\']
    if flood_min_dist != 0.0:
        flood_args += [f'  --flood-min-dist {flood_min_dist} \\']
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
    ] + ([f'#SBATCH --time={time}']    if time    else []) \
      + ([f'#SBATCH --account={account}'] if account else []) \
      + ([f'#SBATCH --qos={qos}']         if qos     else []) + [
        '#SBATCH --export=ALL',
        '',
        'mkdir -p "${CCP4_SCR:-/tmp}"',
        'IDX=$SLURM_ARRAY_TASK_ID',
        f'{sys.executable} {me} \\',
        f'  --sample-id $IDX \\',
        f'  --outdir {outdir} \\',
        f'  --pdb {pdb} \\',
        f'  --mtz {mtz} \\',
        f'  --obs-mtz {obs_mtz} \\',
        f'  --shift-scale {shift_scale} \\',
    ] + flood_args + [
        f'  --ncyc {ncyc} \\',
        f'  --k-conformers {k_conformers} \\',
        f'  --swaps-per-residue {swaps_per_residue} \\',
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
    global FLOOD_NF_MIN, FLOOD_NF_MAX
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
    ap.add_argument('--flood-occ',   type=float, default=DEFAULT_FLOOD_OCC,
                    help='Uniform occupancy for all flood waters (default: %.3f)' % DEFAULT_FLOOD_OCC)
    ap.add_argument('--flood-occ-range', type=float, nargs=2, metavar=('LO', 'HI'),
                    default=None,
                    help='Draw per-water occ uniformly from [LO, HI]; may be negative.'
                         ' Overrides --flood-occ. E.g. --flood-occ-range -0.1 0.1')
    ap.add_argument('--flood-b-range', type=float, nargs=2, metavar=('LO', 'HI'),
                    default=[1.0, 15.0],
                    help='Draw per-water B log-uniformly from [LO, HI] Å².'
                         ' Small B = sharp bumps; large B = diffuse blobs.'
                         ' Default: 1.0 15.0 (matches gt48 atom B range).')
    ap.add_argument('--flood-nf-range', type=int, nargs=2, metavar=('MIN', 'MAX'),
                    default=None,
                    help='For --vary-flood: draw n_flood log-uniformly from [MIN, MAX].'
                         ' Default: %d %d.' % (FLOOD_NF_MIN, FLOOD_NF_MAX))
    ap.add_argument('--flood-min-dist', type=float, default=0.0,
                    help='Min distance from existing atoms when placing flood waters.'
                         ' 0 = everywhere (default). Old default was 2.0 Å.')
    ap.add_argument('--vary-flood',  action='store_true',
                    help='N log-uniform [nf-min,nf-max]; occ scaled to hit target Rfree.'
                         ' Use --flood-rfree-target to set the target (default 0.11).'
                         ' FLOOD_LINE_K is computed analytically from --flood-b-range.')
    ap.add_argument('--flood-rfree-target', type=float, default=None, metavar='RFREE',
                    help='For --vary-flood: target Rfree (default: use hardcoded FLOOD_LINE_K=3.07).'
                         ' FLOOD_LINE_K is derived analytically: K=(target-floor)/slope,'
                         ' slope=_C_RF*Z*sqrt(E_B[ano_ff^2]/3) from --flood-b-range.')
    ap.add_argument('--flood-floor', type=float, default=FLOOD_FLOOR, metavar='RFREE',
                    help=f'For --vary-flood with --flood-rfree-target: Rfree floor'
                         f' (default {FLOOD_FLOOR}, calibrated shift_scale=0, k=32).')
    ap.add_argument('--random-flood', action='store_true',
                    help='N log-uniform [nf-min,nf-max]; B log-uniform [b-lo,b-hi];'
                         ' occ solved per-water to hit target peak amplitude'
                         ' Uniform[0, peak-sigma × σ_gt48] with random sign.'
                         ' Decouples Rfree (peak-sigma), fofc SNR (N), shape (B).')
    ap.add_argument('--flood-occ-max', type=float, default=None,
                    help='For --random-flood: clip per-water |occ| to this maximum.'
                         ' Limits SF contribution of high-B waters that would otherwise'
                         ' blow up Rfree. Calibrate with occ_rms × √N = 1.0 for Rfree~11%%.'
                         ' Default: no clip (occ unconstrained).')
    ap.add_argument('--flood-peak-sigma', type=float, default=3.0,
                    help='For --random-flood: target peak amplitude as multiple of'
                         ' gt48 map RMS (σ_gt48). Each water gets peak drawn from'
                         ' Uniform[0, flood-peak-sigma × σ_gt48] with random sign.'
                         ' Refmac absorbs part of the signal — calibrate empirically.'
                         ' Default: 3.0')
    ap.add_argument('--ncyc',        type=int,   default=DEFAULT_NCYC)
    ap.add_argument('--swaps-per-residue', type=float, default=0.0,
                    help='Expected pairwise conformer swaps per residue in partial model'
                         ' (0=none, 1=one swap per residue, Poisson-sampled; default: 0)')
    ap.add_argument('--seed',        type=int,   default=42)
    ap.add_argument('--workers',     type=int,   default=1)
    ap.add_argument('--max-array',   type=int,   default=300)
    ap.add_argument('--partition',   default='debug')
    ap.add_argument('--account',     default=None,
                    help='SLURM account (e.g. pc_als831)')
    ap.add_argument('--qos',         default=None,
                    help='SLURM QOS (e.g. lr_normal)')
    ap.add_argument('--time',        default=None,
                    help='SLURM walltime per task (omitted by default — no limit)')
    ap.add_argument('--k-conformers', type=int, default=DEFAULT_K_CONFORMERS,
                    help='Number of conformer chains in partial model (default: %d)' % DEFAULT_K_CONFORMERS)
    ap.add_argument('--submit',      action='store_true')
    ap.add_argument('--debug',       action='store_true')
    args = ap.parse_args()

    pdb_path     = Path(args.pdb).resolve()
    mtz_path     = Path(args.mtz).resolve()
    obs_mtz_path = Path(args.obs_mtz).resolve()
    outdir       = Path(args.outdir).resolve()

    flood_occ_lo = args.flood_occ_range[0] if args.flood_occ_range else None
    flood_occ_hi = args.flood_occ_range[1] if args.flood_occ_range else None
    flood_b_lo        = args.flood_b_range[0]
    flood_b_hi        = args.flood_b_range[1]
    flood_peak_sigma  = args.flood_peak_sigma
    flood_occ_max_clip = args.flood_occ_max
    if args.flood_nf_range:
        FLOOD_NF_MIN = args.flood_nf_range[0]
        FLOOD_NF_MAX = args.flood_nf_range[1]

    if args.submit:
        outdir.mkdir(parents=True, exist_ok=True)
        submit_slurm_array(
            args.nsamples, outdir, pdb_path, mtz_path, obs_mtz_path,
            args.shift_scale, args.n_flood, args.flood_occ,
            flood_occ_lo, flood_occ_hi, flood_b_lo, flood_b_hi,
            flood_peak_sigma, flood_occ_max_clip,
            args.flood_min_dist, args.flood_nf_range,
            args.vary_flood, args.random_flood, args.ncyc, args.swaps_per_residue,
            args.max_array, args.seed, args.partition,
            k_conformers=args.k_conformers,
            flood_rfree_target=args.flood_rfree_target, flood_floor=args.flood_floor,
            account=args.account, qos=args.qos, time=args.time,
        )
        return

    common_kw = dict(
        pdb_path=pdb_path, mtz_path=mtz_path, obs_mtz_path=obs_mtz_path,
        k_conformers=args.k_conformers,
        shift_scale=args.shift_scale, n_flood=args.n_flood,
        flood_occ=args.flood_occ, flood_occ_lo=flood_occ_lo, flood_occ_hi=flood_occ_hi,
        flood_b_lo=flood_b_lo, flood_b_hi=flood_b_hi,
        flood_peak_sigma=flood_peak_sigma,
        flood_occ_max_clip=flood_occ_max_clip,
        flood_min_dist=args.flood_min_dist,
        vary_flood=args.vary_flood, random_flood=args.random_flood,
        flood_rfree_target=args.flood_rfree_target, flood_floor=args.flood_floor,
        ncyc=args.ncyc, swaps_per_residue=args.swaps_per_residue,
        seed=args.seed, debug=args.debug,
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
