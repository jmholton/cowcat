#!/usr/bin/env python3
"""
generate_protein.py — Generate protein altloc training data for the CNN.

Pipeline for each sample:
  1.  Random 20-residue amino acid sequence (natural frequencies)
  2.  Build backbone (build_n2c.awk) with Ramachandran-sampled phi/psi
  3.  Build side chains (build_side.awk) with Ponder-Richards chi rotamers
  4.  Add random water molecules
  5.  Add CRYST1 (P1 40×40×40 Å), centre in box, randomise B factors
  6.  phenix.geometry_minimization  →  minimized.pdb (clean geometry)
  7.  jigglepdb.awk × 2 seeds  →  two conformers merged as altloc A/B
  8.  phenix.geometry_minimization  →  truth_full.pdb  (ground truth)
  9.  gemmi sfcalc truth_full.pdb  →  truth.mtz
  10. Build refme.mtz (F=|FC|, SIGF=0.02·|FC|)
  11. Extract single conformer (altloc A)  →  starthere.pdb
  12. refmac5: 20 cycles on starthere.pdb  →  refmacout.mtz
  13. Convert MTZ columns  →  CCP4 .map files

Output per sample directory:
  truth.map        ground-truth density (FC/PHIC of truth_full, both conformers)
  2fofc.map        2Fo-Fc  (FWT/PHWT from refmac, 1-conf model)
  fofc.map         Fo-Fc   (DELFWT/PHDELWT from refmac)
  fc.map           Fc density  (FC/PHIC from refmac, 1-conf model)
  truth_full.pdb   multi-conf ground truth
  partial.pdb      single-conf starting model (= starthere.pdb)
  refmacout.pdb    refmac output coordinates
  refmac.log       refmac log
  metadata.json

Usage:
  # Generate 1000 samples in parallel via SLURM array:
  python generate_protein.py --submit --nsamples 1000 --outdir data_protein_n20_n1000

  # Single sample (used by the array job):
  python generate_protein.py --sample-id 42 --outdir data_protein_n20_n1000

  # Local multiprocess run:
  python generate_protein.py --nsamples 50 --outdir data_test --workers 4
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import gemmi
import numpy as np
from scipy.ndimage import uniform_filter

# ── Tool paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
BUILD_N2C   = SCRIPT_DIR / 'build_n2c.awk'
BUILD_SIDE  = SCRIPT_DIR / 'build_side.awk'
JIGGLEPDB   = SCRIPT_DIR / 'jigglepdb.awk'
PHENIX_GM   = Path(shutil.which('phenix.geometry_minimization') or 'phenix.geometry_minimization')
REFMAC5     = Path(shutil.which('refmac5') or 'refmac5')
UNIQUEIFY   = 'uniqueify'

# ── Crystallographic parameters ────────────────────────────────────────────────
CELL        = (40.0, 40.0, 40.0)   # P1 unit cell (Å)
DMIN        = 2.0                   # resolution cutoff (Å)
SAMPLE_RATE = 3.0                   # oversampling → 60×60×60 grid for 40 Å cell
SPACEGROUP  = 'P 1'                 # space group HM symbol

# ── Flood water iso-Rfree calibration ──────────────────────────────────────────
# Rfree = FLOOD_A * occ * sqrt(n_flood) + FLOOD_B  (calibrated on 1AHO grid)
# Rfree ~11% lies on occ * sqrt(n_flood) = FLOOD_LINE_K
FLOOD_LINE_K  = 5.27
FLOOD_NF_MIN  = 700
FLOOD_NF_MAX  = 4000
BFAC_MU     = np.log(20.0)
BFAC_SIGMA  = 0.7
BFAC_MIN    = 5.0
BFAC_MAX    = 120.0

# Main-chain atoms get lower B factors than side-chain atoms
MAINCHAIN_ATOMS = frozenset({'N', 'CA', 'C', 'O', 'OXT'})
BFAC_MC_MU    = np.log(12.0)   # main chain: lower mean
BFAC_MC_SIGMA = 0.5
BFAC_MC_MIN   = 4.0
BFAC_MC_MAX   = 50.0
BFAC_SC_MU    = np.log(28.0)   # side chain / water: higher mean
BFAC_SC_SIGMA = 0.7
BFAC_SC_MIN   = 5.0
BFAC_SC_MAX   = 120.0

# Per-atom RMSF (Å) derived from 1AHO/gt48.pdb (49 conformers).
# Used by _set_target_bfactors() to override B = 8π²·RMSF²/3 before jigglepdb.
# Atoms not listed fall back to the self-refined B factor.
_GT48_RMSF = {
    ('ALA','N'):0.355,('ALA','CA'):0.380,('ALA','C'):0.333,('ALA','O'):0.412,('ALA','CB'):0.498,
    ('ARG','N'):0.407,('ARG','CA'):0.412,('ARG','C'):0.363,('ARG','O'):0.455,('ARG','CB'):0.545,('ARG','CG'):0.757,('ARG','CD'):0.995,('ARG','NE'):1.332,('ARG','CZ'):1.643,('ARG','NH1'):1.878,('ARG','NH2'):2.277,
    ('ASN','N'):0.336,('ASN','CA'):0.327,('ASN','C'):0.320,('ASN','O'):0.395,('ASN','CB'):0.385,('ASN','CG'):0.392,('ASN','OD1'):0.467,('ASN','ND2'):0.514,
    ('ASP','N'):0.400,('ASP','CA'):0.428,('ASP','C'):0.418,('ASP','O'):0.548,('ASP','CB'):0.539,('ASP','CG'):0.823,('ASP','OD1'):1.060,('ASP','OD2'):1.069,
    ('CYS','N'):0.288,('CYS','CA'):0.277,('CYS','C'):0.270,('CYS','O'):0.360,('CYS','CB'):0.310,('CYS','SG'):0.423,
    ('GLN','N'):0.229,('GLN','CA'):0.262,('GLN','C'):0.258,('GLN','O'):0.336,('GLN','CB'):0.318,('GLN','CG'):0.397,('GLN','CD'):0.437,('GLN','OE1'):0.518,('GLN','NE2'):0.549,
    ('GLU','N'):0.334,('GLU','CA'):0.368,('GLU','C'):0.341,('GLU','O'):0.426,('GLU','CB'):0.450,('GLU','CG'):0.837,('GLU','CD'):1.014,('GLU','OE1'):1.413,('GLU','OE2'):1.717,
    ('GLY','N'):0.342,('GLY','CA'):0.389,('GLY','C'):0.347,('GLY','O'):0.540,
    ('HIS','N'):0.358,('HIS','CA'):0.380,('HIS','C'):0.374,('HIS','O'):0.533,('HIS','CB'):0.433,('HIS','CG'):0.503,('HIS','ND1'):1.010,('HIS','CD2'):0.813,('HIS','CE1'):0.994,('HIS','NE2'):0.744,
    ('ILE','N'):0.269,('ILE','CA'):0.284,('ILE','C'):0.276,('ILE','O'):0.302,('ILE','CB'):0.304,('ILE','CG1'):0.324,('ILE','CG2'):0.389,('ILE','CD1'):0.411,
    ('LEU','N'):0.352,('LEU','CA'):0.348,('LEU','C'):0.354,('LEU','O'):0.435,('LEU','CB'):0.385,('LEU','CG'):0.422,('LEU','CD1'):0.572,('LEU','CD2'):0.596,
    ('LYS','N'):0.345,('LYS','CA'):0.371,('LYS','C'):0.357,('LYS','O'):0.471,('LYS','CB'):0.468,('LYS','CG'):0.598,('LYS','CD'):0.762,('LYS','CE'):0.860,('LYS','NZ'):1.209,
    ('MET','N'):0.340,('MET','CA'):0.360,('MET','C'):0.340,('MET','O'):0.440,('MET','CB'):0.440,('MET','CG'):0.600,('MET','SD'):0.900,('MET','CE'):1.100,
    ('PHE','N'):0.302,('PHE','CA'):0.308,('PHE','C'):0.281,('PHE','O'):0.401,('PHE','CB'):0.416,('PHE','CG'):0.470,('PHE','CD1'):0.559,('PHE','CD2'):0.560,('PHE','CE1'):0.677,('PHE','CE2'):0.675,('PHE','CZ'):0.732,
    ('PRO','N'):0.310,('PRO','CA'):0.353,('PRO','C'):0.325,('PRO','O'):0.406,('PRO','CB'):0.470,('PRO','CG'):0.526,('PRO','CD'):0.404,
    ('SER','N'):0.352,('SER','CA'):0.325,('SER','C'):0.306,('SER','O'):0.358,('SER','CB'):0.342,('SER','OG'):0.367,
    ('THR','N'):0.298,('THR','CA'):0.303,('THR','C'):0.298,('THR','O'):0.373,('THR','CB'):0.373,('THR','OG1'):0.497,('THR','CG2'):0.603,
    ('TRP','N'):0.354,('TRP','CA'):0.382,('TRP','C'):0.376,('TRP','O'):0.418,('TRP','CB'):0.469,('TRP','CG'):0.470,('TRP','CD1'):0.535,('TRP','CD2'):0.482,('TRP','NE1'):0.548,('TRP','CE2'):0.517,('TRP','CE3'):0.541,('TRP','CZ2'):0.574,('TRP','CZ3'):0.605,('TRP','CH2'):0.614,
    ('TYR','N'):0.277,('TYR','CA'):0.277,('TYR','C'):0.263,('TYR','O'):0.350,('TYR','CB'):0.341,('TYR','CG'):0.329,('TYR','CD1'):0.387,('TYR','CD2'):0.399,('TYR','CE1'):0.447,('TYR','CE2'):0.461,('TYR','CZ'):0.454,('TYR','OH'):0.585,
    ('VAL','N'):0.403,('VAL','CA'):0.380,('VAL','C'):0.375,('VAL','O'):0.486,('VAL','CB'):0.458,('VAL','CG1'):0.556,('VAL','CG2'):0.582,
}
_PI2 = np.pi ** 2
def _rmsf_to_b(rmsf):
    return float(8.0 * _PI2 * rmsf**2 / 3.0)

# Per-residue-type conformer count for starthere.pdb, derived from
# 1aho/best_for_0038.pdb mean altloc count per residue type.
# Waters use a separate floor (WATER_MIN_NCONF).
# Residues not listed default to 3.
_GT48_NCONF = {
    'ALA':  4, 'ARG': 13, 'ASN':  7, 'ASP': 10, 'CYS':  7,
    'GLN':  8, 'GLU': 13, 'GLY':  5, 'HIS': 11, 'ILE':  3,
    'LEU':  8, 'LYS': 13, 'MET':  6, 'PHE':  7, 'PRO':  9,
    'SER':  3, 'THR':  7, 'TRP':  3, 'TYR':  6, 'VAL':  4,
}
WATER_MIN_NCONF = 3   # minimum ground-truth conformers per water


def _set_target_bfactors(pdb_path, rmsf_by_resnum=None):
    """Override B-factors in pdb_path with gt48-derived values (B=8π²RMSF²/3).

    Lookup priority:
      1. rmsf_by_resnum[(resnum, atom_name)]  — set when a reference PDB is
         given, encodes per-position disorder from that reference's conformer
         ensemble (overrides the per-restype average).
      2. _GT48_RMSF[(restype, atom_name)]     — per-restype averages from
         1AHO/gt48.pdb (the original generator default).
      3. existing atom B-factor                — used if neither table has an
         entry for this atom (e.g. unusual atoms).

    Waters (HOH) are intentionally skipped for the *table lookup* — their
    resnums often collide with protein resnums and the protein-derived RMSFs
    don't describe water disorder.  But waters that selfref refmac assigned
    a very large B (because the random placement landed in a no-density
    region) get their B capped to WATER_B_MAX so jiggle doesn't fly them
    apart in the subsequent step5 pass.

    Writes the modified structure back in place.
    """
    WATER_B_MAX = 30.0   # B² ⇒ RMSF ≤ ~0.6 Å, jiggle max ~0.6 Å per altloc
    st = gemmi.read_structure(str(pdb_path))
    for chain in st[0]:
        for res in chain:
            if res.name == 'HOH':
                for atom in res:
                    if atom.b_iso > WATER_B_MAX:
                        atom.b_iso = WATER_B_MAX
                continue
            for atom in res:
                if rmsf_by_resnum is not None:
                    rkey = (res.seqid.num, atom.name)
                    if rkey in rmsf_by_resnum:
                        atom.b_iso = _rmsf_to_b(rmsf_by_resnum[rkey])
                        continue
                key = (res.name, atom.name)
                if key in _GT48_RMSF:
                    atom.b_iso = _rmsf_to_b(_GT48_RMSF[key])
    st.write_pdb(str(pdb_path))
def _apply_bfac_table(pdb_path, bfac_table, b_floor=2.0):
    """Write mean-B values from bfac_table onto pdb_path in place.

    bfac_table: {(resnum, atom_name): mean_b_iso} — protein heavy atoms only.
    Waters (HOH) are left at their existing B values; non-table atoms unchanged.
    b_floor: minimum B after assignment (prevents unphysically small values).
    """
    st = gemmi.read_structure(str(pdb_path))
    for chain in st[0]:
        for res in chain:
            if res.name == 'HOH':
                continue
            rn = res.seqid.num
            for atom in res:
                b = bfac_table.get((rn, atom.name))
                if b is not None:
                    atom.b_iso = max(b_floor, float(b))
    st.write_pdb(str(pdb_path))


# ── Reference-PDB parser (boiled-from-reference mode) ────────────────────────
# Extract sequence, per-(resnum, atom) RMSF, and disulfide pairs from a
# multi-conformer reference PDB (e.g. 1aho/gt48.pdb).  Used by generate_sample
# when --reference-pdb is given: synthetic samples then have the reference's
# exact sequence, exact per-atom disorder pattern, and explicit SS bonds.

from functools import lru_cache

SS_DIST_THRESHOLD = 3.0   # Å — chain-A SG-SG distance under which two CYS are
                          #     considered disulfide-bonded in the reference.


@lru_cache(maxsize=4)
def _parse_reference(pdb_path):
    """Parse a multi-conformer reference PDB.

    Returns a dict with:
      sequence:     {resnum: residue_name}  (from chain A)
      rmsf_table:   {(resnum, atom_name): rmsf_angstroms}  (across all chains)
      disulfides:   [(resnum_lo, resnum_hi), ...] from chain-A SG-SG < 3 Å
      unpaired_cys: list of CYS resnums not part of any detected disulfide
      n_conf:       number of conformer chains found
      cell:         (a, b, c, α, β, γ)
      spacegroup_hm: H-M space group symbol

    Cached on pdb_path so re-calls in the same process are free.
    """
    st = gemmi.read_structure(str(pdb_path))
    model = st[0]
    chains = list(model)
    if not chains:
        raise RuntimeError(f'No chains in {pdb_path}')

    # Sequence + disulfide detection from chain A (the reference conformer).
    a = chains[0]
    sequence = {r.seqid.num: r.name for r in a}
    cys_res = [r for r in a if r.name == 'CYS']

    def _sg(res):
        for atom in res:
            if atom.name == 'SG':
                return atom.pos
        return None

    disulfides = []
    paired = set()
    # Pair each CYS with its closest CYS (mutual nearest-neighbour) under SS_DIST_THRESHOLD.
    sgs = [(r.seqid.num, _sg(r)) for r in cys_res]
    sgs = [(n, p) for n, p in sgs if p is not None]
    nearest = {}
    for i, (ni, pi) in enumerate(sgs):
        best_d, best_n = None, None
        for j, (nj, pj) in enumerate(sgs):
            if i == j:
                continue
            d = pi.dist(pj)
            if d <= SS_DIST_THRESHOLD and (best_d is None or d < best_d):
                best_d, best_n = d, nj
        if best_n is not None:
            nearest[ni] = (best_n, best_d)
    # Mutual nearest = disulfide
    for ni, (nj, d) in nearest.items():
        if nj in nearest and nearest[nj][0] == ni and ni < nj:
            disulfides.append((ni, nj))
            paired.add(ni); paired.add(nj)
    unpaired_cys = sorted(n for n, _ in sgs if n not in paired)

    # Per-(resnum, atom_name) RMSF across all conformer chains.
    # PROTEIN ONLY — waters (HOH) live in a separate chain in gt48 but share
    # resnums with protein residues, which would otherwise pollute the table
    # (e.g. a backbone-O entry at resnum N would mix LYS-O across 48 chains
    # with the water O at chain z resnum N, giving a huge fake RMSF).
    coords = {}   # (resnum, atom_name) → list of (x, y, z)
    for chain in chains:
        for res in chain:
            if res.name == 'HOH':
                continue
            rn = res.seqid.num
            for atom in res:
                if atom.element.name == 'H':
                    continue
                coords.setdefault((rn, atom.name), []).append(
                    (atom.pos.x, atom.pos.y, atom.pos.z))

    # k=2 clustering per atom — flag those whose two clusters are clearly
    # separated relative to the intra-cluster RMS.  When an atom is bimodal,
    # `rmsf_table` stores the *intra*-cluster σ (so Gaussian jiggle gives a
    # tight spread within each parent), and `bimodal_atoms` stores the inter-
    # cluster distance d (applied as a one-shot ±d/2 split between two parent
    # models in _apply_bimodal_split before the normal jiggle round).
    def _kmeans2(pts, max_iter=20):
        pts = np.asarray(pts, dtype=np.float64)
        d2 = ((pts[:, None] - pts[None, :]) ** 2).sum(-1)
        i, j = np.unravel_index(d2.argmax(), d2.shape)
        c = np.array([pts[i], pts[j]])
        for _ in range(max_iter):
            d = ((pts[:, None] - c[None]) ** 2).sum(-1)
            labels = d.argmin(1)
            new_c = np.array([pts[labels == k].mean(0) if (labels == k).any() else c[k]
                              for k in (0, 1)])
            if np.allclose(new_c, c):
                break
            c = new_c
        return c, labels

    # Mean B-factor per (resnum, atom_name) across conformers.
    bfac_sums  = {}   # key → [b_iso, ...]
    for chain in chains:
        for res in chain:
            if res.name == 'HOH':
                continue
            rn = res.seqid.num
            for atom in res:
                if atom.element.name == 'H':
                    continue
                bfac_sums.setdefault((rn, atom.name), []).append(atom.b_iso)
    bfac_table = {k: float(np.mean(v)) for k, v in bfac_sums.items()}

    rmsf_table   = {}
    bimodal_atoms = {}
    BIMODAL_MIN_SIGMA = 0.5    # only flag atoms with overall σ > 0.5 Å
    BIMODAL_MIN_SCORE = 3.0    # d_inter / σ_intra > 3 → clearly bimodal
    BIMODAL_MAX_DIST  = 5.0    # cap d_inter at physically plausible value
                               # (anything bigger is almost certainly an artefact
                               # of mis-aligned conformer chains in the reference)
    for key, pts in coords.items():
        if len(pts) < 4:
            if len(pts) >= 2:
                arr  = np.asarray(pts, dtype=np.float64)
                sigma = float(np.sqrt(((arr - arr.mean(0)) ** 2).sum(-1).mean()))
                rmsf_table[key] = sigma
            continue
        arr      = np.asarray(pts, dtype=np.float64)
        sigma_uni = float(np.sqrt(((arr - arr.mean(0)) ** 2).sum(-1).mean()))
        cs, labels = _kmeans2(arr)
        d_inter = float(np.linalg.norm(cs[0] - cs[1]))
        sigma_intra = float(np.sqrt(np.concatenate([
            ((arr[labels == 0] - cs[0]) ** 2).sum(-1),
            ((arr[labels == 1] - cs[1]) ** 2).sum(-1),
        ]).mean()))
        score = d_inter / max(sigma_intra, 1e-6)
        if (sigma_uni > BIMODAL_MIN_SIGMA
                and score > BIMODAL_MIN_SCORE
                and d_inter < BIMODAL_MAX_DIST):
            bimodal_atoms[key] = d_inter
            rmsf_table[key]    = sigma_intra
        else:
            rmsf_table[key]    = sigma_uni

    # Water sites: cluster all HOH-O atoms across conformer chains by spatial
    # proximity (single-linkage at 1.5 Å). Each cluster represents one
    # binding site in the reference; cluster_count = how many distinct
    # ordered-water sites the reference has.  Used by main() to auto-set
    # --nwaters in boiled mode so the synthetic sample matches the reference's
    # water density, even though the actual positions are still random.
    water_positions = []
    for chain in chains:
        for res in chain:
            if res.name != 'HOH':
                continue
            for atom in res:
                if atom.name == 'O':
                    water_positions.append((atom.pos.x, atom.pos.y, atom.pos.z))
    n_water_sites = 0
    if water_positions:
        wp = np.asarray(water_positions, dtype=np.float64)
        used = np.zeros(len(wp), dtype=bool)
        EPS_SQ = 1.5 ** 2
        for i in range(len(wp)):
            if used[i]:
                continue
            queue = [i]
            used[i] = True
            n_water_sites += 1
            while queue:
                head = queue.pop()
                d2 = ((wp - wp[head]) ** 2).sum(-1)
                for j in np.where((~used) & (d2 < EPS_SQ))[0]:
                    used[j] = True
                    queue.append(int(j))

    # High-resolution limit (Å) from REMARK 3 "RESOLUTION RANGE HIGH" if
    # present.  Set to None if the header doesn't expose it.
    dmin = None
    for line in Path(pdb_path).read_text().splitlines():
        if 'RESOLUTION RANGE HIGH' in line:
            try:
                dmin = float(line.split(':')[-1].split()[0])
            except (ValueError, IndexError):
                pass
            break

    return {
        'sequence':      sequence,
        'rmsf_table':    rmsf_table,
        'bfac_table':    bfac_table,
        'bimodal_atoms': bimodal_atoms,
        'disulfides':    disulfides,
        'unpaired_cys':  unpaired_cys,
        'n_conf':        len(chains),
        'n_water_sites': n_water_sites,
        'cell':          (st.cell.a, st.cell.b, st.cell.c,
                          st.cell.alpha, st.cell.beta, st.cell.gamma),
        'spacegroup_hm': st.spacegroup_hm,
        'dmin':          dmin,
    }


def _apply_bimodal_split(in_pdb, bimodal_atoms, parent_A_pdb, parent_B_pdb,
                         tmpdir, rng, chain='A',
                         brace_chain='Z', tight_sigma=0.3, brace_sigma=0.5,
                         brace_exaggerate=2.0,
                         disulfide_pairs=None, max_reasonable_bond=150.0):
    """Split selfref into two "parent" PDBs differing only at the bimodal
    atoms, with bonded geometry kept ideal by phenix.geometry_minimization.

    Mechanism:
      1. Read selfref; make 2 copies.  For each residue with ≥1 bimodal atom,
         pick a random unit vector and displace each bimodal atom by ±d/2
         along it in copy A / copy B.  Non-bimodal atoms stay put.  Bonds
         to non-bimodal partners are now snapped (e.g. GLY C=O).
      2. Combine both copies into one PDB: original chain → 'A', second copy
         → `brace_chain` (default 'Z').
      3. Build a .eff with extra bond restraints between every atom in chain
         A and its counterpart in chain Z:
            - non-bimodal atoms: distance_ideal = 0.001, sigma = tight_sigma
              (= "very short bonds to hold the rest of the molecule together")
            - bimodal atoms:     distance_ideal = d_inter, sigma = brace_sigma
              (= "braces to hold the bimodal atoms apart")
         Plus the standard SS-bond restraints (in BOTH chains) and
         excessive_bond_distance_limit so phenix accepts the snapped inputs.
      4. Run phenix.geometry_minimization on the combined PDB.  Each chain's
         bonded geometry relaxes toward ideal while the cross-chain braces
         hold the bimodal split in place.
      5. Split the minimised output back: chain A → parent_A_pdb,
         chain Z (renamed to A) → parent_B_pdb.

    Returns the number of bimodal atoms split.
    """
    # ── 1. per-residue random direction + per-copy displacement ──────────────
    bimodal_resnums = sorted({rn for (rn, _) in bimodal_atoms})
    res_dir = {}
    for rn in bimodal_resnums:
        phi   = float(rng.uniform(0, 2 * np.pi))
        cos_t = float(rng.uniform(-1, 1))
        sin_t = float(np.sqrt(1 - cos_t * cos_t))
        res_dir[rn] = (sin_t * np.cos(phi),
                       sin_t * np.sin(phi),
                       cos_t)

    st_A = gemmi.read_structure(str(in_pdb))
    st_B = gemmi.read_structure(str(in_pdb))
    n_split = 0
    # Only the protein chain (`chain`, default 'A') gets the bimodal split and
    # the brace-chain duplication.  Waters/other chains stay singular in the
    # combined PDB — they aren't in bimodal_atoms anyway, and duplicating them
    # would collide on resseq with the protein in the brace chain.
    protein_A = next((ch for ch in st_A[0] if ch.name == chain), None)
    protein_B = next((ch for ch in st_B[0] if ch.name == chain), None)
    if protein_A is None or protein_B is None:
        raise RuntimeError(f'_apply_bimodal_split: chain {chain!r} not found in {in_pdb}')
    # Initial split: a small fixed displacement (0.3 Å) in each direction —
    # just enough to break A/Z symmetry so the brace bonds know which way
    # to pull.  Avoids severing bonds in bimodal_combined.pdb (the input
    # to phenix); large d_inter atoms like THR27-CG2 at 2.75 Å would shear
    # off their CB partner if we displaced by full d_inter/2.  The brace
    # pass then pulls them apart to d_inter * brace_exaggerate while bonded
    # chemistry holds bond lengths near-ideal.
    INITIAL_DISP = 0.01
    for res_A, res_B in zip(protein_A, protein_B):
        rn = res_A.seqid.num
        if rn not in res_dir:
            continue
        ux, uy, uz = res_dir[rn]
        for atom_A, atom_B in zip(res_A, res_B):
            if (rn, atom_A.name) not in bimodal_atoms:
                continue
            half = INITIAL_DISP
            p = atom_A.pos
            atom_A.pos = gemmi.Position(p.x + half * ux,
                                        p.y + half * uy,
                                        p.z + half * uz)
            atom_B.pos = gemmi.Position(p.x - half * ux,
                                        p.y - half * uy,
                                        p.z - half * uz)
            n_split += 1

    # Capture (resnum, atom_name, pos) tuples NOW — gemmi invalidates chain
    # iterators after a subsequent add_chain on the same Model.
    protein_atoms = [(res.seqid.num, atom.name,
                      np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
                     for res in protein_A for atom in res]
    # Atoms within BIMODAL_BRACE_LOOSE_R of any bimodal atom get a LOOSE
    # brace (larger sigma) instead of the tight one.  Reason: the bimodal
    # atom moves by ±d/2 between parents; a bonded neighbour (e.g. GLY61-C
    # bonded to bimodal GLY61-O) braced rigidly at 0.001 Å between parents
    # cannot satisfy both C-O bond restraints simultaneously without
    # stretching C-O or the peptide.  A loose brace lets bonded neighbours
    # drift slightly between parents to track their displaced bimodal
    # partner.  We can't *remove* the brace — chain A and chain Z atoms
    # otherwise sit at identical positions and the nonbond repulsion
    # between them blows the structure apart.
    BIMODAL_BRACE_LOOSE_R = 2.5  # Å — covers 1-2 bonds out
    bimodal_xyz = [pos for (rn, name, pos) in protein_atoms
                   if (rn, name) in bimodal_atoms]
    if bimodal_xyz:
        bimodal_xyz = np.stack(bimodal_xyz)
    loose_keys = set()
    if len(bimodal_xyz):
        for (rn, name, pos) in protein_atoms:
            if (rn, name) in bimodal_atoms:
                continue
            if np.min(np.linalg.norm(bimodal_xyz - pos, axis=1)) < BIMODAL_BRACE_LOOSE_R:
                loose_keys.add((rn, name))

    # ── 2. combine into one PDB: both protein copies in chain A as altloc
    #     A / B (occ 0.50 each) so phenix treats them as alternate
    #     conformations and skips nonbond between A↔B.  Waters stay singular.
    protein_B.name = brace_chain
    st_A[0].add_chain(protein_B.clone())
    combined = tmpdir / 'bimodal_combined.pdb'
    st_A.write_pdb(str(combined))

    # Post-process: rewrite chain Z → chain A with altloc B; chain A → altloc A.
    # PDB columns (0-indexed): 16=altLoc, 21=chainID, 54:60=occupancy ('%6.2f').
    new_lines = []
    for line in combined.read_text().splitlines(keepends=True):
        if line.startswith(('ATOM  ', 'HETATM')):
            cid = line[21]
            if cid == chain:
                line = line[:16] + 'A' + line[17:54] + '  0.50' + line[60:]
            elif cid == brace_chain:
                line = (line[:16] + 'B' + line[17:21] + chain
                        + line[22:54] + '  0.50' + line[60:])
        new_lines.append(line)
    combined.write_text(''.join(new_lines))

    # ── 3. build .eff with cross-altloc bond restraints (protein only).
    #     altid A ↔ altid B brace bonds replace cross-chain bonds.  Atoms in
    #     loose_keys (near bimodal) get NO brace — altloc separation already
    #     turns off nonbond between them, so they're free to find their own
    #     bonded ideal in each altloc.
    bond_blocks = []
    for rn, name, _pos in protein_atoms:
        if (rn, name) in loose_keys:
            continue                     # free — nonbond off between altlocs
        if (rn, name) in bimodal_atoms:
            d_ideal = float(bimodal_atoms[(rn, name)]) * brace_exaggerate
            sigma   = brace_sigma
        else:
            d_ideal = 0.001
            sigma   = tight_sigma
        bond_blocks.append(
            f'    bond {{\n'
            f'      action = *add\n'
            f'      atom_selection_1 = chain {chain} and altid A and resseq {rn} and name {name}\n'
            f'      atom_selection_2 = chain {chain} and altid B and resseq {rn} and name {name}\n'
            f'      distance_ideal   = {d_ideal:.4f}\n'
            f'      sigma            = {sigma:.4f}\n'
            f'    }}'
        )
    # SS restraints inside each altloc so disulfides don't blow apart
    if disulfide_pairs:
        for altid in ('A', 'B'):
            for a, b in disulfide_pairs:
                bond_blocks.append(
                    f'    bond {{\n'
                    f'      action = *add\n'
                    f'      atom_selection_1 = chain {chain} and altid {altid} and resseq {a} and name SG\n'
                    f'      atom_selection_2 = chain {chain} and altid {altid} and resseq {b} and name SG\n'
                    f'      distance_ideal   = 2.05\n'
                    f'      sigma            = 0.05\n'
                    f'    }}'
                )
    eff_path = tmpdir / 'bimodal_braces.eff'
    eff_path.write_text(
        f'pdb_interpretation {{\n'
        f'  proceed_with_excessive_length_bonds = True\n'
        f'  max_reasonable_bond_distance        = {max_reasonable_bond}\n'
        f'}}\n'
        f'geometry_restraints {{\n'
        f'  edits {{\n'
        f'    excessive_bond_distance_limit = {max_reasonable_bond}\n'
        + '\n'.join(bond_blocks) + '\n'
        f'  }}\n'
        f'}}\n'
    )

    # ── 4. phenix.geometry_minimization on the combined PDB ──────────────────
    base_cmd = ['cdl=false', 'link_all=False', 'link_none=True',
                'link_ligands=False', 'correct_hydrogens=False']
    cmd = [PHENIX_GM, combined.name, eff_path.name] + base_cmd
    result = run(cmd, cwd=tmpdir, check=False)
    log_text = result.stdout.decode(errors='replace') + result.stderr.decode(errors='replace')
    (tmpdir / 'bimodal_braces.phenix.log').write_text(log_text)
    if result.returncode != 0:
        raise RuntimeError(
            f'phenix.geometry_minimization (bimodal braces) failed:\n{log_text[-2000:]}')
    minim = tmpdir / 'bimodal_combined_minimized.pdb'
    if not minim.exists():
        raise RuntimeError(f'phenix bimodal-brace output not found: {minim}')

    # ── 4b. intermediate pass: drop the LONG bimodal braces, keep only the
    #     short cross-chain "local consistency" braces at sigma=0.3.  With
    #     no long brace pulling them apart, the bimodal atoms relax to
    #     their bonded chemistry ideal in each chain.  Those ideals differ
    #     because the loose-near-bimodal atoms drifted in step 4, so the
    #     gap survives — but the bonded geometry is now clean.
    inter_bonds = []
    for rn, name, _pos in protein_atoms:
        if name != 'CA':
            continue
        if (rn, name) in bimodal_atoms:
            continue
        inter_bonds.append(
            f'    bond {{\n'
            f'      action = *add\n'
            f'      atom_selection_1 = chain {chain} and altid A and resseq {rn} and name {name}\n'
            f'      atom_selection_2 = chain {chain} and altid B and resseq {rn} and name {name}\n'
            f'      distance_ideal   = 0.0010\n'
            f'      sigma            = 0.3000\n'
            f'    }}'
        )
    if disulfide_pairs:
        for altid in ('A', 'B'):
            for a, b in disulfide_pairs:
                inter_bonds.append(
                    f'    bond {{\n'
                    f'      action = *add\n'
                    f'      atom_selection_1 = chain {chain} and altid {altid} and resseq {a} and name SG\n'
                    f'      atom_selection_2 = chain {chain} and altid {altid} and resseq {b} and name SG\n'
                    f'      distance_ideal   = 2.05\n'
                    f'      sigma            = 0.05\n'
                    f'    }}'
                )
    inter_eff = tmpdir / 'bimodal_intermediate.eff'
    inter_eff.write_text(
        f'pdb_interpretation {{\n'
        f'  proceed_with_excessive_length_bonds = True\n'
        f'  max_reasonable_bond_distance        = {max_reasonable_bond}\n'
        f'}}\n'
        f'geometry_restraints {{\n'
        f'  edits {{\n'
        f'    excessive_bond_distance_limit = {max_reasonable_bond}\n'
        + '\n'.join(inter_bonds) + '\n'
        f'  }}\n'
        f'}}\n'
    )
    cmd = [PHENIX_GM, minim.name, inter_eff.name] + base_cmd
    result = run(cmd, cwd=tmpdir, check=False)
    log_text = result.stdout.decode(errors='replace') + result.stderr.decode(errors='replace')
    (tmpdir / 'bimodal_intermediate.phenix.log').write_text(log_text)
    inter_minim = tmpdir / 'bimodal_combined_minimized_minimized.pdb'
    if result.returncode != 0 or not inter_minim.exists():
        print(f'    [bimodal intermediate diverged; using brace-pass output instead]',
              flush=True)
    else:
        minim = inter_minim  # use intermediate output for the split

    # ── 5. split back: minimised chain A altloc A → parent_A's chain A,
    #     minimised chain A altloc B → parent_B's chain A.  Altloc letters
    #     and occupancies are stripped (singular chain A with occ 1.0).
    #     Waters / other non-protein chains come unchanged from selfref.
    #     Done as PDB text rewrite to avoid gemmi altloc-handling quirks.
    minim_lines = minim.read_text().splitlines(keepends=True)

    # Header: CRYST1/SCALE/SSBOND from minim — drop LINK (brace bonds between
    # altlocs A/B — meaningless in the singular-altloc parent PDBs).
    header = []
    for line in minim_lines:
        if line.startswith(('ATOM  ', 'HETATM', 'TER', 'END')):
            break
        if not line.startswith('LINK'):
            header.append(line)

    # Nonchain lines (chain W waters) come from selfref unchanged; compute once
    # since both parents share the same water set.  Writing nonchain_lines AFTER
    # the protein ATOM block (no TER/END in between) ensures gemmi reads chain W
    # when it later parses parent_{A,B}.pdb.
    st_p = gemmi.read_structure(str(in_pdb))
    st_p[0].remove_chain(chain)
    nonchain_pdb = tmpdir / '_nonchain_template.pdb'
    st_p.write_pdb(str(nonchain_pdb))
    nonchain_lines = [ln for ln in nonchain_pdb.read_text().splitlines(keepends=True)
                      if ln.startswith(('ATOM  ', 'HETATM'))]

    for parent_path, keep_alt in ((parent_A_pdb, 'A'), (parent_B_pdb, 'B')):
        kept = []
        for line in minim_lines:
            if not line.startswith(('ATOM  ', 'HETATM')):
                continue
            if line[21] != chain:
                continue
            alt = line[16]
            if alt not in (' ', keep_alt):
                continue
            kept.append(line[:16] + ' ' + line[17:54] + '  1.00' + line[60:])
        parent_path.write_text(''.join(header) + ''.join(kept)
                               + ''.join(nonchain_lines) + 'END\n')

    # Check the bimodal geo for ring-threading clashes that phenix geommin
    # cannot resolve (the brace constraints can push side chains through
    # backbone atoms; gradient descent cannot escape the ring).  Delete
    # offending residues from both parents and from selfref (in_pdb) so the
    # clash does not propagate into jiggled conformers or truth_full.pdb.
    bimodal_geo = minim.with_suffix('.geo')
    bimodal_bad = _parse_geo_bad_nonbonds(bimodal_geo, same_altloc_only=True)
    if bimodal_bad:
        log.info('_apply_bimodal_split: deleting %d residues with unresolvable '
                 'bimodal clashes: %s', len(bimodal_bad), bimodal_bad)
        for p in (parent_A_pdb, parent_B_pdb, Path(in_pdb)):
            _delete_residues_from_pdb(p, bimodal_bad)

    # Remove waters from parent PDBs that now clash with protein atoms.
    # The bimodal brace can displace a side chain into a water position;
    # the resulting protein-water overlap (<2.0 Å) propagates through all
    # jiggle conformers and explodes into thousands of clashes in truth_full.
    WATER_CLASH_DIST = 2.0   # Å — remove water if any protein heavy atom is closer
    for parent_path in (parent_A_pdb, parent_B_pdb, Path(in_pdb)):
        st = gemmi.read_structure(str(parent_path))
        protein_pos = [a.pos
                       for ch in st[0] for res in ch for a in res
                       if ch.name == chain and a.element.name not in ('H', 'X')]
        waters_to_remove = set()
        for ch in st[0]:
            for res in ch:
                if res.name != 'HOH':
                    continue
                for atom in res:
                    if atom.element.name in ('H', 'X'):
                        continue
                    for pp in protein_pos:
                        if atom.pos.dist(pp) < WATER_CLASH_DIST:
                            waters_to_remove.add((ch.name, res.seqid.num))
                            break
        if waters_to_remove:
            log.info('_apply_bimodal_split %s: removing %d waters clashing with protein: %s',
                     parent_path.name, len(waters_to_remove), sorted(waters_to_remove))
            _delete_residues_from_pdb(parent_path, waters_to_remove)

    return n_split


def _ssbond_record(idx, resnum_a, resnum_b, dist=2.04, chain='A'):
    """Format a single SSBOND PDB record (column-precise).

    Columns: 1-6 SSBOND, 8-10 serNum, 12-14 resName, 16 chainID,
             18-21 resNum, 26-28 resName, 30 chainID, 32-35 resNum,
             60-65 sym1, 67-72 sym2, 74-78 length.
    """
    return (f'SSBOND {idx:3d} CYS {chain}{resnum_a:5d}    '
            f'CYS {chain}{resnum_b:5d}                          '
            f'1555   1555 {dist:5.2f}\n')


def _link_record(resnum_a, resnum_b, dist=2.04, chain='A'):
    """Format a PDB LINK record for a CYS-SG to CYS-SG disulfide.

    Column-precise per the PDB v3.3 LINK spec:
      13-16 name1, 17 altLoc1, 18-20 resName1, 22 chainID1, 23-26 resSeq1,
      43-46 name2, 47 altLoc2, 48-50 resName2, 52 chainID2, 53-56 resSeq2,
      60-65 sym1, 67-72 sym2, 74-78 length.
    """
    return (f'LINK         SG  CYS {chain}{resnum_a:4d}'
            f'                 SG  CYS {chain}{resnum_b:4d}'
            f'     1555   1555  {dist:4.2f}\n')


def _inject_ssbonds(pdb_path, disulfide_pairs, chain='A'):
    """Insert SSBOND + LINK records into a PDB file before the first ATOM line.

    Refmac obeys explicit LINK records over its own coordinate-based
    auto-detection; emitting both LINK and SSBOND ensures phenix and refmac
    are on the same page. No-op when disulfide_pairs is empty.  Idempotent if
    called twice (existing SSBOND/LINK lines are replaced).
    """
    if not disulfide_pairs:
        return
    p = Path(pdb_path)
    lines = p.read_text().splitlines(keepends=True)
    # Drop any existing SSBOND / LINK records for SS pairs.
    lines = [ln for ln in lines if not ln.startswith(('SSBOND', 'LINK'))]
    insert_at = next((i for i, ln in enumerate(lines)
                      if ln.startswith(('ATOM', 'HETATM'))), len(lines))
    ss_text   = [_ssbond_record(i + 1, a, b, chain=chain)
                 for i, (a, b) in enumerate(disulfide_pairs)]
    link_text = [_link_record(a, b, chain=chain)
                 for (a, b) in disulfide_pairs]
    lines[insert_at:insert_at] = ss_text + link_text
    p.write_text(''.join(lines))

# Altloc displacement threshold (Å): side chains further than this
# from their centroid stay as separate altloc atoms in the partial model
ALTLOC_DIST_THRESHOLD = 0.0

# ── Amino acid natural frequencies (UniProt statistics) ────────────────────────
_AA_DATA = [
    ('ALA', 0.083), ('ARG', 0.056), ('ASN', 0.040), ('ASP', 0.053),
    ('CYS', 0.017), ('GLN', 0.040), ('GLU', 0.063), ('GLY', 0.073),
    ('HIS', 0.022), ('ILE', 0.052), ('LEU', 0.091), ('LYS', 0.058),
    ('MET', 0.024), ('PHE', 0.039), ('PRO', 0.050), ('SER', 0.066),
    ('THR', 0.053), ('TRP', 0.013), ('TYR', 0.032), ('VAL', 0.066),
]
AA_NAMES = [a for a, _ in _AA_DATA]
AA_PROBS  = np.array([p for _, p in _AA_DATA], dtype=float)
AA_PROBS /= AA_PROBS.sum()

# Number of chi angles per residue type
N_CHI = {
    'GLY': 0, 'ALA': 0,
    'SER': 1, 'CYS': 1, 'VAL': 1, 'THR': 1,
    'LEU': 2, 'ILE': 2, 'PRO': 2, 'ASP': 2, 'ASN': 2,
    'HIS': 2, 'PHE': 2, 'TYR': 2, 'TRP': 2,
    'MET': 3, 'GLU': 3, 'GLN': 3,
    'LYS': 4, 'ARG': 5,
}

# ── Ramachandran mixture model ──────────────────────────────────────────────────
# Each component: (phi_mu, phi_sigma, psi_mu, psi_sigma, weight)
RAMA = [
    (-63,  12, -43,  12, 0.35),   # α-helix
    (-120, 20,  130, 20, 0.25),   # β-sheet
    (-65,  15,  150, 15, 0.25),   # PPII / extended coil
    ( 60,  12,   45, 12, 0.05),   # left-handed α
    (-80,  40,   80, 40, 0.10),   # broad coil / other
]
RAMA_W = np.array([c[4] for c in RAMA]); RAMA_W /= RAMA_W.sum()

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def run(cmd, cwd, input_bytes=None, check=True):
    """Run a subprocess; raise RuntimeError on non-zero exit."""
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        out = result.stdout.decode(errors='replace')
        err = result.stderr.decode(errors='replace')
        raise RuntimeError(
            f"Failed: {' '.join(str(c) for c in cmd)}\n"
            f"STDOUT: {out[-1500:]}\nSTDERR: {err[-1500:]}"
        )
    return result


def sample_phi_psi(rng, restype):
    """Sample (phi, psi) from Ramachandran mixture of Gaussians."""
    if restype == 'PRO':
        return float(rng.normal(-65, 5)), float(rng.normal(-40, 20))
    if restype == 'GLY':
        return float(rng.uniform(-180, 180)), float(rng.uniform(-180, 180))
    idx = rng.choice(len(RAMA), p=RAMA_W)
    mu_phi, sig_phi, mu_psi, sig_psi, _ = RAMA[idx]
    return float(rng.normal(mu_phi, sig_phi)), float(rng.normal(mu_psi, sig_psi))


def sample_chi(rng, n_chi, restype=''):
    """Sample n_chi chi angle labels from simplified P&R 3-state model."""
    opts = ['-', 't', '+']
    w_chi1  = [0.40, 0.35, 0.25]   # chi1: gauche- slightly preferred
    w_chi2p = [0.30, 0.40, 0.30]   # chi2+: trans preferred
    # PRO ring: only + (Cγ-endo) or - (Cγ-exo) puckers are valid
    if restype == 'PRO':
        return [rng.choice(['+', '-']) for _ in range(n_chi)]
    return [rng.choice(opts, p=(w_chi1 if i == 0 else w_chi2p))
            for i in range(n_chi)]


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ══════════════════════════════════════════════════════════════════════════════

def step1_build_backbone(seq, rng, tmpdir):
    """Build main chain with Ramachandran-sampled phi/psi → backbone.pdb."""
    lines = []
    for aa in seq:
        phi, psi = sample_phi_psi(rng, aa)
        lines.append(f"BUILD {aa} {phi:.1f} {psi:.1f} 180")
    cmd_text = '\n'.join(lines) + '\n'
    result = run(['awk', '-f', str(BUILD_N2C)], cwd=tmpdir,
                 input_bytes=cmd_text.encode())
    (tmpdir / 'backbone.pdb').write_bytes(result.stdout)


def step2_build_sidechains(seq, rng, tmpdir):
    """Add side chains with P&R chi rotamers via build_side.awk → side.pdb."""
    lines = []
    for i, aa in enumerate(seq, start=1):
        chis = sample_chi(rng, N_CHI.get(aa, 0), restype=aa)
        lines.append(f"BUILD {aa} {i} {' '.join(chis)}" if chis
                     else f"BUILD {aa} {i}")
    cmd_text = '\n'.join(lines) + '\n'
    backbone = (tmpdir / 'backbone.pdb').read_bytes()
    result = run(['awk', '-f', str(BUILD_SIDE)], cwd=tmpdir,
                 input_bytes=cmd_text.encode() + backbone)
    (tmpdir / 'side.pdb').write_bytes(result.stdout)


def step3_setup_structure(tmpdir, rng, n_waters=10):
    """Read side.pdb → set cell, centre, randomise B, add waters → built.pdb."""
    st = gemmi.read_structure(str(tmpdir / 'side.pdb'))
    st.cell = gemmi.UnitCell(CELL[0], CELL[1], CELL[2], 90, 90, 90)
    st.spacegroup_hm = SPACEGROUP
    is_p1 = (SPACEGROUP.replace(' ', '').upper() == 'P1')

    all_atoms = [a for model in st for chain in model
                 for res in chain for a in res]
    if not all_atoms:
        raise RuntimeError("No atoms after side-chain build")

    # Randomise B factors — main chain lower than side chain
    for a in all_atoms:
        if a.name in MAINCHAIN_ATOMS:
            a.b_iso = float(np.clip(rng.lognormal(BFAC_MC_MU, BFAC_MC_SIGMA),
                                    BFAC_MC_MIN, BFAC_MC_MAX))
        else:
            a.b_iso = float(np.clip(rng.lognormal(BFAC_SC_MU, BFAC_SC_SIGMA),
                                    BFAC_SC_MIN, BFAC_SC_MAX))
        a.occ = 1.0

    # (centering removed — let phenix/refmac handle positioning; for non-P1
    # boiled runs the random-coil centroid often crosses ASU boundaries when
    # forced to (a/4, b/4, c/2), triggering symmetry-mate clashes downstream.)

    # Set protein chain name to 'A'
    for chain in st[0]:
        chain.name = 'A'

    existing = [(a.pos.x, a.pos.y, a.pos.z) for a in all_atoms]

    # Ordered water placement bounds: ASU region for non-P1 to avoid symmetry-mate
    # clashes in refinement; full cell for P1.
    wo_x_hi = CELL[0]/2 - 2.0 if not is_p1 else CELL[0] - 2.0
    wo_y_hi = CELL[1]/2 - 2.0 if not is_p1 else CELL[1] - 2.0
    wo_z_hi = CELL[2] - 2.0
    margin   = 2.0

    # ── Pass 1: full-occupancy waters (avoid all existing atoms + each other) ──
    water_chain = gemmi.Chain('W')
    added = 0
    for _ in range(100000):
        if added >= n_waters:
            break
        x = float(rng.uniform(margin, wo_x_hi))
        y = float(rng.uniform(margin, wo_y_hi))
        z = float(rng.uniform(margin, wo_z_hi))
        if all((x-px)**2 + (y-py)**2 + (z-pz)**2 >= 7.84   # 2.8² Å
               for px, py, pz in existing):
            res = gemmi.Residue()
            res.name = 'HOH'
            res.seqid = gemmi.SeqId(added + 1, ' ')
            atom = gemmi.Atom()
            atom.name = 'O'
            atom.element = gemmi.Element('O')
            atom.pos = gemmi.Position(x, y, z)
            atom.occ = 1.0
            atom.b_iso = float(np.clip(rng.lognormal(BFAC_SC_MU, BFAC_SC_SIGMA),
                                       BFAC_SC_MIN, 80.0))
            res.add_atom(atom)
            water_chain.add_residue(res)
            existing.append((x, y, z))
            added += 1

    if added > 0:
        st[0].add_chain(water_chain)

    st.write_pdb(str(tmpdir / 'built.pdb'))
    return added


def _parse_geo_bad_nonbonds(geo_file, lj_threshold=10.0, bond_delta_threshold=0.3,
                            same_altloc_only=False):
    """Return set of (chain, resnum) pairs involved in severe clashes.

    Parses phenix .geo files for two clash signatures:

    1. Nonbonded: LJ energy > lj_threshold when obs < vdw_ideal.
       Formula from molprobify_runme.com:
         lj0(r, r0) = 4 * ((r0*2^(-1/6)/r)^12 - (r0*2^(-1/6)/r)^6)
         lj(r, r0)  = lj0(r, r0) - lj0(6, r0)   [shifted to 0 at r=6 Å]

    2. Bond lengths: |delta| > bond_delta_threshold (Å).  Severely stretched
       or compressed covalent bonds arise when atoms are threaded through each
       other or pushed apart by an irresolvable clash — both indicate clashing
       geometry that geometry_minimization could not fully relax.

    Only heavy-atom pairs are checked in both cases.
    Atom ID format in .geo: 15-char PDB string; chain at index 9, resnum 10:15,
    altloc at index 4.

    same_altloc_only: if True, skip pairs where the two atoms have different
      altloc letters.  Use when checking a bimodal-combined .geo to avoid
      flagging intentional cross-altloc contacts (A↔B brace restraints) as
      clashes — only same-altloc (A↔A or B↔B) contacts matter there.
    """
    def _lj(r, r0):
        if r <= 0:
            return 1e40
        s = r0 * 2 ** (-1 / 6)
        def lj0(r):
            return 4 * ((s / r) ** 12 - (s / r) ** 6)
        return lj0(r) - lj0(6.0)

    def _altloc(id_str):
        return id_str[4] if len(id_str) >= 5 else ' '

    def _add_pair(id1, id2):
        for id_str in (id1, id2):
            if len(id_str) >= 15:
                chain  = id_str[9]
                try:
                    resnum = int(id_str[10:15])
                    bad.add((chain, resnum))
                except ValueError:
                    pass

    bad = set()
    try:
        lines = Path(geo_file).read_text().splitlines()
    except OSError:
        return bad

    i = 0
    while i < len(lines):
        line = lines[i]
        if 'nonbonded pdb=' in line:
            m1 = re.search(r'"([^"]*)"', line)
            m2 = re.search(r'"([^"]*)"', lines[i + 1]) if i + 1 < len(lines) else None
            if m1 and m2 and i + 3 < len(lines):
                id1, id2 = m1.group(1), m2.group(1)
                atom1 = id1[0:4].strip() if len(id1) >= 4 else ''
                atom2 = id2[0:4].strip() if len(id2) >= 4 else ''
                if not atom1.startswith('H') and not atom2.startswith('H'):
                    if not same_altloc_only or _altloc(id1) == _altloc(id2):
                        parts = lines[i + 3].split()
                        if len(parts) >= 2:
                            try:
                                obs, ideal = float(parts[0]), float(parts[1])
                                if obs < ideal and _lj(obs, ideal) > lj_threshold:
                                    _add_pair(id1, id2)
                            except ValueError:
                                pass
            i += 4
        elif 'bond pdb=' in line:
            m1 = re.search(r'"([^"]*)"', line)
            m2 = re.search(r'"([^"]*)"', lines[i + 1]) if i + 1 < len(lines) else None
            if m1 and m2 and i + 3 < len(lines):
                id1, id2 = m1.group(1), m2.group(1)
                atom1 = id1[0:4].strip() if len(id1) >= 4 else ''
                atom2 = id2[0:4].strip() if len(id2) >= 4 else ''
                if not atom1.startswith('H') and not atom2.startswith('H'):
                    if not same_altloc_only or _altloc(id1) == _altloc(id2):
                        parts = lines[i + 3].split()
                        # bond data line: ideal  model  delta  sigma  weight  residual
                        if len(parts) >= 3:
                            try:
                                if abs(float(parts[2])) > bond_delta_threshold:
                                    _add_pair(id1, id2)
                            except ValueError:
                                pass
            i += 4
        else:
            i += 1
    return bad


def _swap_cryst1(pdb_path, new_a, new_b, new_c, new_sg='P 1', restore=None):
    """Replace the CRYST1 line in pdb_path with the given cell, optionally
    returning the original line so the caller can restore it later.

    If restore is given, it should be the previously returned CRYST1 line
    (including the trailing newline) and the function will swap it back in
    place of whatever CRYST1 line currently exists in the file.
    """
    p = Path(pdb_path)
    lines = p.read_text().splitlines(keepends=True)
    orig = None
    for i, ln in enumerate(lines):
        if ln.startswith('CRYST1'):
            orig = ln
            break
    new_line = (f'CRYST1{new_a:9.3f}{new_b:9.3f}{new_c:9.3f}'
                f'  90.00  90.00  90.00 {new_sg:<11s}\n')
    if restore is not None:
        new_line = restore
    if orig is None:
        lines.insert(0, new_line)
    else:
        lines[i] = new_line
    p.write_text(''.join(lines))
    return orig


def step4_phenix_geommin(pdb_name, tmpdir, log_tag=None,
                         disulfide_pairs=None, chain='A',
                         max_reasonable_bond=150.0):
    """Run phenix.geometry_minimization; return path to *_minimized.pdb.

    When disulfide_pairs is given (list of (resnum_a, resnum_b)), writes a
    .eff parameter file with explicit `geometry_restraints.edits.bond` entries
    for each pair (distance_ideal=2.05, sigma=0.05) plus
    `proceed_with_excessive_length_bonds=True` and a large
    `max_reasonable_bond_distance` so phenix accepts random-coil starting
    geometries.  Phenix hard-caps the bond-distance limit at the shortest
    cell axis, so we also temporarily widen the cell in the input PDB to
    `max_reasonable_bond` and restore the original CRYST1 in the output.
    The minimiser's gradient pulls the SG atoms together.

    Saves stdout+stderr to {stem}{log_tag}.phenix.log in tmpdir.
    """
    cmd = [PHENIX_GM, pdb_name, 'cdl=false',
           'link_all=False', 'link_none=True', 'link_ligands=False',
           'correct_hydrogens=False']
    orig_cryst1 = None
    if disulfide_pairs:
        eff_path = Path(tmpdir) / f'{Path(pdb_name).stem}_disulfides.eff'
        bond_blocks = '\n'.join(f'''    bond {{
      action = *add
      atom_selection_1 = chain {chain} and resseq {a} and name SG
      atom_selection_2 = chain {chain} and resseq {b} and name SG
      distance_ideal   = 2.05
      sigma            = 0.05
    }}''' for a, b in disulfide_pairs)
        eff_text = f'''pdb_interpretation {{
  proceed_with_excessive_length_bonds = True
  max_reasonable_bond_distance        = {max_reasonable_bond}
}}
geometry_restraints {{
  edits {{
    excessive_bond_distance_limit = {max_reasonable_bond}
{bond_blocks}
  }}
}}
'''
        eff_path.write_text(eff_text)
        cmd.append(eff_path.name)
        # Only widen the cell to P1 when the SS bonds are *longer* than the
        # shortest cell axis (phenix's hard upper bound).  After the first
        # pull pass the SGs are within 2-3 Å, so a second symmetry-aware
        # pass can keep the target cell+SG (which is what packs the chain).
        st_in = gemmi.read_structure(str(Path(tmpdir) / pdb_name))
        cur_ss_max = 0.0
        chain_a = list(st_in[0])[0]
        sg_pos = {r.seqid.num: at.pos for r in chain_a if r.name == 'CYS'
                  for at in r if at.name == 'SG'}
        for a, b in disulfide_pairs:
            if a in sg_pos and b in sg_pos:
                cur_ss_max = max(cur_ss_max, sg_pos[a].dist(sg_pos[b]))
        shortest_axis = min(st_in.cell.a, st_in.cell.b, st_in.cell.c)
        if cur_ss_max > shortest_axis * 0.9:
            orig_cryst1 = _swap_cryst1(Path(tmpdir) / pdb_name,
                                       max_reasonable_bond, max_reasonable_bond,
                                       max_reasonable_bond, new_sg='P 1')
    result = run(cmd, cwd=tmpdir, check=False)
    stem = Path(pdb_name).stem
    log_name = f'{stem}{log_tag or ""}.phenix.log'
    log_text = result.stdout.decode(errors='replace') + result.stderr.decode(errors='replace')
    (tmpdir / log_name).write_text(log_text)
    if result.returncode != 0:
        raise RuntimeError(
            f'phenix.geometry_minimization failed for {pdb_name}:\n{log_text[-2000:]}')
    out = tmpdir / f'{stem}_minimized.pdb'
    if not out.exists():
        raise RuntimeError(f'phenix output not found: {out}')

    # Restore the original CRYST1 in both the input and the minimised output
    # so downstream steps see the real cell.
    if orig_cryst1 is not None:
        _swap_cryst1(Path(tmpdir) / pdb_name, 0, 0, 0, restore=orig_cryst1)
        _swap_cryst1(out, 0, 0, 0, restore=orig_cryst1)
    return out


def step4b_selfref_b_factors(minimized_pdb, tmpdir, target_wilson_b=None,
                              b_floor=5.0, water_b_floor=10.0):
    """Refine single-conf model against its own SFs to get realistic B factors.

    Runs refmac for 20 cycles with the model as both PDB and (synthetic) data
    source, allowing B-factor restraints to smooth out uncorrelated random B
    values into physically sensible, bonding-correlated ones.  The refined
    coordinate file is returned; its B factors are used by jigglepdb byB.

    target_wilson_b: if given, measure Wilson B of refined FC and subtract a
    constant offset from every atom's B-factor so the model's overall Wilson B
    matches target.  Preserves all per-atom B-factor correlations (only the
    overall level shifts).  Useful for matching real-data resolution falloff.
    b_floor: minimum B-factor after offset subtraction (default 2.0 Å²).
    """
    # Calculate SFs for the minimised model (no bulk solvent needed for selfref)
    step6_sfcalc(minimized_pdb, tmpdir / 'selfref.mtz', tmpdir, bulk_solvent=False)

    # Build a pseudo-observed MTZ (F=|FC|, SIGF=0.02·|FC|)
    step7_build_refme_mtz(tmpdir / 'selfref.mtz', tmpdir / 'selfref_refme.mtz')

    keywords = (
        b'MAKE HYDR NO NEWLIGAND NOEXIT LINK NO\n'
        b'NCYC 20\n'
        b'LABIN FP=F SIGFP=SIGF\n'
        b'LABOUT FC=FC PHIC=PHIC\n'
        b'MONI DIST 10\n'
        b'VDWREST 10\n'
        b'WEIGHT MATRIX 0.01\n'
        b'END\n'
    )
    result = subprocess.run(
        [str(REFMAC5),
         'XYZIN',  str(minimized_pdb),
         'XYZOUT', 'selfref_out.pdb',
         'HKLIN',  'selfref_refme.mtz',
         'HKLOUT', 'selfref_out.mtz',
         'LIBOUT',  'selfref.lib'],
        input=keywords,
        cwd=str(tmpdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_text = result.stdout.decode(errors='replace')
    (tmpdir / 'selfref_refmac.log').write_text(log_text)
    if result.returncode != 0:
        raise RuntimeError(f'selfref refmac5 failed:\n{log_text[-2000:]}')
    out = tmpdir / 'selfref_out.pdb'
    if not out.exists():
        raise RuntimeError('selfref_out.pdb not found after self-refinement')

    # Optional: shift overall B-factor level to match a target Wilson B
    if target_wilson_b is not None:
        mtz = gemmi.read_mtz_file(str(tmpdir / 'selfref_out.mtz'))
        h = np.asarray(mtz.column_with_label('H'),  dtype=np.int32)
        k = np.asarray(mtz.column_with_label('K'),  dtype=np.int32)
        l = np.asarray(mtz.column_with_label('L'),  dtype=np.int32)
        F = np.asarray(mtz.column_with_label('FC'), dtype=np.float32)
        cell = mtz.cell
        s2 = np.array([cell.calculate_1_d2([int(h_), int(k_), int(l_)])
                       for h_, k_, l_ in zip(h, k, l)], dtype=np.float64)
        gen_B   = _wilson_b(F, s2)
        delta_B = gen_B - target_wilson_b   # subtract this from every atom's B
        st = gemmi.read_structure(str(out))
        for chain in st[0]:
            for res in chain:
                floor = water_b_floor if res.name == 'HOH' else b_floor
                for atom in res:
                    atom.b_iso = max(floor, atom.b_iso - delta_B)
        st.write_pdb(str(out))
        print(f'    selfref Wilson B: {gen_B:.2f} → target {target_wilson_b:.2f}, '
              f'B-factor shift = {-delta_B:+.2f} Å²  '
              f'(b_floor={b_floor:.0f}, water_floor={water_b_floor:.0f})')

    return out


def _apply_flood_signs(pdb_path, rng):
    """Randomly flip the occupancy sign of each chain-F (flood) water.

    Half the flood waters get occ < 0 on average, creating negative peaks in
    the difference map (mirrors spurious solvent in the partial model).
    Sign is assigned per residue and applied uniformly across all its altlocs.
    """
    st = gemmi.read_structure(str(pdb_path))
    for chain in st[0]:
        if chain.name != 'F':
            continue
        for res in chain:
            sign = float(rng.choice([1, 1]))
            for atom in res:
                atom.occ = abs(atom.occ) * sign
    st.write_pdb(str(pdb_path))


def _align_conformers_by_ca(conf_pdbs, chain_name='A'):
    """Kabsch-align all conformers' protein-chain atoms to the first
    conformer using CA positions.

    After per-conformer phenix.geommin, each conformer can be rigid-body
    drifted from the others.  This rotates+translates every conformer
    (except the reference, conf_pdbs[0]) so their CA atoms in chain
    `chain_name` superpose by least squares.  ONLY atoms in `chain_name`
    are moved — waters and other chains stay where the per-conformer
    pipeline left them (e.g. ordered waters in chain W keep their
    independent jiggle positions instead of being dragged by the
    protein's rigid alignment).  Internal protein geometry is preserved.
    """
    if len(conf_pdbs) < 2:
        return
    ref = gemmi.read_structure(str(conf_pdbs[0]))
    ref_ca = np.array([
        (a.pos.x, a.pos.y, a.pos.z)
        for ch in ref[0] if ch.name == chain_name
        for res in ch for a in res if a.name == 'CA'
    ])
    if len(ref_ca) == 0:
        return
    ref_centroid = ref_ca.mean(axis=0)
    P = ref_ca - ref_centroid
    for conf in conf_pdbs[1:]:
        st = gemmi.read_structure(str(conf))
        mob_ca = np.array([
            (a.pos.x, a.pos.y, a.pos.z)
            for ch in st[0] if ch.name == chain_name
            for res in ch for a in res if a.name == 'CA'
        ])
        if len(mob_ca) != len(ref_ca):
            continue
        mob_centroid = mob_ca.mean(axis=0)
        Q = mob_ca - mob_centroid
        H = Q.T @ P
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, float(d)]) @ U.T
        for chain in st[0]:
            if chain.name != chain_name:
                continue  # waters / other chains stay put
            for res in chain:
                for atom in res:
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    q = R @ (p - mob_centroid) + ref_centroid
                    atom.pos = gemmi.Position(float(q[0]), float(q[1]), float(q[2]))
        st.write_pdb(str(conf))


def step5_jigglepdb_and_merge(selfref_pdb, tmpdir, rng, shift_scale=0.5, n_altlocs=2,
                              per_conf_geommin=True, bfac_source_pdb=None,
                              add_h_per_conf=True, jiggle_shift='byB'):
    """Run jigglepdb n_altlocs times, optionally minimize each conformer in
    parallel, then combine → multiconf.pdb with N protein chains (A, B, … occ=1/N)
    and N water chains (a, b, … occ=1/N).

    per_conf_geommin: if True, run phenix.geometry_minimization on each conformer
    (slow with large N — ~13s × n_altlocs). If False, use raw jigglepdb output.
    bfac_source_pdb: if given, replace B-factors in merged output with values
    looked up by (chain, resnum, atom_name) from this PDB. Used to restore
    selfref B-factors after jigglepdb has used gt48 target B-factors for
    displacement amplitude.
    add_h_per_conf: if True, run phenix.reduce on each conformer PDB in parallel
    after geommin so the merged multiconf.pdb already contains hydrogens (saves
    a slow ~5-minute reduce call on the full 20-altloc model downstream).
    """
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_altlocs]
    seeds  = [int(rng.integers(1000, 99999)) for _ in range(n_altlocs)]
    # 'uniformB' is not a built-in jigglepdb shift_opt; translate it.
    if jiggle_shift == 'uniformB':
        jigglepdb_shift = 'byB'
        extra_jiggle_args = ['-v', 'distribution=uniform']
    else:
        jigglepdb_shift = jiggle_shift
        extra_jiggle_args = []
    # selfref_pdb may be a single Path/str OR a list of N parent PDBs (one per
    # altloc).  The list form is used by the bimodal-split branch in
    # generate_sample to seed half the altlocs from each parent.
    if isinstance(selfref_pdb, (str, Path)):
        parent_per_altloc = [selfref_pdb] * n_altlocs
    else:
        parent_per_altloc = list(selfref_pdb)
        if len(parent_per_altloc) != n_altlocs:
            raise ValueError(
                f'step5_jigglepdb: got {len(parent_per_altloc)} parent PDBs '
                f'but n_altlocs={n_altlocs}; lengths must match')
    # Split each parent into protein-only (chain A) and waters (chain W) before
    # jigglepdb — waters are added back AFTER per-conf-geommin + CA-align so
    # they don't get scrambled by the heavy phenix relaxation each altloc gets.
    waters_pristine = tmpdir / 'waters_pristine.pdb'
    waters_written  = False
    protein_parents = []
    for i, parent in enumerate(parent_per_altloc):
        st = gemmi.read_structure(str(parent))
        # Extract waters once (from the first parent — both parents share waters)
        if not waters_written:
            st_w = gemmi.Structure()
            st_w.cell          = st.cell
            st_w.spacegroup_hm = st.spacegroup_hm
            mw = gemmi.Model('1')
            for ch in st[0]:
                if ch.name != 'A':
                    mw.add_chain(ch.clone())
            if len(mw):
                st_w.add_model(mw)
                st_w.write_pdb(str(waters_pristine))
            waters_written = True
        # Protein-only copy of the parent.  Capture chain names first —
        # gemmi invalidates chain references after remove_chain().
        non_a_chains = [ch.name for ch in st[0] if ch.name != 'A']
        for name in non_a_chains:
            st[0].remove_chain(name)
        prot_path = tmpdir / f'parent_protein_{i}.pdb'
        st.write_pdb(str(prot_path))
        protein_parents.append(prot_path)

    conf_pdbs = []
    for parent, seed, label in zip(protein_parents, seeds, labels):
        result = run(
            ['awk', '-f', str(JIGGLEPDB),
             '-v', f'seed={seed}',
             '-v', f'shift={jigglepdb_shift}',
             '-v', f'shift_scale={shift_scale}',
             '-v', 'dry_shift_scale=1.0',
             '-v', 'frac_thrubond=0.1',
             '-v', 'ncyc_thrubond=10',
             '-v', 'frac_magnforce=1.1',
             '-v', 'ncyc_magnforce=10',
             *extra_jiggle_args,
             str(parent)],
            cwd=tmpdir
        )
        p = tmpdir / f'conf{label}.pdb'
        p.write_bytes(result.stdout)
        conf_pdbs.append(p)

    if per_conf_geommin:
        print(f'  per-conf geo min')
        # Minimize each single-conformer PDB independently and in parallel.
        def _minimize_one(conf_pdb):
            return step4_phenix_geommin(conf_pdb.name, tmpdir, log_tag=f'_{conf_pdb.stem}')
        with ThreadPoolExecutor(max_workers=n_altlocs) as pool:
            conf_pdbs = list(pool.map(_minimize_one, conf_pdbs))
        # Each conformer is minimized independently, so they can drift apart
        # rigidly even when local geometry is fine.  Least-squares-align every
        # conformer to the first by CA positions to undo that drift.
        print(f'  aligning CA atoms')
        _align_conformers_by_ca(conf_pdbs)
    # else: use jigglepdb output directly (no per-conformer phenix.GM).
    # Geometry errors from jigglepdb are small (~0.05 Å bonds) given the
    # through-bond correlated displacement; refmac refinement in step9
    # corrects the partial model anyway.

    if add_h_per_conf:
        # phenix.reduce on each single-conformer PDB in parallel. Each call is
        # cheap (~1000 atoms), and 20-way concurrency keeps wall time ≈ 1 reduce
        # call. Replaces the slow single-call reduce on the merged 21K-atom model.
        def _reduce_one(conf_pdb):
            res = subprocess.run(
                [str(PHENIX_GM.parent / 'phenix.reduce'), str(conf_pdb)],
                cwd=str(tmpdir), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if res.returncode not in (0, 1):
                raise RuntimeError(f'phenix.reduce failed on {conf_pdb.name}:\n'
                                   f'{res.stderr.decode(errors="replace")[-1000:]}')
            out = tmpdir / f'{conf_pdb.stem}_H.pdb'
            out.write_bytes(res.stdout)
            return out
        with ThreadPoolExecutor(max_workers=n_altlocs) as pool:
            conf_pdbs = list(pool.map(_reduce_one, conf_pdbs))

    # Now that protein altlocs have been jiggled + minimized + CA-aligned +
    # H-added, jiggle the waters separately (one byB jiggle per altloc seed,
    # no phenix.geommin) and append into each conf*.pdb.  Waters thus get
    # realistic spread without being scrambled by the protein-side phenix
    # passes.  Only the ATOM/HETATM records of the water file are appended;
    # CRYST1/header stays from the protein conf PDB.
    if waters_pristine.exists():
        water_seeds = [int(rng.integers(1000, 99999)) for _ in range(n_altlocs)]
        for conf_pdb, w_seed, label in zip(conf_pdbs, water_seeds, labels):
            water_result = run(
                ['awk', '-f', str(JIGGLEPDB),
                 '-v', f'seed={w_seed}',
                 '-v', f'shift={jigglepdb_shift}',
                 '-v', f'shift_scale={shift_scale}',
                 '-v', 'dry_shift_scale=1.0',
                 '-v', 'frac_thrubond=0.1',
                 '-v', 'ncyc_thrubond=10',
                 '-v', 'frac_magnforce=1.1',
                 '-v', 'ncyc_magnforce=10',
                 *extra_jiggle_args,
                 str(waters_pristine)],
                cwd=tmpdir
            )
            water_path = tmpdir / f'water{label}.pdb'
            water_path.write_bytes(water_result.stdout)
            # Append water ATOM/HETATM lines into the protein conf PDB,
            # stripping its END so the merged file remains valid.
            prot_lines = conf_pdb.read_text().splitlines()
            water_lines = [ln for ln in water_path.read_text().splitlines()
                           if ln.startswith(('ATOM', 'HETATM'))]
            # Drop trailing END (if present) from protein, then append waters.
            while prot_lines and prot_lines[-1].startswith(('END', 'MASTER', 'CONECT')):
                prot_lines.pop()
            merged = '\n'.join(prot_lines + water_lines + ['END', ''])
            conf_pdb.write_text(merged)

    # Build multiconf.pdb in single-chain altloc form to match refmacout.pdb labeling:
    #   chain A: protein, every atom has altloc A,B,…,N (occ=1/N each)
    #   chain S: waters, every atom has altloc A,B,…,N (occ=1/N each)
    # sfcalc sums altloc atoms with their occupancies. Flood waters added later as chain F.
    CONF_LABELS  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    occ_per_conf = 1.0 / n_altlocs

    # Optional B-factor lookup: (chain, seqid_num, atom_name) → b_iso from source PDB
    bfac_lookup = {}
    if bfac_source_pdb is not None:
        src_st = gemmi.read_structure(str(bfac_source_pdb))
        for chain in src_st[0]:
            for res in chain:
                for atom in res:
                    bfac_lookup[(chain.name, res.seqid.num, atom.name)] = atom.b_iso

    confs       = [gemmi.read_structure(str(pdb)) for pdb in conf_pdbs]
    ref_st      = confs[0]
    st_out      = gemmi.Structure()
    st_out.cell          = ref_st.cell
    st_out.spacegroup_hm = ref_st.spacegroup_hm
    model_out   = gemmi.Model('1')

    def _build_altloc_chain(out_name, in_chain_name):
        in_chains = [next((ch for ch in c[0] if ch.name == in_chain_name), None)
                     for c in confs]
        if all(ch is None for ch in in_chains):
            return None
        n_res = max(len(ch) if ch else 0 for ch in in_chains)
        ch_out = gemmi.Chain(out_name)
        for ri in range(n_res):
            ref_res = next((ch[ri] for ch in in_chains if ch and ri < len(ch)), None)
            if ref_res is None:
                continue
            res_out = gemmi.Residue()
            res_out.name        = ref_res.name
            res_out.seqid       = ref_res.seqid
            res_out.entity_type = ref_res.entity_type
            for ci, chain in enumerate(in_chains):
                if chain is None or ri >= len(chain):
                    continue
                for atom in chain[ri]:
                    a_new         = gemmi.Atom()
                    a_new.name    = atom.name
                    a_new.element = atom.element
                    a_new.pos     = atom.pos
                    a_new.b_iso   = bfac_lookup.get(
                        (in_chain_name, ref_res.seqid.num, atom.name),
                        atom.b_iso)
                    a_new.occ     = occ_per_conf
                    a_new.altloc  = CONF_LABELS[ci]
                    res_out.add_atom(a_new)
            ch_out.add_residue(res_out)
        return ch_out

    prot_chain = _build_altloc_chain('A', 'A')
    if prot_chain is not None:
        model_out.add_chain(prot_chain)
    water_chain = _build_altloc_chain('S', 'W')
    if water_chain is not None:
        model_out.add_chain(water_chain)

    st_out.add_model(model_out)

    # Riding-H B-factor fix: phenix.reduce gives H atoms a default low B
    # (~3.87 Å²) and the bfac_source override only repopulates heavy-atom
    # B's, leaving H at the default — Fc(model)+H then mis-weights H and
    # the difference map shows positive density on every H.  For each H,
    # copy the B of the nearest same-altloc heavy atom in the same residue.
    for chain in st_out[0]:
        for res in chain:
            heavy = [a for a in res if a.element.name != 'H']
            if not heavy:
                continue
            hpos = np.array([(a.pos.x, a.pos.y, a.pos.z) for a in heavy])
            for atom in res:
                if atom.element.name != 'H':
                    continue
                # Prefer same-altloc parent; fall back to nearest of any altloc.
                same = [a for a in heavy if a.altloc == atom.altloc]
                pool_atoms = same if same else heavy
                pool_pos = (np.array([(a.pos.x, a.pos.y, a.pos.z) for a in pool_atoms])
                            if same else hpos)
                d = np.linalg.norm(
                    pool_pos - np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                    axis=1)
                atom.b_iso = float(pool_atoms[int(np.argmin(d))].b_iso)
    st_out.write_pdb(str(tmpdir / 'multiconf.pdb'))


def _sample_correlated_protein_occs(chain_residues, n_conf, rng):
    """Generate spatially correlated per-residue Dirichlet occupancy fractions.

    Occupancies along the chain are correlated via an AR(1) latent disorder field.
    Disorder amplitude is larger where sequential CA–CA distances are small
    (compact/helical regions) and smaller where the chain is extended.

    Returns a list of n_residues lists, each of length n_conf, summing to 1.0.
    """
    residues = list(chain_residues)
    n = len(residues)

    # Extract CA positions (None if residue has no CA, e.g. GLY H-only)
    ca_pos = []
    for res in residues:
        ca = next((a for a in res if a.name == 'CA'), None)
        ca_pos.append(np.array([ca.pos.x, ca.pos.y, ca.pos.z]) if ca else None)

    # Sequential CA–CA distances; default 5.0 Å for missing or first residue
    ca_dists = np.full(n, 5.0)
    for i in range(1, n):
        if ca_pos[i] is not None and ca_pos[i - 1] is not None:
            ca_dists[i] = np.linalg.norm(ca_pos[i] - ca_pos[i - 1])

    # AR(1) correlation: higher for close CA–CA (helix ~3.8 Å → ρ≈0.85)
    rho = np.exp(-ca_dists / 5.0)

    # Disorder amplitude: inversely proportional to CA–CA distance
    # helix (~3.8 Å) → amp≈0.66;  extended (~6 Å) → amp≈0.42;  loop (>10 Å) → ~0.25
    amp = 2.5 / np.maximum(ca_dists, 3.5)

    # AR(1) latent disorder field
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = rho[i] * z[i - 1] + np.sqrt(max(0.0, 1.0 - rho[i] ** 2)) * rng.normal(0, amp[i])

    # Map z → Dirichlet concentration α: large |z| → small α (unequal occupancies)
    alpha = np.exp(-0.5 * z ** 2)
    alpha = np.maximum(alpha, 0.05)

    occs_list = []
    for i in range(n):
        raw = rng.dirichlet(np.full(n_conf, alpha[i]))
        raw = np.clip(raw, 0.05, 0.90)
        occs_list.append((raw / raw.sum()).tolist())
    return occs_list


def _merge_altconfs(conf_pdbs, out_pdb, rng=None, flood_occ=None):
    """Combine N single-conf PDBs into an N-altloc PDB (altlocs A, B, C, ...).

    Per-residue occupancies are scaled to a chain-specific total occupancy:
      protein chains:             total = 1.0, correlated along chain via AR(1)
      chain 'W' (ordered waters): total = rng.uniform(0.3, 1.0) per water
      chain 'F' (flood waters):   total = flood_occ (default 0.1)
    """
    labels   = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(conf_pdbs)]
    structs  = [gemmi.read_structure(str(p)) for p in conf_pdbs]
    st_out   = structs[0].clone()
    n_conf   = len(structs)
    _flood_occ = flood_occ if flood_occ is not None else 0.1

    # Pre-compute correlated occupancies for protein chains
    chain_occs = {}  # chain_idx → list of occ vectors (one per residue)
    if rng is not None:
        for ci, chain in enumerate(structs[0][0]):
            if chain.name not in ('W', 'F'):
                chain_occs[ci] = _sample_correlated_protein_occs(chain, n_conf, rng)

    for ci, chains in enumerate(zip(*[s[0] for s in structs])):
        chain_out = st_out[0][ci]
        chain_name = chain_out.name
        for ri, residues in enumerate(zip(*chains)):
            res_out = chain_out[ri]
            # Total occupancy for this chain type
            if chain_name == 'F':
                total_occ = _flood_occ
            elif chain_name == 'W':
                total_occ = float(rng.uniform(0.3, 1.0)) if rng is not None else 1.0
            else:
                total_occ = 1.0

            # Per-conformer occupancies
            if rng is not None:
                if ci in chain_occs:
                    # Protein chain: use pre-computed correlated fractions
                    occs = [o * total_occ for o in chain_occs[ci][ri]]
                else:
                    # Water / flood chains: independent Dirichlet
                    raw = rng.dirichlet(np.ones(n_conf))
                    raw = np.clip(raw, 0.05, 0.90)
                    occs = (raw / raw.sum() * total_occ).tolist()
            else:
                occs = [total_occ / n_conf] * n_conf

            # Tag first-conf atoms (already in st_out)
            for a in res_out:
                a.altloc = labels[0]
                a.occ = occs[0]

            # Add atoms for remaining conformers
            for conf_i, (st_conf, lbl, occ) in enumerate(
                    zip(structs[1:], labels[1:], occs[1:]), start=1):
                res_conf = st_conf[0][ci][ri]
                props = [(a.name, a.element, a.pos.x, a.pos.y, a.pos.z, a.b_iso)
                         for a in res_conf]
                for name, elem, x, y, z, b in props:
                    atom = gemmi.Atom()
                    atom.name    = name
                    atom.element = elem
                    atom.pos     = gemmi.Position(x, y, z)
                    atom.occ     = occ
                    atom.b_iso   = b
                    atom.altloc  = lbl
                    res_out.add_atom(atom)

            # Water/flood altlocs: assign independent B factors so each
            # conformer has a realistic spread rather than all the same value.
            if chain_name in ('W', 'F') and rng is not None:
                for a in res_out:
                    a.b_iso = float(np.clip(
                        rng.lognormal(BFAC_SC_MU, BFAC_SC_SIGMA + 0.3),
                        BFAC_SC_MIN, 120.0))

    st_out.write_pdb(str(out_pdb))


# ── Wilson B reference (cached) ─────────────────────────────────────────────
REF_WILSON_MTZ = SCRIPT_DIR / '1aho' / '1aho.mtz'
_REF_WILSON_B  = None  # cached after first call


def _wilson_b(F, s2, n_bins=20, min_per_bin=10):
    """Wilson B from |F| and s²=1/d² arrays.

    Linear fit of log(<F²>) vs s² in resolution bins.
    Slope = -B/2  →  B = -2·slope.  Returns 0.0 if too few valid bins.
    """
    valid = (F > 0) & np.isfinite(F) & np.isfinite(s2)
    F  = F[valid].astype(np.float64)
    s2 = s2[valid].astype(np.float64)
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
    return -2.0 * float(slope)


def _reference_wilson_b():
    """Compute Wilson B of REF_WILSON_MTZ (1aho FP) once; cache result."""
    global _REF_WILSON_B
    if _REF_WILSON_B is None and REF_WILSON_MTZ.exists():
        mtz = gemmi.read_mtz_file(str(REF_WILSON_MTZ))
        h = np.asarray(mtz.column_with_label('H'),   dtype=np.int32)
        k = np.asarray(mtz.column_with_label('K'),   dtype=np.int32)
        l = np.asarray(mtz.column_with_label('L'),   dtype=np.int32)
        F = np.asarray(mtz.column_with_label('FP'),  dtype=np.float32)
        cell = mtz.cell
        s2 = np.array([cell.calculate_1_d2([int(h_), int(k_), int(l_)])
                       for h_, k_, l_ in zip(h, k, l)], dtype=np.float64)
        _REF_WILSON_B = _wilson_b(F, s2)
    return _REF_WILSON_B


def _split_pdb_by_b_n_ways(pdb_path, n_chunks, out_paths):
    """Sort atoms by B-factor then split into n_chunks contiguous bins.

    Chunk 0 has the n/N lowest-B atoms; chunk N-1 has the n/N highest.
    Returns a list of (b_min, b_max) per chunk.

    Header lines copied into every sub-PDB are restricted to CRYST1/SCALE/
    SSBOND/LINK only.  TER/END/MODEL/ENDMDL/USER MOD records would terminate
    gemmi's model parser before the atoms are read, so they are dropped here
    and a single END is written after the atom block.
    """
    HEADER_RECS = ('CRYST1', 'SCALE1', 'SCALE2', 'SCALE3', 'SSBOND', 'LINK  ')
    headers = []
    atoms   = []
    with open(str(pdb_path)) as fin:
        for line in fin:
            rec6 = line[:6]
            if rec6 in ('ATOM  ', 'HETATM'):
                try:
                    b = float(line[60:66])
                except ValueError:
                    b = 0.0
                atoms.append((b, line))
            elif rec6 in HEADER_RECS:
                headers.append(line)
    atoms.sort(key=lambda x: x[0])
    n = len(atoms)
    b_ranges = []
    for i, path in enumerate(out_paths):
        start = i * n // n_chunks
        end   = (i + 1) * n // n_chunks
        slab  = atoms[start:end]
        with open(str(path), 'w') as fout:
            for h in headers:
                fout.write(h)
            for _, a in slab:
                fout.write(a)
            fout.write('END\n')
        if slab:
            b_ranges.append((slab[0][0], slab[-1][0]))
        else:
            b_ranges.append((0.0, 0.0))
    return b_ranges


def _rate_for_bmin(b_min, safety=1.37):
    """Coarsest gemmi --rate that keeps σ_min ≥ pixel/safety (no aliasing).

    σ_min = sqrt(b_min / 8π²);  pixel = dmin / (2·rate)
    Require pixel ≤ safety·σ_min  →  rate ≥ dmin / (2·safety·σ_min)
    """
    if b_min <= 0:
        return 1.5
    sigma = (b_min / (8.0 * np.pi ** 2)) ** 0.5
    return max(0.5, DMIN / (2.0 * safety * sigma))


def _sfcalc_parallel(pdb_path, mtz_out, tmpdir, n_workers=20):
    """N-way parallel gemmi sfcalc: sort atoms by B, split into bins, run
    concurrent sfcalc with per-bin grid rate, sum complex F values.

    Lowest-B bins need fine grid (rate~1.5); highest-B bins can use coarse
    grid (rate~0.5–1.0).  Wall time is bounded by the slowest (lowest-B) bin.
    """
    sub_pdbs = [tmpdir / f'_sub_{i:02d}.pdb' for i in range(n_workers)]
    sub_mtzs = [tmpdir / f'_sub_{i:02d}.mtz' for i in range(n_workers)]
    b_ranges = _split_pdb_by_b_n_ways(pdb_path, n_workers, sub_pdbs)
    # Use same fine rate everywhere so all chunks produce identical HKL sets;
    # we get the parallel speedup (if any) without the high-res merge headache.
    rates    = [1.5 for _ in b_ranges]
    import time as _time
    _t_par_start = _time.time()

    def _one(i):
        run(['gemmi', 'sfcalc', f'--dmin={DMIN}', f'--rate={rates[i]}',
             f'--to-mtz={sub_mtzs[i].name}', str(sub_pdbs[i])], cwd=tmpdir)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_one, range(n_workers)))
    print(f'    sfcalc {n_workers}x parallel wall time: {_time.time() - _t_par_start:.1f}s')

    def _hkl_key(h, k, l):
        return (h.astype(np.int64) << 42) | \
               ((k.astype(np.int64) & 0x1FFFFF) << 21) | \
               (l.astype(np.int64) & 0x1FFFFF)

    # Pick chunk with the most HKLs (finest grid) as reference HKL set.
    # Coarse-grid chunks may have fewer HKLs (no high-res); they contribute
    # only at HKLs they share with the reference (zero elsewhere).
    chunk_mtzs = [gemmi.read_mtz_file(str(p)) for p in sub_mtzs]
    sizes      = [len(np.asarray(m.column_with_label('H'))) for m in chunk_mtzs]
    ref_idx    = int(np.argmax(sizes))
    m0         = chunk_mtzs[ref_idx]
    h0 = np.asarray(m0.column_with_label('H'), dtype=np.int32)
    k0 = np.asarray(m0.column_with_label('K'), dtype=np.int32)
    l0 = np.asarray(m0.column_with_label('L'), dtype=np.int32)
    key0   = _hkl_key(h0, k0, l0)
    order0 = np.argsort(key0)
    h_p, k_p, l_p = h0[order0], k0[order0], l0[order0]
    ref_key = key0[order0]
    F_total = np.zeros(len(h_p), dtype=np.complex128)

    for m in chunk_mtzs:
        h = np.asarray(m.column_with_label('H'),    dtype=np.int32)
        k = np.asarray(m.column_with_label('K'),    dtype=np.int32)
        l = np.asarray(m.column_with_label('L'),    dtype=np.int32)
        F = np.asarray(m.column_with_label('FC'),   dtype=np.float64)
        P = np.asarray(m.column_with_label('PHIC'), dtype=np.float64)
        chunk_key = _hkl_key(h, k, l)
        idx       = np.searchsorted(ref_key, chunk_key)
        in_range  = idx < len(ref_key)
        match     = in_range & (ref_key[np.where(in_range, idx, 0)] == chunk_key)
        F_complex = F[match] * np.exp(1j * np.radians(P[match]))
        np.add.at(F_total, idx[match], F_complex)

    fc_p  = np.abs(F_total).astype(np.float64)
    phi_p = np.degrees(np.angle(F_total)).astype(np.float64)

    out = gemmi.Mtz()
    out.cell, out.spacegroup = m0.cell, m0.spacegroup
    out.add_dataset('HKL_base')
    for lbl in ('H', 'K', 'L'):
        out.add_column(lbl, 'H')
    out.add_dataset('data')
    out.add_column('FC',   'F')
    out.add_column('PHIC', 'P')
    out.set_data(np.column_stack([h_p, k_p, l_p, fc_p, phi_p]).astype(np.float32))
    out.write_to_file(str(tmpdir / mtz_out))
    return h_p, k_p, l_p, fc_p, phi_p


def _sfcalc_with_bulksolv(pdb_path, mtz_out, tmpdir,
                           solvent_radius=1.41, solvent_scale=0.334, solvent_B=50.0,
                           sfcalc_workers=20, fpart_out=None):
    """Compute structure factors including a bulk solvent contribution.

    Mirrors the model in ano_sfall.com (James Holton):
      1. Protein SFs from gemmi sfcalc (N-way parallel atom split).
      2. Solvent mask via cavenv, scaled to solvent_scale e⁻/Å³
         (default 0.334 = bulk water at 1 g/cm³).
      3. Mask → SFs via gemmi FFT (transform_to_f_phi).
      4. Apply exp(-B_sol * s²/4) Debye-Waller envelope (B_sol = 50 Å²).
      5. F_total = F_protein + F_solvent.

    If fpart_out is given, writes the bulk solvent contribution alone
    (Fpart / PHIpart columns, same Wilson B correction as F_total) to that path.

    H must already be present in pdb_path (call step6_sfcalc which adds H first).
    """
    cell_kw = f'{CELL[0]} {CELL[1]} {CELL[2]} 90 90 90'
    na = round(CELL[0] * SAMPLE_RATE / DMIN)
    nb = round(CELL[1] * SAMPLE_RATE / DMIN)
    nc = round(CELL[2] * SAMPLE_RATE / DMIN)
    grid_kw = f'{na} {nb} {nc}'

    # ── 1. Protein SFs (N-way parallel atom split, complex F summed) ──────────
    h_p, k_p, l_p, fc_p, phi_p = _sfcalc_parallel(
        pdb_path, '_protein_only.mtz', tmpdir, n_workers=sfcalc_workers)
    prot = gemmi.read_mtz_file(str(tmpdir / '_protein_only.mtz'))

    # ── 2. Solvent mask via cavenv ──────────────────────────────────────────────
    # Strip flood-water chain (F) before masking: flood waters are already in
    # F_protein from sfcalc above; including them in cavenv would exclude their
    # positions from the mask, giving an artificially low bulk-solvent scale.
    # Build cavenv mask: single-conformer protein only (no waters/flood).
    # Keep only the first occurrence of each (chain, resnum, atom_name) to
    # strip altlocs. With N=20 altlocs × P212121 symmetry, the full model
    # exceeds cavenv's MAXATM=50000.
    mask_pdb = str(tmpdir / '_mask_input.pdb')
    with open(str(pdb_path)) as fin, open(mask_pdb, 'w') as fout:
        seen = set()
        for line in fin:
            if line[:6] in ('CRYST1',):
                fout.write(line)
                continue
            if line[:6] not in ('ATOM  ', 'HETATM'):
                continue
            chain = line[21]
            if chain in ('S', 'W', 'F') or chain.islower():
                continue
            key = (chain, line[22:26], line[12:16])
            if key in seen:
                continue
            seen.add(key)
            fout.write(line[:16] + ' ' + line[17:])   # blank altloc col
        fout.write('END\n')

    _sg = gemmi.find_spacegroup_by_name(SPACEGROUP)
    sg_num = _sg.number if _sg else 1
    cavenv_kw = (
        f'CELL {cell_kw}\nSYMM {sg_num}\nENVSOLVENT\n'
        f'GRID {grid_kw}\nRADMAX {solvent_radius}\n'
    ).encode()
    cv = subprocess.run(
        ['cavenv', 'xyzin', mask_pdb, 'mapout', '_raw_solvent.map'],
        input=cavenv_kw, cwd=str(tmpdir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if cv.returncode != 0 or not (tmpdir / '_raw_solvent.map').exists():
        raise RuntimeError(f'cavenv failed (rc={cv.returncode}):\n'
                           f'{cv.stdout.decode(errors="replace")[-1000:]}')

    # ── 3. Scale mask to bulk water electron density (via gemmi in-memory) ──────
    ccp4 = gemmi.read_ccp4_map(str(tmpdir / '_raw_solvent.map'))
    ccp4.setup(float('nan'))
    arr = np.array(ccp4.grid, copy=False)
    max_val = float(arr.max()) or 1.0
    arr *= (solvent_scale / max_val)

    # ── 4. Real-space smoothing (box filter, N iterations) ───────────────────────
    # Mirrors ano_sfall.com: N = floor(B_sol / (2*(spacing*pi*1.468)^2))
    # Each 3x3x3 box-filter pass applies effective B of per_iter Å².
    # Remainder is applied in reciprocal space.
    grid_spacing = CELL[0] / arr.shape[0]   # Å per voxel (40/60 for our cell)
    per_iter = 2.0 * (grid_spacing * np.pi * 1.468) ** 2
    n_smooth = int(solvent_B / per_iter) if per_iter > 0 else 0
    if n_smooth < 3:
        n_smooth = 0   # too few iterations: skip (per ano_sfall.com)
    smooth_B = n_smooth * per_iter
    rs_solvent_B = solvent_B - smooth_B
    if n_smooth > 0:
        pad = n_smooth + 1
        padded = np.pad(arr, pad, mode='wrap')
        for _ in range(n_smooth):
            padded = uniform_filter(padded, size=3, mode='nearest')
        arr[:] = padded[pad:-pad, pad:-pad, pad:-pad]

    # ── 5. Mask → SFs via gemmi FFT ─────────────────────────────────────────────
    hkl = gemmi.transform_map_to_f_phi(ccp4.grid, half_l=True)

    # ── 6. Apply B envelope & combine ───────────────────────────────────────────
    a, b, c = CELL[0], CELL[1], CELL[2]
    s_sq = (h_p / a)**2 + (k_p / b)**2 + (l_p / c)**2
    bfac = np.exp(-rs_solvent_B * s_sq / 4.0)

    F_prot = fc_p * np.exp(1j * np.radians(phi_p))
    hkl_array = np.column_stack([h_p, k_p, l_p]).astype(np.int32)
    F_solv = hkl.get_value_by_hkl(hkl_array).astype(complex) * bfac
    F_tot  = F_prot + F_solv

    fc_out  = np.abs(F_tot).astype(np.float32)
    phi_out = np.degrees(np.angle(F_tot)).astype(np.float32)

    # ── 7. Wilson B correction: match overall B of real 1aho data ─────────────
    #     Applies exp(-ΔB · s²/4) where ΔB = B_ref - B_gen.  Brings simulated
    #     <F²> vs s² spectrum into line with the experimental reference so the
    #     CNN sees realistic resolution-dependent intensity falloff.
    wilson_scale = np.ones(len(s_sq), dtype=np.float64)
    ref_B = _reference_wilson_b()
    if ref_B is not None:
        gen_B   = _wilson_b(fc_out, s_sq)
        delta_B = ref_B - gen_B
        wilson_scale = np.exp(-delta_B * s_sq / 4.0)
        fc_out  = (fc_out * wilson_scale).astype(np.float32)
        print(f'    Wilson B: ref={ref_B:.2f} gen={gen_B:.2f} ΔB={delta_B:+.2f} Å² applied')

    out = gemmi.Mtz()
    out.cell       = prot.cell
    out.spacegroup = prot.spacegroup
    out.add_dataset('HKL_base')
    for lbl in ('H', 'K', 'L'):
        out.add_column(lbl, 'H')
    out.add_dataset('data')
    out.add_column('FC',   'F')
    out.add_column('PHIC', 'P')
    out.set_data(np.column_stack([h_p, k_p, l_p, fc_out, phi_out]).astype(np.float32))
    out.write_to_file(str(mtz_out))

    if fpart_out is not None:
        fpart_amp = (np.abs(F_solv) * wilson_scale).astype(np.float32)
        fpart_phi = np.degrees(np.angle(F_solv)).astype(np.float32)
        fp = gemmi.Mtz()
        fp.cell       = prot.cell
        fp.spacegroup = prot.spacegroup
        fp.add_dataset('HKL_base')
        for lbl in ('H', 'K', 'L'):
            fp.add_column(lbl, 'H')
        fp.add_dataset('data')
        fp.add_column('Fpart',   'F')
        fp.add_column('PHIpart', 'P')
        fp.set_data(np.column_stack([h_p, k_p, l_p, fpart_amp, fpart_phi]).astype(np.float32))
        fp.write_to_file(str(fpart_out))


def _pdb_has_hydrogens(pdb_path):
    """Quick scan: True if any ATOM record has element 'H' (cols 77-78)."""
    with open(str(pdb_path)) as f:
        for line in f:
            if line[:6] in ('ATOM  ', 'HETATM') and len(line) >= 78 \
               and line[76:78].strip() == 'H':
                return True
    return False


def step6_sfcalc(pdb_path, mtz_out, tmpdir, bulk_solvent=True, fpart_out=None):
    """Add hydrogens to pdb_path (if not already present), then compute SFs.

    If bulk_solvent=True (default), includes a mask-based bulk solvent
    contribution (cavenv + sfall, matching ano_sfall.com parameters:
    radius=1.41 Å, scale=0.334 e⁻/Å³, B=50 Å²).

    Set bulk_solvent=False for internal steps (e.g. self-refinement B factors)
    where speed matters and absolute realism is not required.

    fpart_out: if given, write the bulk solvent SFs (Fpart/PHIpart) to that path.
    Only meaningful when bulk_solvent=True.

    pdb_path is overwritten with the H-containing model when reduce is run, so
    truth_full.pdb saved to the sample directory includes H either way.
    """
    if not _pdb_has_hydrogens(pdb_path):
        pdb_with_h = tmpdir / '_sfcalc_withH.pdb'
        result = subprocess.run(
            [str(PHENIX_GM.parent / 'phenix.reduce'), str(pdb_path)],
            cwd=str(tmpdir), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode not in (0, 1):  # reduce exits 1 on warnings, normal
            raise RuntimeError(f'phenix.reduce failed:\n{result.stderr.decode(errors="replace")[-1000:]}')
        pdb_with_h.write_bytes(result.stdout)
        pdb_with_h.replace(pdb_path)

    if bulk_solvent:
        _sfcalc_with_bulksolv(pdb_path, mtz_out, tmpdir, fpart_out=fpart_out)
    else:
        run(['gemmi', 'sfcalc', f'--dmin={DMIN}',
             f'--to-mtz={mtz_out}', str(pdb_path)],
            cwd=tmpdir)


def step7_build_refme_mtz(truth_mtz, refme_mtz):
    """Build refme.mtz with F=|FC|, SIGF=0.02·|FC| for refmac."""
    mtz = gemmi.read_mtz_file(str(truth_mtz))
    fc_col = mtz.column_with_label('FC')
    if fc_col is None:
        raise RuntimeError(f"No FC column in {truth_mtz}")

    h_arr   = np.array(mtz.column_with_label('H'),  dtype=np.float32)
    k_arr   = np.array(mtz.column_with_label('K'),  dtype=np.float32)
    l_arr   = np.array(mtz.column_with_label('L'),  dtype=np.float32)
    fc_arr  = np.array(fc_col,                       dtype=np.float32)
    sigf    = np.maximum(0.02 * fc_arr, 0.01).astype(np.float32)

    mtz_out = gemmi.Mtz()
    mtz_out.cell = mtz.cell
    mtz_out.spacegroup = mtz.spacegroup
    mtz_out.add_dataset('HKL_base')
    mtz_out.add_column('H', 'H')
    mtz_out.add_column('K', 'H')
    mtz_out.add_column('L', 'H')
    mtz_out.add_dataset('data')
    mtz_out.add_column('F',    'F')
    mtz_out.add_column('SIGF', 'Q')

    data = np.column_stack([h_arr, k_arr, l_arr, fc_arr, sigf])
    mtz_out.set_data(data)
    mtz_out.write_to_file(str(refme_mtz))


def simulate_missing_data(refme_mtz, frac_missing, frac_never_collected, rng):
    """Simulate two categories of missing data in refme.mtz.

    Must be called AFTER uniqueify so that FreeR_flag is already present.

    Two non-overlapping categories are drawn (Bernoulli per reflection):

      • never_collected : data never measured (blind region, incomplete wedge, etc.)
                          Rows are deleted entirely — refmac has zero knowledge of
                          these HKLs, so they appear in neither FC_ALL_LS nor FWT.

      • missing         : data collected but subsequently rejected/removed.
                          F and SIGF are set to NaN (MNF); the row with FreeR_flag
                          stays in the file.  Refmac sees the HKL with no Fo, Fc-fills
                          FWT/DELFWT, and still computes FC_ALL_LS for these.

    Free-R reflections are NOT missing — they have valid F values and are merely
    flagged as a test set for cross-validation.

    truth.mtz / truth.map are not touched.

    Returns (n_missing, n_never_collected).
    """
    if frac_missing <= 0.0 and frac_never_collected <= 0.0:
        return 0, 0

    mtz    = gemmi.read_mtz_file(str(refme_mtz))
    labels = [col.label for col in mtz.columns]
    data   = np.column_stack([np.array(mtz.column_with_label(lbl), dtype=np.float32)
                               for lbl in labels])
    n = len(data)

    # Draw never_collected first; missing drawn from remainder (non-overlapping)
    never_mask   = rng.random(n) < frac_never_collected
    missing_mask = (rng.random(n) < frac_missing) & ~never_mask

    n_never   = int(never_mask.sum())
    n_missing = int(missing_mask.sum())

    # 'missing': set F and SIGF to NaN — row stays in file with FreeR_flag intact
    f_col    = labels.index('F')
    sigf_col = labels.index('SIGF')
    data[missing_mask, f_col]    = np.nan
    data[missing_mask, sigf_col] = np.nan

    # 'never_collected': delete rows entirely
    data = data[~never_mask]

    # Reconstruct MTZ preserving all datasets and column types
    mtz_out = gemmi.Mtz()
    mtz_out.cell       = mtz.cell
    mtz_out.spacegroup = mtz.spacegroup
    for ds in mtz.datasets:
        ds_out = mtz_out.add_dataset(ds.dataset_name)
        ds_out.project_name = ds.project_name
        ds_out.crystal_name = ds.crystal_name
        ds_out.wavelength   = ds.wavelength
    for col in mtz.columns:
        mtz_out.add_column(col.label, col.type, dataset_id=col.dataset_id)
    mtz_out.set_data(data)
    mtz_out.write_to_file(str(refme_mtz))
    return n_missing, n_never


def step7c_add_freer_flags(refme_mtz, tmpdir):
    """Run uniqueify to complete the unique reflection set and add FreeR_flag.

    uniqueify fills in any missing HKLs (no observed F) and marks ~5% as free.
    With a complete unique set, refmac outputs FC_ALL_LS for ALL reflections,
    giving a bulk-solvent-corrected Fc that is independent of which Fo are present.
    Overwrites refme.mtz in-place.
    """
    tmp_out = tmpdir / '_refme_free.mtz'
    run([str(UNIQUEIFY), str(refme_mtz), str(tmp_out)], cwd=tmpdir)
    tmp_out.replace(refme_mtz)


def _add_collapsed_atom(res_out, atoms):
    """Add a single atom to res_out at the mean position/B of all altloc atoms.
    Occupancy = sum of altloc occs; rounded to 1.0 when sum > 0.95 (floating-point
    tolerance) so that blank-conformer atoms always arrive at refmac with occ=1.0
    and are not picked up by refmac_occupancy_setup.com for occupancy refinement."""
    pos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms])
    cx, cy, cz = pos.mean(axis=0)
    total_occ = sum(a.occ for a in atoms)
    a_out = gemmi.Atom()
    a_out.name    = atoms[0].name
    a_out.element = atoms[0].element
    a_out.pos     = gemmi.Position(cx, cy, cz)
    a_out.b_iso   = float(np.mean([a.b_iso for a in atoms]))
    a_out.occ     = 1.0 if total_occ > 0.95 else float(min(1.0, total_occ))
    a_out.altloc  = '\x00'
    res_out.add_atom(a_out)


def _reduce_conformers(by_name, sc_names, max_confs=3, rng=None):
    """Reduce to ≤ max_confs altloc conformers; returns (updated_by_name, label_list).

    When rng is provided and more than max_confs conformers are present, selects
    max_confs randomly (uniform, without replacement) instead of always keeping the
    highest-occupancy ones.  Occupancies of survivors are renormalised to sum to 1.
    """
    labels = sorted({a.altloc for name in sc_names
                     for a in by_name.get(name, ())
                     if a.altloc and a.altloc != '\x00'})
    if len(labels) <= max_confs:
        return by_name, labels

    # Mean occupancy per conformer label (used for renormalisation regardless of mode)
    occ_by_label = {}
    for l in labels:
        occs = [a.occ for name in sc_names
                for a in by_name.get(name, ()) if a.altloc == l]
        occ_by_label[l] = float(np.mean(occs)) if occs else 0.0

    if rng is not None:
        # Random selection: sample max_confs labels uniformly
        chosen = rng.choice(labels, size=max_confs, replace=False).tolist()
        labels = sorted(chosen)
        occ_by_label = {l: occ_by_label[l] for l in labels}
    else:
        # Default: drop lowest-occupancy conformers until ≤ max_confs remain
        while len(labels) > max_confs:
            drop_l = min(labels, key=lambda l: occ_by_label[l])
            labels.remove(drop_l)
            del occ_by_label[drop_l]

    # Renormalize surviving occupancies to sum to 1.0
    total = sum(occ_by_label.values())
    scale = (1.0 / total) if total > 0 else 1.0
    for l in labels:
        occ_by_label[l] *= scale

    # Rebuild by_name keeping only surviving labels, with renormalized occupancies
    label_set = set(labels)
    new_by_name = dict(by_name)
    for name in sc_names:
        atoms_list = []
        for a in by_name.get(name, ()):
            if a.altloc not in label_set:
                continue
            new_a = gemmi.Atom()
            new_a.name    = a.name
            new_a.element = a.element
            new_a.pos     = a.pos
            new_a.b_iso   = a.b_iso
            new_a.occ     = occ_by_label[a.altloc]
            new_a.altloc  = a.altloc
            atoms_list.append(new_a)
        if atoms_list:
            new_by_name[name] = atoms_list
    return new_by_name, labels


def step8_build_mixed_model(truth_full_pdb, tmpdir, rng, altloc_swaps_per_res=1.0):
    """Build a mixed single/multi-conformer model → starthere.pdb.

    Reads single-chain altloc multiconf format from truth_full.pdb:
      chain A: protein, atoms have altloc A,B,…,N (occ=1/N each)
      chain S: waters,  atoms have altloc A,B,…,N (occ=1/N each)
      chain F: flood waters (skipped)

    For each residue:
      - If max atom displacement across N altlocs > ALTLOC_DIST_THRESHOLD,
        keep all atoms as altlocs (scrambled labels).
      - Otherwise collapse to mean position.
      - Waters (chain S): always collapsed to mean position at occ = 1/N.
    """
    CONF_LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    st_in = gemmi.read_structure(str(truth_full_pdb))

    prot_chain  = next((ch for ch in st_in[0] if ch.name == 'A'), None)
    water_chain = next((ch for ch in st_in[0] if ch.name == 'S'), None)

    if prot_chain is None:
        raise RuntimeError('step8: no chain A (protein) found in truth_full.pdb')

    # Determine n_conf from the first protein residue's altloc count
    altloc_set = set()
    for res in prot_chain:
        for atom in res:
            if atom.altloc and atom.altloc != '\x00':
                altloc_set.add(atom.altloc)
        if altloc_set:
            break
    n_conf = max(len(altloc_set), 1)

    # Build per-residue dicts: each protein residue holds its own atom list (with altlocs)
    def _seqid_str(res):
        return str(res.seqid)

    prot_residues = {}  # seqid_str → list of "synthetic" gemmi.Residue (one per altloc)
    seqid_order   = []
    for res in prot_chain:
        k = _seqid_str(res)
        if k in prot_residues:
            continue
        # Split atoms by altloc into N synthetic residues so downstream code
        # (which iterates "residues" as per-conformer copies) keeps working.
        by_alt = {}
        for atom in res:
            lbl = atom.altloc if (atom.altloc and atom.altloc != '\x00') else CONF_LABELS[0]
            by_alt.setdefault(lbl, []).append(atom)
        synth = []
        for lbl in sorted(by_alt.keys()):
            r2 = gemmi.Residue()
            r2.name = res.name; r2.seqid = res.seqid; r2.entity_type = res.entity_type
            for a in by_alt[lbl]:
                a_copy = gemmi.Atom()
                a_copy.name    = a.name
                a_copy.element = a.element
                a_copy.pos     = a.pos
                a_copy.b_iso   = a.b_iso
                a_copy.occ     = a.occ
                a_copy.altloc  = '\x00'  # cleared so downstream re-labels
                r2.add_atom(a_copy)
            synth.append(r2)
        prot_residues[k] = synth
        seqid_order.append(k)

    water_residues = {}
    if water_chain is not None:
        for res in water_chain:
            k = _seqid_str(res)
            # Each water residue's altloc atoms become "N residues, one atom each"
            # to keep the existing reduction logic happy.
            by_alt = {}
            for atom in res:
                lbl = atom.altloc if (atom.altloc and atom.altloc != '\x00') else CONF_LABELS[0]
                by_alt.setdefault(lbl, []).append(atom)
            synth = []
            for lbl in sorted(by_alt.keys()):
                r2 = gemmi.Residue()
                r2.name = res.name; r2.seqid = res.seqid; r2.entity_type = res.entity_type
                for a in by_alt[lbl]:
                    a_copy = gemmi.Atom()
                    a_copy.name    = a.name
                    a_copy.element = a.element
                    a_copy.pos     = a.pos
                    a_copy.b_iso   = a.b_iso
                    a_copy.occ     = a.occ
                    a_copy.altloc  = '\x00'
                    r2.add_atom(a_copy)
                synth.append(r2)
            water_residues.setdefault(k, []).extend(synth)

    # ── Fragment clustering for backbone-safe altloc scrambling ──────────────
    # The bimodal split creates two parent structures (A and B) with backbone
    # atoms at significantly different positions.  Scrambling altloc labels
    # independently for adjacent residues can put C of residue i from parent A
    # and N of residue i+1 from parent B in the same altloc → broken peptide
    # bond.  Fix: identify residues with large backbone spread (coming from two
    # parents), cluster consecutive such residues into fragments, and apply one
    # consistent label permutation across each fragment.
    _BB_ATOMS          = {'N', 'CA', 'C', 'O'}
    _BB_BIMODAL_THRESH = 0.3   # Å — backbone spread threshold to flag a residue
    _FULL_LABELS       = list(CONF_LABELS[:n_conf])

    # Pass 1: measure backbone spread per residue (quick scan, no atom copies yet)
    bb_spread = {}   # seqid_str → max backbone spread across altlocs
    for key in seqid_order:
        residues = prot_residues.get(key, [])
        if len(residues) < 2:
            bb_spread[key] = 0.0
            continue
        max_spread = 0.0
        atom_pts = {}  # atom_name → list of (x,y,z)
        for res in residues:
            for atom in res:
                if atom.name in _BB_ATOMS:
                    atom_pts.setdefault(atom.name, []).append(
                        (atom.pos.x, atom.pos.y, atom.pos.z))
        for pts in atom_pts.values():
            if len(pts) < 2:
                continue
            arr = np.array(pts)
            centroid = arr.mean(0)
            spread = float(np.sqrt(((arr - centroid) ** 2).sum(1)).max())
            max_spread = max(max_spread, spread)
        bb_spread[key] = max_spread

    bimodal_bb_keys = {k for k, s in bb_spread.items() if s > _BB_BIMODAL_THRESH}

    # Parse sequence numbers for adjacency (ignore insertion codes for grouping)
    def _seqnum(key):
        try:
            return int(key)
        except ValueError:
            try:
                return int(''.join(c for c in key if c.isdigit()))
            except ValueError:
                return None

    key_to_sn = {k: _seqnum(k) for k in seqid_order}
    sn_to_key = {}
    for k, sn in key_to_sn.items():
        if sn is not None:
            sn_to_key.setdefault(sn, k)

    # Union-find: merge bimodal residue with its sequence neighbours
    uf_parent = {k: k for k in seqid_order}
    def _find(x):
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x
    def _union(a, b):
        uf_parent[_find(a)] = _find(b)

    for key in bimodal_bb_keys:
        sn = key_to_sn.get(key)
        if sn is None:
            continue
        for adj_sn in (sn - 1, sn + 1):
            adj_key = sn_to_key.get(adj_sn)
            if adj_key is not None:
                _union(key, adj_key)

    # Assign one rng-drawn permutation per bimodal fragment (others get None)
    fragment_perm = {}   # root_key → {old_label: new_label} or None
    seen_roots = set()
    for key in seqid_order:
        root = _find(key)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        # Only generate a shared perm if the fragment contains bimodal bb residues
        members = [k for k in seqid_order if _find(k) == root]
        if any(k in bimodal_bb_keys for k in members):
            sh = list(_FULL_LABELS)
            n_sw = int(rng.poisson(altloc_swaps_per_res))
            for _ in range(n_sw):
                ii, jj = rng.choice(len(sh), size=2, replace=False)
                sh[ii], sh[jj] = sh[jj], sh[ii]
            fragment_perm[root] = dict(zip(_FULL_LABELS, sh))
        else:
            fragment_perm[root] = None   # will use per-residue swap

    st_out = gemmi.Structure()
    st_out.cell          = st_in.cell
    st_out.spacegroup_hm = st_in.spacegroup_hm
    model_out = gemmi.Model('1')

    # ── Protein chain → output as single chain A ──────────────────────────────
    chain_out = gemmi.Chain('A')
    for key in seqid_order:
        residues = prot_residues.get(key, [])
        if not residues:
            continue
        res0    = residues[0]
        res_out = gemmi.Residue()
        res_out.name        = res0.name
        res_out.seqid       = res0.seqid
        res_out.entity_type = res0.entity_type

        # Build by_name assigning conformer labels A, B, … so _reduce_conformers works
        by_name = {}
        for conf_i, res in enumerate(residues):
            lbl = CONF_LABELS[conf_i]
            for atom in res:
                a_copy         = gemmi.Atom()
                a_copy.name    = atom.name
                a_copy.element = atom.element
                a_copy.pos     = atom.pos
                a_copy.b_iso   = atom.b_iso
                a_copy.occ     = atom.occ
                a_copy.altloc  = lbl
                by_name.setdefault(atom.name, []).append(a_copy)

        all_names = list(by_name.keys())

        # Residue-wide spatial spread: max displacement of any atom from its centroid
        res_spread = 0.0
        for n in all_names:
            atoms = by_name.get(n, [])
            if len(atoms) > 1:
                pos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms])
                centroid = pos.mean(axis=0)
                dists = np.sqrt(((pos - centroid) ** 2).sum(axis=1))
                res_spread = max(res_spread, float(dists.max()))

        if res_spread <= ALTLOC_DIST_THRESHOLD:
            for name in all_names:
                _add_collapsed_atom(res_out, by_name[name])
        else:
            # Floor 2 (for bimodal coverage), cap 10, per-residue _GT48_NCONF
            # in between.  The 2-floor matches one altloc per bimodal cluster;
            # the 10-cap keeps ARG/LYS/GLU/HIS (which gt48 gives 11-13) from
            # bloating starthere.
            res_max = min(n_conf, 10, max(2, _GT48_NCONF.get(res0.name, 2)))
            by_name, present = _reduce_conformers(by_name, all_names, max_confs=res_max)

            root = _find(key)
            shared_perm = fragment_perm.get(root)
            if shared_perm is not None:
                # Bimodal fragment: apply the shared permutation so adjacent
                # residues consistently use conformers from the same parent.
                label_map = {orig: shared_perm.get(orig, orig) for orig in present}
            else:
                # Non-bimodal: independent per-residue swap (original behaviour).
                shuffled = list(present)
                n_swaps = int(rng.poisson(altloc_swaps_per_res))
                for _ in range(n_swaps):
                    i, j = rng.choice(len(shuffled), size=2, replace=False)
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
                label_map = dict(zip(present, shuffled))

            for name in all_names:
                for atom in by_name.get(name, []):
                    orig  = atom.altloc if (atom.altloc and atom.altloc != '\x00') else (present[0] if present else 'A')
                    a_out         = gemmi.Atom()
                    a_out.name    = atom.name
                    a_out.element = atom.element
                    a_out.pos     = atom.pos
                    a_out.b_iso   = atom.b_iso
                    a_out.occ     = atom.occ
                    a_out.altloc  = label_map.get(orig, orig)
                    res_out.add_atom(a_out)

        chain_out.add_residue(res_out)
    model_out.add_chain(chain_out)

    # ── Waters → keep WATER_MIN_NCONF altloc positions in chain S ───────────────
    # Keep at least WATER_MIN_NCONF ground-truth conformers per water so the
    # network sees realistic multi-position water density. Capped at n_conf.
    # refmac_occupancy_setup.com gives each altloc an independent incomplete group.
    if water_residues:
        n_water_conf = min(n_conf, WATER_MIN_NCONF)
        water_occ = 1.0 / n_water_conf
        water_chain_out = gemmi.Chain('S')
        water_seqids = sorted(water_residues.keys(),
                              key=lambda k: water_residues[k][0].seqid.num)
        for key in water_seqids:
            residues = water_residues[key][:n_water_conf]
            res0    = residues[0]
            res_out = gemmi.Residue()
            res_out.name        = res0.name
            res_out.seqid       = res0.seqid
            res_out.entity_type = res0.entity_type
            for ci, res in enumerate(residues):
                lbl = CONF_LABELS[ci]
                for atom in res:
                    a_out         = gemmi.Atom()
                    a_out.name    = atom.name
                    a_out.element = atom.element
                    a_out.pos     = atom.pos
                    a_out.b_iso   = atom.b_iso
                    a_out.occ     = water_occ
                    a_out.altloc  = lbl
                    res_out.add_atom(a_out)
            water_chain_out.add_residue(res_out)
        model_out.add_chain(water_chain_out)

    # Strip H atoms before writing — riding H from phenix geommin have partial occ
    # when collapsed (present in only k of N conformers → occ=k/N), which creates
    # spurious incomplete groups in refmac_occupancy_setup.com.
    # refmac MAKE HYDR A adds them back with correct occ relative to their heavy atom.
    H = gemmi.Element('H')
    for chain in model_out:
        for res in chain:
            to_del = [i for i, a in enumerate(res) if a.element == H]
            for i in reversed(to_del):
                del res[i]

    st_out.add_model(model_out)
    st_out.write_pdb(str(tmpdir / 'starthere.pdb'))


def _parse_unused_links(log_text):
    """Parse refmac's 'Automatic generation of links' section.

    Returns a set of (chain, resnum_int) tuples for every residue that appears
    in an 'Unused' link entry (both ends of each clashing pair).

    Log line format:
      Unused  :  <link>  Mon1  At1  alt1  Ch1  Res1  Mon2  At2  alt2  Ch2  Res2  distM  distI
    """
    to_delete = set()
    in_section = False
    for line in log_text.splitlines():
        if 'Automatic generation of links' in line:
            in_section = True
        if not in_section:
            continue
        if not line.strip().startswith('Unused'):
            continue
        parts = line.split()
        # parts: [0]Unused [1]: [2]link [3]Mon1 [4]At1 [5]alt1 [6]Ch1 [7]Res1
        #        [8]Mon2 [9]At2 [10]alt2 [11]Ch2 [12]Res2 [13]distM [14]distI
        try:
            to_delete.add((parts[6],  int(parts[7])))
            to_delete.add((parts[11], int(parts[12])))
        except (IndexError, ValueError):
            pass
    return to_delete


def _add_extra_b(pdb_path, extra_b):
    """Add extra_b to every atom's B factor in pdb_path (in-place)."""
    st = gemmi.read_structure(str(pdb_path))
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    atom.b_iso = max(0.0, atom.b_iso + extra_b)
    st.write_pdb(str(pdb_path))


def _delete_residues_from_pdb(pdb_path, to_delete):
    """Delete residues from a PDB file in-place.

    to_delete: set of (chain_name, resnum_int) tuples.
    """
    st = gemmi.read_structure(str(pdb_path))
    for chain in st[0]:
        drop = [ri for ri, res in enumerate(chain)
                if (chain.name, res.seqid.num) in to_delete]
        for ri in reversed(drop):
            del chain[ri]
    st.write_pdb(str(pdb_path))


def step9_probe(tmpdir):
    """NCYC 0 refmac probe to detect geometry clashes (unused links).

    Runs refmac with NCYC 0 on starthere.pdb and parses the log for 'Unused'
    link entries, which indicate impossible geometry (e.g. side chain threading
    through a ring).  Saves the probe log to probe_refmac.log.

    Returns a set of (chain, resnum_int) tuples that should be deleted.
    """
    def _build_occ_bytes():
        run([str(SCRIPT_DIR / 'refmac_occupancy_setup.com'),
             'starthere.pdb', 'allhet'],
            cwd=tmpdir)
        b = (tmpdir / 'refmac_opts_occ.txt').read_bytes()
        return b if b.endswith(b'\n') else b + b'\n'

    probe_kw = (
        _build_occ_bytes() +
        b'MAKE HYDR A NEWLIGAND NOEXIT\n'
        b'NCYC 0\n'
        b'LABIN FP=F SIGFP=SIGF FREE=FreeR_flag\n'
        b'MONI DIST 10\n'
        b'END\n'
    )
    probe = subprocess.run(
        [str(REFMAC5),
         'XYZIN',  'starthere.pdb',
         'XYZOUT', '_probe.pdb',
         'HKLIN',  'refme.mtz',
         'HKLOUT', '_probe.mtz',
         'LIBOUT',  'refmac.lib'],
        input=probe_kw,
        cwd=str(tmpdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    probe_log = probe.stdout.decode(errors='replace')
    (tmpdir / 'probe_refmac.log').write_text(probe_log)
    return _parse_unused_links(probe_log)


def _max_abs_fofc_sigma(mtz_path):
    """Return max |Fo-Fc| / σ(Fo-Fc) over the full unit cell map."""
    mtz = gemmi.read_mtz_file(str(mtz_path))
    grid = mtz.transform_f_phi_to_map('DELFWT', 'PHDELWT', sample_rate=SAMPLE_RATE)
    arr  = np.asarray(grid, dtype=np.float64)
    s    = float(arr.std())
    return float(np.max(np.abs(arr)) / s) if s > 0 else 0.0


def step9_refmac(tmpdir, ncyc_per_round=(20, 40), refine_occ=(False, True),
                 weight_matrix=None,
                 water_round_ncyc=20, max_water_rounds=2, water_peak_threshold=5.0):
    """Run refmac in sequential rounds with per-round NCYC and occupancy control.

    ncyc_per_round: list of ints, one per round
    refine_occ:     list of bools (same length as ncyc_per_round); True enables
                    OCCUpancy GROUP keywords from refmac_occupancy_setup.com
    weight_matrix:  optional WEIGHT MATRIX override (default: refmac auto)

    After the base rounds, optional "water-fill" rounds: if the Fo-Fc map
    has any peak > water_peak_threshold σ, run add_waters.com to place new
    waters at the peaks, then refmac with VDWREST 0 (so adjacent partial-
    occ waters don't fight each other into the "water cannon" failure mode
    where two overlapping waters refine to occ-sum>1, then suddenly repel
    each other and smash into neighbouring atoms).  Repeat up to
    max_water_rounds times; stop early when peaks drop below threshold.

    Splitting occupancy refinement off the first round avoids a refmac
    intermediate-state bond-stretching bug: when occ is refined in round 1,
    the round-2 input has perturbed occupancies that confuse the geometry
    minimiser and stretch bonds (e.g. CB-CA → 1.73 Å on high-occ altlocs).

    Returns the concatenated log text.
    """
    n_rounds = len(ncyc_per_round)
    assert len(refine_occ) == n_rounds, "ncyc_per_round and refine_occ must have same length"

    def _build_occ_bytes(xyzin):
        run([str(SCRIPT_DIR / 'refmac_occupancy_setup.com'), xyzin, 'allhet'],
            cwd=tmpdir)
        b = (tmpdir / 'refmac_opts_occ.txt').read_bytes()
        return b if b.endswith(b'\n') else b + b'\n'

    def _rwork_rfree(log):
        # Refmac's canonical R-factor report (one per refinement step):
        #   Overall R factor                     =     0.0825
        #   Free R factor                        =     0.1166
        # The last occurrence is the final cycle.  Don't match the per-cycle
        # loggraph table line ("R factor   prev   curr") — that prints
        # *weighted* R-factors, not Rwork.  Match by '=' sign + leading
        # keyword so the parser stays robust.
        rw = rf = None
        for line in log.splitlines():
            s = line.strip()
            if s.startswith('Overall R factor') and '=' in s:
                try: rw = float(s.split('=')[-1].split()[0])
                except Exception: pass
            elif s.startswith('Free R factor') and '=' in s:
                try: rf = float(s.split('=')[-1].split()[0])
                except Exception: pass
        return rw, rf

    full_log = ''
    xyzin = 'starthere.pdb'

    for rnd in range(n_rounds):
        xyzout = 'refmacout.pdb'
        occ_bytes = _build_occ_bytes(xyzin) if refine_occ[rnd] else b''
        hout_bytes = b'MAKE HOUT Y\n' if rnd == n_rounds - 1 else b''
        # If the previous round did occupancy refinement, the partial-occ
        # waters may have overlapping occupancies > 1; turning off the
        # vdW restraint here prevents the "water cannon" repulsion that
        # would otherwise launch them into neighbouring atoms.
        vdw_bytes = b'VDWREST 0\n' if (rnd > 0 and refine_occ[rnd - 1]) else b''
        keywords = (
            occ_bytes +
            b'MAKE HYDR A NEWLIGAND NOEXIT\n' +
            hout_bytes +
            vdw_bytes +
            f'NCYC {ncyc_per_round[rnd]}\n'.encode() +
            (f'WEIGHT MATRIX {weight_matrix}\n'.encode() if weight_matrix is not None else b'') +
            b'LABIN FP=F SIGFP=SIGF FREE=FreeR_flag\n'
            b'LABOUT FC=FC PHIC=PHIC FWT=FWT PHWT=PHWT '
            b'DELFWT=DELFWT PHDELWT=PHDELWT\n'
            b'MONI DIST 10\n'
            b'END\n'
        )
        result = subprocess.run(
            [str(REFMAC5),
             'XYZIN',  xyzin,
             'XYZOUT', xyzout,
             'HKLIN',  'refme.mtz',
             'HKLOUT', 'refmacout.mtz',
             'LIBOUT',  'refmac.lib'],
            input=keywords,
            cwd=str(tmpdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_text = result.stdout.decode(errors='replace')
        full_log += f'\n{"="*60}\n refmac round {rnd+1}/{n_rounds}\n{"="*60}\n' + log_text
        if result.returncode != 0:
            (tmpdir / 'refmac.log').write_text(full_log)
            raise RuntimeError(f'refmac5 round {rnd+1} failed:\n{log_text[-3000:]}')
        rw, rf = _rwork_rfree(log_text)
        r_str = f'R={rw:.4f} Rf={rf:.4f}' if rw is not None else 'R=n/a'
        print(f'    refmac round {rnd+1}/{n_rounds}: {r_str}')
        xyzin = xyzout  # feed output into next round

    # ── Water-fill rounds (optional) ─────────────────────────────────────────
    # After the base refinement rounds, look for Fo-Fc peaks > threshold σ in
    # refmacout.mtz.  If any exist, run add_waters.com to place waters at
    # the peaks → new.pdb, re-generate occupancy groups, and refmac again
    # with VDWREST 0 to prevent water-cannon repulsion between overlapping
    # partial-occ waters.  Up to max_water_rounds passes.
    for wrnd in range(max_water_rounds):
        peak_sigma = _max_abs_fofc_sigma(tmpdir / 'refmacout.mtz')
        if peak_sigma <= water_peak_threshold:
            print(f'    water-fill skipped: max |Fo-Fc| = {peak_sigma:.2f}σ '
                  f'≤ {water_peak_threshold:.1f}σ threshold')
            break
        # add_waters.com  <distance>  <pdb>  <mtz>   →  new.pdb
        aw = subprocess.run(
            ['add_waters.com', '0.8A', 'refmacout.pdb', 'refmacout.mtz'],
            cwd=str(tmpdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (tmpdir / f'add_waters_{wrnd+1}.log').write_text(
            aw.stdout.decode(errors='replace'))
        new_pdb = tmpdir / 'new.pdb'
        if aw.returncode != 0 or not new_pdb.exists():
            print(f'    water-fill round {wrnd+1}: add_waters.com failed; '
                  f'skipping remaining water rounds')
            break
        # Regenerate occ-group keywords from new.pdb (allhet → independent water occ)
        occ_bytes = _build_occ_bytes('new.pdb')
        # Final round → keep MAKE HOUT Y so H stay in refmacout.pdb
        is_last = (wrnd == max_water_rounds - 1)
        hout_bytes = b'MAKE HOUT Y\n' if is_last else b''
        kw = (
            occ_bytes +
            b'MAKE HYDR A NEWLIGAND NOEXIT\n' +
            hout_bytes +
            f'NCYC {water_round_ncyc}\n'.encode() +
            b'VDWREST 0\n' +
            (f'WEIGHT MATRIX {weight_matrix}\n'.encode() if weight_matrix is not None else b'') +
            b'LABIN FP=F SIGFP=SIGF FREE=FreeR_flag\n'
            b'LABOUT FC=FC PHIC=PHIC FWT=FWT PHWT=PHWT '
            b'DELFWT=DELFWT PHDELWT=PHDELWT\n'
            b'MONI DIST 10\n'
            b'END\n'
        )
        wresult = subprocess.run(
            [str(REFMAC5),
             'XYZIN',  'new.pdb',
             'XYZOUT', 'refmacout.pdb',
             'HKLIN',  'refme.mtz',
             'HKLOUT', 'refmacout.mtz',
             'LIBOUT',  'refmac.lib'],
            input=kw,
            cwd=str(tmpdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        wlog = wresult.stdout.decode(errors='replace')
        full_log += (f'\n{"="*60}\n refmac water-fill round {wrnd+1}'
                     f' (vdwrest=0, peak was {peak_sigma:.2f}σ)\n{"="*60}\n' + wlog)
        if wresult.returncode != 0:
            (tmpdir / 'refmac.log').write_text(full_log)
            raise RuntimeError(f'refmac water round {wrnd+1} failed:\n{wlog[-3000:]}')
        rw, rf = _rwork_rfree(wlog)
        r_str = f'R={rw:.4f} Rf={rf:.4f}' if rw is not None else 'R=n/a'
        print(f'    refmac water-fill round {wrnd+1}: peak was {peak_sigma:.2f}σ; {r_str}')

    (tmpdir / 'refmac.log').write_text(full_log)
    return full_log


def _read_grid_shape(map_path):
    """Read (NS, NR, NC) grid dimensions from CCP4 map header."""
    import struct
    with open(map_path, 'rb') as f:
        nc, nr, ns = struct.unpack('3i', f.read(12))
    return [int(ns), int(nr), int(nc)]


def step10_convert_maps(tmpdir, outdir):
    """Compute CCP4 .map files.

    truth.map  — FC/PHIC from truth.mtz (full model + H, all HKLs)
    2fofc.map  — FWT/PHWT from refmacout.mtz (sigma-A weighted 2Fo-Fc)
    fofc.map   — DELFWT/PHDELWT from refmacout.mtz (sigma-A weighted Fo-Fc)
    fc.map     — FC_ALL_LS/PHIC_ALL_LS from refmacout.mtz: bulk-solvent-corrected
                 Fc for ALL unique HKLs (present and missing), provided because
                 refme.mtz was completed with uniqueify before refinement.
    """
    def mtz_to_ccp4(mtz_path, f_col, phi_col, out_path):
        mtz  = gemmi.read_mtz_file(str(mtz_path))
        grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=SAMPLE_RATE)
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header()
        ccp4.write_ccp4_map(str(out_path))

    mtz_r = tmpdir / 'refmacout.mtz'
    mtz_t = tmpdir / 'truth.mtz'
    mtz_to_ccp4(mtz_t, 'FC',         'PHIC',         outdir / 'truth.map')
    mtz_to_ccp4(mtz_r, 'FWT',        'PHWT',         outdir / '2fofc.map')
    mtz_to_ccp4(mtz_r, 'DELFWT',     'PHDELWT',      outdir / 'fofc.map')
    mtz_to_ccp4(mtz_r, 'FC_ALL_LS',  'PHIC_ALL_LS',  outdir / 'fc.map')

    # True phased difference map: truth.map - fc.map
    # This is the *correct* Fo-Fc the network is being trained to predict.
    truth_grid = gemmi.read_ccp4_map(str(outdir / 'truth.map')).grid
    fc_grid    = gemmi.read_ccp4_map(str(outdir / 'fc.map')).grid
    diff_arr   = np.array(truth_grid, copy=False) - np.array(fc_grid, copy=False)
    diff_grid  = gemmi.FloatGrid(diff_arr.astype(np.float32),
                                 truth_grid.unit_cell, truth_grid.spacegroup)
    diff_ccp4 = gemmi.Ccp4Map()
    diff_ccp4.grid = diff_grid
    diff_ccp4.update_ccp4_header()
    diff_ccp4.write_ccp4_map(str(outdir / 'truediff.map'))


def _generate_flood_waters(truth_full_pdb, rng, n_flood, flood_occ,
                            collision_pdb=None):
    """Append chain F flood waters to truth_full.pdb, avoiding existing atoms.

    Flood waters are checked only against the protein/ordered-water atoms in
    collision_pdb (or truth_full_pdb if not given); they do NOT exclude each
    other, so overlapping flood waters are allowed.  Collision_pdb should be a
    single-conformer model (e.g. the self-refined PDB) — ~20× cheaper than the
    20-altloc truth model and essentially equivalent since altlocs are within
    ~1 Å of single-conformer positions.

    Returns the number of flood waters actually placed.
    """
    _occ = float(flood_occ) if flood_occ is not None else 0.1
    st = gemmi.read_structure(str(truth_full_pdb))
    coll_src = gemmi.read_structure(str(collision_pdb)) if collision_pdb else st
    # Skip H atoms — they sit inside heavy-atom exclusion radii anyway.
    existing_xyz = np.array(
        [(a.pos.x, a.pos.y, a.pos.z)
         for chain in coll_src[0] for res in chain for a in res
         if a.element.name != 'H'],
        dtype=np.float64,
    )
    margin = 2.0
    flood_chain = gemmi.Chain('F')
    added = 0
    for _ in range(n_flood * 20):
        if added >= n_flood:
            break
        x = float(rng.uniform(margin, CELL[0] - margin))
        y = float(rng.uniform(margin, CELL[1] - margin))
        z = float(rng.uniform(margin, CELL[2] - margin))
        dx = existing_xyz[:, 0] - x
        dy = existing_xyz[:, 1] - y
        dz = existing_xyz[:, 2] - z
        if np.any(dx * dx + dy * dy + dz * dz < 7.84):
            continue
        b = float(np.clip(rng.lognormal(BFAC_SC_MU, BFAC_SC_SIGMA + 0.3),
                          BFAC_SC_MIN, 120.0))
        res = gemmi.Residue()
        res.name  = 'HOH'
        res.seqid = gemmi.SeqId(added + 1, ' ')
        atom         = gemmi.Atom()
        atom.name    = 'O'
        atom.element = gemmi.Element('O')
        atom.pos     = gemmi.Position(x, y, z)
        atom.occ     = _occ
        atom.b_iso   = b
        res.add_atom(atom)
        flood_chain.add_residue(res)
        added += 1
    if added > 0:
        st[0].add_chain(flood_chain)
    st.write_pdb(str(truth_full_pdb))
    return added


# ══════════════════════════════════════════════════════════════════════════════
# Sample orchestration
# ══════════════════════════════════════════════════════════════════════════════

def generate_sample(sample_idx, outdir, n_residues=20, n_waters=10, n_flood=0,
                    flood_avoid_fullocc=True, flood_occ=None, vary_flood=False,
                    shift_scale=0.5, n_altlocs=2, missing_fraction=0.05,
                    never_collected_fraction=0.0, extra_b=0.0,
                    altloc_swaps_per_res=1.0, weight_matrix=None,
                    per_conf_geommin=True, reference_pdb=None,
                    jiggle_distribution='byB',
                    phenix_refine_starthere=False,
                    skip_refmac=False,
                    fast_refmac=False,
                    seed=None, debug=False):
    """Run the full pipeline for one sample. Returns (sample_idx, ok, info).

    If debug=True, the entire tmpdir is copied to sample_dir/debug/ before
    cleanup, giving access to all intermediate PDB and log files.
    """
    t0 = time.time()
    outdir = Path(outdir).resolve()
    sample_dir = outdir / f'sample_{sample_idx:05d}'

    if sample_dir.exists() and (sample_dir / 'metadata.json').exists():
        return sample_idx, True, 'already done'

    # Remove any stale prot_* dirs left by previously aborted runs on this node
    ccp4_scr = Path(os.environ.get('CCP4_SCR', '/tmp'))
    os.makedirs(ccp4_scr, exist_ok=True)

    effective_seed = sample_idx if seed is None else seed + sample_idx
    rng = np.random.default_rng(seed=effective_seed)

    if vary_flood and n_flood > 0:
        rng_flood = np.random.default_rng(seed=effective_seed + 4)
        log_nf  = rng_flood.uniform(np.log(FLOOD_NF_MIN), np.log(FLOOD_NF_MAX))
        n_flood = int(np.round(np.exp(log_nf)))
        flood_occ = rng_flood.uniform(0.01, 0.05)

    # Reference-PDB mode: sequence + per-(resnum,atom) RMSF come from the
    # reference instead of random AA sampling / per-restype defaults.
    ref = _parse_reference(reference_pdb) if reference_pdb else None
    if ref is not None:
        # Sort residues by seqid.num so build_n2c sees them in chain order.
        ordered = sorted(ref['sequence'].items())
        seq = [restype for _, restype in ordered]
        n_residues = len(seq)
    else:
        seq = list(rng.choice(AA_NAMES, size=n_residues, p=AA_PROBS))

    tmpdir = Path(tempfile.mkdtemp(prefix=f'prot_{sample_idx:05d}_',
                                   dir=ccp4_scr))
    timings = {}
    def _t(label, t_prev):
        now = time.time()
        timings[label] = round(now - t_prev, 1)
        return now

    try:
        t = time.time()

        # 1-2: Build backbone + side chains
        step1_build_backbone(seq, rng, tmpdir)
        step2_build_sidechains(seq, rng, tmpdir)
        t = _t('build_seq', t)

        # 3: Set up structure (cell, waters, centre, B factors)
        n_water_added = step3_setup_structure(tmpdir, rng, n_waters=n_waters)
        t = _t('setup_struct', t)

        # In boiled-from-reference mode, phenix.geometry_minimization gets an
        # .eff file with explicit bond edits to pull random-coil SGs together
        # into the reference's disulfide pairs.  After minimisation we inject
        # SSBOND+LINK records into the result so refmac downstream applies the
        # same restraint.
        ss_pairs = ref['disulfides'] if ref is not None else []

        # 4: Geometry minimisation.  When ss_pairs are present and any SG-SG
        #    distance exceeds the shortest cell axis, step4 temporarily widens
        #    the cell to P1 / max_reasonable_bond Å so the long bond edits are
        #    allowed; otherwise it runs in the target cell+SG directly.
        minimized_pdb = step4_phenix_geommin(
            'built.pdb', tmpdir, log_tag='_1st',
            disulfide_pairs=ss_pairs,
        )
        if ss_pairs:
            _inject_ssbonds(minimized_pdb, ss_pairs)
        t = _t('phenix_gm_1st', t)

        # 4a (boiled only): second symmetry-aware minimisation pass.
        # The first pass ran in widened-cell P1 (so it could pull long SS
        # bonds within phenix's "shortest cell axis" hard cap), which means
        # no symmetry pressure compacted the chain.  This second pass runs in
        # the target cell+SG with the SS bonds already short, so phenix's
        # symmetry awareness packs the chain into the ASU before refmac sees
        # it.  Without this, selfref refmac inherits a chain that crosses ASU
        # boundaries and crushes the structure (and tangles the SGs because
        # auto-link detection on the compacted ball finds bogus SS pairs).
        # TODO: replace with a pre-positioning step on built.pdb so this
        # second call is unnecessary.
        if ss_pairs:
            shutil.copy2(minimized_pdb, tmpdir / 'built_pack.pdb')
            packed_pdb = step4_phenix_geommin(
                'built_pack.pdb', tmpdir, log_tag='_2nd',
                disulfide_pairs=ss_pairs, max_reasonable_bond=10.0,
            )
            _inject_ssbonds(packed_pdb, ss_pairs)
            minimized_pdb = packed_pdb
            t = _t('phenix_gm_2nd', t)

        # 4c: Check .geo file for severe heavy-atom clashes — nonbond LJ > 10
        #     and bond |delta| > 0.3 Å (stretched/compressed bonds indicate
        #     irresolvable clashes).  Delete offenders NOW before building
        #     altlocs — cheaper than re-running sfcalc/refmac later.
        geo_file = minimized_pdb.with_suffix('.geo')  # built_minimized or built_pack_minimized
        geo_bad = _parse_geo_bad_nonbonds(geo_file)
        if geo_bad:
            log.info('step4c: deleting %d residues with severe nonbond clashes: %s',
                     len(geo_bad), geo_bad)
            _delete_residues_from_pdb(minimized_pdb, geo_bad)
        t = _t('geo_clash_check', t)

        # 4b: Assign B factors from reference:
        #   selfref_pdb (jigglepdb amplitude): B = 8π²·RMSF²/3 per atom
        #   selfref_bfac_pdb (truth_full.pdb): mean B across conformers per atom
        selfref_pdb = minimized_pdb
        _set_target_bfactors(selfref_pdb,
                             rmsf_by_resnum=ref['rmsf_table'] if ref else None)
        selfref_bfac_pdb = tmpdir / 'selfref_bfac_truth.pdb'
        shutil.copy2(minimized_pdb, selfref_bfac_pdb)
        if ref is not None:
            _apply_bfac_table(selfref_bfac_pdb, ref['bfac_table'])

        # 4d (boiled only): if the reference flagged bimodal atoms, split the
        # selfref model into two "parent" models that differ at those atoms
        # by ±d_inter/2 along a fresh random direction.  Half of the altloc
        # jiggle seeds will run from parent_A, half from parent_B — this
        # reproduces the inter-cluster spread (which jiggle alone cannot,
        # because it samples a unimodal distribution around one position).
        parent_pdbs = selfref_pdb
        if ref is not None and ref.get('bimodal_atoms'):
            parent_A = tmpdir / 'parent_A.pdb'
            parent_B = tmpdir / 'parent_B.pdb'
            n_split  = _apply_bimodal_split(selfref_pdb,
                                            ref['bimodal_atoms'],
                                            parent_A, parent_B,
                                            tmpdir, rng,
                                            disulfide_pairs=ref.get('disulfides'))
            print(f'  bimodal split: {n_split} atoms placed at ±d/2; '
                  f'phenix braces relaxed bonded geometry')
            # Interleave parents A/B so altloc labels alternate: A=parent_A,
            # B=parent_B, C=parent_A, ...  _reduce_conformers drops the
            # alphabetically-lowest labels first (ties broken by label), so
            # grouping [A]*n_A + [B]*n_B would leave starthere with altlocs
            # all from parent_B (e.g. R,S,T at n_altlocs=20).  Alternating
            # guarantees the surviving max_confs altlocs straddle both
            # clusters, so the partial model covers each bimodal mode.
            parent_pdbs = [parent_A if i % 2 == 0 else parent_B
                           for i in range(n_altlocs)]

        # 5: jigglepdb using RMSF-derived B → N full chains in multiconf.pdb;
        #    truth B-factors in merged output come from mean reference B (selfref_bfac_pdb).
        step5_jigglepdb_and_merge(parent_pdbs, tmpdir, rng,
                                  shift_scale=shift_scale, n_altlocs=n_altlocs,
                                  per_conf_geommin=per_conf_geommin,
                                  bfac_source_pdb=selfref_bfac_pdb,
                                  jiggle_shift=jiggle_distribution)
        t = _t('jiggle_and_merge', t)

        # 6: Each conformer was already minimized independently inside
        #    step5_jigglepdb_and_merge; multiconf.pdb is the truth structure.
        shutil.copy2(tmpdir / 'multiconf.pdb', tmpdir / 'truth_full.pdb')

        # 6a: Inject flood waters into truth_full.pdb now that all protein/water
        #     atoms are finalized; avoids their positions. Then flip half the signs.
        n_flood_added = 0
        if n_flood > 0:
            n_flood_added = _generate_flood_waters(
                tmpdir / 'truth_full.pdb', rng, n_flood, flood_occ,
                collision_pdb=selfref_pdb)
            _apply_flood_signs(tmpdir / 'truth_full.pdb', rng)

        # 6b: Apply extra_b to all truth atoms — broadens the target density,
        #     simulating lower effective resolution.  Modifies truth_full.pdb
        #     in-place so the saved PDB, truth.mtz, and truth.map are consistent.
        if extra_b:
            _add_extra_b(tmpdir / 'truth_full.pdb', extra_b)

        # 7: sfcalc on truth_full → truth.mtz + Fpart.mtz (bulk solvent SFs)
        step6_sfcalc(tmpdir / 'truth_full.pdb', tmpdir / 'truth.mtz', tmpdir,
                     fpart_out=tmpdir / 'Fpart.mtz')
        t = _t('sfcalc', t)

        # 8: Build refme.mtz
        step7_build_refme_mtz(tmpdir / 'truth.mtz', tmpdir / 'refme.mtz')

        # 8b: Complete unique set + FreeR flags via uniqueify
        #     Must run BEFORE simulate_missing_data so FreeR_flag is in the file.
        step7c_add_freer_flags(tmpdir / 'refme.mtz', tmpdir)

        # 8c: Simulate missing data on the uniqueify-completed MTZ.
        #     'missing'         → F/SIGF set to NaN, row retained; refmac Fc-fills.
        #     'never_collected' → row deleted entirely; refmac has no knowledge of them.
        n_missing, n_never = simulate_missing_data(
            tmpdir / 'refme.mtz', missing_fraction, never_collected_fraction, rng)
        t = _t('refme_mtz', t)

        # 9: Build mixed single/multi-conformer model → starthere.pdb
        step8_build_mixed_model(tmpdir / 'truth_full.pdb', tmpdir, rng,
                                 altloc_swaps_per_res=altloc_swaps_per_res)
        t = _t('build_mixed_model', t)

        # 9b (optional, gated by phenix_refine_starthere): vanilla phenix.refine
        # on starthere.pdb against refme.mtz, so refmac inherits a partially-
        # refined model (helps when starthere has large positional offsets from
        # the truth).  Costs ~220 s; off by default since it has worsened R
        # in current testing.  Output overwrites starthere.pdb.  Failure is
        # non-fatal — fall through to refmac.  starthere0.pdb is always
        # written as a backup (regardless of whether this step runs).
        shutil.copy2(tmpdir / 'starthere.pdb', tmpdir / 'starthere0.pdb')
        if phenix_refine_starthere:
            try:
                res = subprocess.run(
                    ['phenix.refine', 'starthere.pdb', 'refme.mtz',
                     'prefix=starthere_phx',
                     'refinement.input.xray_data.r_free_flags.label=FreeR_flag',
                     'main.number_of_macro_cycles=3',
                     '--overwrite'],
                    cwd=str(tmpdir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                (tmpdir / 'starthere_phx.log').write_text(res.stdout.decode(errors='replace'))
                refined = tmpdir / 'starthere_phx_001.pdb'
                if res.returncode == 0 and refined.exists():
                    shutil.copy2(refined, tmpdir / 'starthere.pdb')
                    print(f'  phenix.refine starthere: ok (replaced starthere.pdb)')
                else:
                    print(f'  phenix.refine starthere: failed rc={res.returncode}; '
                          f'using unrefined starthere.pdb')
            except FileNotFoundError:
                print(f'  phenix.refine not on PATH; using unrefined starthere.pdb')
            t = _t('phenix_refine_starthere', t)

        # 10: Refmac NCYC=20 × 2 rounds with occupancy refinement on both
        if skip_refmac:
            sample_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmpdir / 'refme.mtz',     sample_dir / 'refme.mtz')
            shutil.copy2(tmpdir / 'starthere.pdb', sample_dir / 'starthere.pdb')
            shutil.copy2(tmpdir / 'truth_full.pdb', sample_dir / 'truth_full.pdb')
            if debug:
                debug_dir = sample_dir / 'debug'
                if debug_dir.exists():
                    subprocess.run(['rm', '-rf', str(debug_dir)], check=False)
                shutil.copytree(str(tmpdir), str(debug_dir))
            elapsed = time.time() - t0
            timing_str = '  '.join(f'{k}={v}s' for k, v in timings.items())
            return sample_idx, True, f'ok (refmac skipped) in {elapsed:.1f}s\n  {timing_str}'

        if fast_refmac:
            refmac_log = step9_refmac(tmpdir,
                                      ncyc_per_round=(5,), refine_occ=(True,),
                                      max_water_rounds=0,
                                      weight_matrix=weight_matrix)
            t = _t('refmac_fast1x5', t)
        else:
            refmac_log = step9_refmac(tmpdir,
                                      ncyc_per_round=(20, 20), refine_occ=(True, True),
                                      weight_matrix=weight_matrix)
            t = _t('refmac_2x20', t)

        # Parse final Rwork from refmac log
        # "Overall R factor = 0.0201" appears once per cycle; last is final cycle.
        rwork = None
        hits = re.findall(r'Overall R factor\s*=\s*(\d+\.\d+)', refmac_log)
        if hits:
            rwork = float(hits[-1])

        # 11: Convert to maps
        sample_dir.mkdir(parents=True, exist_ok=True)
        step10_convert_maps(tmpdir, sample_dir)
        fofc_peak_sigma = None   # set below after refmacout.pdb is in sample_dir

        # Copy PDB and log files
        shutil.copy2(tmpdir / 'truth_full.pdb',  sample_dir / 'truth_full.pdb')
        shutil.copy2(tmpdir / 'starthere.pdb',    sample_dir / 'partial.pdb')
        if (tmpdir / 'refmacout.pdb').exists():
            shutil.copy2(tmpdir / 'refmacout.pdb', sample_dir / 'refmacout.pdb')
        if (tmpdir / 'refmacout.mtz').exists():
            shutil.copy2(tmpdir / 'refmacout.mtz', sample_dir / 'refmacout.mtz')
        shutil.copy2(tmpdir / 'refme.mtz',        sample_dir / 'refme.mtz')
        if (tmpdir / 'Fpart.mtz').exists():
            shutil.copy2(tmpdir / 'Fpart.mtz',    sample_dir / 'Fpart.mtz')
        shutil.copy2(tmpdir / 'refmac.log',        sample_dir / 'refmac.log')
        # (phenix logs land in debug/ via the debug copytree below — don't
        # copy them into the main sample dir; with per_conf_geommin and the
        # bimodal pipeline there are 20+ of them and they bloat large runs.)

        # 11a: Pick the 5 most extreme Fo-Fc peaks and report their nearest
        # atom — catches model/data inconsistencies before they reach the
        # CNN. Runs against fofc.map + refmacout.pdb in sample_dir (must be
        # after the copies above). Skips silently if pick.com isn't on PATH.
        if shutil.which('pick.com') and (sample_dir / 'fofc.map').exists() and \
                (sample_dir / 'refmacout.pdb').exists():
            try:
                pick_res = subprocess.run(
                    ['pick.com', '-extreme=5', '-fast', 'fofc.map', 'refmacout.pdb'],
                    cwd=str(sample_dir),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=120,
                )
                txt = pick_res.stdout.decode(errors='replace')
                (sample_dir / 'pick_details.log').write_text(txt)
                sigmas = []
                for line in txt.splitlines():
                    parts = line.split()
                    if len(parts) >= 7 and parts[3].isdigit():
                        try:
                            sigmas.append(abs(float(parts[4])))
                        except ValueError:
                            pass
                if sigmas:
                    fofc_peak_sigma = float(max(sigmas))
                    log.info('step11a: max |Fo-Fc| peak = %.2fσ',
                             fofc_peak_sigma)
            except Exception as e:
                log.warning('step11a pick.com failed: %s', e)

        # Debug: dump entire tmpdir
        if debug:
            debug_dir = sample_dir / 'debug'
            if debug_dir.exists():
                # subprocess rm -rf instead of shutil.rmtree — the latter
                # raises ENOTEMPTY on NFS due to a directory-cache race.
                subprocess.run(['rm', '-rf', str(debug_dir)], check=False)
            shutil.copytree(str(tmpdir), str(debug_dir))

        t = _t('maps_and_copy', t)

        # Metadata
        meta = dict(
            sample_idx=int(sample_idx),
            sequence=seq,
            n_residues=n_residues,
            n_waters_requested=n_waters,
            n_waters_added=n_water_added,
            n_flood_added=n_flood_added,
            n_clashing_residues_deleted=len(geo_bad),
            rwork_final=rwork,
            max_fofc_peak_sigma=fofc_peak_sigma,
            missing_fraction=missing_fraction,
            n_reflections_missing=n_missing,
            never_collected_fraction=never_collected_fraction,
            n_reflections_never_collected=n_never,
            extra_b=extra_b,
            altloc_swaps_per_res=altloc_swaps_per_res,
            vary_flood=vary_flood,
            n_flood_actual=n_flood,
            flood_occ_actual=flood_occ,
            cell=list(CELL),
            spacegroup=SPACEGROUP,
            dmin=DMIN,
            grid_shape=list(_read_grid_shape(sample_dir / 'truth.map')),
            step_timings=timings,
        )
        (sample_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))

        elapsed = time.time() - t0
        timing_str = '  '.join(f'{k}={v}s' for k, v in timings.items())
        return sample_idx, True, f'ok in {elapsed:.1f}s  Rwork={rwork}\n  {timing_str}'

    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / 'error.log').write_text(msg)
        debug_dir = sample_dir / 'debug'
        if debug_dir.exists():
            subprocess.run(['rm', '-rf', str(debug_dir)], check=False)
        try:
            shutil.copytree(str(tmpdir), str(debug_dir))
        except Exception:
            pass
        elapsed = time.time() - t0
        return sample_idx, False, f'FAILED in {elapsed:.1f}s: {e}'

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLURM array submission helper
# ══════════════════════════════════════════════════════════════════════════════

def submit_slurm_array(nsamples, outdir, n_residues, n_waters, n_flood=0,
                       flood_avoid_fullocc=True, shift_scale=0.5, n_altlocs=2,
                       missing_fraction=0.05, never_collected_fraction=0.05,
                       extra_b=0.0, altloc_swaps_per_res=1.0, vary_flood=False,
                       max_array=300, seed=None, flood_occ=None, cell=None,
                       dmin=2.0, spacegroup='P 1', partition='debug',
                       account=None, qos=None, time='00:10:00',
                       weight_matrix=None, per_conf_geommin=True,
                       exclude_nodes=None, reference_pdb=None,
                       phenix_refine_starthere=False, skip_refmac=False,
                       fast_refmac=False, debug=False):
    """Write and submit a SLURM array job script."""
    script = SCRIPT_DIR / f'_slurm_{outdir.name}.sh'
    python  = sys.executable
    me      = Path(__file__).resolve()

    # Each task needs one CPU per altloc conformer so phenix.GM runs truly in
    # parallel inside step5_jigglepdb_and_merge (ThreadPoolExecutor).
    cpus_per_task = max(n_altlocs, 2)

    seed_line      = f'    --seed {seed} \\\n'                     if seed                is not None else ''
    flood_occ_line = f'    --flood-occ {flood_occ} \\\n'           if flood_occ           is not None else ''
    extra_b_line   = f'    --extra-b {extra_b} \\\n'               if extra_b                         else ''
    scramble_line  = f'    --altloc-swaps-per-res {altloc_swaps_per_res} \\\n' if altloc_swaps_per_res != 1.0 else ''
    weight_line    = f'    --weight-matrix {weight_matrix} \\\n'              if weight_matrix is not None   else ''
    varflood_line  = f'    --vary-flood \\\n'                       if vary_flood                      else ''
    sg_line        = f'    --spacegroup "{spacegroup}" \\\n'        if spacegroup != 'P 1'             else ''
    _cell = cell if cell is not None else (40.0, 40.0, 40.0)
    cell_line      = f'    --cell {_cell[0]} {_cell[1]} {_cell[2]} \\\n'
    dmin_line      = f'    --dmin {dmin} \\\n'
    account_line   = f'#SBATCH --account={account}\n'    if account              else ''
    qos_line       = f'#SBATCH --qos={qos}\n'            if qos                  else ''
    exclude_line   = f'#SBATCH --exclude={exclude_nodes}\n' if exclude_nodes     else ''
    pcg_line       = f'    --per-conf-geommin \\\n'      if per_conf_geommin     else ''
    refpdb_line    = f'    --reference-pdb {Path(reference_pdb).resolve()} \\\n' if reference_pdb else ''
    phxstart_line  = f'    --phenix-refine-starthere \\\n'                        if phenix_refine_starthere else ''
    skiprefmac_line  = f'    --skip-refmac \\\n'                                   if skip_refmac   else ''
    fastrefmac_line  = f'    --fast-refmac \\\n'                                  if fast_refmac   else ''
    debug_line       = f'    --debug \\\n'                                        if debug         else ''
    script_text = f"""\
#!/bin/bash
#SBATCH --job-name=prot_data
#SBATCH --partition={partition}
{account_line}{qos_line}{exclude_line}#SBATCH --array=0-{nsamples-1}%{max_array}
#SBATCH --output={outdir}/logs/%A_%a.log
#SBATCH --error={outdir}/logs/%A_%a.log
#SBATCH --cpus-per-task={cpus_per_task}

mkdir -p {outdir}/logs
mkdir -p "${{CCP4_SCR:-/tmp}}"
{python} {me} \\
    --sample-id $SLURM_ARRAY_TASK_ID \\
    --outdir {outdir} \\
    --nresidues {n_residues} \\
    --nwaters {n_waters} \\
    --n-flood {n_flood} \\
    {'--flood-avoid-fullocc' if flood_avoid_fullocc else '--no-flood-avoid-fullocc'} \\
    --shift-scale {shift_scale} \\
    --n-altlocs {n_altlocs} \\
    --missing-fraction {missing_fraction} \\
    --never-collected-fraction {never_collected_fraction} \\
{cell_line}{dmin_line}{sg_line}{flood_occ_line}{varflood_line}{extra_b_line}{scramble_line}{weight_line}{pcg_line}{refpdb_line}{phxstart_line}{skiprefmac_line}{fastrefmac_line}{debug_line}{seed_line}"""
    script.write_text(script_text)
    script.chmod(0o755)

    # Pre-create logs dir so SLURM can open the log files before the script body runs
    (outdir / 'logs').mkdir(parents=True, exist_ok=True)

    result = subprocess.run(['sbatch', str(script)],
                            capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode == 0


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate protein altloc training data for the CNN.')
    parser.add_argument('--outdir',     default='./data/data_protein')
    parser.add_argument('--nsamples',   type=int, default=100)
    parser.add_argument('--nresidues',  type=int, default=20)
    parser.add_argument('--nwaters',    type=int, default=30)
    parser.add_argument('--spacegroup', default='P 1',
                        help='Space group HM symbol (default: "P 1"). '
                             'For P2₁2₁2₁ use "P 21 21 21". '
                             'Protein is centred in the ASU; waters/floods '
                             'are restricted to the ASU region.')
    parser.add_argument('--workers',    type=int, default=1)
    parser.add_argument('--sample-id',  type=int, default=None,
                        help='Run a single sample (for SLURM array jobs)')
    parser.add_argument('--submit',     action='store_true',
                        help='Submit a SLURM array job instead of running locally')
    parser.add_argument('--partition',  default='debug',
                        help='SLURM partition (default: debug)')
    parser.add_argument('--account',    default=None,
                        help='SLURM account (e.g. pc_als831)')
    parser.add_argument('--qos',        default=None,
                        help='SLURM QOS (e.g. lr_normal)')
    parser.add_argument('--time',       default='00:15:00',
                        help='SLURM walltime per task (default: 00:15:00)')
    parser.add_argument('--max-array',  type=int, default=300,
                        help='SLURM --array concurrency limit')
    parser.add_argument('--n-flood',     type=int,   default=0,
                        help='Number of partial-occ flood waters to add (default 0)')
    parser.add_argument('--flood-occ',   type=float, default=None,
                        help='Fixed occupancy for flood waters (default: random 0.1-0.8)')
    parser.add_argument('--vary-flood', action='store_true', default=False,
                        help='Per-sample: draw n_flood ~ LogUniform(%d,%d), '
                             'set occ=%.2f/sqrt(n_flood) for Rfree~11%%' %
                             (FLOOD_NF_MIN, FLOOD_NF_MAX, FLOOD_LINE_K))
    parser.add_argument('--flood-avoid-fullocc', action='store_true', default=True,
                        help='Flood waters avoid full-occ atoms (default True)')
    parser.add_argument('--no-flood-avoid-fullocc', dest='flood_avoid_fullocc',
                        action='store_false',
                        help='Allow flood waters to overlap full-occ atoms')
    parser.add_argument('--shift-scale', type=float, default=0.5,
                        help='jigglepdb shift_scale (scales byB displacement; default 0.5)')
    parser.add_argument('--n-altlocs',   type=int,   default=2,
                        help='Number of alternate conformers to generate (default 2)')
    parser.add_argument('--missing-fraction', type=float, default=0.05,
                        help='Fraction of reflections collected but rejected (F→NaN, default 0.05)')
    parser.add_argument('--extra-b',  type=float, default=0.0,
                        help='Extra B factor added to all truth_full.pdb atoms before sfcalc')
    parser.add_argument('--weight-matrix', type=float, default=None,
                        help='refmac WEIGHT MATRIX value (default: auto)')
    parser.add_argument('--never-collected-fraction', type=float, default=0.05,
                        help='Fraction of reflections never measured (rows deleted, default 0.05)')
    parser.add_argument('--cell', nargs=3, type=float, default=[40.0, 40.0, 40.0],
                        metavar=('A', 'B', 'C'),
                        help='P1 unit cell dimensions in Å (default: 40 40 40)')
    parser.add_argument('--dmin', type=float, default=2.0,
                        help='Resolution cutoff in Å (default: 2.0)')
    parser.add_argument('--altloc-swaps-per-res', type=float, default=1.0,
                        help='Expected number of random pairwise altloc swaps per residue '
                             '(Poisson); 0=no scrambling, 1=~1 swap/res, >1=more scrambled '
                             '(default: 1.0)')
    parser.add_argument('--debug',      action='store_true',
                        help='Copy entire tmpdir to sample_dir/debug/ for inspection')
    parser.add_argument('--per-conf-geommin', action='store_true',
                        help='Run phenix.geometry_minimization on each conformer '
                             '(parallel via ThreadPoolExecutor; ~13s × n_altlocs)')
    parser.add_argument('--phenix-refine-starthere', action='store_true',
                        help='Run vanilla phenix.refine on starthere.pdb before '
                             'the first refmac round (off by default)')
    parser.add_argument('--skip-refmac', action='store_true',
                        help='Skip refmac (and downstream map conversion); copy '
                             'refme.mtz, starthere.pdb, and truth_full.pdb into '
                             'the sample dir and exit early')
    parser.add_argument('--fast-refmac', action='store_true',
                        help='Run one refmac round at NCYC=5 (no water fill) '
                             'instead of the default 2×NCYC=20 + water rounds; '
                             'faster data generation at lower model quality')
    parser.add_argument('--exclude-nodes',  default=None,
                        help='SLURM --exclude= node list (e.g. "voltron,graphics2" '
                             'to keep CPU jobs off GPU nodes)')
    parser.add_argument('--seed',       type=int, default=None,
                        help='Fixed RNG seed (overrides sample-id as seed); '
                             'use to hold the protein structure constant while varying other params')
    parser.add_argument('--reference-pdb', default=None,
                        help='Multi-conformer PDB whose sequence, per-(resnum,atom) RMSF, '
                             'and disulfide connectivity are used in place of the random/'
                             'per-restype defaults. Cell+spacegroup also taken from the '
                             'reference unless --cell/--spacegroup are explicit. Intended '
                             'for "boiled-from-reference" runs (e.g. 1aho/gt48.pdb).')
    parser.add_argument('--jiggle-distribution', default='byB',
                        choices=['byB', 'LorentzB', 'uniformB'],
                        help='jigglepdb displacement distribution. All three derive '
                             'the magnitude from the per-atom B factor. byB = isotropic '
                             'Gaussian (default); LorentzB = Lorentzian (heavy tails, '
                             'occasional far outliers); uniformB = uniform within a '
                             'sphere of radius shift (bounded, more "rotamer-like" '
                             'jumps). Lorentz/uniform may give more bimodal-looking '
                             'spread for high-B atoms.')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    global CELL, DMIN, SPACEGROUP
    CELL       = tuple(args.cell)
    DMIN       = args.dmin
    SPACEGROUP = args.spacegroup

    # Reference PDB overrides cell + spacegroup if neither was explicitly given.
    # (argparse default is [40,40,40] / 'P 1' / 2.0; we treat those as "not set".)
    if args.reference_pdb is not None:
        ref = _parse_reference(args.reference_pdb)
        if tuple(args.cell) == (40.0, 40.0, 40.0):
            CELL = ref['cell'][:3]
            print(f'  reference cell:       {CELL} (overrides default)')
        if args.spacegroup == 'P 1':
            SPACEGROUP = ref['spacegroup_hm']
            print(f'  reference spacegroup: {SPACEGROUP!r} (overrides default)')
        if args.dmin == 2.0 and ref.get('dmin') is not None:
            DMIN = ref['dmin']
            print(f'  reference dmin:       {DMIN} Å (overrides default 2.0)')
        n_ref = len(ref['sequence'])
        if args.nresidues != n_ref:
            print(f'  reference sequence has {n_ref} residues → overriding --nresidues')
            args.nresidues = n_ref
        # Boiled mode defaults shift_scale=1.0 (vs the protein-v4 default 0.5).
        # The reference's per-atom σ already encodes the full disorder, so the
        # jiggle amplitude should hit it 1:1, not be cut in half.
        if args.shift_scale == 0.5:
            args.shift_scale = 1.0
            print(f'  reference mode:       shift_scale → 1.0 (overrides default 0.5)')
        # Auto-set n_waters to the reference's distinct-water-site count.
        # (argparse default 30 ⇒ "not set"; user-provided values pass through.)
        n_ref_waters = int(ref.get('n_water_sites', 0) or 0)
        if args.nwaters == 30 and n_ref_waters > 0:
            args.nwaters = n_ref_waters
            print(f'  reference waters:     {args.nwaters} distinct ordered-water sites (overrides default 30)')
        print(f'  reference disulfides: {ref["disulfides"]}')
        print(f'  reference unpaired Cys: {ref["unpaired_cys"]}')
        print(f'  reference bimodal:    {len(ref.get("bimodal_atoms", {}))} atoms')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

    # ── Single sample (SLURM array task) ──────────────────────────────────────
    if args.sample_id is not None:
        idx, ok, msg = generate_sample(
            args.sample_id, outdir,
            n_residues=args.nresidues,
            n_waters=args.nwaters,
            n_flood=args.n_flood,
            flood_avoid_fullocc=args.flood_avoid_fullocc,
            flood_occ=args.flood_occ,
            vary_flood=args.vary_flood,
            shift_scale=args.shift_scale,
            n_altlocs=args.n_altlocs,
            missing_fraction=args.missing_fraction,
            never_collected_fraction=args.never_collected_fraction,
            extra_b=args.extra_b,
            altloc_swaps_per_res=args.altloc_swaps_per_res,
            weight_matrix=args.weight_matrix,
            per_conf_geommin=args.per_conf_geommin,
            reference_pdb=args.reference_pdb,
            jiggle_distribution=args.jiggle_distribution,
            phenix_refine_starthere=args.phenix_refine_starthere,
            skip_refmac=args.skip_refmac,
            fast_refmac=args.fast_refmac,
            seed=args.seed,
            debug=args.debug,
        )
        print(f'Sample {idx:05d}: {msg}')
        sys.exit(0 if ok else 1)

    # ── SLURM array submission ─────────────────────────────────────────────────
    if args.submit:
        ok = submit_slurm_array(
            args.nsamples, outdir.resolve(),
            args.nresidues, args.nwaters, args.n_flood, args.flood_avoid_fullocc,
            args.shift_scale, args.n_altlocs, args.missing_fraction,
            args.never_collected_fraction,
            extra_b=args.extra_b, altloc_swaps_per_res=args.altloc_swaps_per_res,
            weight_matrix=args.weight_matrix,
            vary_flood=args.vary_flood, max_array=args.max_array,
            seed=args.seed, flood_occ=args.flood_occ,
            cell=CELL, dmin=DMIN, spacegroup=SPACEGROUP,
            partition=args.partition, account=args.account, qos=args.qos,
            time=args.time,
            per_conf_geommin=args.per_conf_geommin,
            exclude_nodes=args.exclude_nodes,
            reference_pdb=args.reference_pdb,
            phenix_refine_starthere=args.phenix_refine_starthere,
            skip_refmac=args.skip_refmac,
            fast_refmac=args.fast_refmac,
            debug=args.debug,
        )
        sys.exit(0 if ok else 1)

    # ── Local parallel run ─────────────────────────────────────────────────────
    sample_ids = list(range(args.nsamples))
    done = ok_count = err_count = 0

    _kw = dict(
        n_residues=args.nresidues, n_waters=args.nwaters,
        n_flood=args.n_flood, flood_avoid_fullocc=args.flood_avoid_fullocc,
        flood_occ=args.flood_occ, vary_flood=args.vary_flood,
        shift_scale=args.shift_scale, n_altlocs=args.n_altlocs,
        missing_fraction=args.missing_fraction,
        never_collected_fraction=args.never_collected_fraction,
        extra_b=args.extra_b, altloc_swaps_per_res=args.altloc_swaps_per_res,
        weight_matrix=args.weight_matrix,
        reference_pdb=args.reference_pdb,
        jiggle_distribution=args.jiggle_distribution,
        phenix_refine_starthere=args.phenix_refine_starthere,
        skip_refmac=args.skip_refmac,
        fast_refmac=args.fast_refmac,
        seed=args.seed, debug=args.debug,
    )

    if args.workers <= 1:
        for sid in sample_ids:
            idx, ok, msg = generate_sample(sid, outdir, **_kw)
            done += 1
            status = 'OK' if ok else 'ERR'
            ok_count += ok; err_count += (not ok)
            log.info(f'[{done}/{args.nsamples}] {status} sample {idx:05d}: {msg}')
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(generate_sample, sid, str(outdir), **_kw): sid
                for sid in sample_ids
            }
            for fut in as_completed(futures):
                idx, ok, msg = fut.result()
                done += 1
                status = 'OK' if ok else 'ERR'
                ok_count += ok; err_count += (not ok)
                log.info(f'[{done}/{args.nsamples}] {status} sample {idx:05d}: {msg}')

    log.info(f'Done. ok={ok_count}  errors={err_count}')


if __name__ == '__main__':
    main()
