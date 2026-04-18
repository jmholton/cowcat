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
from concurrent.futures import ProcessPoolExecutor, as_completed

import gemmi
import numpy as np

# ── Tool paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
BUILD_N2C   = Path('/home/jamesh/projects/git/build_pdb/build_n2c.awk')
BUILD_SIDE  = Path('/home/jamesh/projects/git/build_pdb/build_side.awk')
JIGGLEPDB   = SCRIPT_DIR / 'jigglepdb.awk'
PHENIX_GM   = Path('/programs/phenix-2.0-5936/phenix_bin/phenix.geometry_minimization')
REFMAC5     = Path('/programs/ccp4-8.0/bin/refmac5')
UNIQUEIFY   = 'uniqueify'

# ── Crystallographic parameters ────────────────────────────────────────────────
CELL        = (40.0, 40.0, 40.0)   # P1 unit cell (Å)
DMIN        = 2.0                   # resolution cutoff (Å)
SAMPLE_RATE = 3.0                   # oversampling → 60×60×60 grid for 40 Å cell
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

# Altloc displacement threshold (Å): side chains further than this
# from their centroid stay as separate altloc atoms in the partial model
ALTLOC_DIST_THRESHOLD = 1.2

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


def step3_setup_structure(tmpdir, rng, n_waters=10, n_flood=0, flood_avoid_fullocc=True, flood_occ=None):
    """Read side.pdb → set cell, centre, randomise B, add waters → built.pdb."""
    st = gemmi.read_structure(str(tmpdir / 'side.pdb'))
    st.cell = gemmi.UnitCell(CELL[0], CELL[1], CELL[2], 90, 90, 90)
    st.spacegroup_hm = 'P 1'

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

    # Centre at (20, 20, 20)
    cx = sum(a.pos.x for a in all_atoms) / len(all_atoms)
    cy = sum(a.pos.y for a in all_atoms) / len(all_atoms)
    cz = sum(a.pos.z for a in all_atoms) / len(all_atoms)
    dx, dy, dz = CELL[0]/2 - cx, CELL[1]/2 - cy, CELL[2]/2 - cz
    for a in all_atoms:
        a.pos = gemmi.Position(a.pos.x + dx, a.pos.y + dy, a.pos.z + dz)

    # Set protein chain name to 'A'
    for chain in st[0]:
        chain.name = 'A'

    existing = [(a.pos.x, a.pos.y, a.pos.z) for a in all_atoms]

    # ── Pass 1: full-occupancy waters (avoid all existing atoms + each other) ──
    water_chain = gemmi.Chain('W')
    added = 0
    margin = 2.0
    for _ in range(100000):
        if added >= n_waters:
            break
        x = float(rng.uniform(margin, CELL[0] - margin))
        y = float(rng.uniform(margin, CELL[1] - margin))
        z = float(rng.uniform(margin, CELL[2] - margin))
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

    # ── Pass 2: flood waters in chain 'F' (partial occ, random B) ──────────
    # Chain 'F' is kept separate so _merge_altconfs can scale occupancies
    # to flood_occ total rather than 1.0.
    flood_added = 0
    if n_flood > 0:
        flood_chain = gemmi.Chain('F')
        fullocc_positions = existing if flood_avoid_fullocc else []
        for _ in range(n_flood * 20):
            if flood_added >= n_flood:
                break
            x = float(rng.uniform(margin, CELL[0] - margin))
            y = float(rng.uniform(margin, CELL[1] - margin))
            z = float(rng.uniform(margin, CELL[2] - margin))
            if flood_avoid_fullocc and not all(
                    (x-px)**2 + (y-py)**2 + (z-pz)**2 >= 7.84
                    for px, py, pz in fullocc_positions):
                continue
            b = float(np.clip(rng.lognormal(BFAC_SC_MU, BFAC_SC_SIGMA + 0.3),
                               BFAC_SC_MIN, 120.0))
            res = gemmi.Residue()
            res.name = 'HOH'
            res.seqid = gemmi.SeqId(flood_added + 1, ' ')
            atom = gemmi.Atom()
            atom.name = 'O'
            atom.element = gemmi.Element('O')
            atom.pos = gemmi.Position(x, y, z)
            atom.occ = 1.0   # jigglepdb/merge_altconfs will rescale to flood_occ
            atom.b_iso = b
            res.add_atom(atom)
            flood_chain.add_residue(res)
            flood_added += 1
        if flood_added > 0:
            st[0].add_chain(flood_chain)

    st.write_pdb(str(tmpdir / 'built.pdb'))
    return added, flood_added


def step4_phenix_geommin(pdb_name, tmpdir, log_tag=None):
    """Run phenix.geometry_minimization; return path to *_minimized.pdb.

    Saves stdout+stderr to {stem}{log_tag}.phenix.log in tmpdir.
    """
    result = run([PHENIX_GM, pdb_name, 'write_geo_file=False', 'cdl=false',
                  'link_all=False', 'link_none=True', 'link_ligands=False'],
                 cwd=tmpdir, check=False)
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
    return out


def step4b_selfref_b_factors(minimized_pdb, tmpdir):
    """Refine single-conf model against its own SFs to get realistic B factors.

    Runs refmac for 20 cycles with the model as both PDB and (synthetic) data
    source, allowing B-factor restraints to smooth out uncorrelated random B
    values into physically sensible, bonding-correlated ones.  The refined
    coordinate file is returned; its B factors are used by jigglepdb byB.
    """
    # Calculate SFs for the minimised model
    step6_sfcalc(minimized_pdb, tmpdir / 'selfref.mtz', tmpdir)

    # Build a pseudo-observed MTZ (F=|FC|, SIGF=0.02·|FC|)
    step7_build_refme_mtz(tmpdir / 'selfref.mtz', tmpdir / 'selfref_refme.mtz')

    keywords = (
        b'MAKE HYDR NO NEWLIGAND NOEXIT\n'
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
    return out


def step5_jigglepdb_and_merge(selfref_pdb, tmpdir, rng, shift_scale=0.5, n_altlocs=2, flood_occ=None):
    """Run jigglepdb n_altlocs times on the self-refined model, merge → multiconf.pdb."""
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_altlocs]
    seeds  = [int(rng.integers(1000, 99999)) for _ in range(n_altlocs)]
    conf_pdbs = []
    for seed, label in zip(seeds, labels):
        result = run(
            ['awk', '-f', str(JIGGLEPDB),
             '-v', f'seed={seed}',
             '-v', 'shift=byB',
             '-v', f'shift_scale={shift_scale}',
             '-v', 'dry_shift_scale=1.0',
             '-v', 'frac_thrubond=0.9',
             '-v', 'ncyc_thrubond=500',
             '-v', 'frac_magnforce=1.1',
             '-v', 'ncyc_magnforce=500',
             str(selfref_pdb)],
            cwd=tmpdir
        )
        p = tmpdir / f'conf{label}.pdb'
        p.write_bytes(result.stdout)
        conf_pdbs.append(p)

    _merge_altconfs(conf_pdbs, tmpdir / 'multiconf.pdb', rng=rng, flood_occ=flood_occ)


def _merge_altconfs(conf_pdbs, out_pdb, rng=None, flood_occ=None):
    """Combine N single-conf PDBs into an N-altloc PDB (altlocs A, B, C, ...).

    Per-residue occupancies are Dirichlet-distributed, scaled to a chain-specific
    total occupancy:
      chain 'A' (protein):  total = 1.0
      chain 'W' (ordered waters): total = rng.uniform(0.3, 1.0) per water
      chain 'F' (flood waters):   total = flood_occ (default 0.1)
    """
    labels   = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(conf_pdbs)]
    structs  = [gemmi.read_structure(str(p)) for p in conf_pdbs]
    st_out   = structs[0].clone()
    n_conf   = len(structs)
    _flood_occ = flood_occ if flood_occ is not None else 0.1

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
            # Dirichlet-distributed per-residue occupancies, scaled to total_occ
            if rng is not None:
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

    st_out.write_pdb(str(out_pdb))


def step6_sfcalc(pdb_path, mtz_out, tmpdir):
    """Add hydrogens to pdb_path in-place via hgen, then compute structure factors.

    pdb_path is overwritten with the H-containing model so that truth_full.pdb
    saved to the sample directory includes H.  FC amplitudes (and thus the
    'observed' Fo in refme.mtz) include the H contribution.
    """
    pdb_with_h = tmpdir / '_sfcalc_withH.pdb'
    result = subprocess.run(
        [str(PHENIX_GM.parent / 'phenix.reduce'), str(pdb_path)],
        cwd=str(tmpdir), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode not in (0, 1):  # reduce exits 1 on warnings, which is normal
        raise RuntimeError(f'phenix.reduce failed:\n{result.stderr.decode(errors="replace")[-1000:]}')
    pdb_with_h.write_bytes(result.stdout)
    # Replace truth_full.pdb with the H-containing version
    pdb_with_h.replace(pdb_path)
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
    Occupancy = sum of altloc occs (preserves partial occupancy)."""
    pos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms])
    cx, cy, cz = pos.mean(axis=0)
    a_out = gemmi.Atom()
    a_out.name    = atoms[0].name
    a_out.element = atoms[0].element
    a_out.pos     = gemmi.Position(cx, cy, cz)
    a_out.b_iso   = float(np.mean([a.b_iso for a in atoms]))
    a_out.occ     = float(min(1.0, sum(a.occ for a in atoms)))
    a_out.altloc  = '\x00'
    res_out.add_atom(a_out)


def _reduce_conformers(by_name, sc_names, max_confs=3):
    """Merge closest conformer pairs until ≤ max_confs altlocs remain.

    Distance between two conformers = max atomic displacement across all
    shared side-chain atoms (same metric as the keep_altloc threshold).
    Returns (updated_by_name, remaining_label_list).
    """
    labels = sorted({a.altloc for name in sc_names
                     for a in by_name.get(name, ())
                     if a.altloc and a.altloc != '\x00'})
    if len(labels) <= max_confs:
        return by_name, labels

    # Represent each conformer as {atom_name: [x, y, z, b, occ, element, atom_name_str]}
    conf = {}
    for l in labels:
        conf[l] = {}
        for name in sc_names:
            for a in by_name.get(name, ()):
                if a.altloc == l:
                    conf[l][name] = [a.pos.x, a.pos.y, a.pos.z,
                                     a.b_iso, a.occ, a.element, a.name]

    while len(labels) > max_confs:
        # Find the pair of conformers with the smallest max atomic displacement
        best_li, best_lj, best_dist = labels[0], labels[1], float('inf')
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                li, lj = labels[i], labels[j]
                shared = set(conf[li]) & set(conf[lj])
                d = max((np.linalg.norm(np.array(conf[li][n][:3]) - np.array(conf[lj][n][:3]))
                         for n in shared), default=0.0)
                if d < best_dist:
                    best_dist = d
                    best_li, best_lj = li, lj

        # Merge best_lj into best_li (average position/B, sum occ)
        for name in set(conf[best_li]) | set(conf[best_lj]):
            if name in conf[best_li] and name in conf[best_lj]:
                xi, yi, zi, bi, oi, el, an = conf[best_li][name]
                xj, yj, zj, bj, oj = conf[best_lj][name][:5]
                conf[best_li][name] = [(xi+xj)/2, (yi+yj)/2, (zi+zj)/2,
                                       (bi+bj)/2, min(1.0, oi+oj), el, an]
            elif name in conf[best_lj]:
                conf[best_li][name] = conf[best_lj][name]
        del conf[best_lj]
        labels.remove(best_lj)

    # Rebuild by_name with reduced conformers (mc atoms are not in sc_names, unchanged)
    new_by_name = dict(by_name)
    for name in sc_names:
        atoms_list = []
        for l in labels:
            if name not in conf[l]:
                continue
            x, y, z, b, occ, el, an = conf[l][name]
            a = gemmi.Atom()
            a.name    = an
            a.element = el
            a.pos     = gemmi.Position(x, y, z)
            a.b_iso   = b
            a.occ     = occ
            a.altloc  = l
            atoms_list.append(a)
        if atoms_list:
            new_by_name[name] = atoms_list
    return new_by_name, labels


def step8_build_mixed_model(truth_full_pdb, tmpdir, rng):
    """Build a mixed single/multi-conformer model → starthere.pdb.

    For each residue:
      - Main chain (N, CA, C, O, OXT): always collapse all altlocs to mean position.
      - Side chain: if ANY atom across conformers is > ALTLOC_DIST_THRESHOLD Å from
        the centroid, keep the FULL side chain as alternates (with scrambled labels).
        Otherwise collapse the full side chain to mean positions.
      - Waters: always collapse.
    This mimics a realistic partial model where main-chain order is assumed and
    only genuinely disordered side chains are modelled as altlocs.
    """
    st_in  = gemmi.read_structure(str(truth_full_pdb))

    st_out = gemmi.Structure()
    st_out.cell         = st_in.cell
    st_out.spacegroup_hm = st_in.spacegroup_hm
    model_out = gemmi.Model('1')

    for chain_in in st_in[0]:
        if chain_in.name == 'F':   # flood waters — excluded from partial model
            continue
        chain_out = gemmi.Chain(chain_in.name)
        for res_in in chain_in:
            res_out = gemmi.Residue()
            res_out.name        = res_in.name
            res_out.seqid       = res_in.seqid
            res_out.entity_type = res_in.entity_type

            is_solvent = res_in.name in ('HOH', 'WAT', 'H2O')

            # Group atoms by name (one list per atom name, one entry per altloc)
            by_name = {}
            for atom in res_in:
                by_name.setdefault(atom.name, []).append(atom)

            if is_solvent:
                for atoms in by_name.values():
                    _add_collapsed_atom(res_out, atoms)
                chain_out.add_residue(res_out)
                continue

            mc_names = [n for n in by_name if n in MAINCHAIN_ATOMS]
            sc_names = [n for n in by_name if n not in MAINCHAIN_ATOMS]

            # Check if any side-chain atom exceeds threshold from its centroid
            keep_altloc = False
            for name in sc_names:
                atoms = by_name[name]
                if len(atoms) > 1:
                    pos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms])
                    centroid = pos.mean(axis=0)
                    if np.max(np.linalg.norm(pos - centroid, axis=1)) > ALTLOC_DIST_THRESHOLD:
                        keep_altloc = True
                        break

            # Main chain: always collapse
            for name in mc_names:
                _add_collapsed_atom(res_out, by_name[name])

            if not keep_altloc:
                # Side chain: collapse
                for name in sc_names:
                    _add_collapsed_atom(res_out, by_name[name])
            else:
                # Merge closest conformer pairs until ≤ 3 altlocs remain
                by_name, present = _reduce_conformers(by_name, sc_names, max_confs=3)

                # Scramble altloc labels per residue
                shuffled = list(present)
                rng.shuffle(shuffled)
                label_map = dict(zip(present, shuffled))

                for name in sc_names:
                    for atom in by_name.get(name, []):
                        orig = atom.altloc if (atom.altloc and atom.altloc != '\x00') else (present[0] if present else 'A')
                        a_out = gemmi.Atom()
                        a_out.name    = atom.name
                        a_out.element = atom.element
                        a_out.pos     = atom.pos
                        a_out.b_iso   = atom.b_iso
                        a_out.occ     = atom.occ
                        a_out.altloc  = label_map.get(orig, orig)
                        res_out.add_atom(a_out)

            chain_out.add_residue(res_out)
        model_out.add_chain(chain_out)

    st_out.add_model(model_out)
    st_out.write_pdb(str(tmpdir / 'starthere.pdb'))


def step9_refmac(tmpdir):
    """Run refmac5 (20 cycles) → refmacout.mtz and refmac.log."""
    keywords = (
        b'MAKE HYDR NO NEWLIGAND NOEXIT\n'
        b'NCYC 20\n'
        b'LABIN FP=F SIGFP=SIGF FREE=FreeR_flag\n'
        b'LABOUT FC=FC PHIC=PHIC FWT=FWT PHWT=PHWT '
        b'DELFWT=DELFWT PHDELWT=PHDELWT\n'
        b'MONI DIST 10\n'
        b'END\n'
    )
    result = subprocess.run(
        [str(REFMAC5),
         'XYZIN',  'starthere.pdb',
         'XYZOUT', 'refmacout.pdb',
         'HKLIN',  'refme.mtz',
         'HKLOUT', 'refmacout.mtz',
         'LIBOUT',  'refmac.lib'],
        input=keywords,
        cwd=str(tmpdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_text = result.stdout.decode(errors='replace')
    (tmpdir / 'refmac.log').write_text(log_text)
    if result.returncode != 0:
        raise RuntimeError(f'refmac5 failed:\n{log_text[-3000:]}')
    return log_text


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


# ══════════════════════════════════════════════════════════════════════════════
# Sample orchestration
# ══════════════════════════════════════════════════════════════════════════════

def generate_sample(sample_idx, outdir, n_residues=20, n_waters=10, n_flood=0,
                    flood_avoid_fullocc=True, flood_occ=None,
                    shift_scale=0.5, n_altlocs=2, missing_fraction=0.05,
                    never_collected_fraction=0.05, debug=False):
    """Run the full pipeline for one sample. Returns (sample_idx, ok, info).

    If debug=True, the entire tmpdir is copied to sample_dir/debug/ before
    cleanup, giving access to all intermediate PDB and log files.
    """
    t0 = time.time()
    outdir = Path(outdir).resolve()
    sample_dir = outdir / f'sample_{sample_idx:05d}'

    if sample_dir.exists() and (sample_dir / 'metadata.json').exists():
        return sample_idx, True, 'already done'

    rng = np.random.default_rng(seed=sample_idx)
    seq = list(rng.choice(AA_NAMES, size=n_residues, p=AA_PROBS))

    tmpdir = Path(tempfile.mkdtemp(prefix=f'prot_{sample_idx:05d}_'))
    try:
        # 1-2: Build backbone + side chains
        step1_build_backbone(seq, rng, tmpdir)
        step2_build_sidechains(seq, rng, tmpdir)

        # 3: Set up structure (cell, waters, centre, B factors)
        n_water_added, n_flood_added = step3_setup_structure(
            tmpdir, rng, n_waters=n_waters,
            n_flood=n_flood, flood_avoid_fullocc=flood_avoid_fullocc,
            flood_occ=flood_occ)

        # 4: First geometry minimisation
        minimized_pdb = step4_phenix_geommin('built.pdb', tmpdir, log_tag='_1st')

        # 4b: Self-refine B factors (20 refmac cycles against own SFs)
        #     Gives chemically correlated B factors before jigglepdb
        selfref_pdb = step4b_selfref_b_factors(minimized_pdb, tmpdir)

        # 5: jigglepdb using refined B factors → altloc A/B/...
        step5_jigglepdb_and_merge(selfref_pdb, tmpdir, rng,
                                  shift_scale=shift_scale, n_altlocs=n_altlocs,
                                  flood_occ=flood_occ)

        # 6: Second geometry minimisation → truth_full.pdb
        truth_full_pdb = step4_phenix_geommin('multiconf.pdb', tmpdir, log_tag='_2nd')
        shutil.copy2(truth_full_pdb, tmpdir / 'truth_full.pdb')

        # 7: sfcalc on truth_full → truth.mtz
        step6_sfcalc(tmpdir / 'truth_full.pdb', tmpdir / 'truth.mtz', tmpdir)

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

        # 9: Build mixed single/multi-conformer model → starthere.pdb
        step8_build_mixed_model(tmpdir / 'truth_full.pdb', tmpdir, rng)

        # 10: refmac refinement
        refmac_log = step9_refmac(tmpdir)

        # Parse final Rwork from refmac log
        # "Overall R factor = 0.0201" appears once per cycle; last is final cycle.
        rwork = None
        hits = re.findall(r'Overall R factor\s*=\s*(\d+\.\d+)', refmac_log)
        if hits:
            rwork = float(hits[-1])

        # 11: Convert to maps
        sample_dir.mkdir(parents=True, exist_ok=True)
        step10_convert_maps(tmpdir, sample_dir)

        # Copy PDB and log files
        shutil.copy2(tmpdir / 'truth_full.pdb',  sample_dir / 'truth_full.pdb')
        shutil.copy2(tmpdir / 'starthere.pdb',    sample_dir / 'partial.pdb')
        if (tmpdir / 'refmacout.pdb').exists():
            shutil.copy2(tmpdir / 'refmacout.pdb', sample_dir / 'refmacout.pdb')
        if (tmpdir / 'refmacout.mtz').exists():
            shutil.copy2(tmpdir / 'refmacout.mtz', sample_dir / 'refmacout.mtz')
        shutil.copy2(tmpdir / 'refmac.log',        sample_dir / 'refmac.log')
        for plog in tmpdir.glob('*.phenix.log'):
            shutil.copy2(plog, sample_dir / plog.name)

        # Debug: dump entire tmpdir
        if debug:
            debug_dir = sample_dir / 'debug'
            if debug_dir.exists():
                shutil.rmtree(str(debug_dir))
            shutil.copytree(str(tmpdir), str(debug_dir))

        # Metadata
        meta = dict(
            sample_idx=int(sample_idx),
            sequence=seq,
            n_residues=n_residues,
            n_waters_requested=n_waters,
            n_waters_added=n_water_added,
            n_flood_added=n_flood_added,
            rwork_final=rwork,
            missing_fraction=missing_fraction,
            n_reflections_missing=n_missing,
            never_collected_fraction=never_collected_fraction,
            n_reflections_never_collected=n_never,
            cell=list(CELL),
            dmin=DMIN,
            grid_shape=[60, 60, 60],
        )
        (sample_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))

        elapsed = time.time() - t0
        return sample_idx, True, f'ok in {elapsed:.1f}s  Rwork={rwork}'

    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / 'error.log').write_text(msg)
        if debug:
            debug_dir = sample_dir / 'debug'
            if debug_dir.exists():
                shutil.rmtree(str(debug_dir))
            shutil.copytree(str(tmpdir), str(debug_dir))
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
                       max_array=300):
    """Write and submit a SLURM array job script."""
    script = SCRIPT_DIR / '_slurm_protein.sh'
    python  = sys.executable
    me      = Path(__file__).resolve()

    script_text = f"""\
#!/bin/bash
#SBATCH --job-name=prot_data
#SBATCH --partition=debug
#SBATCH --array=0-{nsamples-1}%{max_array}
#SBATCH --output={outdir}/logs/%A_%a.log
#SBATCH --error={outdir}/logs/%A_%a.log
#SBATCH --time=02:00:00

mkdir -p {outdir}/logs
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
    --never-collected-fraction {never_collected_fraction}
"""
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
    parser.add_argument('--outdir',     default='./data_protein')
    parser.add_argument('--nsamples',   type=int, default=100)
    parser.add_argument('--nresidues',  type=int, default=20)
    parser.add_argument('--nwaters',    type=int, default=10)
    parser.add_argument('--workers',    type=int, default=1)
    parser.add_argument('--sample-id',  type=int, default=None,
                        help='Run a single sample (for SLURM array jobs)')
    parser.add_argument('--submit',     action='store_true',
                        help='Submit a SLURM array job instead of running locally')
    parser.add_argument('--max-array',  type=int, default=300,
                        help='SLURM --array concurrency limit')
    parser.add_argument('--n-flood',     type=int,   default=0,
                        help='Number of partial-occ flood waters to add (default 0)')
    parser.add_argument('--flood-occ',   type=float, default=None,
                        help='Fixed occupancy for flood waters (default: random 0.1-0.8)')
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
    parser.add_argument('--never-collected-fraction', type=float, default=0.05,
                        help='Fraction of reflections never measured (rows deleted, default 0.05)')
    parser.add_argument('--debug',      action='store_true',
                        help='Copy entire tmpdir to sample_dir/debug/ for inspection')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Single sample (SLURM array task) ──────────────────────────────────────
    if args.sample_id is not None:
        idx, ok, msg = generate_sample(
            args.sample_id, outdir,
            n_residues=args.nresidues,
            n_waters=args.nwaters,
            n_flood=args.n_flood,
            flood_avoid_fullocc=args.flood_avoid_fullocc,
            flood_occ=args.flood_occ,
            shift_scale=args.shift_scale,
            n_altlocs=args.n_altlocs,
            missing_fraction=args.missing_fraction,
            never_collected_fraction=args.never_collected_fraction,
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
            args.never_collected_fraction, args.max_array,
        )
        sys.exit(0 if ok else 1)

    # ── Local parallel run ─────────────────────────────────────────────────────
    sample_ids = list(range(args.nsamples))
    done = ok_count = err_count = 0

    if args.workers <= 1:
        for sid in sample_ids:
            idx, ok, msg = generate_sample(sid, outdir,
                                           args.nresidues, args.nwaters,
                                           args.n_flood, args.flood_avoid_fullocc,
                                           args.flood_occ, args.shift_scale,
                                           args.n_altlocs, args.missing_fraction,
                                           args.never_collected_fraction,
                                           debug=args.debug)
            done += 1
            status = 'OK' if ok else 'ERR'
            ok_count += ok; err_count += (not ok)
            log.info(f'[{done}/{args.nsamples}] {status} sample {idx:05d}: {msg}')
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(generate_sample, sid, str(outdir),
                            args.nresidues, args.nwaters, args.n_flood,
                            args.flood_avoid_fullocc, args.flood_occ,
                            args.shift_scale, args.n_altlocs, args.missing_fraction,
                            args.never_collected_fraction, args.debug): sid
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
