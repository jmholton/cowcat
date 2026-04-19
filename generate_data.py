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
import shutil
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


def step2_renumber(tmpdir):
    """Assign chain 'S' and ordinal residue numbers 1-N to truth_full.pdb."""
    st = gemmi.read_structure(str(tmpdir / 'truth_full.pdb'))
    for model in st:
        for chain in model:
            chain.name = 'S'
            for i, residue in enumerate(chain):
                residue.seqid = gemmi.SeqId(i + 1, ' ')
    st.write_pdb(str(tmpdir / 'truth_full.pdb'))


_ALTLOC_LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def step2b_insert_altconf(tmpdir, n_altconfs=2, altconf_rms=0.5, input_pdb='truth_full.pdb',
                          n_clusters=1):
    """
    Replace n_clusters randomly selected atoms with alt-conf clusters of n_altconfs atoms each.

    Truth model  — for each cluster: n_altconfs atoms at displaced positions around the original,
                   each with an independently randomised occupancy (Dirichlet proportions × random
                   total in [0.5, 1.5]; sum need not equal 1) and an independent log-normal B factor.
    Partial model — single centroid atom per cluster (average position), with occupancy equal to the
                   truth total scattered by a log-normal factor (sigma=0.2, i.e. ~20% RMS) and an
                   independent log-normal B factor.

    Displacement: per-axis sigma = altconf_rms / sqrt(3), so the RMS 3D distance of each
    alt conf from the centroid equals altconf_rms.

    Overwrites truth_full.pdb (each cluster adds n_altconfs - 1 extra atoms).
    Writes partial.pdb with the original atom count (each cluster → single centroid atom).

    Returns a list of dicts (one per cluster) with cluster info for metadata.
    """
    st = gemmi.read_structure(str(tmpdir / input_pdb))
    rng = np.random.default_rng()

    chain = st[0][0]
    n_atoms = len(chain)
    if n_atoms < n_clusters:
        raise RuntimeError(f'Not enough atoms ({n_atoms}) for {n_clusters} alt-conf clusters')

    # Select n_clusters distinct target atoms
    target_ris = sorted(int(i) for i in rng.choice(n_atoms, size=n_clusters, replace=False))

    # Collect cluster parameters before modifying anything
    sigma = altconf_rms / np.sqrt(3.0)
    cluster_info = []
    for target_ri in target_ris:
        target_res   = chain[target_ri]
        orig_atom    = target_res[0]
        center       = np.array([orig_atom.pos.x, orig_atom.pos.y, orig_atom.pos.z])
        displacements = rng.normal(0.0, sigma, size=(n_altconfs, 3))
        positions    = center + displacements
        b_factors    = np.clip(
            np.exp(rng.normal(BFAC_MU, BFAC_SIGMA, size=n_altconfs)), BFAC_MIN, BFAC_MAX
        ).tolist()
        total_occ    = float(rng.uniform(0.5, 1.5))
        props        = rng.dirichlet(np.ones(n_altconfs))
        occs         = (total_occ * props).tolist()
        centroid     = positions.mean(axis=0)
        partial_occ  = float(np.clip(total_occ * np.exp(rng.normal(0.0, 0.2)), 0.05, 3.0))
        partial_b    = float(np.clip(np.exp(rng.normal(BFAC_MU, BFAC_SIGMA)), BFAC_MIN, BFAC_MAX))
        cluster_info.append({
            'target_ri':      target_ri,
            'orig_seqid_num': int(str(target_res.seqid).strip()),
            'res_name':       target_res.name,
            'orig_name':      orig_atom.name,
            'orig_element_z': orig_atom.element.atomic_number,
            'positions':      positions,
            'b_factors':      b_factors,
            'total_occ':      total_occ,
            'occs':           occs,
            'centroid':       centroid,
            'partial_occ':    partial_occ,
            'partial_b':      partial_b,
        })

    # --- Build partial.pdb BEFORE modifying st (clone preserves original) ---
    st_partial  = st.clone()
    p_chain_obj = st_partial[0][0]
    centroid_chain = p_chain_obj.name   # same for all residues in randompdb.com output
    for ci in cluster_info:
        p_res_obj    = p_chain_obj[ci['target_ri']]
        ci['centroid_seqid'] = str(p_res_obj.seqid)
        p_atom       = p_res_obj[0]
        p_atom.pos   = gemmi.Position(float(ci['centroid'][0]),
                                      float(ci['centroid'][1]),
                                      float(ci['centroid'][2]))
        p_atom.occ   = ci['partial_occ']
        p_atom.b_iso = ci['partial_b']
        p_atom.altloc = '\0'
    st_partial.write_pdb(str(tmpdir / 'partial.pdb'))

    # --- Modify truth_full.pdb: remove originals (reverse order), append alt-conf residues ---
    # Each cluster becomes ONE residue with the original seqid, containing n_altconfs O atoms
    # (altlocs A/B/C/…) — standard PDB alt-conf convention.
    for ci in reversed(cluster_info):    # reverse so earlier indices stay valid
        del chain[ci['target_ri']]
    for ci in cluster_info:
        new_res       = gemmi.Residue()
        new_res.name  = ci['res_name']
        new_res.seqid = gemmi.SeqId(ci['orig_seqid_num'], ' ')
        for i in range(n_altconfs):
            a         = gemmi.Atom()
            a.name    = ci['orig_name']
            a.element = gemmi.Element(ci['orig_element_z'])
            a.pos     = gemmi.Position(ci['positions'][i, 0],
                                       ci['positions'][i, 1],
                                       ci['positions'][i, 2])
            a.b_iso   = ci['b_factors'][i]
            a.occ     = ci['occs'][i]
            a.altloc  = _ALTLOC_LABELS[i % len(_ALTLOC_LABELS)]
            new_res.add_atom(a)
        chain.add_residue(new_res)
    st.write_pdb(str(tmpdir / 'truth_full.pdb'))

    # Build and return metadata list (one dict per cluster)
    return [
        {
            'n_atoms':                n_atoms,
            'altconf_n':              n_altconfs,
            'altconf_rms':            round(float(altconf_rms), 4),
            'altconf_target_res_idx': ci['target_ri'],
            'altconf_centroid':       [round(float(v), 4) for v in ci['centroid']],
            'altconf_positions':      [[round(float(v), 4) for v in p] for p in ci['positions']],
            'altconf_b_factors':      [round(float(b), 2) for b in ci['b_factors']],
            'altconf_occs':           [round(float(o), 4) for o in ci['occs']],
            'altconf_total_occ':      round(ci['total_occ'], 4),
            'partial_atom_occ':       round(ci['partial_occ'], 4),
            'partial_atom_b':         round(ci['partial_b'], 2),
            'centroid_chain':         centroid_chain,
            'centroid_seqid':         ci['centroid_seqid'],
        }
        for ci in cluster_info
    ]


def step2c_add_hydrogens(tmpdir, pdb_name, rng=None):
    """Add 2 H atoms per O atom using the twirl quaternion protocol.

    Water geometry (from rebuild_water_hydrogens.com):
      H1 local offset: [ 0.957,  0.000,  0.000 ] Å
      H2 local offset: [-0.248,  0.924,  0.000 ] Å
    A random rotation matrix (via unit quaternion) is drawn per O atom.
    H atoms inherit the parent O atom's occupancy, B factor, and altloc so
    that refmac's residue-based occupancy groups cover O+H1+H2 together.
    """
    if rng is None:
        rng = np.random.default_rng()
    H1_local = np.array([ 0.957,  0.000, 0.0], dtype=np.float64)
    H2_local = np.array([-0.248,  0.924, 0.0], dtype=np.float64)

    st = gemmi.read_structure(str(tmpdir / pdb_name))
    for model in st:
        for chain in model:
            for residue in chain:
                # Cache all O atom properties before any add_atom() call.
                # A residue may contain multiple O atoms (alt-conf clusters), and
                # add_atom() can reallocate gemmi's internal atom vector, invalidating
                # C++ references held by Python atom objects.
                o_props = [
                    (np.array([a.pos.x, a.pos.y, a.pos.z]), a.occ, a.b_iso, a.altloc)
                    for a in residue if a.element == gemmi.Element('O')
                ]
                for o_pos, o_occ, o_b, o_altloc in o_props:
                    q = rng.standard_normal(4)
                    q /= np.linalg.norm(q)
                    w, x, y, z = q
                    R = np.array([
                        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)      ],
                        [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)      ],
                        [2*(x*z - w*y),       2*(y*z + w*x),      1 - 2*(x*x + y*y)  ],
                    ])
                    for h_name, h_local in (('H1', H1_local), ('H2', H2_local)):
                        h_pos = o_pos + R @ h_local
                        h = gemmi.Atom()
                        h.name    = h_name
                        h.element = gemmi.Element('H')
                        h.pos     = gemmi.Position(float(h_pos[0]), float(h_pos[1]), float(h_pos[2]))
                        h.occ     = o_occ
                        h.b_iso   = o_b
                        h.altloc  = o_altloc
                        residue.add_atom(h)
    st.write_pdb(str(tmpdir / pdb_name))


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
    Set nmissing atoms to random occupancies drawn from Uniform(0.8, 1.0); write partial.pdb.
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
    occs = rng.uniform(0.8, 1.0, size=n_partial).tolist()

    occ_map = {idx: occ for idx, occ in zip(selected, occs)}
    for res_idx, residue in enumerate(chain):
        if res_idx in occ_map:
            for atom in residue:
                atom.occ = float(occ_map[res_idx])

    st_full.write_pdb(str(tmpdir / 'partial.pdb'))
    return selected, [round(o, 4) for o in occs], n_atoms


def step4_bfac_shift(tmpdir, sigma, n_modify=None, atom_indices=None, input_pdb='truth_full.pdb'):
    """
    Add Gaussian noise N(0, sigma) Å² to B factors of selected heavy atoms; clip to
    [BFAC_MIN, BFAC_MAX]; write partial.pdb.
    atom_indices: residue-index list from step4_partial_occupancy — all atoms in each
                  named residue (O + H1 + H2) get the same B-factor shift applied.
    n_modify:     count for random selection when atom_indices is None (selects O atoms).
    Returns (n_modified, n_residues_total).
    """
    st = gemmi.read_structure(str(tmpdir / input_pdb))
    chain = st[0][0]
    rng = np.random.default_rng()
    if atom_indices is not None:
        # atom_indices are residue indices; shift all atoms in those residues together
        n_mod = len(atom_indices)
        for res_idx in atom_indices:
            shift = float(rng.normal(0.0, sigma))
            for atom in chain[res_idx]:
                atom.b_iso = float(np.clip(atom.b_iso + shift, BFAC_MIN, BFAC_MAX))
    else:
        n_residues = len(chain)
        if n_modify is None or int(n_modify) >= n_residues:
            target_idxs = range(n_residues)
            n_mod = n_residues
        else:
            n_mod = int(n_modify)
            target_idxs = rng.choice(n_residues, size=n_mod, replace=False)
        for res_idx in target_idxs:
            shift = float(rng.normal(0.0, sigma))
            for atom in chain[res_idx]:
                atom.b_iso = float(np.clip(atom.b_iso + shift, BFAC_MIN, BFAC_MAX))
    st.write_pdb(str(tmpdir / 'partial.pdb'))
    return n_mod, len(chain)


def step4_xyz_shift(tmpdir, sigma=0.01, n_modify=None, atom_indices=None, input_pdb='truth_full.pdb'):
    """
    Add Gaussian positional noise N(0, sigma) Å to x,y,z of selected O atoms; write partial.pdb.
    H atoms in the same residue are shifted by the same delta so water geometry is preserved.
    atom_indices: residue-index list from step4_partial_occupancy — whole residue is shifted.
    n_modify:     count for random selection when atom_indices is None (selects by residue).
    Returns (n_modified, n_residues_total).
    """
    st = gemmi.read_structure(str(tmpdir / input_pdb))
    chain = st[0][0]
    rng = np.random.default_rng()
    if atom_indices is not None:
        n_mod = len(atom_indices)
        for res_idx in atom_indices:
            dx, dy, dz = rng.normal(0.0, sigma, size=3)
            for atom in chain[res_idx]:
                atom.pos = gemmi.Position(atom.pos.x + dx, atom.pos.y + dy, atom.pos.z + dz)
    else:
        n_residues = len(chain)
        if n_modify is None or int(n_modify) >= n_residues:
            target_idxs = range(n_residues)
            n_mod = n_residues
        else:
            n_mod = int(n_modify)
            target_idxs = rng.choice(n_residues, size=n_mod, replace=False)
        for res_idx in target_idxs:
            dx, dy, dz = rng.normal(0.0, sigma, size=3)
            for atom in chain[res_idx]:
                atom.pos = gemmi.Position(atom.pos.x + dx, atom.pos.y + dy, atom.pos.z + dz)
    st.write_pdb(str(tmpdir / 'partial.pdb'))
    return n_mod, len(chain)


def step5_sfcalc_partial(tmpdir):
    """gemmi sfcalc on partial model → partial.mtz with FC/PHIC columns."""
    run(['gemmi', 'sfcalc', f'--dmin={DMIN}', '--to-mtz=partial.mtz', 'partial.pdb'],
        tmpdir)
    if not (tmpdir / 'partial.mtz').exists():
        raise RuntimeError('gemmi sfcalc did not produce partial.mtz')


def step5_build_refme_mtz(tmpdir):
    """Build refme.mtz from truth.mtz: F=|FC|, SIGF=0.02*|FC|, for refmac input."""
    mtz = gemmi.read_mtz_file(str(tmpdir / 'truth.mtz'))
    fc_lbl, _ = find_fc_phi_labels(mtz)
    H  = col_array(mtz, 'H').astype(np.int32)
    K  = col_array(mtz, 'K').astype(np.int32)
    L  = col_array(mtz, 'L').astype(np.int32)
    Fc = col_array(mtz, fc_lbl)

    mtz_out = gemmi.Mtz()
    mtz_out.cell       = mtz.cell
    mtz_out.spacegroup = mtz.spacegroup
    ds0 = mtz_out.add_dataset('HKL_base'); ds0.wavelength = 0.0
    ds1 = mtz_out.add_dataset('data');     ds1.wavelength = 1.0
    for lbl in ('H', 'K', 'L'):
        mtz_out.add_column(lbl, 'H', dataset_id=0)
    mtz_out.add_column('F',    'F', dataset_id=1)
    mtz_out.add_column('SIGF', 'Q', dataset_id=1)
    mtz_out.set_data(np.column_stack([
        H, K, L, Fc, np.maximum(0.02 * Fc, 1e-6),
    ]).astype(np.float32))
    mtz_out.write_to_file(str(tmpdir / 'refme.mtz'))


def _refmac_occ_single(tmpdir, centroids, ncyc=5,
                       pdb_in='partial.pdb', pdb_out='refmacout.pdb',
                       mtz_out='refmacout.mtz', occ_only=False):
    """
    Run one refmac5 occupancy-refinement pass.

    centroids : list of (chain_id, resnum_str) tuples.
    occ_only  : if True, add 'REFI OCCC' to suppress positional/B refinement
                (used for batch sub-structures that lack a complete Fc).
    Returns (rwork, {resnum_str: refined_occ}).
    """
    lines = [
        'LABIN FP=F SIGFP=SIGF',
        f'NCYC {ncyc}',
        'MAKE HYDR Y',
        'MAKE HOUT Y',
        'VDWREST 0',
        'WEIGHT MATRIX 50',
    ]
    if occ_only:
        lines.append('REFI OCCC')
    lines.append('occupancy refine')
    for gid, (chain_id, resnum_str) in enumerate(centroids, start=1):
        chain_kw = f' chain {chain_id}' if chain_id.strip() else ''
        lines.append(f'occupancy group id {gid}{chain_kw} residue {resnum_str}')
        lines.append(f'occupancy group alts incomplete {gid}')
    lines.append('END')
    keywords = '\n'.join(lines) + '\n'

    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)
    result = subprocess.run(
        ['refmac5',
         'HKLIN',  'refme.mtz',
         'XYZIN',  pdb_in,
         'HKLOUT', mtz_out,
         'XYZOUT', pdb_out],
        input=keywords.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(tmpdir),
    )
    log_text = result.stdout.decode(errors='replace')
    # Append refmac log to tmpdir/refmac.log (accumulates across staged batches)
    with open(tmpdir / 'refmac.log', 'a') as _lf:
        _lf.write(f'=== refmac5 {pdb_in} -> {pdb_out} ===\n')
        _lf.write(log_text)
        _lf.write('\n')

    if result.returncode != 0:
        err_text = result.stderr.decode(errors='replace')
        raise RuntimeError(
            f'refmac5 failed:\n{log_text[-3000:]}\n--- stderr ---\n{err_text[-1000:]}'
        )

    pdb_out_path = tmpdir / pdb_out
    if not pdb_out_path.exists():
        raise RuntimeError(f'refmac5 did not produce {pdb_out}')

    # Parse Rwork from REMARK 3
    rwork = None
    with open(pdb_out_path) as f:
        for line in f:
            if 'R VALUE' in line and 'WORKING SET' in line:
                m = re.search(r'(\d+\.\d+)\s*$', line.strip())
                if m:
                    rwork = float(m.group(1))

    # Read refined occupancies — take first (O) atom per centroid residue
    resnum_set = {resnum_str for _, resnum_str in centroids}
    refined_occs = {}
    st_ref = gemmi.read_structure(str(pdb_out_path))
    for chain in st_ref[0]:
        for res in chain:
            sid = str(res.seqid)
            if sid in resnum_set:
                for atom in res:          # first atom = O
                    refined_occs[sid] = float(atom.occ)
                    break

    return rwork, refined_occs


def step5_refmac_occ(tmpdir, centroids, ncyc=5, max_atoms=4999):
    """
    Run refmac5 occupancy refinement, staging automatically when partial.pdb
    has more than max_atoms atoms (refmac5 hard limit ~5000).

    For staged runs each batch PDB contains only the batch's centroid residues
    (with their H atoms); occ-only refinement is used since the sub-structure
    Fc is incomplete.  Batch outputs are merged into refmacout.pdb so that
    step5_merge_refmac_centroid can work unchanged.

    Returns (rwork, {resnum_str: refined_occ}).
    """
    st = gemmi.read_structure(str(tmpdir / 'partial.pdb'))
    chain0 = st[0][0]
    n_residues  = len(chain0)
    n_total_atoms = sum(len(res) for res in chain0)

    if n_total_atoms <= max_atoms:
        return _refmac_occ_single(tmpdir, centroids, ncyc)

    # ── Staged mode ──────────────────────────────────────────────────────────
    atoms_per_res = max(1, n_total_atoms / max(1, n_residues))
    batch_n_res   = max(1, int(max_atoms / atoms_per_res))
    log.info('Staged refmac: %d atoms in partial.pdb, batch_size=%d residues',
             n_total_atoms, batch_n_res)

    # Build seqid → residue map
    res_by_seqid = {str(res.seqid): res for res in chain0}

    all_refined  = {}
    all_rwork    = []
    n_batches    = (len(centroids) + batch_n_res - 1) // batch_n_res

    for k in range(n_batches):
        batch = centroids[k * batch_n_res : (k + 1) * batch_n_res]

        # Write batch sub-structure
        st_b = gemmi.Structure()
        st_b.cell          = st.cell
        st_b.spacegroup_hm = st.spacegroup_hm
        m_b  = gemmi.Model('1')
        ch_b = gemmi.Chain(chain0.name)
        for _, seqid in batch:
            if seqid in res_by_seqid:
                ch_b.add_residue(res_by_seqid[seqid].clone())
        m_b.add_chain(ch_b)
        st_b.add_model(m_b)
        batch_pdb_name = f'partial_batch_{k}.pdb'
        st_b.write_pdb(str(tmpdir / batch_pdb_name))

        pdb_out_b = f'refmacout_batch_{k}.pdb'
        mtz_out_b = f'refmacout_batch_{k}.mtz'
        rw, refined = _refmac_occ_single(
            tmpdir, batch, ncyc,
            pdb_in=batch_pdb_name, pdb_out=pdb_out_b, mtz_out=mtz_out_b,
            occ_only=True,
        )
        all_refined.update(refined)
        if rw is not None:
            all_rwork.append(rw)
        log.info('  batch %d/%d: %d centroids  Rwork=%.4f',
                 k + 1, n_batches, len(batch), rw or 0.0)

    # Merge batch refinements → refmacout.pdb (required by step5_merge_refmac_centroid)
    # Apply refined occ from each batch back to the full partial structure, then save.
    for res in chain0:
        sid = str(res.seqid)
        if sid in all_refined:
            for atom in res:
                atom.occ = all_refined[sid]
    st.write_pdb(str(tmpdir / 'refmacout.pdb'))

    rwork_mean = float(np.mean(all_rwork)) if all_rwork else None
    return rwork_mean, all_refined


def step5_merge_refmac_centroid(tmpdir, resnum_strs):
    """
    Build partial_refined.pdb: non-disordered atoms keep truth values from
    the pre-refmac partial.pdb; each centroid atom (identified by residue seqid
    in resnum_strs) gets x/y/z/occ/B from refmacout.pdb.
    Writes partial_refined.pdb to tmpdir.
    """
    st_pre = gemmi.read_structure(str(tmpdir / 'partial.pdb'))
    st_ref = gemmi.read_structure(str(tmpdir / 'refmacout.pdb'))

    # Extract refined parameters for each centroid from refmacout.pdb,
    # keyed by (resnum_str, atom_name) so H atoms are updated to their own
    # refined positions, not the O atom's position.
    resnum_set = set(resnum_strs)
    ref_params = {}   # resnum_str -> {atom_name: (pos, occ, b)}
    for chain in st_ref[0]:
        for res in chain:
            sid = str(res.seqid)
            if sid in resnum_set:
                ref_params[sid] = {}
                for atom in res:
                    ref_params[sid][atom.name] = (
                        gemmi.Position(atom.pos.x, atom.pos.y, atom.pos.z),
                        float(atom.occ),
                        float(atom.b_iso),
                    )
    for sid in resnum_set:
        if sid not in ref_params:
            raise RuntimeError(f'Centroid residue {sid} not found in refmacout.pdb')

    # Apply refined values to centroid atoms in pre-refmac partial.pdb,
    # matching atom by name so each atom (O, H1, H2) gets its own parameters.
    for chain in st_pre[0]:
        for res in chain:
            sid = str(res.seqid)
            if sid in ref_params:
                atom_map = ref_params[sid]
                for atom in res:
                    if atom.name in atom_map:
                        pos, occ, b = atom_map[atom.name]
                        atom.pos   = pos
                        atom.occ   = occ
                        atom.b_iso = b

    st_pre.write_pdb(str(tmpdir / 'partial_refined.pdb'))


def step6_build_maps_refmac(tmpdir, outdir):
    """
    Build maps from refmacout.mtz (FWT/PHWT, DELFWT/PHDELWT, FC_ALL/PHIC_ALL)
    and truth.mtz.  Computes cross-Patterson and diff-Patterson from fofc/fc grids.
    Returns grid shape tuple.
    """
    mtz_ref = gemmi.read_mtz_file(str(tmpdir / 'refmacout.mtz'))
    mtz_t   = gemmi.read_mtz_file(str(tmpdir / 'truth.mtz'))

    fc_lbl_t, phi_lbl_t = find_fc_phi_labels(mtz_t)
    fc_lbl_r, phi_lbl_r = find_fc_phi_labels(mtz_ref)

    grid_2fofc = mtz_ref.transform_f_phi_to_map('FWT',    'PHWT',    sample_rate=SAMPLE_RATE)
    grid_fofc  = mtz_ref.transform_f_phi_to_map('DELFWT', 'PHDELWT', sample_rate=SAMPLE_RATE)
    grid_fc    = mtz_ref.transform_f_phi_to_map(fc_lbl_r, phi_lbl_r, sample_rate=SAMPLE_RATE)
    grid_truth = mtz_t.transform_f_phi_to_map(fc_lbl_t,  phi_lbl_t, sample_rate=SAMPLE_RATE)

    arr_fofc = np.array(grid_fofc, copy=False)
    arr_fc   = np.array(grid_fc,   copy=False)
    F_fofc   = np.fft.rfftn(arr_fofc)
    F_fc     = np.fft.rfftn(arr_fc)

    crossp = np.fft.irfftn(
        F_fofc * np.conj(F_fc), s=arr_fofc.shape,
    ).real.astype(np.float32)
    diffp = np.fft.irfftn(
        np.abs(F_fofc) ** 2, s=arr_fofc.shape,
    ).real.astype(np.float32)
    np.save(str(outdir / 'crossp.npy'), crossp)
    np.save(str(outdir / 'diffp.npy'),  diffp)

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

    # R-factor: Σ|Fc_partial - Fobs_scaled| / Σ|Fc_partial|
    R_factor = float(np.sum(np.abs(Fc - Fobs_scaled)) / np.sum(np.abs(Fc)))

    return Fobs_scaled, scale_k, scale_B, R_factor


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

def generate_sample(sample_idx, outdir_root, natoms=None, cell=None, nmissing=None,
                    partial_occ=False, xyz_shift=None, xyz_natoms=None,
                    bfac_shift=None, bfac_natoms=None,
                    n_altconfs=1, altconf_rms=0.5, n_clusters=1,
                    all_clusters=False, no_refmac=False):
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
        step2_renumber(tmpdir)

        altconf_metas = None
        if n_altconfs >= 2:
            # Resolve all_clusters: set n_clusters = actual atom count
            actual_n_clusters = n_clusters
            if all_clusters:
                st_tmp = gemmi.read_structure(str(tmpdir / 'truth_full.pdb'))
                actual_n_clusters = len(st_tmp[0][0])
            log.info('[%05d] step 2b/6 inserting %d×%d-conf clusters (rms=%.3f Å) ...',
                     sample_idx, actual_n_clusters, n_altconfs, altconf_rms)
            altconf_metas = step2b_insert_altconf(tmpdir, n_altconfs=n_altconfs,
                                                  altconf_rms=altconf_rms,
                                                  n_clusters=actual_n_clusters)

        log.info('[%05d] step 2c/6 adding H atoms (twirl quaternion) ...', sample_idx)
        rng_h = np.random.default_rng()
        step2c_add_hydrogens(tmpdir, 'truth_full.pdb', rng_h)
        if (tmpdir / 'partial.pdb').exists():
            step2c_add_hydrogens(tmpdir, 'partial.pdb', rng_h)

        log.info('[%05d] step 3/6 sfcalc full model ...', sample_idx)
        step3_sfcalc_full(tmpdir)

        # ── Step 4: modify partial model (chain of optional steps) ────────────
        # Each sub-step reads from partial.pdb (if written) or truth_full.pdb,
        # and overwrites partial.pdb.  Apply in order: occ → bfac → xyz.
        # Alt-conf mode: partial.pdb already written by step2b; skip all step4 modes.
        partial_written = altconf_metas is not None
        n_atoms  = None
        selected = None
        occs     = None
        removed  = None
        n_partial = None
        bfac_n_modified = None
        xyz_n_modified  = None

        if partial_occ:
            log.info('[%05d] step 4/6 setting partial occupancies (%s atoms) ...',
                     sample_idx, nmissing if nmissing is not None else 'default')
            selected, occs, n_atoms = step4_partial_occupancy(tmpdir, nmissing=nmissing)
            n_partial = n_atoms
            log.info('[%05d] atoms full=%d  partial-occ=%d', sample_idx, n_atoms, len(selected))
            partial_written = True

        if bfac_shift is not None:
            in_pdb = 'partial.pdb' if partial_written else 'truth_full.pdb'
            # When partial_occ is active, apply bfac to the SAME atoms (single-atom residues assumed)
            link_indices = selected if partial_occ else None
            n_bfac_desc = str(len(link_indices)) if link_indices is not None else (str(bfac_natoms) if bfac_natoms is not None else 'all')
            log.info('[%05d] step 4b/6 B-shifting %s atoms (sigma=%.2f Å²) ...',
                     sample_idx, n_bfac_desc, bfac_shift)
            bfac_n_modified, n_at = step4_bfac_shift(
                tmpdir, sigma=bfac_shift, n_modify=bfac_natoms,
                atom_indices=link_indices, input_pdb=in_pdb)
            if n_atoms is None:
                n_atoms = n_at
                n_partial = n_at
            log.info('[%05d] bfac_shift: modified %d/%d atoms', sample_idx, bfac_n_modified, n_at)
            partial_written = True

        if xyz_shift is not None:
            in_pdb = 'partial.pdb' if partial_written else 'truth_full.pdb'
            # When partial_occ is active, apply xyz to the SAME atoms (single-atom residues assumed)
            link_indices = selected if partial_occ else None
            n_xyz_desc = str(len(link_indices)) if link_indices is not None else (str(xyz_natoms) if xyz_natoms is not None else 'all')
            log.info('[%05d] step 4c/6 xyz-shifting %s atoms (sigma=%.4f Å) ...',
                     sample_idx, n_xyz_desc, xyz_shift)
            xyz_n_modified, n_at = step4_xyz_shift(
                tmpdir, sigma=xyz_shift, n_modify=xyz_natoms,
                atom_indices=link_indices, input_pdb=in_pdb)
            if n_atoms is None:
                n_atoms = n_at
                n_partial = n_at
            partial_written = True

        if not partial_written:
            log.info('[%05d] step 4/6 deleting atoms ...', sample_idx)
            removed, n_atoms = step4_delete_atoms(tmpdir, nmissing=nmissing)
            n_partial = n_atoms - len(removed)
            log.info('[%05d] atoms full=%d partial=%d', sample_idx, n_atoms, n_partial)

        # Initialise variables so both paths leave them defined for the metadata block
        scale_k = scale_B = R_factor = Fobs_scaled = None
        refmac_rwork = refined_occs = None

        if altconf_metas is not None and not no_refmac:
            # ── Alt-conf path: refmac occupancy refinement → maps from refmacout.mtz ──
            log.info('[%05d] step 5/6 building refme.mtz ...', sample_idx)
            step5_build_refme_mtz(tmpdir)

            centroids = [(m['centroid_chain'], m['centroid_seqid']) for m in altconf_metas]
            log.info('[%05d] step 5b/6 refmac occ refinement (%d centroids: %s) ...',
                     sample_idx, len(centroids),
                     ', '.join(f'res={s}' for _, s in centroids))
            refmac_rwork, refined_occs = step5_refmac_occ(tmpdir, centroids=centroids)
            log.info('[%05d] Rwork=%.4f  occ_refined=%s',
                     sample_idx, refmac_rwork or 0.0,
                     {k: round(v, 4) for k, v in refined_occs.items()})

            step5_merge_refmac_centroid(tmpdir, [m['centroid_seqid'] for m in altconf_metas])

            log.info('[%05d] step 6/6 building maps from refmacout.mtz ...', sample_idx)
            grid_shape = step6_build_maps_refmac(tmpdir, outdir)

            shutil.copy2(tmpdir / 'truth_full.pdb',      outdir / 'truth_full.pdb')
            shutil.copy2(tmpdir / 'partial_refined.pdb', outdir / 'partial.pdb')
            shutil.copy2(tmpdir / 'refmacout.pdb',       outdir / 'refmacout.pdb')
            if (tmpdir / 'refmac.log').exists():
                shutil.copy2(tmpdir / 'refmac.log', outdir / 'refmac.log')

        elif altconf_metas is not None and no_refmac:
            # ── Alt-conf + no-refmac: sfcalc on centroids + scaleit → maps ──────────
            log.info('[%05d] step 5/6 sfcalc partial (centroid) model ...', sample_idx)
            step5_sfcalc_partial(tmpdir)

            log.info('[%05d] step 5b/6 scaling Ftrue to Fcalc with scaleit ...', sample_idx)
            Fobs_scaled, scale_k, scale_B, R_factor = step5b_scale_ftrue(tmpdir)
            log.info('[%05d] scale_k=%.6f  scale_B=%.4f  R=%.4f',
                     sample_idx, scale_k, scale_B, R_factor)

            log.info('[%05d] step 6/6 building maps ...', sample_idx)
            grid_shape = step6_build_maps(tmpdir, outdir, Fo_scaled=Fobs_scaled)

            shutil.copy2(tmpdir / 'truth_full.pdb', outdir / 'truth_full.pdb')
            shutil.copy2(tmpdir / 'partial.pdb',    outdir / 'partial.pdb')

        else:
            # ── Standard path: sfcalc + scaleit + manual map coefficients ──────────
            log.info('[%05d] step 5/6 sfcalc partial model ...', sample_idx)
            step5_sfcalc_partial(tmpdir)

            log.info('[%05d] step 5b/6 scaling Ftrue to Fcalc with scaleit ...', sample_idx)
            Fobs_scaled, scale_k, scale_B, R_factor = step5b_scale_ftrue(tmpdir)
            log.info('[%05d] scale_k=%.6f  scale_B=%.4f  R=%.4f',
                     sample_idx, scale_k, scale_B, R_factor)

            log.info('[%05d] step 6/6 building maps ...', sample_idx)
            grid_shape = step6_build_maps(tmpdir, outdir, Fo_scaled=Fobs_scaled)

            shutil.copy2(tmpdir / 'truth_full.pdb', outdir / 'truth_full.pdb')
            shutil.copy2(tmpdir / 'partial.pdb',    outdir / 'partial.pdb')

    cell_list = [float(v) for v in (cell or ('40','40','40','90','90','90'))]

    # Unified metadata — record whichever modifications were applied
    meta = {
        'n_atoms':    n_atoms,
        'cell':       cell_list,
        'dmin':       DMIN,
        'grid_shape': list(grid_shape),
    }
    if scale_k is not None:
        meta['scale_k']  = round(float(scale_k), 6)
        meta['scale_B']  = round(float(scale_B), 4)
        meta['R_factor'] = round(R_factor, 4)
    if refmac_rwork is not None:
        meta['rwork_refmac'] = round(refmac_rwork, 4)
    if refined_occs is not None:
        meta['occ_refined'] = {k: round(float(v), 4) for k, v in refined_occs.items()}
    if partial_occ:
        meta['partial_occ_mode']         = True
        meta['partial_occ_natoms']       = len(selected)
        meta['partial_occ_atom_indices'] = selected
        meta['partial_occ_values']       = occs
    if bfac_shift is not None:
        meta['bfac_shift_sigma']   = round(float(bfac_shift), 4)
        meta['bfac_shift_natoms']  = bfac_natoms   # None = all atoms
        meta['bfac_shift_n_modified'] = bfac_n_modified
    if xyz_shift is not None:
        meta['xyz_sigma']      = round(float(xyz_shift), 6)
        meta['xyz_natoms']     = xyz_natoms        # None = all atoms
        meta['xyz_n_modified'] = xyz_n_modified
    if removed is not None:
        meta['n_atoms_partial']      = n_partial
        meta['deletion_fraction']    = round(len(removed) / n_atoms, 4)
        meta['removed_atom_indices'] = removed
    if altconf_metas is not None:
        meta['n_clusters']   = len(altconf_metas)
        meta['altconf_n']    = altconf_metas[0]['altconf_n']
        meta['altconf_rms']  = altconf_metas[0]['altconf_rms']
        meta['all_clusters'] = all_clusters
        meta['no_refmac']    = no_refmac
        if not no_refmac:
            meta['clusters'] = altconf_metas   # omit from large all-cluster runs (too big)

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
                  natoms, nmissing, cell_size, verbose, partial_occ=False,
                  natoms_range=None, modified_range=None,
                  xyz_shift=None, xyz_natoms=None,
                  bfac_shift=None, bfac_natoms=None,
                  n_altconfs=1, altconf_rms=0.5, n_clusters=1,
                  all_clusters=False, no_refmac=False):
    """Submit a sbatch array job; return the SLURM job-id string."""
    array_spec = _make_array_spec(pending)
    if max_array and max_array > 0:
        array_spec += f'%{max_array}'

    extra = []
    if natoms          is not None: extra += [f'--natoms {natoms}']
    if natoms_range    is not None: extra += [f'--natoms-range {natoms_range[0]} {natoms_range[1]}']
    if nmissing        is not None: extra += [f'--modified {nmissing}']
    if modified_range  is not None: extra += [f'--modified-range {modified_range[0]} {modified_range[1]}']
    if cell_size != 40.0:           extra += [f'--cell-size {cell_size}']
    if verbose:                     extra += ['--verbose']
    if partial_occ:                 extra += ['--partial-occ']
    if xyz_shift       is not None: extra += [f'--xyz-shift {xyz_shift}']
    if xyz_natoms      is not None: extra += [f'--xyz-natoms {xyz_natoms}']
    if bfac_shift      is not None: extra += [f'--bfac-shift {bfac_shift}']
    if bfac_natoms     is not None: extra += [f'--bfac-natoms {bfac_natoms}']
    if n_altconfs      >= 2:        extra += [f'--n-altconfs {n_altconfs}']
    if n_altconfs      >= 2:        extra += [f'--altconf-rms {altconf_rms}']
    if n_clusters      >  1:        extra += [f'--n-clusters {n_clusters}']
    if all_clusters:               extra += ['--all-clusters']
    if no_refmac:                  extra += ['--no-refmac']
    extra_str = ' '.join(extra)

    logs_dir = os.path.join(outdir_abs, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    lines = [
        '#!/bin/bash',
        '#SBATCH --job-name=cnn_gen',
        '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=1',
        '#SBATCH --export=ALL',
        f'#SBATCH --output={logs_dir}/%A_%a.log',
        f'#SBATCH --error={logs_dir}/%A_%a.log',
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
    parser.add_argument('--natoms',         type=int, default=None,
                        help='Fix number of atoms via -N (default: use -Vm)')
    parser.add_argument('--natoms-range',   type=int, nargs=2, metavar=('MIN', 'MAX'),
                        help='Draw natoms uniformly from [MIN, MAX] per sample')
    parser.add_argument('--modified',       type=int, default=None,
                        help='Fix number of deleted/modified atoms (default: DELETE_FRAC * n_atoms)')
    parser.add_argument('--modified-range', type=int, nargs=2, metavar=('MIN', 'MAX'),
                        help='Draw n_modified uniformly from [MIN, MAX] per sample')
    parser.add_argument('--cell-size',   type=float, default=40.0,
                        help='Cubic cell edge in Å (default: 40)')
    parser.add_argument('--verbose',     action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--partial-occ', action='store_true',
                        help='Partial occupancy mode: give --modified atoms random occ in [0.8,1.0] '
                             'instead of deleting them (--modified controls how many)')
    parser.add_argument('--xyz-shift',   type=float, default=None, metavar='SIGMA',
                        help='Add Gaussian noise N(0, SIGMA) Å to atom xyz coordinates')
    parser.add_argument('--xyz-natoms',  type=int, default=None, metavar='N',
                        help='Number of atoms to apply --xyz-shift to (default: all atoms)')
    parser.add_argument('--bfac-shift',  type=float, default=None, metavar='SIGMA',
                        help='Add Gaussian noise N(0, SIGMA) Å² to B factors of --bfac-natoms atoms')
    parser.add_argument('--bfac-natoms', type=int, default=None, metavar='N',
                        help='Number of atoms to apply --bfac-shift to (default: all atoms)')
    parser.add_argument('--n-altconfs', type=int, default=1, metavar='N',
                        help='Number of alt confs in the truth cluster (default: 1 = disabled). '
                             'When >=2: truth has N atoms with independent random occupancies '
                             '(Dirichlet props × total~U(0.5,1.5)); partial has one atom at '
                             'centroid with occ = total_occ * lognormal(sigma=0.2) and random B.')
    parser.add_argument('--altconf-rms', type=float, default=0.5, metavar='SIGMA',
                        help='RMS 3D displacement of each alt conf from the cluster centroid (Å, default: 0.5)')
    parser.add_argument('--n-clusters', type=int, default=1, metavar='N',
                        help='Number of atoms to split into alt-conf clusters (default: 1). '
                             'Requires --n-altconfs >= 2.')
    parser.add_argument('--all-clusters', action='store_true',
                        help='Split every atom into an alt-conf cluster (overrides --n-clusters). '
                             'Requires --n-altconfs >= 2.')
    parser.add_argument('--no-refmac', action='store_true',
                        help='Use sfcalc+scaleit instead of refmac for the alt-conf path. '
                             'Required (or strongly advised) when --all-clusters is set.')
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
        # Resolve natoms and nmissing (fixed values or drawn from ranges)
        rng = np.random.default_rng()
        natoms  = args.natoms
        nmissing = args.modified
        if args.natoms_range:
            lo, hi = args.natoms_range
            natoms = int(rng.integers(lo, hi + 1))
        if args.modified_range:
            lo, hi = args.modified_range
            nmissing = int(rng.integers(lo, hi + 1))
        elif args.natoms_range:
            # default: 1/4 of drawn natoms
            nmissing = natoms // 4
        try:
            generate_sample(i, outdir_root=outdir, natoms=natoms,
                            cell=cell, nmissing=nmissing,
                            partial_occ=args.partial_occ,
                            xyz_shift=args.xyz_shift, xyz_natoms=args.xyz_natoms,
                            bfac_shift=args.bfac_shift, bfac_natoms=args.bfac_natoms,
                            n_altconfs=args.n_altconfs, altconf_rms=args.altconf_rms,
                            n_clusters=args.n_clusters,
                            all_clusters=args.all_clusters, no_refmac=args.no_refmac)
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
        args.natoms, args.modified, args.cell_size, args.verbose, args.partial_occ,
        natoms_range=args.natoms_range, modified_range=args.modified_range,
        xyz_shift=args.xyz_shift, xyz_natoms=args.xyz_natoms,
        bfac_shift=args.bfac_shift, bfac_natoms=args.bfac_natoms,
        n_altconfs=args.n_altconfs, altconf_rms=args.altconf_rms,
        n_clusters=args.n_clusters,
        all_clusters=args.all_clusters, no_refmac=args.no_refmac,
    )
    log.info('Submitted SLURM array job %s  (%d tasks)', job_id, len(pending))

    _wait_for_job(job_id, len(pending))

    # NFS async: the compute nodes' kernel writeback may still be in-flight
    # after the job exits squeue.  Poll until all metadata.json files appear
    # (or 60 s elapses), flushing the login node's attribute cache each round.
    deadline = time.time() + 60
    while True:
        for i in pending:
            try:
                os.stat(outdir / f'sample_{i:05d}' / 'metadata.json')
            except OSError:
                pass
        ok = sum(1 for i in pending
                 if (outdir / f'sample_{i:05d}' / 'metadata.json').exists())
        if ok == len(pending) or time.time() > deadline:
            break
        time.sleep(3)

    errors = len(pending) - ok
    log.info('Done. ok=%d  skipped=%d  errors=%d', ok, skipped, errors)


if __name__ == '__main__':
    main()
