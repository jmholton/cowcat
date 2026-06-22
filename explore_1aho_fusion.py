#!/usr/bin/env python3
"""
explore_1aho_fusion.py — Compare conformer-reduction strategies for the 1AHO
48-conformer model.

Each strategy reduces the 48-chain truth model (refmacout_minRfree.pdb) to a
simpler partial model.  Refmac is run with NCYC 0 (initial R only) and NCYC 50
(short refinement) against fixed Fobs = F(truth protein) + Fpart from
refme_minRfree.mtz.

Run with:
    ccp4-python explore_1aho_fusion.py [--pdb 1aho/refmacout_minRfree.pdb]
                                       [--mtz 1aho/refme_minRfree.mtz]
                                       [--outdir explore_fusion_out]
                                       [--damp]   # add damp for NCYC5 run
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

import numpy as np
import gemmi

SCRIPT_DIR = Path(__file__).resolve().parent
import shutil as _shutil
REFMAC5    = _shutil.which('refmac5') or '/programs/ccp4-8.0/bin/refmac5'
del _shutil
DMIN       = 0.965

MAINCHAIN_ATOMS = frozenset({'N', 'CA', 'C', 'O', 'OXT', 'H', 'HA', 'HA2', 'HA3'})

# Atoms beyond Cβ (distal side chain) for bouquet detection
DISTAL_SC = frozenset({
    'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2',
    'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2',
    'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'ND1', 'ND2',
    'OG', 'OG1', 'OD1', 'OD2', 'OE1', 'OE2', 'OH',
    'SD', 'SG',
})

# All altloc labels available for output. Written via direct PDB formatter (not
# gemmi.write_pdb) so case is preserved exactly as given here.
ALL_ALT_LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

# Maximum altloc groups passed to refmac for bouquet/disulfide residues.
BOUQUET_MAX_ALT = 48
# Minimum altloc groups for disulfide pairs (correlated motion, SG-SG constraint
# makes individual CYS look ordered even when the pair samples multiple positions).
DISULF_MIN_NCONF = 16


def adaptive_k(spread, min_k=8, max_k=48, scale=0.5):
    """k ~ spread / scale; floor=8, ceiling=48 (full ensemble)."""
    return min(max_k, max(min_k, int(round(spread / scale))))


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd, cwd, input_bytes=None, check=True):
    r = subprocess.run(
        [str(c) for c in cmd], input=input_bytes, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if check and r.returncode != 0:
        raise RuntimeError(
            f'{cmd[0]} failed (rc={r.returncode}):\n'
            f'{r.stdout.decode(errors="replace")[-2000:]}'
        )
    return r.stdout.decode(errors='replace')


def load_density_map(mtz_path, f_col='FWT', phi_col='PHWT'):
    """Load a CCP4 density map from an MTZ file via gemmi FFT."""
    mtz  = gemmi.read_mtz_file(str(mtz_path))
    grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=3.0)
    grid.normalize()   # z-score so interpolated values are comparable across runs
    print(f'  Density map: {grid.nu}×{grid.nv}×{grid.nw} from {f_col}/{phi_col}')
    return grid


def density_score(valid_chains, reskey, anames, conf_data, grid):
    """Mean density at atom positions for each of the n_conf conformers.

    Returns array of shape (n_conf,), weakest→strongest when argsorted ascending.
    """
    scores = []
    for cn in valid_chains:
        atoms  = conf_data[cn].get(reskey, {}).get('atoms', {})
        vals   = [grid.interpolate_value(atoms[an].pos)
                  for an in anames if an in atoms]
        scores.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(scores)


def _maximin_select(centroids, k):
    """Greedy maximin: pick k rows of centroids maximising minimum pairwise distance.

    Starts with the point farthest from the global centroid (most outlying),
    then iteratively selects the point with the largest minimum distance to all
    already-selected points.  Returns a list of k row indices.
    """
    n = len(centroids)
    k = min(k, n)
    if k == 1:
        return [0]
    mean = centroids.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(centroids - mean, axis=1)))
    selected = [start]
    min_dists = np.linalg.norm(centroids - centroids[start], axis=1).copy()
    for _ in range(k - 1):
        nxt = int(np.argmax(min_dists))
        selected.append(nxt)
        d = np.linalg.norm(centroids - centroids[nxt], axis=1)
        np.minimum(min_dists, d, out=min_dists)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Fobs MTZ construction
# ─────────────────────────────────────────────────────────────────────────────

def _water_sf_dict(pdb_path, tmpdir):
    """Compute SFs for coordinate waters only; return {(h,k,l): complex_F}."""
    st = gemmi.read_structure(str(pdb_path))
    st_w = gemmi.Structure()
    st_w.cell = st.cell
    st_w.spacegroup_hm = st.spacegroup_hm
    mdl = gemmi.Model('1')
    ch  = gemmi.Chain('W')
    for chain in st[0]:
        for res in chain:
            if res.name in ('HOH', 'WAT', 'H2O'):
                ch.add_residue(res.clone())
    mdl.add_chain(ch)
    st_w.add_model(mdl)
    water_pdb = tmpdir / '_waters.pdb'
    st_w.write_pdb(str(water_pdb))

    water_mtz = tmpdir / '_water_sf.mtz'
    run(['gemmi', 'sfcalc', f'--dmin={DMIN}', f'--to-mtz={water_mtz}', str(water_pdb)],
        cwd=tmpdir)
    m   = gemmi.read_mtz_file(str(water_mtz))
    h_w = np.array(m.column_with_label('H'),    dtype=np.int32)
    k_w = np.array(m.column_with_label('K'),    dtype=np.int32)
    l_w = np.array(m.column_with_label('L'),    dtype=np.int32)
    fc_w = np.array(m.column_with_label('FC'),  dtype=np.float64)
    ph_w = np.array(m.column_with_label('PHIC'), dtype=np.float64)
    F_w  = fc_w * np.exp(1j * np.radians(ph_w))
    n_wat = sum(len(list(res)) for res in ch)
    print(f'  Water SFs: {len(h_w)} reflections from {n_wat} water atoms')
    return {(int(h_w[i]), int(k_w[i]), int(l_w[i])): F_w[i] for i in range(len(h_w))}


def build_fobs_mtz(pdb_path, refme_path, tmpdir):
    """Return path to fobs.mtz: FP=|F_prot+F_solv|, SIGFP, FreeR_flag, Fpart, PHIpart.

    Fpart = bulk-solvent Fpart from refme_mtz + coordinate-water SFs from pdb_path.
    Coordinate waters are excluded from starthere.pdb (they appear as missing
    density for the CNN to recover) and enter Fc only via the Fpart fixed term.
    Fobs is computed from a protein-only PDB (no HOH) + Fpart, so waters are
    counted exactly once on both sides.
    """
    # Write protein-only PDB (no waters) for Fobs sfcalc
    st = gemmi.read_structure(str(pdb_path))
    st_prot = gemmi.Structure()
    st_prot.cell = st.cell
    st_prot.spacegroup_hm = st.spacegroup_hm
    mdl = gemmi.Model('1')
    for chain in st[0]:
        ch_out = gemmi.Chain(chain.name)
        for res in chain:
            if res.name not in ('HOH', 'WAT', 'H2O'):
                ch_out.add_residue(res.clone())
        if len(list(ch_out)) > 0:
            mdl.add_chain(ch_out)
    st_prot.add_model(mdl)
    prot_only_pdb = tmpdir / '_prot_only.pdb'
    st_prot.write_pdb(str(prot_only_pdb))

    prot_mtz = tmpdir / '_prot_sf.mtz'
    run(['gemmi', 'sfcalc', f'--dmin={DMIN}', f'--to-mtz={prot_mtz}', str(prot_only_pdb)],
        cwd=tmpdir)

    water_dict = _water_sf_dict(pdb_path, tmpdir)

    prot  = gemmi.read_mtz_file(str(prot_mtz))
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

    fp_out = np.zeros(len(h_p), dtype=np.float32)
    sp_out = np.zeros(len(h_p), dtype=np.float32)
    fr_out = np.zeros(len(h_p), dtype=np.float32)
    fa_out = np.zeros(len(h_p), dtype=np.float32)
    pa_out = np.zeros(len(h_p), dtype=np.float32)

    for i in range(len(h_p)):
        hkl = (int(h_p[i]), int(k_p[i]), int(l_p[i]))
        fpa, ppa, fra = refme_dict.get(hkl, (0.0, 0.0, 0.0))
        F_solv = fpa * np.exp(1j * np.radians(ppa)) + water_dict.get(hkl, 0.0)
        F_total = F_prot[i] + F_solv
        amp = float(np.abs(F_total))
        fp_out[i] = amp
        sp_out[i] = max(0.01, 0.02 * amp)
        fr_out[i] = float(fra)
        fa_out[i] = float(np.abs(F_solv))
        pa_out[i] = float(np.degrees(np.angle(F_solv))) if abs(F_solv) > 0 else 0.0

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
    data = np.column_stack([h_p, k_p, l_p, fp_out, sp_out, fr_out, fa_out, pa_out])
    out.set_data(data.astype(np.float32))

    out_mtz = tmpdir / 'fobs.mtz'
    out.write_to_file(str(out_mtz))
    print(f'  Fobs MTZ: {len(h_p)} reflections, dmin={DMIN} Å, '
          f'mean FP={fp_out[fp_out > 0].mean():.1f}')
    return out_mtz


# ─────────────────────────────────────────────────────────────────────────────
# 48-conformer parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_conformers(pdb_path):
    """Return (st, chain_names, conf_data).

    conf_data[chain_name][(seqnum, icode)]['atoms'][atom_name] = gemmi.Atom
    conf_data[chain_name][(seqnum, icode)]['resname'] = str
    chain_names: ordered list of 48 chain IDs (template = chain_names[0]).
    """
    st = gemmi.read_structure(str(pdb_path))
    # Exclude water-only chains (e.g. chain 'z' in 1AHO which holds all HOH).
    chain_names = [ch.name for ch in st[0]
                   if any(res.name not in ('HOH', 'WAT', 'H2O') for res in ch)]
    conf_data = {}
    for chain in st[0]:
        residues = {}
        for res in chain:
            key = (res.seqid.num, res.seqid.icode.strip())
            residues[key] = {
                'resname': res.name,
                'seqid':   res.seqid,
                'etype':   res.entity_type,
                'atoms':   {atom.name: atom for atom in res},
            }
        conf_data[chain.name] = residues
    return st, chain_names, conf_data


# ─────────────────────────────────────────────────────────────────────────────
# Bouquet detection
# ─────────────────────────────────────────────────────────────────────────────

def bouquet_max_spread(reskey, conf_data, chain_names):
    """Max distance from centroid for distal SC atoms across all 48 conformers."""
    pts = []
    for cn in chain_names:
        rd = conf_data[cn].get(reskey)
        if rd is None:
            continue
        for aname, atom in rd['atoms'].items():
            if aname in DISTAL_SC:
                pts.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if len(pts) < 2:
        return 0.0
    pts = np.array(pts)
    center = pts.mean(axis=0)
    return float(np.max(np.linalg.norm(pts - center, axis=1)))


def heavy_atom_max_dev(reskey, conf_data, chain_names):
    """Per-atom centroid deviation (heavy atoms only), max across all atoms in residue."""
    all_anames = set()
    for cn in chain_names:
        rd = conf_data[cn].get(reskey)
        if rd:
            all_anames.update(rd['atoms'].keys())
    max_dev = 0.0
    for aname in all_anames:
        if aname.startswith('H'):
            continue
        pts = []
        for cn in chain_names:
            rd = conf_data[cn].get(reskey)
            if rd and aname in rd['atoms']:
                a = rd['atoms'][aname]
                pts.append([a.pos.x, a.pos.y, a.pos.z])
        if len(pts) < 2:
            continue
        pts = np.array(pts)
        dev = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
        if dev > max_dev:
            max_dev = dev
    return max_dev


def dev_to_nconf(dev):
    """Map per-residue heavy-atom max deviation (Å) to number of altloc groups."""
    if dev < 1.0:
        return 3
    elif dev < 1.5:
        return 8
    elif dev < 3.0:
        return 16
    elif dev < 4.0:
        return 32
    else:
        return 48


def ca_max_spread(reskey, conf_data, chain_names):
    """Max distance of CA from its centroid across all conformers.

    Used for MC bouquet detection independently of SC disorder.
    """
    pts = []
    for cn in chain_names:
        rd = conf_data[cn].get(reskey)
        if rd is None:
            continue
        ca = rd['atoms'].get('CA')
        if ca:
            pts.append([ca.pos.x, ca.pos.y, ca.pos.z])
    if len(pts) < 2:
        return 0.0
    pts = np.array(pts)
    center = pts.mean(axis=0)
    return float(np.max(np.linalg.norm(pts - center, axis=1)))


def detect_disulfides(conf_data, chain_names, threshold=2.5):
    """Return {reskey: partner_reskey} for disulfide-bonded CYS pairs.

    Uses mean SG-SG distance across all conformers.
    """
    ref = chain_names[0]
    cys_keys = [rk for rk, rd in conf_data[ref].items() if rd['resname'] == 'CYS']
    disulf = {}
    for i, rk1 in enumerate(cys_keys):
        for rk2 in cys_keys[i + 1:]:
            dists = []
            for cn in chain_names:
                sg1 = conf_data[cn].get(rk1, {}).get('atoms', {}).get('SG')
                sg2 = conf_data[cn].get(rk2, {}).get('atoms', {}).get('SG')
                if sg1 and sg2:
                    d = np.linalg.norm([sg1.pos.x - sg2.pos.x,
                                        sg1.pos.y - sg2.pos.y,
                                        sg1.pos.z - sg2.pos.z])
                    dists.append(d)
            if dists and np.mean(dists) < threshold:
                disulf[rk1] = rk2
                disulf[rk2] = rk1
                print(f'    Disulfide: {rk1} — {rk2}  mean SG-SG={np.mean(dists):.2f} Å')
    return disulf


# ─────────────────────────────────────────────────────────────────────────────
# Atom-building helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_pdb_direct(st, path):
    """Write a gemmi Structure to PDB without going through gemmi's writer.

    Gemmi uppercases altloc chars on write, which limits us to 26 labels.
    Writing ATOM records directly preserves case and allows a–z, 0–9 too.
    """
    CRYST1 = (f"CRYST1{st.cell.a:9.3f}{st.cell.b:9.3f}{st.cell.c:9.3f}"
              f"{st.cell.alpha:7.2f}{st.cell.beta:7.2f}{st.cell.gamma:7.2f}"
              f" {st.spacegroup_hm:<11s}1\n")
    lines = [CRYST1]
    serial = 1
    for model in st:
        for chain in model:
            for res in chain:
                is_het = res.entity_type == gemmi.EntityType.NonPolymer or \
                         res.name in ('HOH', 'WAT', 'H2O')
                rec = 'HETATM' if is_het else 'ATOM  '
                seqnum = res.seqid.num
                icode  = res.seqid.icode if res.seqid.icode != ' ' else ' '
                for atom in res:
                    altloc = atom.altloc if atom.altloc != '\x00' else ' '
                    elem   = atom.element.name.upper()
                    aname  = atom.name
                    # PDB atom name column (cols 13-16): 4 chars
                    if len(elem) == 2 or len(aname) == 4:
                        name4 = f'{aname:<4s}'
                    else:
                        name4 = f' {aname:<3s}'
                    line = (f'{rec}{serial:5d} {name4}{altloc}'
                            f'{res.name:<3s} {chain.name:1s}'
                            f'{seqnum:4d}{icode:1s}   '
                            f'{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}'
                            f'{atom.occ:6.2f}{atom.b_iso:6.2f}'
                            f'          {elem:>2s}\n')
                    lines.append(line)
                    serial += 1
    lines.append('END\n')
    Path(path).write_text(''.join(lines))


def make_atom(template, x, y, z, occ, b_iso, altloc):
    a = gemmi.Atom()
    a.name    = template.name
    a.element = template.element
    a.pos     = gemmi.Position(x, y, z)
    a.occ     = float(occ)
    a.b_iso   = float(b_iso)
    a.altloc  = altloc   # '\x00' for no altloc, else 'A'/'B'/...
    return a


# ─────────────────────────────────────────────────────────────────────────────
# Reduced model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_reduced_pdb(st_orig, chain_names, conf_data, strategy,
                      density_grid,
                      bouquet_threshold=1.5,
                      mc_bouq_threshold=0.5,
                      out_pdb=None, tmpdir=None,
                      rng=None, max_k=None):
    """Build starthere.pdb using the given fusion strategy.

    Returns path to the written PDB.

    strategy encoding:
      mc_mode : 'cluster_k1' (centroid) or 'cluster_k2'
      sc_mode_ordered  : 'cluster_k1', 'cluster_k2', 'cluster_k3'
      sc_mode_bouquet  : 'cluster_k1', 'cluster_k2', 'cluster_k4', 'cluster_k8'
      bouquet_threshold: float (Å), spread > this → bouquet residue
    """
    mc_ord, mc_bouq, sc_ord, sc_bouq = strategy   # unpack 4-tuple

    ref_chain_name = chain_names[0]
    ref_chain_data = conf_data[ref_chain_name]

    # Pre-compute per-residue heavy-atom max deviation for nconf scheduling
    res_devs = {reskey: heavy_atom_max_dev(reskey, conf_data, chain_names)
                for reskey in ref_chain_data}

    # ── Pre-compute shared SC groups for disulfide pairs ─────────────────────
    disulfides   = detect_disulfides(conf_data, chain_names)
    disulf_groups = {}  # reskey → list of lists of chain names (one list per group)
    processed_pairs = set()
    for rk1, rk2 in disulfides.items():
        pair = tuple(sorted([str(rk1), str(rk2)]))
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)
        common = [cn for cn in chain_names
                  if rk1 in conf_data[cn] and rk2 in conf_data[cn]]
        if not common:
            continue
        # Randomly select half of common conformers (same half for both CYS)
        if rng is not None and len(common) >= 2:
            n_half = max(1, len(common) // 2)
            chosen = sorted(rng.choice(len(common), n_half, replace=False).tolist())
            common = [common[i] for i in chosen]
        k = min(len(common), max(DISULF_MIN_NCONF, dev_to_nconf(max(res_devs.get(rk1, 0.0), res_devs.get(rk2, 0.0)))))
        # Combined SG density score for ordering
        scores = []
        for cn in common:
            sg1 = conf_data[cn].get(rk1, {}).get('atoms', {}).get('SG')
            sg2 = conf_data[cn].get(rk2, {}).get('atoms', {}).get('SG')
            v1  = float(density_grid.interpolate_value(sg1.pos)) if sg1 else 0.0
            v2  = float(density_grid.interpolate_value(sg2.pos)) if sg2 else 0.0
            scores.append((v1 + v2) / 2.0)
        order  = np.argsort(scores)
        groups = [arr for arr in np.array_split(order, k) if len(arr) > 0]
        for rk in (rk1, rk2):
            disulf_groups[rk] = [[common[i] for i in grp] for grp in groups]

    st_out = gemmi.Structure()
    st_out.cell         = st_orig.cell
    st_out.spacegroup_hm = st_orig.spacegroup_hm
    model_out = gemmi.Model('1')
    chain_out = gemmi.Chain('A')

    n_bouquet = 0
    n_altloc_res = 0

    for reskey, ref_rd in ref_chain_data.items():
        resname = ref_rd['resname']
        is_solvent = resname in ('HOH', 'WAT', 'H2O')

        # Gather valid conformers (chains that have this residue)
        valid_chains = [cn for cn in chain_names if reskey in conf_data[cn]]
        n_conf = len(valid_chains)

        # Conformer occupancies: mean occ of all atoms in that conformer/residue
        conf_occs = []
        for cn in valid_chains:
            atoms = conf_data[cn][reskey]['atoms']
            occ_vals = [a.occ for a in atoms.values() if a.occ > 0]
            conf_occs.append(np.mean(occ_vals) if occ_vals else 1.0 / n_conf)
        conf_occs = np.array(conf_occs, dtype=np.float64)
        if conf_occs.sum() > 0:
            conf_occs /= conf_occs.sum()

        # All atom names present in the reference conformer
        ref_atoms = ref_rd['atoms']
        all_anames = list(ref_atoms.keys())
        mc_anames  = [n for n in all_anames if n in MAINCHAIN_ATOMS]
        sc_anames  = [n for n in all_anames if n not in MAINCHAIN_ATOMS]

        res_out = gemmi.Residue()
        res_out.name        = resname
        res_out.seqid       = ref_rd['seqid']
        res_out.entity_type = ref_rd['etype']

        def weighted_atom(aname, indices, weights):
            """Mean-position atom from subset of conformers, no altloc."""
            template = ref_atoms.get(aname)
            if template is None:
                return None
            xyzs, bs, ws = [], [], []
            for idx in indices:
                cn = valid_chains[idx]
                a  = conf_data[cn][reskey]['atoms'].get(aname)
                if a is None:
                    continue
                xyzs.append([a.pos.x, a.pos.y, a.pos.z])
                bs.append(a.b_iso)
                ws.append(weights[idx] if idx < len(weights) else 1.0)
            if not xyzs:
                return None
            ws = np.array(ws)
            ws /= ws.sum()
            xyz = np.average(np.array(xyzs), weights=ws, axis=0)
            b   = float(np.average(np.array(bs), weights=ws))
            occ_sum = conf_occs[indices].sum() if len(conf_occs) > 0 else 1.0
            return make_atom(template, *xyz, occ_sum, b, '\x00')

        def split_by_density(anames, k):
            """Order conformers weakest→strongest by mean density at anames,
            split into k equal groups, return list of member-index arrays."""
            scores = density_score(valid_chains, reskey, anames, conf_data, density_grid)
            order  = np.argsort(scores)          # weakest first
            return [arr for arr in np.array_split(order, k) if len(arr) > 0]

        def split_by_maximin(anames, k):
            """Select k representative conformers by maximin; assign Voronoi occupancy.

            Returns (singleton_groups, voronoi_occs):
              singleton_groups[i] = np.array([rep_i])   — actual conformer index
              voronoi_occs[i]     = sum of conf_occs for all conformers nearest to rep_i
            Using the actual conformer's atomic positions (no averaging blur).
            """
            n = len(valid_chains)
            k_actual = min(k, n)
            centroids = np.zeros((n, 3))
            for i, cn in enumerate(valid_chains):
                pts = []
                for aname in anames:
                    a = conf_data[cn][reskey]['atoms'].get(aname)
                    if a:
                        pts.append([a.pos.x, a.pos.y, a.pos.z])
                if pts:
                    centroids[i] = np.mean(pts, axis=0)
            selected = _maximin_select(centroids, k_actual)
            # Voronoi partition: assign each conformer to its nearest representative
            sel_pts = centroids[selected]
            dists   = np.linalg.norm(
                centroids[:, None, :] - sel_pts[None, :, :], axis=2)  # (n, k)
            assign  = np.argmin(dists, axis=1)                         # (n,)
            voronoi_occs = [
                float(conf_occs[assign == gi].sum()) for gi in range(k_actual)
            ]
            singleton_groups = [np.array([rep]) for rep in selected]
            return singleton_groups, voronoi_occs

        def atoms_from_groups(anames, groups, multi_altloc, labels, group_occs=None):
            """Write averaged atoms for each group into res_out.

            labels: character sequence to use for altloc labels (MC_ALT_LABELS
            or SC_ALT_LABELS).  Every atom name in anames is written for every
            group (using global mean as fallback) so all altlocs are complete.
            group_occs: if provided, override per-group occupancy (e.g. Voronoi sums
            when groups are singletons from maximin selection).
            """
            for gi, members in enumerate(groups):
                lbl = labels[gi] if multi_altloc else '\x00'
                ws  = conf_occs[members]
                ws  = ws / ws.sum() if ws.sum() > 0 else np.ones(len(ws)) / len(ws)
                occ = group_occs[gi] if group_occs is not None else float(conf_occs[members].sum())
                for aname in anames:
                    template = ref_atoms.get(aname)
                    if template is None:
                        continue
                    xyzs, bs, w2 = [], [], []
                    for idx, w in zip(members, ws):
                        cn = valid_chains[idx]
                        a  = conf_data[cn][reskey]['atoms'].get(aname)
                        if a:
                            xyzs.append([a.pos.x, a.pos.y, a.pos.z])
                            bs.append(a.b_iso)
                            w2.append(w)
                    if not xyzs:
                        # Completeness fallback: use global mean across all conformers
                        for cn in valid_chains:
                            a = conf_data[cn][reskey]['atoms'].get(aname)
                            if a:
                                xyzs.append([a.pos.x, a.pos.y, a.pos.z])
                                bs.append(a.b_iso)
                                w2.append(1.0)
                    if not xyzs:
                        continue   # atom absent from all 48 conformers
                    w2  = np.array(w2) / sum(w2)
                    xyz = np.average(np.array(xyzs), weights=w2, axis=0)
                    b   = float(np.average(np.array(bs), weights=w2))
                    eff_occ = occ if multi_altloc else 1.0
                    res_out.add_atom(make_atom(template, *xyz, eff_occ, b, lbl))

        # ── Solvent: excluded from model (waters go into Fpart) ──────────────
        if is_solvent:
            continue

        all_idx = np.arange(n_conf)

        spread  = bouquet_max_spread(reskey, conf_data, chain_names)
        is_bouq = (spread > bouquet_threshold)
        if is_bouq:
            n_bouquet += 1

        # ── Bouquet and disulfide residues: all conformers, MC+SC together ──
        # Write up to 26 altlocs covering the full 48-conformer ensemble so that
        # refmac sees both main-chain and side-chain disorder from the same
        # conformers (no independent MC/SC clustering that creates ghost combos).
        if is_bouq or reskey in disulf_groups:
            if reskey in disulf_groups:
                # Shared ordering by combined SG density of both CYS in pair
                cn_to_idx = {cn: i for i, cn in enumerate(valid_chains)}
                all_groups = [
                    np.array([cn_to_idx[cn] for cn in grp if cn in cn_to_idx], dtype=int)
                    for grp in disulf_groups[reskey]
                ]
                all_groups = [g for g in all_groups if len(g) > 0]
            else:
                k = min(n_conf, dev_to_nconf(res_devs.get(reskey, 0.0)))
                if max_k is not None:
                    k = min(k, max_k)
                all_groups, bouq_voccs = split_by_maximin(all_anames, k)
            multi = len(all_groups) > 1
            if multi:
                n_altloc_res += 1
            bouq_occs = bouq_voccs if (reskey not in disulf_groups and multi) else None
            atoms_from_groups(all_anames, all_groups, multi_altloc=multi,
                              labels=ALL_ALT_LABELS, group_occs=bouq_occs)
            chain_out.add_residue(res_out)
            continue

        # ── Regular residues: unified MC+SC processing (same altloc labels) ──
        mode = sc_ord
        k = int(mode.split('_k')[1]) if '_k' in mode else 1
        if max_k is not None:
            k = min(k, max_k)
        k = min(k, n_conf)
        if k <= 1:
            atoms_from_groups(all_anames, [all_idx], multi_altloc=False,
                              labels=ALL_ALT_LABELS)
            chain_out.add_residue(res_out)
            continue
        all_groups, all_voccs = split_by_maximin(all_anames, k)
        multi = len(all_groups) > 1
        if multi:
            n_altloc_res += 1
        atoms_from_groups(all_anames, all_groups, multi_altloc=multi,
                          labels=ALL_ALT_LABELS, group_occs=all_voccs)

        chain_out.add_residue(res_out)

    model_out.add_chain(chain_out)
    st_out.add_model(model_out)

    if out_pdb is None:
        out_pdb = tmpdir / 'starthere.pdb'
    _write_pdb_direct(st_out, out_pdb)
    print(f'    bouquet residues: {n_bouquet}  altloc residues: {n_altloc_res}')
    return out_pdb, n_bouquet, n_altloc_res


# ─────────────────────────────────────────────────────────────────────────────
# Whole-chain maximin condensation (k=N chain selection via combine_pdbs_runme.com)
# ─────────────────────────────────────────────────────────────────────────────

COMBINE_PDBS = SCRIPT_DIR / 'combine_pdbs_runme.com'


def select_chains_maximin(chain_names, conf_data, k):
    """Select k representative chains by maximin over per-chain CA centroid.

    Returns (selected_chain_names, voronoi_occs) where each occ = fraction
    of the 48 conformers assigned to that representative.
    """
    n = len(chain_names)
    k = min(k, n)
    centroids = np.zeros((n, 3))
    for i, cn in enumerate(chain_names):
        pts = [[a.pos.x, a.pos.y, a.pos.z]
               for rd in conf_data[cn].values()
               for a in [rd['atoms'].get('CA')] if a]
        if pts:
            centroids[i] = np.mean(pts, axis=0)
    selected_idx = _maximin_select(centroids, k)
    sel_pts = centroids[selected_idx]
    assign = np.argmin(
        np.linalg.norm(centroids[:, None] - sel_pts[None], axis=2), axis=1)
    w = 1.0 / n
    return ([chain_names[i] for i in selected_idx],
            [float((assign == gi).sum()) * w for gi in range(k)])


def _write_chain_pdb(chain_name, occ, conf_data, ref_chain_data,
                     cell, spacegroup_hm, outpath):
    """Write one conformer chain as a PDB with chain_id='A' and altloc=chain_name."""
    lines = [f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
             f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f}"
             f" {spacegroup_hm:<11s}1\n"]
    serial = 1
    rd_this = conf_data[chain_name]
    for reskey, ref_rd in ref_chain_data.items():
        if reskey not in rd_this:
            continue
        rd = rd_this[reskey]
        resname = rd['resname']
        is_het = resname in ('HOH', 'WAT', 'H2O')
        rec = 'HETATM' if is_het else 'ATOM  '
        seqnum = ref_rd['seqid'].num
        icode = ref_rd['seqid'].icode if ref_rd['seqid'].icode != ' ' else ' '
        for aname, ref_atom in ref_rd['atoms'].items():
            atom = rd['atoms'].get(aname, ref_atom)
            elem = atom.element.name.upper()
            name4 = f' {aname:<3s}' if len(elem) == 1 and len(aname) < 4 else f'{aname:<4s}'
            lines.append(
                f'{rec}{serial:5d} {name4}{chain_name}'
                f'{resname:<3s} A'
                f'{seqnum:4d}{icode:1s}   '
                f'{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}'
                f'{occ:6.2f}{atom.b_iso:6.2f}'
                f'          {elem:>2s}\n'
            )
            serial += 1
    lines.append('END\n')
    Path(outpath).write_text(''.join(lines))


def write_full_conf_pdb(conf_data, chain_names, st_for_hoh,
                        flood_pos, flood_occ, cell, spacegroup_hm, out_pdb,
                        flood_biso=20.0):
    """Write truth PDB: all protein conformers from conf_data + HOH + flood waters.

    Each conformer chain gets occ = 1/n_chains.  HOH-only chains are copied
    verbatim from st_for_hoh.  Flood waters are written as chain W.
    flood_occ and flood_biso may each be a scalar or a 1-D array (one per water).
    flood_occ may be negative.
    """
    n = len(chain_names)
    occ_each = 1.0 / n
    cryst = (f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
             f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f}"
             f" {spacegroup_hm:<11s}1\n")
    lines = [cryst]
    serial = 1
    ref_chain_data = conf_data[chain_names[0]]

    for cn in chain_names:
        rd_this = conf_data[cn]
        for reskey, ref_rd in ref_chain_data.items():
            if reskey not in rd_this:
                continue
            rd = rd_this[reskey]
            resname = rd['resname']
            is_het = resname in ('HOH', 'WAT', 'H2O')
            rec = 'HETATM' if is_het else 'ATOM  '
            seqnum = ref_rd['seqid'].num
            icode = ref_rd['seqid'].icode if ref_rd['seqid'].icode != ' ' else ' '
            for aname, ref_atom in ref_rd['atoms'].items():
                atom = rd['atoms'].get(aname, ref_atom)
                elem = atom.element.name.upper()
                name4 = f' {aname:<3s}' if len(elem) == 1 and len(aname) < 4 else f'{aname:<4s}'
                lines.append(
                    f'{rec}{serial:5d} {name4}{cn}'
                    f'{resname:<3s} {cn:1s}'
                    f'{seqnum:4d}{icode:1s}   '
                    f'{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}'
                    f'{occ_each:6.2f}{atom.b_iso:6.2f}'
                    f'          {elem:>2s}\n'
                )
                serial += 1

    for chain in st_for_hoh[0]:
        if not all(res.name in ('HOH', 'WAT', 'H2O') for res in chain):
            continue
        for res in chain:
            seqnum = res.seqid.num
            icode = res.seqid.icode if res.seqid.icode != ' ' else ' '
            for atom in res:
                elem = atom.element.name.upper()
                name4 = f' {atom.name:<3s}' if len(elem) == 1 and len(atom.name) < 4 else f'{atom.name:<4s}'
                lines.append(
                    f'HETATM{serial:5d} {name4} '
                    f'HOH {chain.name:1s}'
                    f'{seqnum:4d}{icode:1s}   '
                    f'{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}'
                    f'{atom.occ:6.2f}{atom.b_iso:6.2f}'
                    f'          {elem:>2s}\n'
                )
                serial += 1

    import numpy as _np
    _occ_arr  = (flood_occ  if hasattr(flood_occ,  '__len__')
                 else _np.full(len(flood_pos), flood_occ))
    _biso_arr = (flood_biso if hasattr(flood_biso, '__len__')
                 else _np.full(len(flood_pos), flood_biso))
    for i, (x, y, z) in enumerate(flood_pos):
        lines.append(
            f'HETATM{serial:5d}  O   HOH W{i+1:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}'
            f'{_occ_arr[i]:6.2f}{_biso_arr[i]:6.2f}'
            f'           O\n'
        )
        serial += 1

    lines.append('END\n')
    Path(out_pdb).write_text(''.join(lines))


def build_starthere_pdb(chain_names, conf_data, st_orig, k, ref_pdb, out_pdb, workdir):
    """Select k chains by maximin; concatenate chain PDBs directly.

    Returns n_chains selected.
    """
    selected, occs = select_chains_maximin(chain_names, conf_data, k)
    ref_chain_data = conf_data[chain_names[0]]
    all_lines = []
    cryst_written = False
    for cn, occ in zip(selected, occs):
        p = Path(workdir) / f'_chain_{cn}.pdb'
        _write_chain_pdb(cn, occ, conf_data, ref_chain_data,
                         st_orig.cell, st_orig.spacegroup_hm, p)
        for line in p.read_text().splitlines(keepends=True):
            if line.startswith('CRYST1'):
                if not cryst_written:
                    all_lines.append(line)
                    cryst_written = True
            elif not line.startswith('END'):
                all_lines.append(line)
    all_lines.append('END\n')
    Path(out_pdb).write_text(''.join(all_lines))
    print(f'    k={k}: selected {selected}')
    return len(selected)


def _select_residue_maximin(chain_names, conf_data, reskey, k):
    """Select k chains by maximin on a single residue's heavy-atom centroid."""
    coords, valid = [], []
    for cn in chain_names:
        rd = conf_data[cn].get(reskey)
        if rd is None:
            continue
        pts = [[a.pos.x, a.pos.y, a.pos.z]
               for a in rd['atoms'].values()
               if a.element.name not in ('H', 'D')]
        if pts:
            coords.append(np.mean(pts, axis=0))
            valid.append(cn)
    if not valid:
        return chain_names[:k]
    k = min(k, len(valid))
    idx = _maximin_select(np.array(coords), k)
    return [valid[i] for i in idx]


_SLOT_CHAINS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')


def build_varconf_pdb(chain_names, conf_data, st_orig, out_pdb, workdir, max_k=None,
                      water_pdb=None, limit_o=True):
    """Build variable-conformer-count PDB using chain_id=altloc format.

    Per-residue maximin selects the best k_r source chains independently for
    each residue.  Output uses canonical slot chain_ids (A, B, C, …) so each
    output chain is a complete peptide with consistent covalent connectivity.
    max_k caps the per-residue conformer count (None = no cap).
    water_pdb: path to PDB containing HETATM water records to append verbatim.

    Returns (slot_chains_used, res_k) where res_k maps reskey → k.
    """
    ref_chain = chain_names[0]
    ref_chain_data = conf_data[ref_chain]
    residue_keys = list(ref_chain_data.keys())

    res_devs = {rk: heavy_atom_max_dev(rk, conf_data, chain_names)
                for rk in residue_keys}
    cap = min(len(chain_names), max_k) if max_k is not None else len(chain_names)
    res_k = {rk: min(cap, dev_to_nconf(res_devs[rk]))
             for rk in residue_keys}

    # Carbonyl O of residue r forms the peptide bond with N of residue r+1.
    # Limit O to the same slots as r+1 so inter-residue geometry restraints exist.
    next_reskey = {rk: residue_keys[j + 1]
                   for j, rk in enumerate(residue_keys) if j + 1 < len(residue_keys)}
    if limit_o:
        res_k_O = {rk: min(res_k[rk], res_k[next_reskey[rk]]) if rk in next_reskey else res_k[rk]
                   for rk in residue_keys}
    else:
        res_k_O = res_k  # no O-limiting: O appears in all chains (exposes blowup bug)

    k_max = max(res_k.values())
    slot_names = _SLOT_CHAINS[:k_max]

    # Per-residue maximin: for each residue independently select k_r source chains
    per_res_sel = {rk: _select_residue_maximin(chain_names, conf_data, rk, res_k[rk])
                   for rk in residue_keys}

    cryst = (f"CRYST1{st_orig.cell.a:9.3f}{st_orig.cell.b:9.3f}{st_orig.cell.c:9.3f}"
             f"{st_orig.cell.alpha:7.2f}{st_orig.cell.beta:7.2f}{st_orig.cell.gamma:7.2f}"
             f" {st_orig.spacegroup_hm:<11s}1\n")
    all_lines = [cryst]
    serial = 1

    for slot_i, slot_cn in enumerate(slot_names):
        for reskey, ref_rd in ref_chain_data.items():
            if slot_i >= res_k[reskey]:
                continue
            if slot_i >= len(per_res_sel[reskey]):
                # residue exists in fewer chains than desired k
                continue
            src_cn = per_res_sel[reskey][slot_i]
            rd_this = conf_data[src_cn]
            if reskey not in rd_this:
                continue
            rd = rd_this[reskey]
            resname = rd['resname']
            is_het = resname in ('HOH', 'WAT', 'H2O')
            rec = 'HETATM' if is_het else 'ATOM  '
            seqnum = ref_rd['seqid'].num
            icode = ref_rd['seqid'].icode if ref_rd['seqid'].icode != ' ' else ' '
            for aname, ref_atom in ref_rd['atoms'].items():
                eff_k = res_k_O[reskey] if (aname == 'O' and not is_het) else res_k[reskey]
                if slot_i >= eff_k:
                    continue
                occ = 1.0 / eff_k
                atom = rd['atoms'].get(aname, ref_atom)
                elem = atom.element.name.upper()
                name4 = f' {aname:<3s}' if len(elem) == 1 and len(aname) < 4 else f'{aname:<4s}'
                all_lines.append(
                    f'{rec}{serial:5d} {name4}{slot_cn}'
                    f'{resname:<3s} {slot_cn:1s}'
                    f'{seqnum:4d}{icode:1s}   '
                    f'{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}'
                    f'{occ:6.2f}{atom.b_iso:6.2f}'
                    f'          {elem:>2s}\n'
                )
                serial += 1

    if water_pdb is not None:
        for wline in Path(water_pdb).read_text().splitlines(keepends=True):
            if wline.startswith('HETATM') or wline.startswith('ATOM'):
                all_lines.append(wline)
    all_lines.append('END\n')
    Path(out_pdb).write_text(''.join(all_lines))
    print(f'    k_max={k_max}: {len(slot_names)} slot chains (per-residue maximin)')
    print(f'    res_k distribution: ' +
          ', '.join(f'{k}×{sum(1 for v in res_k.values() if v==k)}'
                    for k in sorted(set(res_k.values()))))
    return slot_names, res_k, per_res_sel


# ─────────────────────────────────────────────────────────────────────────────
# Fo-Fc density probing and conformer rebuild
# ─────────────────────────────────────────────────────────────────────────────

_MC_ATOMS = frozenset({'N', 'CA', 'C', 'O', 'CB'})


def _probe_density_gemmi(grid, sigma, gemmi_res, sidechain_only=True):
    """Return (min, max) Fo-Fc in sigma units over heavy atoms of a gemmi Residue.

    Any atom exceeding a threshold triggers action — we don't average.
    """
    vals = []
    for atom in gemmi_res:
        if atom.element.name in ('H', 'D'):
            continue
        if sidechain_only and atom.name in _MC_ATOMS:
            continue
        vals.append(grid.interpolate_value(
            gemmi.Position(atom.pos.x, atom.pos.y, atom.pos.z)))
    if not vals:  # Gly or no sidechain: fall back to all heavy atoms
        for atom in gemmi_res:
            if atom.element.name not in ('H', 'D'):
                vals.append(grid.interpolate_value(
                    gemmi.Position(atom.pos.x, atom.pos.y, atom.pos.z)))
    if not vals:
        return 0.0, 0.0
    return float(min(vals)) / sigma, float(max(vals)) / sigma


def _probe_density_conf(grid, sigma, atom_dict, sidechain_only=True):
    """Return (min, max) Fo-Fc in sigma units over atoms from a conf_data atom dict."""
    vals = []
    for aname, atom in atom_dict.items():
        if atom.element.name in ('H', 'D'):
            continue
        if sidechain_only and aname in _MC_ATOMS:
            continue
        vals.append(grid.interpolate_value(
            gemmi.Position(atom.pos.x, atom.pos.y, atom.pos.z)))
    if not vals:
        for aname, atom in atom_dict.items():
            if atom.element.name not in ('H', 'D'):
                vals.append(grid.interpolate_value(
                    gemmi.Position(atom.pos.x, atom.pos.y, atom.pos.z)))
    if not vals:
        return 0.0, 0.0
    return float(min(vals)) / sigma, float(max(vals)) / sigma


def _find_disulfide_pairs(conf_data, chain_names, sg_dist=2.5):
    """Return dict mapping each CYS reskey to its disulfide partner reskey."""
    cn = chain_names[0]
    cys_res = [(rk, rd) for rk, rd in conf_data[cn].items()
               if rd['resname'] == 'CYS']
    pairs = {}
    for i, (rk1, rd1) in enumerate(cys_res):
        sg1 = rd1['atoms'].get('SG')
        if sg1 is None:
            continue
        for rk2, rd2 in cys_res[i + 1:]:
            sg2 = rd2['atoms'].get('SG')
            if sg2 is None:
                continue
            d = np.sqrt((sg1.pos.x - sg2.pos.x) ** 2 +
                        (sg1.pos.y - sg2.pos.y) ** 2 +
                        (sg1.pos.z - sg2.pos.z) ** 2)
            if d < sg_dist:
                pairs[rk1] = rk2
                pairs[rk2] = rk1
    return pairs


# ── per_res_sel JSON persistence ─────────────────────────────────────────────

def _rk_to_str(rk):
    return f'{rk[0]}_{rk[1]}'


def _str_to_rk(s):
    num, icode = s.split('_', 1)
    return (int(num), icode)


def save_per_res_sel(per_res_sel, path):
    """Serialise per_res_sel {(seqnum,icode): [chain,...]} → JSON file."""
    Path(path).write_text(json.dumps(
        {_rk_to_str(rk): chains for rk, chains in per_res_sel.items()},
        indent=2))


def load_per_res_sel(path):
    """Load per_res_sel from JSON written by save_per_res_sel."""
    return {_str_to_rk(k): v for k, v in json.loads(Path(path).read_text()).items()}


# ── Density scoring ───────────────────────────────────────────────────────────

def _load_slot_res(refmacout_pdb, hoh_names=('HOH', 'WAT', 'H2O')):
    """Return slot_cn → {reskey: gemmi.Residue} from a refmacout PDB."""
    st_ref = gemmi.read_structure(str(refmacout_pdb))
    slot_res = {}
    for ch in st_ref[0]:
        rmap = {}
        for res in ch:
            if res.name in hoh_names:
                continue
            rk = (res.seqid.num, res.seqid.icode.strip())
            rmap[rk] = res
        slot_res[ch.name] = rmap
    return slot_res


def score_density_outliers(refmacout_pdb, refmacout_mtz,
                            conf_data, chain_names,
                            per_res_sel, residue_keys, ref_chain_data,
                            neg_thresh=-3.0, pos_thresh=3.0):
    """Score every prune/add candidate by its peak Fo-Fc excess.

    Returns (candidates, sigma).  candidates is sorted by excess descending:
      [{'action': 'prune'|'add', 'excess': float, 'score': float,
        'rk': tuple, 'slot_i': int|None, 'gt48_cn': str,
        'resname': str, 'dmin': float, 'dmax': float}, ...]
    excess = how far past threshold (larger = higher priority).
    """
    HOH_NAMES = ('HOH', 'WAT', 'H2O')
    mtz = gemmi.read_mtz_file(str(refmacout_mtz))
    grid = mtz.transform_f_phi_to_map('DELFWT', 'PHDELWT', sample_rate=3.0)
    sigma = float(np.array(grid).std())

    slot_res = _load_slot_res(refmacout_pdb, HOH_NAMES)
    slot_order = {cn: i for i, cn in enumerate(_SLOT_CHAINS)}

    candidates = []

    # Prune candidates: current (slot_i, rk) with any atom below neg_thresh.
    for slot_cn, rmap in slot_res.items():
        slot_i = slot_order.get(slot_cn)
        if slot_i is None:
            continue
        for rk, gemmi_res in rmap.items():
            if rk not in per_res_sel or slot_i >= len(per_res_sel[rk]):
                continue
            dmin, dmax = _probe_density_gemmi(grid, sigma, gemmi_res)
            if dmin < neg_thresh:
                gt48_cn = per_res_sel[rk][slot_i]
                candidates.append({
                    'action': 'prune', 'excess': neg_thresh - dmin,
                    'score': dmin, 'rk': rk, 'slot_i': slot_i,
                    'gt48_cn': gt48_cn, 'resname': gemmi_res.name,
                    'dmin': dmin, 'dmax': dmax,
                })

    # Add candidates: gt48 conformers not in model with any atom above pos_thresh.
    for rk in residue_keys:
        current = set(per_res_sel.get(rk, []))
        rd_ref = ref_chain_data.get(rk)
        if rd_ref is None:
            continue
        resname = rd_ref['resname']
        if resname in HOH_NAMES:
            continue
        for gt48_cn in chain_names:
            if gt48_cn in current:
                continue
            rd = conf_data[gt48_cn].get(rk)
            if rd is None:
                continue
            dmin, dmax = _probe_density_conf(grid, sigma, rd['atoms'])
            if dmax > pos_thresh:
                candidates.append({
                    'action': 'add', 'excess': dmax - pos_thresh,
                    'score': dmax, 'rk': rk, 'slot_i': None,
                    'gt48_cn': gt48_cn, 'resname': resname,
                    'dmin': dmin, 'dmax': dmax,
                })

    candidates.sort(key=lambda x: x['excess'], reverse=True)
    return candidates, sigma


def find_map_peak_candidate(refmacout_mtz, conf_data, chain_names,
                             per_res_sel, residue_keys, ref_chain_data,
                             min_sigma=1.0):
    """Return the absent gt48 conformer with the highest Fo-Fc atom-density.

    Samples DELFWT/PHDELWT at every atom of every absent conformer and picks
    the one whose peak atom has the highest density.  min_sigma guards against
    adding from flat noise.

    Returns a candidate dict (same format as score_density_outliers) or None.
    """
    HOH_NAMES = ('HOH', 'WAT', 'H2O')
    mtz  = gemmi.read_mtz_file(str(refmacout_mtz))
    grid = mtz.transform_f_phi_to_map('DELFWT', 'PHDELWT', sample_rate=3.0)
    sigma = float(np.array(grid).std())

    best_dmax = min_sigma
    best = None

    for rk in residue_keys:
        current = set(per_res_sel.get(rk, []))
        rd_ref = ref_chain_data.get(rk)
        if rd_ref is None:
            continue
        resname = rd_ref['resname']
        if resname in HOH_NAMES:
            continue
        for gt48_cn in chain_names:
            if gt48_cn in current:
                continue
            rd = conf_data[gt48_cn].get(rk)
            if rd is None:
                continue
            dmin, dmax = _probe_density_conf(grid, sigma, rd['atoms'])
            if dmax > best_dmax:
                best_dmax = dmax
                best = {
                    'action': 'add', 'excess': dmax - 3.0,
                    'score': dmax, 'rk': rk, 'slot_i': None,
                    'gt48_cn': gt48_cn, 'resname': resname,
                    'dmin': dmin, 'dmax': dmax,
                    'peak_sigma': dmax,
                }

    return best


def find_swap_candidate(refmacout_mtz, refmacout_pdb, peak_cand,
                        per_res_sel, neg_thresh=-3.0):
    """If the peak candidate's residue has a current slot below neg_thresh in
    the Fo-Fc map (using refined coordinates), return a 'swap' candidate that
    replaces that slot with the peak conformer in one top-N action.

    Probes refined coords from refmacout_pdb (not gt48 coords) so the check
    reflects the actual current model density, not the gt48 template.
    Note: the new conformer's gt48 xyz must differ from remaining slots for
    refmac to separate them in occupancy refinement.

    Returns None if no slot is significantly negative.
    """
    rk      = peak_cand['rk']
    current = per_res_sel.get(rk, [])
    if not current:
        return None

    mtz   = gemmi.read_mtz_file(str(refmacout_mtz))
    grid  = mtz.transform_f_phi_to_map('DELFWT', 'PHDELWT', sample_rate=3.0)
    sigma = float(np.array(grid).std())

    slot_res = _load_slot_res(str(refmacout_pdb))

    worst_i    = None
    worst_cn   = None
    worst_dmin = 0.0  # only update if strictly below neg_thresh
    for i, gt48_cn in enumerate(current):
        slot_cn   = _SLOT_CHAINS[i]
        gemmi_res = slot_res.get(slot_cn, {}).get(rk)
        if gemmi_res is None:
            continue
        dmin, _ = _probe_density_gemmi(grid, sigma, gemmi_res)
        if dmin < worst_dmin:
            worst_dmin = dmin
            worst_cn   = gt48_cn
            worst_i    = i

    if worst_cn is None or worst_dmin >= neg_thresh:
        return None

    return {
        'action':      'swap',
        'excess':      (-worst_dmin) + peak_cand['dmax'],
        'score':       -worst_dmin,
        'rk':          rk,
        'slot_i':      worst_i,
        'old_gt48_cn': worst_cn,
        'gt48_cn':     peak_cand['gt48_cn'],
        'resname':     peak_cand['resname'],
        'dmin':        worst_dmin,
        'dmax':        peak_cand['dmax'],
    }


# ── PDB writing for rebuilt models ───────────────────────────────────────────

def _write_rebuilt_pdb(new_per_res_sel, orig_per_res_sel, slot_res,
                       ref_chain_data, conf_data, residue_keys, st_orig,
                       out_pdb, water_pdb=None):
    """Write rebuilt multi-conformer PDB.

    Conformers that were in orig_per_res_sel use slot_res (refmacout.pdb) coords.
    New conformers (not in orig) use conf_data gt48 coords.
    Returns new_res_k dict.
    """
    HOH_NAMES = ('HOH', 'WAT', 'H2O')
    new_res_k = {rk: len(sel) for rk, sel in new_per_res_sel.items()}

    next_reskey = {rk: residue_keys[j + 1]
                   for j, rk in enumerate(residue_keys) if j + 1 < len(residue_keys)}
    new_res_k_O = {rk: min(new_res_k[rk], new_res_k[next_reskey[rk]])
                   if rk in next_reskey else new_res_k[rk]
                   for rk in residue_keys}

    k_max = max(new_res_k.values()) if new_res_k else 1
    new_slot_names = _SLOT_CHAINS[:k_max]

    # Map (rk, gt48_cn) → slot_cn in refmacout.pdb (based on orig_per_res_sel).
    orig_gt48_to_slot = {}  # rk → {gt48_cn: slot_cn}
    for rk, sel in orig_per_res_sel.items():
        orig_gt48_to_slot[rk] = {cn: _SLOT_CHAINS[i] for i, cn in enumerate(sel)}

    cryst = (f"CRYST1{st_orig.cell.a:9.3f}{st_orig.cell.b:9.3f}{st_orig.cell.c:9.3f}"
             f"{st_orig.cell.alpha:7.2f}{st_orig.cell.beta:7.2f}{st_orig.cell.gamma:7.2f}"
             f" {st_orig.spacegroup_hm:<11s}1\n")
    all_lines = [cryst]
    serial = 1

    for slot_i, slot_cn in enumerate(new_slot_names):
        for rk, ref_rd in ref_chain_data.items():
            sel = new_per_res_sel.get(rk, [])
            if slot_i >= len(sel):
                continue
            gt48_cn = sel[slot_i]
            eff_k   = new_res_k[rk]
            eff_k_O = new_res_k_O[rk]

            resname = ref_rd['resname']
            is_het  = resname in HOH_NAMES
            rec     = 'HETATM' if is_het else 'ATOM  '
            seqnum  = ref_rd['seqid'].num
            icode   = ref_rd['seqid'].icode if ref_rd['seqid'].icode != ' ' else ' '

            # Coordinate source: refined if this gt48_cn was in orig_per_res_sel.
            orig_slot_cn = orig_gt48_to_slot.get(rk, {}).get(gt48_cn)
            ref_slot_r   = slot_res.get(orig_slot_cn, {}).get(rk) if orig_slot_cn else None

            for aname, ref_atom in ref_rd['atoms'].items():
                is_O       = (aname == 'O' and not is_het)
                eff_k_atom = eff_k_O if is_O else eff_k
                if slot_i >= eff_k_atom:
                    continue
                occ = 1.0 / eff_k_atom

                if ref_slot_r is not None:
                    gatom = ref_slot_r.find_atom(aname, '\x00')
                    if gatom is None:
                        gatom = ref_slot_r.find_atom(aname, '*')
                    src = (gatom if gatom is not None
                           else conf_data[gt48_cn].get(rk, {}).get('atoms', {}).get(aname, ref_atom))
                    x, y, z, b = src.pos.x, src.pos.y, src.pos.z, src.b_iso
                else:
                    src = conf_data[gt48_cn].get(rk, {}).get('atoms', {}).get(aname, ref_atom)
                    x, y, z, b = src.pos.x, src.pos.y, src.pos.z, src.b_iso

                elem  = ref_atom.element.name.upper()
                name4 = f' {aname:<3s}' if len(elem) == 1 and len(aname) < 4 else f'{aname:<4s}'
                all_lines.append(
                    f'{rec}{serial:5d} {name4}{slot_cn}'
                    f'{resname:<3s} {slot_cn:1s}'
                    f'{seqnum:4d}{icode:1s}   '
                    f'{x:8.3f}{y:8.3f}{z:8.3f}'
                    f'{occ:6.2f}{b:6.2f}'
                    f'          {elem:>2s}\n'
                )
                serial += 1

    if water_pdb is not None:
        for wline in Path(water_pdb).read_text().splitlines(keepends=True):
            if wline.startswith('HETATM') or wline.startswith('ATOM'):
                all_lines.append(wline)
    all_lines.append('END\n')
    Path(out_pdb).write_text(''.join(all_lines))
    return new_res_k


# ── Apply top-N rebuild ───────────────────────────────────────────────────────

def apply_rebuild_topn(candidates, top_n, per_res_sel, orig_per_res_sel, slot_res,
                       residue_keys, ref_chain_data, conf_data, ss_pairs, st_orig,
                       out_pdb, water_pdb=None):
    """Apply the top_n candidates (by excess) and write rebuilt PDB.

    Returns (new_per_res_sel, new_res_k, actions_applied).
    """
    prune_by_rk = {}  # rk → set of gt48_cn to remove
    add_by_rk   = {}  # rk → list of gt48_cn to add
    actions_applied = []
    n_done = 0

    for cand in candidates:
        if n_done >= top_n:
            break
        rk         = cand['rk']
        partner_rk = ss_pairs.get(rk)

        if cand['action'] == 'prune':
            gt48_cn = cand['gt48_cn']
            prune_by_rk.setdefault(rk, set()).add(gt48_cn)
            actions_applied.append(cand)
            n_done += 1
            print(f'  PRUNE slot {_SLOT_CHAINS[cand["slot_i"]]} (gt48:{gt48_cn}) '
                  f'res {rk[0]} {cand["resname"]}: '
                  f'min={cand["dmin"]:.2f}σ max={cand["dmax"]:.2f}σ')
            if partner_rk is not None:
                prune_by_rk.setdefault(partner_rk, set()).add(gt48_cn)
                print(f'  PRUNE SS-partner res {partner_rk[0]} gt48:{gt48_cn}')

        elif cand['action'] == 'add':
            gt48_cn = cand['gt48_cn']
            # Skip if already queued for addition
            if gt48_cn in add_by_rk.get(rk, []):
                continue
            add_by_rk.setdefault(rk, []).append(gt48_cn)
            actions_applied.append(cand)
            n_done += 1
            print(f'  ADD   gt48:{gt48_cn} res {rk[0]} {cand["resname"]}: '
                  f'max={cand["dmax"]:.2f}σ min={cand["dmin"]:.2f}σ')
            if partner_rk is not None and conf_data[gt48_cn].get(partner_rk):
                if gt48_cn not in add_by_rk.get(partner_rk, []):
                    add_by_rk.setdefault(partner_rk, []).append(gt48_cn)
                    print(f'  ADD   gt48:{gt48_cn} res {partner_rk[0]} (SS-partner)')

        elif cand['action'] == 'swap':
            old_cn = cand['old_gt48_cn']
            new_cn = cand['gt48_cn']
            if old_cn in prune_by_rk.get(rk, set()):
                continue  # old slot already being pruned
            if new_cn in add_by_rk.get(rk, []):
                continue  # new conformer already being added
            prune_by_rk.setdefault(rk, set()).add(old_cn)
            add_by_rk.setdefault(rk, []).append(new_cn)
            actions_applied.append(cand)
            n_done += 1
            print(f'  SWAP  slot {_SLOT_CHAINS[cand["slot_i"]]} '
                  f'(gt48:{old_cn}→{new_cn}) res {rk[0]} {cand["resname"]}: '
                  f'worst={cand["dmin"]:.2f}σ peak={cand["dmax"]:.2f}σ')
            if partner_rk is not None:
                if old_cn in per_res_sel.get(partner_rk, []):
                    prune_by_rk.setdefault(partner_rk, set()).add(old_cn)
                if conf_data[new_cn].get(partner_rk) and \
                        new_cn not in add_by_rk.get(partner_rk, []):
                    add_by_rk.setdefault(partner_rk, []).append(new_cn)
                    print(f'  SWAP  SS-partner res {partner_rk[0]} '
                          f'gt48:{old_cn}→{new_cn}')

    # Build new_per_res_sel.
    new_per_res_sel = {}
    for rk in residue_keys:
        sel      = list(per_res_sel.get(rk, []))
        prune    = prune_by_rk.get(rk, set())
        sel      = [cn for cn in sel if cn not in prune]
        existing = set(sel)
        for cn in add_by_rk.get(rk, []):
            if cn not in existing:
                sel.append(cn)
                existing.add(cn)
        new_per_res_sel[rk] = sel

    new_res_k = _write_rebuilt_pdb(
        new_per_res_sel, orig_per_res_sel, slot_res,
        ref_chain_data, conf_data, residue_keys, st_orig,
        out_pdb, water_pdb)
    return new_per_res_sel, new_res_k, actions_applied


# ── rebuild_conformers (applies ALL outliers; kept for backward compat) ───────

def rebuild_conformers(refmacout_pdb, refmacout_mtz,
                       conf_data, chain_names, per_res_sel,
                       residue_keys, ref_chain_data, st_orig,
                       out_pdb, water_pdb=None,
                       neg_thresh=-3.0, pos_thresh=3.0):
    """Prune negative-density conformers; add positive-density ones from gt48.

    Applies ALL outliers (no top-N limit).
    Returns (new_per_res_sel, new_res_k, n_pruned, n_added).
    """
    ss_pairs = _find_disulfide_pairs(conf_data, chain_names)
    print('  Disulfide pairs: ' +
          ', '.join(f'{a[0]}-{b[0]}' for a, b in sorted(ss_pairs.items()) if a < b))

    candidates, sigma = score_density_outliers(
        refmacout_pdb, refmacout_mtz, conf_data, chain_names,
        per_res_sel, residue_keys, ref_chain_data,
        neg_thresh=neg_thresh, pos_thresh=pos_thresh)
    n_prune = sum(1 for c in candidates if c['action'] == 'prune')
    n_add   = sum(1 for c in candidates if c['action'] == 'add')
    print(f'  Fo-Fc sigma={sigma:.4f} e/Å³  outliers: {n_prune} prune, {n_add} add')

    slot_res = _load_slot_res(refmacout_pdb)
    new_per_res_sel, new_res_k, actions = apply_rebuild_topn(
        candidates, top_n=len(candidates),
        per_res_sel=per_res_sel, orig_per_res_sel=per_res_sel,
        slot_res=slot_res, residue_keys=residue_keys,
        ref_chain_data=ref_chain_data, conf_data=conf_data,
        ss_pairs=ss_pairs, st_orig=st_orig, out_pdb=out_pdb, water_pdb=water_pdb)

    n_pruned = sum(1 for a in actions if a['action'] == 'prune')
    n_added  = sum(1 for a in actions if a['action'] == 'add')
    print(f'  Rebuilt: {n_pruned} pruned, {n_added} added  '
          f'(neg<{neg_thresh}σ, pos>{pos_thresh}σ)')
    return new_per_res_sel, new_res_k, n_pruned, n_added


# ─────────────────────────────────────────────────────────────────────────────
# Refmac runner
# ─────────────────────────────────────────────────────────────────────────────

def parse_rfactors(log_text):
    """Extract final R / Rfree from refmac log (last occurrence wins)."""
    r_work = rfree = None
    for line in log_text.splitlines():
        m = re.search(r'Overall R factor\s*=\s*([\d.]+)', line)
        if m:
            r_work = float(m.group(1))
        m = re.search(r'Free R factor\s*=\s*([\d.]+)', line)
        if m:
            rfree = float(m.group(1))
    return r_work, rfree


def fofc_extrema_report(mtz_path, pdb_path, search_radius=3.0):
    """For each model atom, sample Fo-Fc at its position; report the worst peak
    and worst hole.  Also report global map extrema with nearest-atom search
    (using NeighborSearch for proper PBC handling).

    Returns a multi-line string.
    """
    mtz  = gemmi.read_mtz_file(str(mtz_path))
    grid = mtz.transform_f_phi_to_map('DELFWT', 'PHDELWT', sample_rate=3.0)
    arr  = np.array(grid, copy=False)
    sigma = arr.std()
    if sigma < 1e-8:
        return '    Fo-Fc map is flat (sigma≈0)'

    st = gemmi.read_structure(str(pdb_path))
    ns = gemmi.NeighborSearch(st[0], st.cell, search_radius).populate(include_h=True)

    def atom_desc(chain, res, atom):
        alt = atom.altloc if atom.altloc != '\x00' else '-'
        return (f'{res.name}{res.seqid.num} {atom.name}[{alt}]'
                f' occ={atom.occ:.2f}')

    # Per-atom Fo-Fc interpolation
    best_peak_sig, best_peak_desc = 0.0, ''
    best_hole_sig, best_hole_desc = 0.0, ''
    for chain in st[0]:
        for res in chain:
            for atom in res:
                val = grid.interpolate_value(atom.pos) / sigma
                desc = atom_desc(chain, res, atom)
                if val > best_peak_sig:
                    best_peak_sig, best_peak_desc = val, desc
                if val < best_hole_sig:
                    best_hole_sig, best_hole_desc = val, desc

    # Global map extrema with NeighborSearch for PBC-correct nearest atom
    def global_nearest(idx):
        frac = gemmi.Fractional(idx[0]/grid.nu, idx[1]/grid.nv, idx[2]/grid.nw)
        pos  = grid.unit_cell.orthogonalize(frac)
        mark = ns.find_nearest_atom(pos)
        if mark is None:
            return pos, '(no nearby atom)', 999.0
        cra  = mark.to_cra(st[0])
        dist = mark.pos.dist(pos)   # mark.pos is the PBC image position
        desc = atom_desc(cra.chain, cra.residue, cra.atom) + f' d={dist:.2f}Å'
        return pos, desc, dist

    peak_idx = np.unravel_index(arr.argmax(), arr.shape)
    hole_idx = np.unravel_index(arr.argmin(), arr.shape)
    _, peak_near, peak_d = global_nearest(peak_idx)
    _, hole_near, hole_d = global_nearest(hole_idx)
    global_peak = arr[peak_idx] / sigma
    global_hole = arr[hole_idx] / sigma

    lines = []
    if best_peak_desc:
        lines.append(f'    Fo-Fc at atoms  peak: +{best_peak_sig:.1f}σ  {best_peak_desc}')
    if best_hole_desc:
        lines.append(f'    Fo-Fc at atoms  hole: {best_hole_sig:.1f}σ  {best_hole_desc}')
    lines.append(f'    Fo-Fc global    peak: +{global_peak:.1f}σ  nearest: {peak_near}')
    lines.append(f'    Fo-Fc global    hole: {global_hole:.1f}σ  nearest: {hole_near}')
    return '\n'.join(lines)


def generate_occ_groups(pdb_path):
    """Generate refmac occupancy group keywords from altloc structure of pdb_path.

    Multi-chain conformer format (chain_id == altloc):
        Per-residue, per-chain groups.  O atoms are handled separately when they
        appear in fewer chains than the rest of the residue (due to the O-limiting
        strategy that requires residue r+1 to exist).

        For each residue r:
          - If all chains that have any atom also have O: one 'alts complete' group
            per chain for the whole residue.
          - Otherwise: two independent 'alts complete' blocks —
              (a) O atom only, over the subset of chains that have O
              (b) all non-O atoms, enumerated explicitly, over all chains

    Single-chain altloc format:
        MC altlocs and SC altlocs treated as separate complete groups per residue.

    Returns bytes ready to append to refmac keyword input.
    """
    HOH_NAMES = ('HOH', 'WAT', 'H2O')
    st = gemmi.read_structure(str(pdb_path))
    lines = ['occupancy refine']
    gid = 1

    prot_chains = [ch for ch in st[0]
                   if any(res.name not in HOH_NAMES for res in ch)]

    # Detect multi-chain conformer format: chain names are single alphabetic
    # letters (slot chains A-Z, a-z) and at least two chains share some residues.
    is_multichain = False
    if len(prot_chains) >= 2:
        slot_like = all(len(ch.name) == 1 and ch.name.isalpha()
                        for ch in prot_chains)
        if slot_like:
            def _rknums(ch):
                return {res.seqid.num for res in ch if res.name not in HOH_NAMES}
            ref_rk = _rknums(prot_chains[0])
            has_overlap = any(ref_rk & _rknums(ch) for ch in prot_chains[1:])
            is_multichain = has_overlap

    if is_multichain:
        # Collect per-residue atom inventory across all chains.
        # res_info: seqnum_key → {chain_name: set(atom_names)}
        res_info = {}
        res_order = []
        for ch in prot_chains:
            for res in ch:
                if res.name in HOH_NAMES:
                    continue
                key = (res.seqid.num, res.seqid.icode.strip())
                if key not in res_info:
                    res_info[key] = {}
                    res_order.append(key)
                res_info[key][ch.name] = {a.name for a in res}

        for key in res_order:
            chain_atoms = res_info[key]
            resnum = key[0]
            icode = key[1]
            res_id = f'{resnum}{icode}' if icode else str(resnum)
            all_chains = sorted(chain_atoms.keys())
            o_chains = [c for c in all_chains if 'O' in chain_atoms[c]]

            if len(o_chains) == len(all_chains):
                # Simple case: O present in every chain — one group per chain.
                group_ids = []
                for cn in all_chains:
                    lines.append(f'occupancy group id {gid} chain {cn}'
                                 f' residue {res_id}')
                    group_ids.append(gid)
                    gid += 1
                lines.append('occupancy group alts complete ' +
                              ' '.join(map(str, group_ids)))
            else:
                # O missing from some chains: two independent complete blocks.
                # (a) O-only group over o_chains
                if o_chains:
                    o_ids = []
                    for cn in o_chains:
                        lines.append(f'occupancy group id {gid} chain {cn}'
                                     f' residue {res_id} atom O')
                        o_ids.append(gid)
                        gid += 1
                    lines.append('occupancy group alts complete ' +
                                 ' '.join(map(str, o_ids)))
                # (b) Non-O atoms: one group per chain, atoms listed explicitly.
                non_o_ids = []
                for cn in all_chains:
                    non_o_atoms = sorted(chain_atoms[cn] - {'O'})
                    if not non_o_atoms:
                        continue
                    for aname in non_o_atoms:
                        lines.append(f'occupancy group id {gid} chain {cn}'
                                     f' residue {res_id} atom {aname}')
                    non_o_ids.append(gid)
                    gid += 1
                if non_o_ids:
                    lines.append('occupancy group alts complete ' +
                                 ' '.join(map(str, non_o_ids)))
        return ('\n'.join(lines) + '\n').encode()

    # Single-chain altloc format: group MC and SC altlocs per residue.
    # If mc_alts == sc_alts (whole-conformer alternates, as in the 1AHO pipeline),
    # use one group per altloc covering all atoms so MC and SC share the same occ.
    # If they differ (generate_protein style with independent MC/SC disorder),
    # emit two separate complete groups.
    for chain in st[0]:
        for res in chain:
            resnum = res.seqid.num
            icode  = res.seqid.icode.strip()
            res_id = f'{resnum}{icode}' if icode else str(resnum)
            mc_alts, sc_alts = set(), set()
            for atom in res:
                if atom.altloc != '\x00':
                    target = mc_alts if atom.name in MAINCHAIN_ATOMS else sc_alts
                    target.add(atom.altloc)
            all_alts = mc_alts | sc_alts
            if len(all_alts) < 2:
                continue
            if mc_alts == sc_alts:
                # Whole-conformer alternates: one group per altloc, all atoms together.
                group_ids = []
                for alt in sorted(all_alts):
                    lines.append(f'occupancy group id {gid} chain {chain.name}'
                                 f' residue {res_id} alt {alt}')
                    group_ids.append(gid)
                    gid += 1
                lines.append('occupancy group alts complete ' +
                              ' '.join(map(str, group_ids)))
            else:
                # Independent MC/SC disorder: emit separate complete groups.
                for alt_set in (sorted(mc_alts), sorted(sc_alts)):
                    if len(alt_set) < 2:
                        continue
                    group_ids = []
                    for alt in alt_set:
                        lines.append(f'occupancy group id {gid} chain {chain.name}'
                                     f' residue {res_id} alt {alt}')
                        group_ids.append(gid)
                        gid += 1
                    lines.append('occupancy group alts complete ' +
                                 ' '.join(map(str, group_ids)))
    return ('\n'.join(lines) + '\n').encode()


def run_refmac_quick(xyzin, fobs_mtz, ncyc, weight_matrix, tmpdir,
                     occ_refine=True, fp_col='FP'):
    """Single refmac call for iterative rebuild rounds.

    Returns (R, Rfree, log_text, out_mtz|None, out_pdb|None).
    """
    xyzin    = Path(xyzin).resolve()
    fobs_mtz = Path(fobs_mtz).resolve()
    damp = min(0.5, 0.5 / weight_matrix) if weight_matrix > 1.0 else 0.5
    kw  = f'LABIN FP={fp_col} FPART1=Fpart PHIP1=PHIpart FREE=FreeR_flag\n'.encode()
    kw += b'LABOUT FC=FC PHIC=PHIC FWT=FWT PHWT=PHWT DELFWT=DELFWT PHDELWT=PHDELWT\n'
    kw += f'solvent no\nscpart 1\ndamp {damp:.4f} {damp:.4f}\nmake hout Y\nmake hydr Y\n'.encode()
    kw += f'weight matrix {weight_matrix}\nNCYC {ncyc}\n'.encode()
    if occ_refine:
        kw += generate_occ_groups(xyzin)
    kw += b'END\n'

    xyzout = tmpdir / '_quick_out.pdb'
    hklout = tmpdir / '_quick_out.mtz'
    log = run(
        [REFMAC5,
         'XYZIN', xyzin, 'XYZOUT', xyzout,
         'HKLIN', fobs_mtz, 'HKLOUT', hklout,
         'LIBOUT', tmpdir / '_quick.lib'],
        input_bytes=kw, cwd=tmpdir, check=False,
    )
    r, rf = parse_rfactors(log)
    return (r, rf, log,
            hklout if hklout.exists() else None,
            xyzout if xyzout.exists() else None)


def run_refmac(starthere_pdb, fobs_mtz, ncyc, tmpdir):
    """Run refmac, return (R, Rfree, log_text)."""
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

    out_mtz = tmpdir / f'_rfac_n{ncyc}.mtz'
    log_path = tmpdir / f'_rfac_n{ncyc}.log'
    try:
        log = run(
            [REFMAC5,
             'XYZIN', starthere_pdb,
             'XYZOUT', tmpdir / '_out.pdb',
             'HKLIN',  fobs_mtz,
             'HKLOUT', out_mtz,
             'LIBOUT', tmpdir / '_refmac.lib'],
            input_bytes=kw, cwd=tmpdir, check=False,
        )
    except Exception as e:
        return None, None, str(e), None, None
    log_path.write_text(log)
    r, rf = parse_rfactors(log)
    out_pdb = tmpdir / '_out.pdb'
    return r, rf, log, (out_mtz if out_mtz.exists() else None), (out_pdb if out_pdb.exists() else None)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each strategy: (name, desc, mc_ord, mc_bouq, sc_ord, sc_bouq, sc_bouq_thr, mc_bouq_thr)
STRATEGIES = [
    ('S1',  'MC k2 | SC k2 everywhere',
     'cluster_k2', 'cluster_k2', 'cluster_k2', 'cluster_k2',  1.5, 0.5),
    ('S2',  'MC k2 | SC k2 ord | SC bouq k4',
     'cluster_k2', 'cluster_k2', 'cluster_k2', 'cluster_k4',  1.5, 0.5),
    ('S3',  'MC k2 | SC k2 ord | SC bouq k8',
     'cluster_k2', 'cluster_k2', 'cluster_k2', 'cluster_k8',  1.5, 0.5),
    ('S4',  'MC k2 | SC k3 ord | SC bouq k8',
     'cluster_k2', 'cluster_k2', 'cluster_k3', 'cluster_k8',  1.5, 0.5),
    ('S5',  'MC k2/k4 CA>0.5 | SC k3 ord | SC bouq k8',
     'cluster_k2', 'cluster_k4', 'cluster_k3', 'cluster_k8',  1.5, 0.5),
    ('S6',  'MC k2/k8 CA>0.5 | SC k3 ord | SC bouq k8',
     'cluster_k2', 'cluster_k8', 'cluster_k3', 'cluster_k8',  1.5, 0.5),
    ('S7',  'MC k2/k8 CA>0.5 | SC k3 ord | SC bouq adaptive',
     'cluster_k2', 'cluster_k8', 'cluster_k3', 'adaptive',    1.5, 0.5),
    ('S8',  'MC k3/k8 CA>0.5 | SC k3 ord | SC bouq adaptive',
     'cluster_k3', 'cluster_k8', 'cluster_k3', 'adaptive',    1.5, 0.5),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb',    default='1aho/gt48.pdb')
    ap.add_argument('--mtz',    default='1aho/gt48.mtz')
    ap.add_argument('--outdir', default='1aho/explore_fusion')
    ap.add_argument('--strategies', nargs='*',
                    help='Subset of strategy names (default: all)')
    args = ap.parse_args()

    pdb_path  = Path(args.pdb).resolve()
    refme_path = Path(args.mtz).resolve()
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected = set(args.strategies) if args.strategies else None

    print('=' * 70)
    print('1AHO conformer fusion exploration')
    print(f'  PDB  : {pdb_path}')
    print(f'  MTZ  : {refme_path}')
    print(f'  dmin : {DMIN} Å')
    print('=' * 70)

    with tempfile.TemporaryDirectory(prefix='1aho_fobs_') as _td:
        tmpdir_fobs = Path(_td)
        print('\nBuilding Fobs MTZ...')
        fobs_mtz = build_fobs_mtz(pdb_path, refme_path, tmpdir_fobs)
        # Copy to outdir so it persists
        fobs_final = outdir / 'fobs.mtz'
        shutil.copy2(fobs_mtz, fobs_final)
        print(f'  → saved to {fobs_final}')

        print('\nParsing 48-conformer PDB...')
        st_orig, chain_names, conf_data = parse_conformers(pdb_path)
        print(f'  {len(chain_names)} chains (conformers), '
              f'{len(conf_data[chain_names[0]])} residues in ref chain')

        print('\nLoading density map for conformer scoring...')
        refmac_mtz = pdb_path.parent / 'gt48.mtz'
        density_grid = load_density_map(str(refmac_mtz))

        # Header
        print()
        hdr = (f"{'Strategy':<6}  {'Description':<52}  "
               f"{'Nbouq':>5}  {'Nalt':>4}  "
               f"{'R0':>6}  {'Rf0':>6}  {'R50':>6}  {'Rf50':>7}")
        print(hdr)
        print('-' * len(hdr))

        for row in STRATEGIES:
            name, desc, mc_ord, mc_bouq, sc_ord, sc_bouq, bouq_thresh, mc_bouq_thresh = row
            if selected and name not in selected:
                continue

            print(f'\n[{name}] {desc}')
            strat_dir = outdir / name
            strat_dir.mkdir(exist_ok=True)

            out_pdb = strat_dir / 'starthere.pdb'
            try:
                _, n_bouq, n_alt = build_reduced_pdb(
                    st_orig, chain_names, conf_data,
                    strategy=(mc_ord, mc_bouq, sc_ord, sc_bouq),
                    density_grid=density_grid,
                    bouquet_threshold=bouq_thresh,
                    mc_bouq_threshold=mc_bouq_thresh,
                    out_pdb=out_pdb, tmpdir=strat_dir,
                )
            except Exception as e:
                print(f'  ERROR building model: {e}')
                continue

            # NCYC 0
            with tempfile.TemporaryDirectory(prefix=f'rfac_{name}_n0_') as td0:
                td0 = Path(td0)
                shutil.copy2(out_pdb, td0 / 'starthere.pdb')
                shutil.copy2(fobs_final, td0 / 'fobs.mtz')
                print(f'  Running refmac NCYC 0...')
                r0, rf0, log0, _, pdb0 = run_refmac(
                    td0 / 'starthere.pdb', td0 / 'fobs.mtz',
                    ncyc=0, tmpdir=td0)
                (strat_dir / 'refmac_n0.log').write_text(log0 or '')
                if pdb0:
                    shutil.copy2(pdb0, strat_dir / 'refmac_n0.pdb')
                r0_s  = f'{r0:.4f}' if r0  is not None else '  —  '
                rf0_s = f'{rf0:.4f}' if rf0 is not None else '  —  '
                print(f'    NCYC 0: R={r0_s}  Rfree={rf0_s}')

            # NCYC 50
            with tempfile.TemporaryDirectory(prefix=f'rfac_{name}_n50_') as td5:
                td5 = Path(td5)
                shutil.copy2(out_pdb, td5 / 'starthere.pdb')
                shutil.copy2(fobs_final, td5 / 'fobs.mtz')
                print(f'  Running refmac NCYC 50...')
                r5, rf5, log5, mtz5, pdb5 = run_refmac(
                    td5 / 'starthere.pdb', td5 / 'fobs.mtz',
                    ncyc=50, tmpdir=td5)
                (strat_dir / 'refmac_n50.log').write_text(log5 or '')
                if mtz5:
                    shutil.copy2(mtz5, strat_dir / 'refmac_n50.mtz')
                if pdb5:
                    shutil.copy2(pdb5, strat_dir / 'refmac_n50.pdb')
                r5_s  = f'{r5:.4f}' if r5  is not None else '  —  '
                rf5_s = f'{rf5:.4f}' if rf5 is not None else '  —  '
                print(f'    NCYC 50: R={r5_s}  Rfree={rf5_s}')
                if mtz5 and pdb5:
                    print(fofc_extrema_report(mtz5, pdb5))

            print(f"  {name:<6}  {desc[:52]:<52}  "
                  f"{n_bouq:>5}  {n_alt:>4}  "
                  f"{r0_s:>6}  {rf0_s:>6}  {r5_s:>6}  {rf5_s:>6}")

    print('\n' + '=' * 70)
    print('Done. Starthere PDBs and refmac logs in:', outdir)


if __name__ == '__main__':
    main()
