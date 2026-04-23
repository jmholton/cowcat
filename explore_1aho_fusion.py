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
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import gemmi

SCRIPT_DIR = Path(__file__).parent
REFMAC5    = Path('/programs/ccp4-8.0/bin/refmac5')
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

# All altloc labels: MC gets the first mc_k letters, SC gets the next sc_k letters.
# Gemmi capitalises all altloc chars when writing PDB, so we must keep MC and SC
# in disjoint uppercase ranges rather than upper/lower-case.
ALL_ALT_LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Maximum altloc groups passed to refmac for bouquet/disulfide residues.
# Refmac MX1ALT=20 is the hard limit; we use fewer to reduce per-cycle cost.
BOUQUET_MAX_ALT = 8


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
                      rng=None):
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
        k = min(len(common), BOUQUET_MAX_ALT)
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

        def atoms_from_groups(anames, groups, multi_altloc, labels):
            """Write averaged atoms for each group into res_out.

            labels: character sequence to use for altloc labels (MC_ALT_LABELS
            or SC_ALT_LABELS).  Every atom name in anames is written for every
            group (using global mean as fallback) so all altlocs are complete.
            """
            for gi, members in enumerate(groups):
                lbl = labels[gi] if multi_altloc else '\x00'
                ws  = conf_occs[members]
                ws  = ws / ws.sum() if ws.sum() > 0 else np.ones(len(ws)) / len(ws)
                occ = float(conf_occs[members].sum())
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
                scores = density_score(valid_chains, reskey, all_anames, conf_data, density_grid)
                if rng is not None and n_conf >= 2:
                    n_half = max(1, n_conf // 2)
                    half_idx = np.sort(rng.choice(n_conf, n_half, replace=False))
                else:
                    half_idx = np.arange(n_conf)
                k = min(len(half_idx), BOUQUET_MAX_ALT)
                half_order = half_idx[np.argsort(scores[half_idx])]
                all_groups = [arr for arr in np.array_split(half_order, k) if len(arr) > 0]
            multi = len(all_groups) > 1
            if multi:
                n_altloc_res += 1
            atoms_from_groups(all_anames, all_groups, multi_altloc=multi,
                              labels=ALL_ALT_LABELS)
            chain_out.add_residue(res_out)
            continue

        # ── Regular residues: separate MC/SC processing ──────────────────────
        spread_for_mc = ca_max_spread(reskey, conf_data, chain_names)
        mc_mode = mc_bouq if spread_for_mc > mc_bouq_threshold else mc_ord
        mc_k = int(mc_mode.split('_k')[1]) if '_k' in mc_mode else 1
        mc_multi = (mc_k > 1 and n_conf >= 2)
        if mc_multi:
            mc_groups = split_by_density(mc_anames, mc_k)
            n_altloc_res += 1
        else:
            mc_groups = [all_idx]
            mc_k = 1

        mc_letters = mc_k if mc_multi else 0
        mc_labels  = ALL_ALT_LABELS[:mc_letters] if mc_multi else ALL_ALT_LABELS
        atoms_from_groups(mc_anames, mc_groups, multi_altloc=mc_multi, labels=mc_labels)

        if not sc_anames:
            chain_out.add_residue(res_out)
            continue

        sc_max_k  = 26 - mc_letters
        sc_labels = ALL_ALT_LABELS[mc_letters:]

        mode = sc_ord
        if mode == 'adaptive':
            k = 1
        else:
            k = int(mode.split('_k')[1]) if '_k' in mode else 1
        k = min(k, sc_max_k)
        if k <= 1:
            atoms_from_groups(sc_anames, [all_idx], multi_altloc=False, labels=sc_labels)
            chain_out.add_residue(res_out)
            continue
        sc_groups = split_by_density(sc_anames, k)
        multi = len(sc_groups) > 1
        if multi:
            n_altloc_res += 1
        atoms_from_groups(sc_anames, sc_groups, multi_altloc=multi, labels=sc_labels)

        chain_out.add_residue(res_out)

    model_out.add_chain(chain_out)
    st_out.add_model(model_out)

    if out_pdb is None:
        out_pdb = tmpdir / 'starthere.pdb'
    st_out.write_pdb(str(out_pdb))
    print(f'    bouquet residues: {n_bouquet}  altloc residues: {n_altloc_res}')
    return out_pdb, n_bouquet, n_altloc_res


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

    MC altlocs (uppercase) and SC altlocs (lowercase) are treated as separate
    complete groups so each set sums to occupancy 1.0 independently.
    Returns bytes ready to append to refmac keyword input.
    """
    st = gemmi.read_structure(str(pdb_path))
    lines = ['occupancy refine']
    gid = 1
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
    ap.add_argument('--pdb',    default='1aho/refmacout_minRfree.pdb')
    ap.add_argument('--mtz',    default='1aho/refme_minRfree.mtz')
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
        refmac_mtz = pdb_path.parent / 'refmacout_minRfree.mtz'
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
