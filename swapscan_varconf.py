#!/usr/bin/env ccp4-python
"""Broad scan of internal conformer-letter swap moves on a varconf PDB.

For every pair of chain letters (s, t) in the starting PDB, tries swapping a
specific atom subset at a specific residue between the two chains and measures
whether Rfree / agreement with ground-truth SFs improves after NCYC refinement.

Four move types (all atoms come from the starting PDB — no external library):
  sc   "CA and out": swap CA + sidechain between chains s and t at residue i
         (backbone N, C, O stay in their original chains)
  pep  "CA to CA":  swap {C, O} at residue i and {N} at residue i+1
  o    carbonyl O only: swap O at residue i
  ss   disulfide unit: swap SC of both CYS in a disulfide pair simultaneously

Workflow:
  # Submit scan (baseline + SLURM arrays):
  ccp4-python swapscan_varconf.py --submit \\
      --pdb  1aho/varconf_opt1.pdb \\
      --fobs 1aho/refme.mtz \\
      --truth 1aho/gt48.mtz \\
      --outdir 1aho/swapscan_opt1 \\
      --move-types sc,pep,o,ss \\
      --ncyc 50 \\
      --partition lr6 --account pc_als831 --qos lr_normal

  # Run single task (called by SLURM):
  ccp4-python swapscan_varconf.py --task N --outdir 1aho/swapscan_opt1

  # Collate results:
  ccp4-python swapscan_varconf.py --collate --outdir 1aho/swapscan_opt1
"""

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import gemmi
import numpy as np

from explore_1aho_fusion import run_refmac_quick

HOH_NAMES = frozenset({'HOH', 'WAT', 'H2O'})
MAX_ARRAY  = 1000

# Atoms kept in place during an SC swap (pure backbone bonds fixed at N and C ends)
SC_BACKBONE_KEEP = frozenset({'N', 'C', 'O'})
# Pep swap: {C, O} at residue i, {N} at residue i+1
PEP_ATOMS_I = frozenset({'C', 'O'})
PEP_ATOMS_J = frozenset({'N'})
# O-only swap
O_ATOMS = frozenset({'O'})


# ── PDB atom parsing / writing ────────────────────────────────────────────────

def parse_atoms(pdb_path):
    """Return list of atom dicts and non-atom header lines from a PDB."""
    atom_recs = []
    header    = []
    for line in Path(pdb_path).read_text().splitlines(keepends=True):
        rec = line[:6]
        if rec in ('ATOM  ', 'HETATM'):
            try:
                atom_recs.append({
                    'rec':     rec,
                    'chain':   line[21],
                    'seqnum':  int(line[22:26]),
                    'icode':   line[26].strip(),
                    'aname':   line[12:16].strip(),
                    'resname': line[17:20].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occ':     float(line[54:60]),
                    'b':       float(line[60:66]),
                    'rest':    line[66:],   # element, charge, newline
                    'name4':   line[12:16], # preserve original spacing
                    'altloc':  line[16],
                })
            except (ValueError, IndexError):
                header.append(line)
        elif line.strip() not in ('END', 'TER', ''):
            header.append(line)
    return atom_recs, header


def write_atoms(atom_recs, header_lines, out_pdb):
    """Write PDB from atom record list."""
    lines = list(header_lines)
    for i, a in enumerate(atom_recs, 1):
        lines.append(
            f'{a["rec"]}{i:5d} {a["name4"]}{a["altloc"]}'
            f'{a["resname"]:<3s} {a["chain"]}{a["seqnum"]:4d}{a["icode"] or " ":1s}   '
            f'{a["x"]:8.3f}{a["y"]:8.3f}{a["z"]:8.3f}'
            f'{a["occ"]:6.2f}{a["b"]:6.2f}{a["rest"]}'
        )
    lines.append('END\n')
    Path(out_pdb).write_text(''.join(lines))


def atom_index(atom_recs):
    """Return dict (chain, seqnum, icode, aname, altloc) → list-of-indices."""
    idx = {}
    for i, a in enumerate(atom_recs):
        key = (a['chain'], a['seqnum'], a['icode'], a['aname'], a['altloc'])
        idx.setdefault(key, []).append(i)
    return idx


def _chain_lookup(idx, chain, sn, ic, aname):
    """Find any atoms matching (chain, sn, ic, aname) regardless of altloc."""
    prefix = (chain, sn, ic, aname)
    for key, idxs in idx.items():
        if key[:4] == prefix:
            return idxs
    return []


# ── Disulfide pair detection ─────────────────────────────────────────────────

def find_ss_pairs(base_atoms, ref_chain='A', sg_cutoff=2.5):
    """Return list of (rk_i, rk_j) CYS pairs with SG-SG < sg_cutoff in ref_chain."""
    altloc_mode = any(a['altloc'].strip() for a in base_atoms
                      if a['resname'] not in HOH_NAMES)
    sg_pos = {}
    for a in base_atoms:
        if a['resname'] != 'CYS' or a['aname'] != 'SG':
            continue
        if altloc_mode:
            if a['chain'] != ref_chain or a['altloc'] not in (' ', 'A'):
                continue
        else:
            if a['chain'] != ref_chain:
                continue
        sg_pos[(a['seqnum'], a['icode'])] = (a['x'], a['y'], a['z'])
    pairs = []
    rks = sorted(sg_pos)
    for i, rk_i in enumerate(rks):
        for rk_j in rks[i + 1:]:
            xi, yi, zi = sg_pos[rk_i]
            xj, yj, zj = sg_pos[rk_j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 < sg_cutoff ** 2:
                pairs.append((rk_i, rk_j))
    return pairs


# ── Swap PDB construction ─────────────────────────────────────────────────────

def _swap_xyz(recs, i, j):
    """Exchange x/y/z between records i and j (in-place)."""
    recs[i]['x'], recs[j]['x'] = recs[j]['x'], recs[i]['x']
    recs[i]['y'], recs[j]['y'] = recs[j]['y'], recs[i]['y']
    recs[i]['z'], recs[j]['z'] = recs[j]['z'], recs[i]['z']


def _apply_one_swap(recs, idx, sw):
    """Apply a single swap spec to recs (in-place). Returns number of atoms swapped."""
    altloc_mode = 'altloc_s' in sw
    move_type   = sw['move_type']
    seqnum      = sw['rk'][0]
    icode       = sw['rk'][1]
    rk_next     = sw.get('rk_next')
    rk_partner  = sw.get('rk_partner')

    if altloc_mode:
        altloc_s = sw['altloc_s']
        altloc_t = sw['altloc_t']
        chain    = sw['chain']
    else:
        chain_s  = sw['chain_s']
        chain_t  = sw['chain_t']

    if move_type == 'sc':
        atom_specs = [(seqnum, icode, None)]
    elif move_type == 'pep':
        if rk_next is None:
            return 0
        atom_specs = [(seqnum, icode, PEP_ATOMS_I),
                      (rk_next[0], rk_next[1], PEP_ATOMS_J)]
    elif move_type == 'o':
        atom_specs = [(seqnum, icode, O_ATOMS)]
    elif move_type == 'ss':
        if rk_partner is None:
            return 0
        atom_specs = [(seqnum, icode, None),
                      (rk_partner[0], rk_partner[1], None)]
    else:
        return 0

    n = 0
    for (sn, ic, atom_set) in atom_specs:
        for aname_key in idx:
            c, sn2, ic2, aname, alt = aname_key
            if altloc_mode:
                if c != chain or sn2 != sn or ic2 != ic or alt != altloc_s:
                    continue
            else:
                if c != chain_s or sn2 != sn or ic2 != ic:
                    continue
            if atom_set is None:
                if aname in SC_BACKBONE_KEEP:
                    continue
            elif aname not in atom_set:
                continue
            if altloc_mode:
                s_idxs = idx.get((chain,   sn, ic, aname, altloc_s), [])
                t_idxs = idx.get((chain,   sn, ic, aname, altloc_t), [])
            else:
                s_idxs = _chain_lookup(idx, chain_s, sn, ic, aname)
                t_idxs = _chain_lookup(idx, chain_t, sn, ic, aname)
            if not s_idxs or not t_idxs:
                continue
            _swap_xyz(recs, s_idxs[0], t_idxs[0])
            n += 1
    return n


def make_swap_pdb(base_atoms, idx, swaps):
    """Apply a list of swap specs to a fresh copy of base_atoms.

    Returns (recs, success).  Empty swaps list = identity (no change).
    """
    recs = [dict(a) for a in base_atoms]
    if not swaps:
        return recs, True
    n_total = sum(_apply_one_swap(recs, idx, sw) for sw in swaps)
    return recs, n_total > 0


# ── Metrics ───────────────────────────────────────────────────────────────────

MOLPROBIFY = Path.home() / 'Develop' / 'molprobify_runme.com'

def run_molprobity(pdb_path, tdir):
    """Run molprobify_runme.com on pdb_path (absolute), return wE float or None."""
    if not MOLPROBIFY.exists():
        return None
    try:
        proc = subprocess.run(
            [str(MOLPROBIFY), str(pdb_path.resolve())],
            cwd=str(tdir), capture_output=True, text=True, timeout=1800)
        for line in proc.stdout.splitlines():
            if 'weighted energy (wE):' in line:
                return float(line.split(':')[-1].strip())
    except Exception as e:
        print(f'  molprobify error: {e}')
    return None


def compute_metrics(trial_mtz_path, gt48_mtz_path):
    """RMSD (electrons) and R-true between |FC_ALL_LS| and |Fgt|.

    Least-squares scales Fgt onto FC, then:
      rmsd_e = sqrt(mean((FC - sc*Fgt)^2))   [electrons]
      r_true = sum|FC - sc*Fgt| / sum(sc*Fgt)
    """
    try:
        t   = gemmi.read_mtz_file(str(trial_mtz_path))
        g   = gemmi.read_mtz_file(str(gt48_mtz_path))
        t_a = np.array(t, copy=False)
        g_a = np.array(g, copy=False)
        fc_idx  = t.column_with_label('FC_ALL_LS').idx
        fgt_idx = g.column_with_label('Fgt').idx
        g_dict  = {(int(r[0]), int(r[1]), int(r[2])): float(r[fgt_idx])
                   for r in g_a if np.isfinite(r[fgt_idx]) and r[fgt_idx] > 0}
        fc_v, fgt_v = [], []
        for row in t_a:
            hkl = (int(row[0]), int(row[1]), int(row[2]))
            fc  = float(row[fc_idx])
            if hkl in g_dict and np.isfinite(fc):
                fc_v.append(fc)
                fgt_v.append(g_dict[hkl])
        if len(fc_v) < 100:
            return None, None
        fc   = np.array(fc_v)
        fgt  = np.array(fgt_v)
        sc   = np.dot(fc, fgt) / np.dot(fgt, fgt)
        diff = fc - sc * fgt
        rmsd_e = float(np.sqrt(np.mean(diff ** 2)))
        r_t    = float(np.sum(np.abs(diff)) / np.sum(sc * fgt))
        return rmsd_e, r_t
    except Exception as e:
        print(f'  compute_metrics error: {e}')
        return None, None


# ── Baseline ──────────────────────────────────────────────────────────────────

def run_baseline(base_pdb, fobs_mtz, truth_mtz, outdir, ncyc=50, weight=0.5):
    bl_dir  = outdir / 'baseline'
    bl_dir.mkdir(parents=True, exist_ok=True)
    bl_json = bl_dir / 'result.json'
    if bl_json.exists():
        print(f'Baseline already exists.')
        return json.loads(bl_json.read_text())

    print(f'Running baseline refmac NCYC {ncyc} on {base_pdb}...')
    bl_mtz = bl_dir / 'refmacout.mtz'
    bl_pdb = bl_dir / 'refmacout.pdb'
    with tempfile.TemporaryDirectory(prefix='swapscan_bl_') as td:
        r, rf, log, mtz_out, pdb_out = run_refmac_quick(
            base_pdb, fobs_mtz, ncyc, weight, Path(td))
        if pdb_out and pdb_out.exists():
            shutil.copy2(pdb_out, bl_pdb)
        if mtz_out and mtz_out.exists():
            shutil.copy2(mtz_out, bl_mtz)
        (bl_dir / 'refmac.log').write_text(log or '')

    rmsd_e, r_true = (None, None)
    if bl_mtz.exists() and truth_mtz:
        rmsd_e, r_true = compute_metrics(bl_mtz, truth_mtz)

    res = {'r': r, 'rf': rf, 'rmsd_e': rmsd_e, 'r_true': r_true,
           'mtz': str(bl_mtz), 'pdb': str(bl_pdb)}
    rmsd_str = f'{rmsd_e:.4f}' if rmsd_e is not None else 'N/A'
    print(f'  R={r:.4f}  Rf={rf:.4f}  rmsd_e={rmsd_str}')
    bl_json.write_text(json.dumps(res, indent=2))
    return res


# ── Trial enumeration ─────────────────────────────────────────────────────────

def _build_swap_catalog(base_atoms, move_types):
    """All valid single-swap specs (no trial_id).  Each spec is a dict.

    Auto-detects altloc mode (single protein chain with altloc labels)
    vs chain mode (conformers as separate chains).
    """
    protein_chains = set(a['chain'] for a in base_atoms
                         if a['resname'] not in HOH_NAMES)
    altloc_mode = (len(protein_chains) == 1 and
                   any(a['altloc'].strip() for a in base_atoms
                       if a['resname'] not in HOH_NAMES))

    if altloc_mode:
        # Conformers encoded as altloc labels within a single chain.
        confs_at_res = {}   # (seqnum, icode) → set of altloc labels
        resname_at   = {}
        chain_of_res = {}
        for a in base_atoms:
            if a['resname'] in HOH_NAMES or not a['altloc'].strip():
                continue
            rk = (a['seqnum'], a['icode'])
            confs_at_res.setdefault(rk, set()).add(a['altloc'])
            resname_at[rk]  = a['resname']
            chain_of_res[rk] = a['chain']

        residue_keys = sorted(confs_at_res.keys())
        rk_next_of   = {rk: residue_keys[i + 1]
                        for i, rk in enumerate(residue_keys[:-1])}
        ss_pairs = find_ss_pairs(base_atoms) if 'ss' in move_types else []

        catalog = []
        for move in move_types:
            if move == 'ss':
                for rk_i, rk_j in ss_pairs:
                    alts  = sorted(confs_at_res.get(rk_i, set())
                                   & confs_at_res.get(rk_j, set()))
                    chain = chain_of_res.get(rk_i, 'A')
                    for altloc_s, altloc_t in combinations(alts, 2):
                        catalog.append({
                            'move_type': 'ss',
                            'altloc_s': altloc_s, 'altloc_t': altloc_t, 'chain': chain,
                            'rk': list(rk_i), 'rk_next': None, 'rk_partner': list(rk_j),
                            'resname': resname_at.get(rk_i, 'CYS'),
                        })
                continue
            for rk in residue_keys:
                alts = sorted(confs_at_res[rk])
                if len(alts) < 2:
                    continue
                rk_next = rk_next_of.get(rk)
                if move == 'pep' and rk_next is None:
                    continue
                chain = chain_of_res.get(rk, 'A')
                for altloc_s, altloc_t in combinations(alts, 2):
                    catalog.append({
                        'move_type': move,
                        'altloc_s': altloc_s, 'altloc_t': altloc_t, 'chain': chain,
                        'rk': list(rk), 'rk_next': list(rk_next) if rk_next else None,
                        'rk_partner': None, 'resname': resname_at[rk],
                    })
        return catalog

    else:
        # Conformers encoded as separate chains (original behaviour).
        chains_at_res = {}
        resname_at    = {}
        for a in base_atoms:
            if a['resname'] in HOH_NAMES:
                continue
            rk = (a['seqnum'], a['icode'])
            chains_at_res.setdefault(rk, set()).add(a['chain'])
            resname_at[rk] = a['resname']

        residue_keys = sorted(chains_at_res.keys())
        rk_next_of   = {rk: residue_keys[i + 1]
                        for i, rk in enumerate(residue_keys[:-1])}
        ss_pairs = find_ss_pairs(base_atoms) if 'ss' in move_types else []

        catalog = []
        for move in move_types:
            if move == 'ss':
                for rk_i, rk_j in ss_pairs:
                    chains = sorted(chains_at_res.get(rk_i, set())
                                    & chains_at_res.get(rk_j, set()))
                    for chain_s, chain_t in combinations(chains, 2):
                        catalog.append({
                            'move_type': 'ss', 'chain_s': chain_s, 'chain_t': chain_t,
                            'rk': list(rk_i), 'rk_next': None, 'rk_partner': list(rk_j),
                            'resname': resname_at.get(rk_i, 'CYS'),
                        })
                continue
            for rk in residue_keys:
                chains = sorted(chains_at_res[rk])
                if len(chains) < 2:
                    continue
                rk_next = rk_next_of.get(rk)
                if move == 'pep' and rk_next is None:
                    continue
                for chain_s, chain_t in combinations(chains, 2):
                    catalog.append({
                        'move_type': move, 'chain_s': chain_s, 'chain_t': chain_t,
                        'rk': list(rk), 'rk_next': list(rk_next) if rk_next else None,
                        'rk_partner': None, 'resname': resname_at[rk],
                    })
        return catalog


def enumerate_trials(base_atoms, move_types, spr=1.0, n_trials=1000, seed=42):
    """Random multi-swap trials with spr (swaps-per-residue) controlling density.

    spr < 1  → each trial covers spr×N_res residues (random subset), 1 swap each.
    spr >= 1 → each trial covers all residues, floor(spr) or ceil(spr) swaps each
               (randomly rounding to match the exact target count per residue).

    Multiple swaps at the same residue use distinct chain-pair draws.
    """
    catalog_by_res = {}
    for s in _build_swap_catalog(base_atoms, move_types):
        rk = tuple(s['rk'])
        catalog_by_res.setdefault(rk, []).append(s)

    residue_keys = sorted(catalog_by_res.keys())
    N_res        = len(residue_keys)
    rng          = np.random.default_rng(seed)

    # Compute per-residue swap count (fractional → stochastic rounding)
    k_lo  = int(spr)
    k_hi  = k_lo + 1
    frac  = spr - k_lo          # probability of getting k_hi instead of k_lo

    total_est = round(spr * N_res) if spr < 1 else round(spr * N_res)
    print(f'  spr={spr}  N_res={N_res}  ~{total_est} swaps/trial  {n_trials} trials')

    trials = [{'trial_id': 0, 'swaps': []}]   # trial 0 = no-swap baseline

    for tid in range(1, n_trials + 1):
        swaps = []
        if spr < 1.0:
            # Cover a random fraction of residues, 1 swap each
            n_cover = max(1, round(spr * N_res))
            chosen_res = rng.choice(N_res, size=n_cover, replace=False)
            for ri in chosen_res:
                rk  = residue_keys[ri]
                cat = catalog_by_res[rk]
                ci  = rng.integers(len(cat))
                swaps.append(cat[ci])
        else:
            for rk in residue_keys:
                cat     = catalog_by_res[rk]
                k       = k_hi if rng.random() < frac else k_lo
                k       = max(1, min(k, len(cat)))
                idxs    = rng.choice(len(cat), size=k, replace=False)
                for ci in idxs:
                    swaps.append(cat[ci])
        trials.append({'trial_id': tid, 'swaps': swaps})
    return trials


def enumerate_trials_exhaust(base_atoms, move_types):
    """One trial per unique single swap — exhaustive enumeration of the catalog."""
    catalog = _build_swap_catalog(base_atoms, move_types)
    print(f'  {len(catalog)} unique single-swap trials (exhaustive)')
    trials = [{'trial_id': 0, 'swaps': []}]   # trial 0 = no-swap baseline
    for i, sw in enumerate(catalog, 1):
        trials.append({'trial_id': i, 'swaps': [sw]})
    return trials


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(trial, base_atoms, header_lines, atom_idx,
              fobs_mtz, truth_mtz, outdir, ncyc=50, weight=0.5,
              no_molprobify=False):
    tid   = trial['trial_id']
    tdir  = outdir / f'trial_{tid:05d}'
    tdir.mkdir(parents=True, exist_ok=True)
    rjson    = tdir / 'result.json'
    out_mtz  = tdir / 'refmacout.mtz'
    out_pdb  = tdir / 'refmacout.pdb'
    if rjson.exists() and out_mtz.exists():
        return json.loads(rjson.read_text())

    swapped, ok = make_swap_pdb(base_atoms, atom_idx, trial['swaps'])

    if not ok:
        res = dict(trial, r=None, rf=None, cc=None, r_true=None, status='no_atoms')
        rjson.write_text(json.dumps(res))
        return res

    with tempfile.TemporaryDirectory(prefix=f'swap_{tid}_') as td:
        td = Path(td)
        swap_pdb = td / 'swap.pdb'
        write_atoms(swapped, header_lines, swap_pdb)

        r, rf, log, mtz_out, pdb_out = run_refmac_quick(
            swap_pdb, fobs_mtz, ncyc, weight, td)

        rmsd_e, r_true = (None, None)
        if mtz_out and mtz_out.exists() and truth_mtz:
            rmsd_e, r_true = compute_metrics(mtz_out, truth_mtz)

        wE = None
        if not no_molprobify and pdb_out and pdb_out.exists():
            wE = run_molprobity(pdb_out, td)

        if mtz_out and mtz_out.exists():
            shutil.copy2(mtz_out, out_mtz)
        if pdb_out and pdb_out.exists():
            shutil.copy2(pdb_out, out_pdb)

    res = dict(trial, r=r, rf=rf, rmsd_e=rmsd_e, r_true=r_true, wE=wE,
               status='ok' if r is not None else 'refmac_failed')
    rjson.write_text(json.dumps(res))
    return res


# ── Collation ─────────────────────────────────────────────────────────────────

def collate(outdir):
    # Trial 0 is the no-swap baseline; fall back to baseline/ for old-format runs
    bl_json = outdir / 'trial_00000' / 'result.json'
    if not bl_json.exists():
        bl_json = outdir / 'baseline' / 'result.json'
    bl      = json.loads(bl_json.read_text())
    bl_r    = bl.get('r')
    bl_rf   = bl.get('rf')
    bl_rmsd = bl.get('rmsd_e')
    bl_wE   = bl.get('wE')
    rmsd_str = f'{bl_rmsd:.4f}' if bl_rmsd is not None else 'N/A'
    wE_str   = f'{bl_wE:.3f}' if bl_wE is not None else 'N/A'
    print(f'Baseline (no-swap): R={bl_r:.4f}  Rf={bl_rf:.4f}  rmsd_e={rmsd_str}  wE={wE_str}')

    trials_json = outdir / 'trials.json'
    trials = json.loads(trials_json.read_text())

    results = []
    n_miss  = 0
    for t in trials:
        if t['trial_id'] == 0:
            continue   # no-swap baseline — not a candidate
        rj = outdir / f'trial_{t["trial_id"]:05d}' / 'result.json'
        if not rj.exists():
            n_miss += 1
            continue
        res = json.loads(rj.read_text())
        if res.get('status') == 'no_atoms' or res.get('rf') is None:
            continue
        res['delta_rf']    = res['rf'] - bl_rf if bl_rf else None
        res['delta_rmsd_e'] = (res['rmsd_e'] - bl_rmsd) if (res.get('rmsd_e') is not None and bl_rmsd is not None) else None
        res['delta_wE']    = (res['wE'] - bl_wE) if (res.get('wE') is not None and bl_wE is not None) else None
        results.append(res)

    if n_miss:
        print(f'  ({n_miss} trials missing)')

    results.sort(key=lambda x: x.get('delta_rf') or 0)

    summary = {'baseline': bl, 'n_trials': len(results),
               'n_missing': n_miss, 'results': results}
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))

    def _swap_label(swaps):
        if swaps and 'altloc_s' in swaps[0]:
            parts = [f'{s["move_type"]} {s["altloc_s"]}→{s["altloc_t"]} '
                     f'{s["rk"][0]}{s["resname"]}' for s in swaps[:4]]
        else:
            parts = [f'{s["move_type"]} {s["chain_s"]}→{s["chain_t"]} '
                     f'{s["rk"][0]}{s["resname"]}' for s in swaps[:4]]
        suffix = f' +{len(swaps)-4}more' if len(swaps) > 4 else ''
        return '  '.join(parts) + suffix

    def _print_table(title, rows):
        hdr = f'{"tid":>7}  {"R":>7}  {"Rf":>7}  {"ΔRf":>7}  {"rmsd_e":>7}  {"Δrmsd_e":>8}  {"wE":>7}  {"ΔwE":>7}'
        print(f'\n{title}')
        print(hdr)
        print('-' * len(hdr))
        for res in rows[:20]:
            label = _swap_label(res.get('swaps', []))
            wE_val  = res.get('wE')
            dwE_val = res.get('delta_wE')
            wE_s  = f'{wE_val:7.3f}' if wE_val is not None else '    N/A'
            dwE_s = f'{dwE_val:+7.3f}' if dwE_val is not None else '    N/A'
            print(f'{res["trial_id"]:>7}  '
                  f'{res.get("r") or 0:7.4f}  {res.get("rf") or 0:7.4f}  '
                  f'{res.get("delta_rf") or 0:+7.4f}  '
                  f'{res.get("rmsd_e") or 0:7.4f}  '
                  f'{res.get("delta_rmsd_e") or 0:+8.4f}  '
                  f'{wE_s}  {dwE_s}  {label}')

    _print_table('Top 20 by ΔRf:', results[:20])

    by_wE = [r for r in results if r.get('wE') is not None]
    by_wE.sort(key=lambda x: x['wE'])
    _print_table('Top 20 by wE:', by_wE)

    find_compatible_combos(by_wE)

    print(f'\nSummary → {outdir}/summary.json')


def find_compatible_combos(by_wE, top_n=20, max_gap=1):
    """Find non-overlapping swap combinations among top-wE trials.

    Two trials are compatible if no residue in one is within max_gap
    sequence positions of any residue in the other.  Prints the best
    compatible pairs and triples ranked by combined ΔwE.
    """
    top = [r for r in by_wE if r.get('swaps') and r.get('delta_wE') is not None][:top_n]
    if len(top) < 2:
        return

    # Only show combos if trials are small enough to be interpretable
    max_swaps = max(len(r['swaps']) for r in top)
    if max_swaps > 8:
        print(f'\n(Compatible-combo search skipped: trials have up to {max_swaps} swaps)')
        return

    def res_set(trial):
        return set(s['rk'][0] for s in trial['swaps'])

    def compatible(r1, r2):
        for a in res_set(r1):
            for b in res_set(r2):
                if abs(a - b) <= max_gap:
                    return False
        return True

    def short_label(trial):
        return '+'.join(f'{s["resname"]}{s["rk"][0]}{s["move_type"]}'
                        for s in trial['swaps'])

    pairs = []
    for i, t1 in enumerate(top):
        for t2 in top[i+1:]:
            if compatible(t1, t2):
                pairs.append((t1['delta_wE'] + t2['delta_wE'], t1, t2))
    pairs.sort(key=lambda x: x[0])

    triples = []
    for i, t1 in enumerate(top):
        for j, t2 in enumerate(top[i+1:], i+1):
            if not compatible(t1, t2):
                continue
            for t3 in top[j+1:]:
                if compatible(t1, t3) and compatible(t2, t3):
                    triples.append((
                        t1['delta_wE'] + t2['delta_wE'] + t3['delta_wE'],
                        t1, t2, t3))
    triples.sort(key=lambda x: x[0])

    if pairs:
        print(f'\nCompatible pairs (top 5 by combined ΔwE):')
        for dwE, t1, t2 in pairs[:5]:
            print(f'  t{t1["trial_id"]:03d}+t{t2["trial_id"]:03d}  ΔwE≈{dwE:+.3f}'
                  f'  [{short_label(t1)}] + [{short_label(t2)}]')

    if triples:
        print(f'\nCompatible triples (top 3 by combined ΔwE):')
        for dwE, t1, t2, t3 in triples[:3]:
            print(f'  t{t1["trial_id"]:03d}+t{t2["trial_id"]:03d}+t{t3["trial_id"]:03d}'
                  f'  ΔwE≈{dwE:+.3f}')
            print(f'    [{short_label(t1)}]')
            print(f'    [{short_label(t2)}]')
            print(f'    [{short_label(t3)}]')


# ── SLURM submission ──────────────────────────────────────────────────────────

def submit(outdir, base_pdb, fobs_mtz, truth_mtz, move_types,
           ncyc, weight, partition, account, qos,
           spr=1.0, n_trials=1000, seed=42, exhaust=False,
           no_molprobify=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Enumerate trials (trial 0 = no-swap, serves as baseline)
    print('Parsing base PDB...')
    base_atoms, header_lines = parse_atoms(base_pdb)
    print(f'  {len(base_atoms)} atoms, '
          f'{len(set(a["chain"] for a in base_atoms))} chains')

    if exhaust:
        print(f'Enumerating trials (move types: {move_types}, exhaustive single-swap)...')
        trials = enumerate_trials_exhaust(base_atoms, move_types)
    else:
        print(f'Enumerating trials (move types: {move_types}, spr={spr}, {n_trials} trials)...')
        trials = enumerate_trials(base_atoms, move_types, spr=spr,
                                  n_trials=n_trials, seed=seed)
    print(f'  {len(trials)} trials')
    (outdir / 'trials.json').write_text(json.dumps(trials, indent=2))

    cfg = {'base_pdb': str(Path(base_pdb).resolve()),
           'fobs_mtz': str(Path(fobs_mtz).resolve()),
           'truth_mtz': str(Path(truth_mtz).resolve()) if truth_mtz else None,
           'ncyc': ncyc, 'weight': weight, 'spr': spr, 'exhaust': exhaust,
           'no_molprobify': no_molprobify}
    (outdir / 'config.json').write_text(json.dumps(cfg, indent=2))

    script     = Path(__file__).resolve()
    outdir_abs = outdir.resolve()
    n_batches  = (len(trials) + MAX_ARRAY - 1) // MAX_ARRAY

    for b in range(n_batches):
        start      = b * MAX_ARRAY
        end        = min(start + MAX_ARRAY - 1, len(trials) - 1)
        batch_size = end - start + 1
        sh    = outdir / f'_batch{b}.sh'
        log   = outdir_abs / f'slurm_b{b}_%a.out'
        lines = ['#!/bin/bash',
                 f'#SBATCH --job-name=swapscan',
                 f'#SBATCH --partition={partition}',
                 f'#SBATCH --ntasks=1',
                 f'#SBATCH --array=0-{batch_size - 1}',
                 f'#SBATCH --output={log}',
                 '#SBATCH --export=ALL']
        if account:
            lines.append(f'#SBATCH --account={account}')
        if qos:
            lines.append(f'#SBATCH --qos={qos}')
        no_mp_flag = ' --no-molprobify' if no_molprobify else ''
        lines += ['mkdir -p "${CCP4_SCR:-/tmp}"',
                  f'cd {SCRIPT_DIR}',
                  f'ccp4-python {script} --task $(( {start} + $SLURM_ARRAY_TASK_ID )) --outdir {outdir_abs}{no_mp_flag}']
        sh.write_text('\n'.join(lines) + '\n')
        r = subprocess.run(['sbatch', str(sh)], capture_output=True, text=True)
        print(f'  Batch {b} (trials {start}–{end}): {r.stdout.strip() or r.stderr.strip()}')


def rescore_trial(outdir, trial_id):
    """Run molprobify on an existing trial's refmacout.pdb and update result.json."""
    tdir  = outdir / f'trial_{trial_id:05d}'
    rjson = tdir / 'result.json'
    pdb   = tdir / 'refmacout.pdb'
    if not rjson.exists() or not pdb.exists():
        return
    res = json.loads(rjson.read_text())
    if res.get('wE') is not None:
        return  # already scored
    wE = run_molprobity(pdb, tdir)
    if wE is not None:
        res['wE'] = wE
        rjson.write_text(json.dumps(res))
        print(f'trial {trial_id:05d}: wE={wE:.3f}')


def rescore_submit(outdir, partition, account, qos):
    """Submit SLURM array to run molprobify on all existing trials in outdir."""
    outdir = Path(outdir).resolve()
    trials = json.loads((outdir / 'trials.json').read_text())
    n      = len(trials)
    script = Path(__file__).resolve()
    n_batches = (n + MAX_ARRAY - 1) // MAX_ARRAY

    for b in range(n_batches):
        start = b * MAX_ARRAY
        end   = min(start + MAX_ARRAY - 1, n - 1)
        sh    = outdir / f'_rescore_batch{b}.sh'
        log   = outdir / f'slurm_rescore_b{b}_%a.out'
        lines = ['#!/bin/bash',
                 '#SBATCH --job-name=rescore',
                 f'#SBATCH --partition={partition}',
                 '#SBATCH --ntasks=1',
                 f'#SBATCH --array={start}-{end}',
                 f'#SBATCH --output={log}',
                 '#SBATCH --export=ALL']
        if account:
            lines.append(f'#SBATCH --account={account}')
        if qos:
            lines.append(f'#SBATCH --qos={qos}')
        lines += ['mkdir -p "${CCP4_SCR:-/tmp}"',
                  f'cd {SCRIPT_DIR}',
                  f'ccp4-python {script} --rescore-task $SLURM_ARRAY_TASK_ID --outdir {outdir}']
        sh.write_text('\n'.join(lines) + '\n')
        r = subprocess.run(['sbatch', str(sh)], capture_output=True, text=True)
        print(f'  Batch {b} (trials {start}–{end}): {r.stdout.strip() or r.stderr.strip()}')


def targeted_submit(outdir, partition, account, qos):
    """Submit a small array of hand-chosen wE-optimised swap combinations.

    Building blocks come from the top group-wE improvers found in
    swapscan_opt5/spr_0p06.  All use varconf_opt6.pdb as the base.

    Groups (non-overlapping residue sets):
      A = t147: {ASP9,VAL10,PHE15,GLN37}  ΔwE=-1.333
      B = t135: {ASN19,CYS26,GLY31,PRO60} ΔwE=-1.317
      C = t116: {ASP3,GLY4,GLU24,ARG62}   ΔwE=-1.313
      D = t239: {CYS26,PRO41,HIS54,GLY59} ΔwE=-1.240  (shares CYS26 with B)
      E = t033: {ASN19,THR27,ARG62,HIS64} ΔwE=-1.227  (shares with B,C)
    """
    outdir   = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base_pdb  = str(SCRIPT_DIR / '1aho' / 'varconf_opt6.pdb')
    fobs_mtz  = str(SCRIPT_DIR / '1aho' / 'refme.mtz')
    truth_mtz = str(SCRIPT_DIR / '1aho' / 'gt48.mtz')
    ncyc, weight = 50, 0.5

    # Exact swap dicts extracted from swapscan_opt5/spr_0p06 result.json files
    # Group A — t147: ASP9, VAL10, PHE15, GLN37
    A = [
        {'move_type': 'o',  'chain_s': 'A', 'chain_t': 'D', 'rk': [15, ''], 'rk_next': [16, ''], 'rk_partner': None, 'resname': 'PHE'},
        {'move_type': 'sc', 'chain_s': 'A', 'chain_t': 'G', 'rk': [37, ''], 'rk_next': [38, ''], 'rk_partner': None, 'resname': 'GLN'},
        {'move_type': 'sc', 'chain_s': 'E', 'chain_t': 'N', 'rk': [10, ''], 'rk_next': [11, ''], 'rk_partner': None, 'resname': 'VAL'},
        {'move_type': 'o',  'chain_s': 'E', 'chain_t': 'L', 'rk': [9,  ''], 'rk_next': [10, ''], 'rk_partner': None, 'resname': 'ASP'},
    ]
    # Group B — t135: GLY31, CYS26, ASN19, PRO60
    B = [
        {'move_type': 'sc',  'chain_s': 'A', 'chain_t': 'B', 'rk': [31, ''], 'rk_next': [32, ''], 'rk_partner': None, 'resname': 'GLY'},
        {'move_type': 'pep', 'chain_s': 'A', 'chain_t': 'B', 'rk': [26, ''], 'rk_next': [27, ''], 'rk_partner': None, 'resname': 'CYS'},
        {'move_type': 'sc',  'chain_s': 'A', 'chain_t': 'C', 'rk': [19, ''], 'rk_next': [20, ''], 'rk_partner': None, 'resname': 'ASN'},
        {'move_type': 'pep', 'chain_s': 'A', 'chain_t': 'H', 'rk': [60, ''], 'rk_next': [61, ''], 'rk_partner': None, 'resname': 'PRO'},
    ]
    # Group C — t116: GLU24, ARG62, GLY4, ASP3
    C = [
        {'move_type': 'o', 'chain_s': 'I', 'chain_t': 'M', 'rk': [24, ''], 'rk_next': [25, ''], 'rk_partner': None, 'resname': 'GLU'},
        {'move_type': 'o', 'chain_s': 'A', 'chain_t': 'E', 'rk': [62, ''], 'rk_next': [63, ''], 'rk_partner': None, 'resname': 'ARG'},
        {'move_type': 'o', 'chain_s': 'A', 'chain_t': 'B', 'rk': [4,  ''], 'rk_next': [5,  ''], 'rk_partner': None, 'resname': 'GLY'},
        {'move_type': 'o', 'chain_s': 'A', 'chain_t': 'E', 'rk': [3,  ''], 'rk_next': [4,  ''], 'rk_partner': None, 'resname': 'ASP'},
    ]
    # Group D — t239: CYS26, GLY59, HIS54, PRO41  (shares CYS26 with B)
    D = [
        {'move_type': 'o',   'chain_s': 'A', 'chain_t': 'C', 'rk': [26, ''], 'rk_next': [27, ''], 'rk_partner': None, 'resname': 'CYS'},
        {'move_type': 'o',   'chain_s': 'B', 'chain_t': 'C', 'rk': [59, ''], 'rk_next': [60, ''], 'rk_partner': None, 'resname': 'GLY'},
        {'move_type': 'sc',  'chain_s': 'A', 'chain_t': 'C', 'rk': [54, ''], 'rk_next': [55, ''], 'rk_partner': None, 'resname': 'HIS'},
        {'move_type': 'pep', 'chain_s': 'J', 'chain_t': 'M', 'rk': [41, ''], 'rk_next': [42, ''], 'rk_partner': None, 'resname': 'PRO'},
    ]
    # Group E — t033: THR27, HIS64, ARG62, ASN19  (shares with B,C)
    E = [
        {'move_type': 'o', 'chain_s': 'C', 'chain_t': 'D', 'rk': [27, ''], 'rk_next': [28, ''], 'rk_partner': None, 'resname': 'THR'},
        {'move_type': 'o', 'chain_s': 'G', 'chain_t': 'O', 'rk': [64, ''], 'rk_next': None,     'rk_partner': None, 'resname': 'HIS'},
        {'move_type': 'o', 'chain_s': 'H', 'chain_t': 'M', 'rk': [62, ''], 'rk_next': [63, ''], 'rk_partner': None, 'resname': 'ARG'},
        {'move_type': 'sc', 'chain_s': 'B', 'chain_t': 'C', 'rk': [19, ''], 'rk_next': [20, ''], 'rk_partner': None, 'resname': 'ASN'},
    ]

    # Combinations: label → swap list
    # A+B+C are fully non-overlapping: {9,10,15,37} | {19,26,31,60} | {3,4,24,62}
    combos = [
        ('baseline',    []),
        ('A',           A),
        ('B',           B),
        ('C',           C),
        ('D',           D),
        ('A+B',         A + B),
        ('A+C',         A + C),
        ('B+C',         B + C),
        ('A+D',         A + D),
        ('A+B+C',       A + B + C),
        ('A+B+D',       A + B + D),
        ('A+C+D',       A + C + D),
        ('A+B+C+D',     A + B + C + D),
    ]

    trials = [{'trial_id': i, 'label': label, 'swaps': swaps}
              for i, (label, swaps) in enumerate(combos)]

    cfg = {'base_pdb': base_pdb, 'fobs_mtz': fobs_mtz, 'truth_mtz': truth_mtz,
           'ncyc': ncyc, 'weight': weight, 'targeted': True}
    (outdir / 'config.json').write_text(json.dumps(cfg, indent=2))
    (outdir / 'trials.json').write_text(json.dumps(trials, indent=2))

    print(f'Targeted trials: {len(trials)} combos → {outdir}')
    for t in trials:
        res_tags = '+'.join(f'{s["resname"]}{s["rk"][0]}{s["move_type"]}'
                            for s in t['swaps']) or '(baseline)'
        print(f'  {t["trial_id"]:3d}  {t["label"]:<30}  {res_tags}')

    script     = Path(__file__).resolve()
    n          = len(trials)
    sh         = outdir / '_targeted_batch.sh'
    log        = outdir / 'slurm_%a.out'
    lines = ['#!/bin/bash',
             '#SBATCH --job-name=targeted',
             f'#SBATCH --partition={partition}',
             '#SBATCH --ntasks=1',
             f'#SBATCH --array=0-{n-1}',
             f'#SBATCH --output={log}',
             '#SBATCH --export=ALL']
    if account:
        lines.append(f'#SBATCH --account={account}')
    if qos:
        lines.append(f'#SBATCH --qos={qos}')
    lines += ['mkdir -p "${CCP4_SCR:-/tmp}"',
              f'cd {SCRIPT_DIR}',
              f'ccp4-python {script} --task $SLURM_ARRAY_TASK_ID --outdir {outdir}']
    sh.write_text('\n'.join(lines) + '\n')
    r = subprocess.run(['sbatch', str(sh)], capture_output=True, text=True)
    print(f'Submitted: {r.stdout.strip() or r.stderr.strip()}')


def submit_sweep(outdir, base_pdb, fobs_mtz, truth_mtz, move_types,
                 ncyc, weight, partition, account, qos,
                 spr_values, n_trials=1000, seed=42):
    """Submit one swapscan per spr value into outdir/spr_{val}/."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for spr in spr_values:
        tag    = f'spr_{spr:.2f}'.replace('.', 'p')
        subdir = outdir / tag
        print(f'\n── spr={spr}  →  {subdir} ──')
        submit(subdir, base_pdb, fobs_mtz, truth_mtz, move_types,
               ncyc, weight, partition, account, qos,
               spr=spr, n_trials=n_trials, seed=seed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pdb',         help='Base varconf PDB')
    ap.add_argument('--fobs',        help='Fobs MTZ (refme.mtz)')
    ap.add_argument('--truth',       help='Ground-truth MTZ (gt48.mtz)', default=None)
    ap.add_argument('--outdir',      required=True)
    ap.add_argument('--move-types',  default='sc,pep,o,ss',
                    help='Comma-separated list of move types: sc,pep,o,ss')
    ap.add_argument('--ncyc',        type=int,   default=50)
    ap.add_argument('--weight',      type=float, default=0.5)
    ap.add_argument('--partition',   default='lr6')
    ap.add_argument('--account',     default=None)
    ap.add_argument('--qos',         default=None)
    ap.add_argument('--spr',         type=float, default=1.0,
                    help='Swaps per residue per trial (default 1.0)')
    ap.add_argument('--sweep-spr',   default=None,
                    help='Comma-separated spr values to sweep, e.g. 0.5,1,2,5,10,20,50')
    ap.add_argument('--n-trials',    type=int, default=1000,
                    help='Number of random trials per spr value (default 1000)')
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--exhaust',        action='store_true',
                    help='Enumerate every unique single swap (one trial per catalog entry)')
    ap.add_argument('--no-molprobify', action='store_true',
                    help='Skip molprobify (wE=None); ~10x faster. Use --rescore-submit afterwards on top candidates.')
    ap.add_argument('--submit',        action='store_true')
    ap.add_argument('--task',          type=int, default=None)
    ap.add_argument('--collate',       action='store_true')
    ap.add_argument('--rescore-task',  type=int, default=None,
                    help='Run molprobify on trial N (called by SLURM)')
    ap.add_argument('--rescore-submit', action='store_true',
                    help='Submit SLURM array to rescore all trials in --outdir')
    ap.add_argument('--targeted-submit', action='store_true',
                    help='Submit hand-chosen wE-optimised swap combinations')
    args = ap.parse_args()

    outdir = Path(args.outdir)

    if args.collate:
        subdirs = sorted(outdir.glob('spr_*/'))
        if subdirs:
            for sd in subdirs:
                print(f'\n{"="*60}\n{sd.name}\n{"="*60}')
                collate(sd)
        else:
            collate(outdir)
        return

    if args.rescore_task is not None:
        rescore_trial(outdir, args.rescore_task)
        return

    if args.rescore_submit:
        # Accept a glob of spr_* subdirs if outdir is a sweep parent
        subdirs = sorted(outdir.glob('spr_*/'))
        targets = subdirs if subdirs else [outdir]
        for td in targets:
            print(f'\n── rescoring {td} ──')
            rescore_submit(td, args.partition, args.account, args.qos)
        return

    if args.targeted_submit:
        targeted_submit(outdir, args.partition, args.account, args.qos)
        return

    if args.task is not None:
        cfg    = json.loads((outdir / 'config.json').read_text())
        trials = json.loads((outdir / 'trials.json').read_text())
        trial  = trials[args.task]

        base_atoms, header_lines = parse_atoms(cfg['base_pdb'])
        idx = atom_index(base_atoms)

        res = run_trial(trial, base_atoms, header_lines, idx,
                        cfg['fobs_mtz'], cfg.get('truth_mtz'),
                        outdir, ncyc=cfg['ncyc'], weight=cfg['weight'],
                        no_molprobify=cfg.get('no_molprobify', False))
        n_sw = len(trial.get('swaps', []))
        print(f'trial {args.task}: {n_sw} swaps  '
              f'R={res.get("r")}  Rf={res.get("rf")}  rmsd_e={res.get("rmsd_e")}  wE={res.get("wE")}')
        return

    if args.submit:
        if not args.pdb or not args.fobs:
            ap.error('--submit requires --pdb and --fobs')
        move_types = [m.strip() for m in args.move_types.split(',')]
        kw = dict(base_pdb       = Path(args.pdb).resolve(),
                  fobs_mtz       = Path(args.fobs).resolve(),
                  truth_mtz      = Path(args.truth).resolve() if args.truth else None,
                  move_types     = move_types,
                  ncyc           = args.ncyc,
                  weight         = args.weight,
                  partition      = args.partition,
                  account        = args.account,
                  qos            = args.qos,
                  n_trials       = args.n_trials,
                  seed           = args.seed,
                  no_molprobify  = args.no_molprobify)
        if args.sweep_spr:
            spr_values = [float(v) for v in args.sweep_spr.split(',')]
            submit_sweep(outdir, spr_values=spr_values, **kw)
        elif args.exhaust:
            submit(outdir, exhaust=True, **kw)
        else:
            submit(outdir, spr=args.spr, **kw)
        return

    ap.print_help()


if __name__ == '__main__':
    main()
