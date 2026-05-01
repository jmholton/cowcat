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
    """Return dict (chain, seqnum, icode, aname) → list-of-indices."""
    idx = {}
    for i, a in enumerate(atom_recs):
        key = (a['chain'], a['seqnum'], a['icode'], a['aname'])
        idx.setdefault(key, []).append(i)
    return idx


# ── Disulfide pair detection ─────────────────────────────────────────────────

def find_ss_pairs(base_atoms, ref_chain='A', sg_cutoff=2.5):
    """Return list of (rk_i, rk_j) CYS pairs with SG-SG < sg_cutoff in ref_chain."""
    sg_pos = {}
    for a in base_atoms:
        if a['chain'] != ref_chain or a['resname'] != 'CYS' or a['aname'] != 'SG':
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
    chain_s    = sw['chain_s']
    chain_t    = sw['chain_t']
    move_type  = sw['move_type']
    seqnum     = sw['rk'][0]
    icode      = sw['rk'][1]
    rk_next    = sw.get('rk_next')
    rk_partner = sw.get('rk_partner')

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
            c, sn2, ic2, aname = aname_key
            if c != chain_s or sn2 != sn or ic2 != ic:
                continue
            if atom_set is None:
                if aname in SC_BACKBONE_KEEP:
                    continue
            elif aname not in atom_set:
                continue
            s_idxs = idx.get((chain_s, sn, ic, aname), [])
            t_idxs = idx.get((chain_t, sn, ic, aname), [])
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
    """All valid single-swap specs (no trial_id).  Each spec is a dict."""
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


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(trial, base_atoms, header_lines, atom_idx,
              fobs_mtz, truth_mtz, outdir, ncyc=50, weight=0.5):
    tid   = trial['trial_id']
    tdir  = outdir / f'trial_{tid:05d}'
    tdir.mkdir(parents=True, exist_ok=True)
    rjson = tdir / 'result.json'
    if rjson.exists():
        return json.loads(rjson.read_text())

    swapped, ok = make_swap_pdb(base_atoms, atom_idx, trial['swaps'])

    if not ok:
        res = dict(trial, r=None, rf=None, cc=None, r_true=None, status='no_atoms')
        rjson.write_text(json.dumps(res))
        return res

    swap_pdb = tdir / 'swap.pdb'
    write_atoms(swapped, header_lines, swap_pdb)

    out_mtz = tdir / 'refmacout.mtz'
    out_pdb = tdir / 'refmacout.pdb'
    with tempfile.TemporaryDirectory(prefix=f'swap_{tid}_') as td:
        r, rf, log, mtz_out, pdb_out = run_refmac_quick(
            swap_pdb, fobs_mtz, ncyc, weight, Path(td))
        if pdb_out and pdb_out.exists():
            shutil.copy2(pdb_out, out_pdb)
        if mtz_out and mtz_out.exists():
            shutil.copy2(mtz_out, out_mtz)
        (tdir / 'refmac.log').write_text(log or '')

    rmsd_e, r_true = (None, None)
    if out_mtz.exists() and truth_mtz:
        rmsd_e, r_true = compute_metrics(out_mtz, truth_mtz)

    res = dict(trial, r=r, rf=rf, rmsd_e=rmsd_e, r_true=r_true,
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
    rmsd_str = f'{bl_rmsd:.4f}' if bl_rmsd is not None else 'N/A'
    print(f'Baseline (no-swap): R={bl_r:.4f}  Rf={bl_rf:.4f}  rmsd_e={rmsd_str}')

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
        results.append(res)

    if n_miss:
        print(f'  ({n_miss} trials missing)')

    results.sort(key=lambda x: x.get('delta_rf') or 0)

    summary = {'baseline': bl, 'n_trials': len(results),
               'n_missing': n_miss, 'results': results}
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))

    def _swap_label(swaps):
        parts = [f'{s["move_type"]} {s["chain_s"]}→{s["chain_t"]} '
                 f'{s["rk"][0]}{s["resname"]}' for s in swaps[:4]]
        suffix = f' +{len(swaps)-4}more' if len(swaps) > 4 else ''
        return '  '.join(parts) + suffix

    hdr = f'{"tid":>7}  {"R":>7}  {"Rf":>7}  {"ΔRf":>7}  {"rmsd_e":>7}  {"Δrmsd_e":>8}'
    print(f'\nTop 20 by ΔRf:')
    print(hdr)
    print('-' * len(hdr))
    for res in results[:20]:
        label = _swap_label(res.get('swaps', []))
        print(f'{res["trial_id"]:>7}  '
              f'{res.get("r") or 0:7.4f}  {res.get("rf") or 0:7.4f}  '
              f'{res.get("delta_rf") or 0:+7.4f}  '
              f'{res.get("rmsd_e") or 0:7.4f}  '
              f'{res.get("delta_rmsd_e") or 0:+8.4f}  {label}')
    print(f'\nSummary → {outdir}/summary.json')


# ── SLURM submission ──────────────────────────────────────────────────────────

def submit(outdir, base_pdb, fobs_mtz, truth_mtz, move_types,
           ncyc, weight, partition, account, qos,
           spr=1.0, n_trials=1000, seed=42):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Enumerate trials (trial 0 = no-swap, serves as baseline)
    print('Parsing base PDB...')
    base_atoms, header_lines = parse_atoms(base_pdb)
    print(f'  {len(base_atoms)} atoms, '
          f'{len(set(a["chain"] for a in base_atoms))} chains')

    print(f'Enumerating trials (move types: {move_types}, spr={spr}, {n_trials} trials)...')
    trials = enumerate_trials(base_atoms, move_types, spr=spr,
                              n_trials=n_trials, seed=seed)
    print(f'  {len(trials)} trials')
    (outdir / 'trials.json').write_text(json.dumps(trials, indent=2))

    cfg = {'base_pdb': str(Path(base_pdb).resolve()),
           'fobs_mtz': str(Path(fobs_mtz).resolve()),
           'truth_mtz': str(Path(truth_mtz).resolve()) if truth_mtz else None,
           'ncyc': ncyc, 'weight': weight, 'spr': spr}
    (outdir / 'config.json').write_text(json.dumps(cfg, indent=2))

    script     = Path(__file__).resolve()
    outdir_abs = outdir.resolve()
    n_batches  = (len(trials) + MAX_ARRAY - 1) // MAX_ARRAY

    for b in range(n_batches):
        start = b * MAX_ARRAY
        end   = min(start + MAX_ARRAY - 1, len(trials) - 1)
        sh    = outdir / f'_batch{b}.sh'
        log   = outdir_abs / f'slurm_b{b}_%a.out'
        lines = ['#!/bin/bash',
                 f'#SBATCH --job-name=swapscan',
                 f'#SBATCH --partition={partition}',
                 f'#SBATCH --ntasks=1',
                 f'#SBATCH --array={start}-{end}',
                 f'#SBATCH --output={log}',
                 '#SBATCH --export=ALL']
        if account:
            lines.append(f'#SBATCH --account={account}')
        if qos:
            lines.append(f'#SBATCH --qos={qos}')
        lines += [f'cd {SCRIPT_DIR}',
                  f'ccp4-python {script} --task $SLURM_ARRAY_TASK_ID --outdir {outdir_abs}']
        sh.write_text('\n'.join(lines) + '\n')
        r = subprocess.run(['sbatch', str(sh)], capture_output=True, text=True)
        print(f'  Batch {b} (trials {start}–{end}): {r.stdout.strip() or r.stderr.strip()}')


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
    ap.add_argument('--partition',   default='refmac')
    ap.add_argument('--account',     default=None)
    ap.add_argument('--qos',         default=None)
    ap.add_argument('--spr',         type=float, default=1.0,
                    help='Swaps per residue per trial (default 1.0)')
    ap.add_argument('--sweep-spr',   default=None,
                    help='Comma-separated spr values to sweep, e.g. 0.5,1,2,5,10,20,50')
    ap.add_argument('--n-trials',    type=int, default=1000,
                    help='Number of random trials per spr value (default 1000)')
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--submit',      action='store_true')
    ap.add_argument('--task',        type=int, default=None)
    ap.add_argument('--collate',     action='store_true')
    args = ap.parse_args()

    outdir = Path(args.outdir)

    if args.collate:
        collate(outdir)
        return

    if args.task is not None:
        cfg    = json.loads((outdir / 'config.json').read_text())
        trials = json.loads((outdir / 'trials.json').read_text())
        trial  = trials[args.task]

        base_atoms, header_lines = parse_atoms(cfg['base_pdb'])
        idx = atom_index(base_atoms)

        res = run_trial(trial, base_atoms, header_lines, idx,
                        cfg['fobs_mtz'], cfg.get('truth_mtz'),
                        outdir, ncyc=cfg['ncyc'], weight=cfg['weight'])
        n_sw = len(trial.get('swaps', []))
        print(f'trial {args.task}: {n_sw} swaps  '
              f'R={res.get("r")}  Rf={res.get("rf")}  rmsd_e={res.get("rmsd_e")}')
        return

    if args.submit:
        if not args.pdb or not args.fobs:
            ap.error('--submit requires --pdb and --fobs')
        move_types = [m.strip() for m in args.move_types.split(',')]
        kw = dict(base_pdb   = Path(args.pdb).resolve(),
                  fobs_mtz   = Path(args.fobs).resolve(),
                  truth_mtz  = Path(args.truth).resolve() if args.truth else None,
                  move_types = move_types,
                  ncyc       = args.ncyc,
                  weight     = args.weight,
                  partition  = args.partition,
                  account    = args.account,
                  qos        = args.qos,
                  n_trials   = args.n_trials,
                  seed       = args.seed)
        if args.sweep_spr:
            spr_values = [float(v) for v in args.sweep_spr.split(',')]
            submit_sweep(outdir, spr_values=spr_values, **kw)
        else:
            submit(outdir, spr=args.spr, **kw)
        return

    ap.print_help()


if __name__ == '__main__':
    main()
