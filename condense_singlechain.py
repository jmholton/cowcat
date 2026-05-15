#!/usr/bin/env ccp4-python
"""
condense_singlechain.py — Per-residue altloc maximin on a single-chain altloc
PDB (e.g. deconform output). Avoids the multichain "union-template" trick that
caused duplicate-position altlocs.

For each residue, takes its existing altlocs (positions across the model),
applies per-residue maximin to pick k_target of them, and writes a new
single-chain altloc PDB with selected altlocs renormalized to sum=1.

Then runs the standard refmac weight-snap schedule.
"""
import json
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import gemmi

import explore_1aho_fusion as ef
import explore_condensation as ec
from explore_1aho_fusion import (
    DMIN, MAINCHAIN_ATOMS, run, _maximin_select,
)
from explore_condensation import run_weightsnap
from condense_bb import build_fobs_calc_only

# Use refmac5-newhess
ec.REFMAC5 = Path('/programs/ccp4-8.0/bin/refmac5-newhess')

REOCCUPY    = Path('/home/jamesh/Develop/reoccupy.awk')
OCCSETUP    = Path('/home/jamesh/Develop/refmac_occupancy_setup.com')

THRESHOLD_SETS = {
    'default':   [(0.6, 2), (0.8, 4), (1.0, 6), (1.5, 8), (2.5, 12), (99, 16)],
    'lean':      [(0.6, 1), (0.8, 2), (1.0, 4), (1.5, 6), (2.5, 8),  (99, 12)],
    'ultralean': [(0.6, 1), (0.8, 1), (1.0, 2), (1.5, 4), (2.5, 6),  (99, 8)],
    'midrich':   [(0.6, 3), (0.8, 5), (1.0, 7), (1.5, 10),(2.5, 14), (99, 20)],
    'rich':      [(0.6, 4), (0.8, 6), (1.0, 8), (1.5, 12),(2.5, 16), (99, 24)],
    # bottom k=1 for rigid residues, scaling up to handle very flexible loops
    'floor1':    [(0.4, 1), (0.6, 2), (0.8, 3), (1.2, 5), (2.0, 8),  (99, 12)],
    'floor1lean':[(0.5, 1), (0.8, 2), (1.2, 3), (1.8, 5), (2.5, 7),  (99, 10)],
    # same upper schedule as floor1 but bottoms at k=2 instead of k=1
    'floor2':    [(0.6, 2), (0.8, 3), (1.2, 5), (2.0, 8),  (99, 12)],
}

def make_dev_to_nconf(threshold_set):
    table = THRESHOLD_SETS[threshold_set]
    def f(dev):
        for lim, n in table:
            if dev < lim:
                return n
        return table[-1][1]
    return f


# ── Backbone+SS strip on single-chain-altloc PDB ───────────────────────────────
def strip_to_backbone_singlechain(in_pdb, out_pdb, keep_disulfides=True):
    """Same atom selection as condense_bb.strip_to_backbone but preserves
    single-chain-altloc layout (no chain-id rewrite)."""
    cys_extra = frozenset({'CB', 'SG', 'HB', 'HB1', 'HB2', 'HB3', 'HG'}) if keep_disulfides else frozenset()
    out_lines = []
    with open(in_pdb) as f:
        for line in f:
            if line.startswith('ATOM  ') or line.startswith('HETATM'):
                resname = line[17:20].strip()
                if resname in ('HOH', 'WAT', 'H2O'):
                    continue
                atom_name = line[12:16].strip()
                keep_set = MAINCHAIN_ATOMS | (cys_extra if resname == 'CYS' else frozenset())
                if atom_name not in keep_set:
                    continue
            out_lines.append(line)
    with open(out_pdb, 'w') as f:
        f.writelines(out_lines)


# ── Parse single-chain altloc PDB ──────────────────────────────────────────────
def parse_singlechain_altloc(pdb_path):
    """Return (header, residues) where:
        residues[(resnum, icode, resname)] = {altloc: [pdb_lines]}
    """
    header = []
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM  ') or line.startswith('HETATM')):
                if not (line.startswith('END') or line.startswith('TER')):
                    header.append(line)
                continue
            resname = line[17:20].strip()
            if resname in ('HOH', 'WAT', 'H2O'):
                continue
            seqnum = int(line[22:26])
            icode = line[26]
            alt = line[16]
            key = (seqnum, icode, resname)
            residues.setdefault(key, {}).setdefault(alt, []).append(line)
    return header, residues


def residue_max_dev(altlocs):
    """Max heavy-atom centroid deviation across altlocs of one residue."""
    by_atom = {}
    for alt, lines in altlocs.items():
        for line in lines:
            aname = line[12:16].strip()
            if aname.startswith('H'):
                continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            by_atom.setdefault(aname, []).append((x, y, z))
    max_dev = 0.0
    for aname, pts in by_atom.items():
        if len(pts) < 2:
            continue
        arr = np.array(pts)
        dev = float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)))
        if dev > max_dev:
            max_dev = dev
    return max_dev


def select_altlocs_maximin(altlocs, k):
    """Maximin-pick k altlocs from those available (or all if fewer)."""
    keys = list(altlocs.keys())
    if len(keys) <= k:
        return keys
    centroids = []
    for alt in keys:
        pts = []
        for line in altlocs[alt]:
            aname = line[12:16].strip()
            if aname.startswith('H'):
                continue
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        centroids.append(np.mean(pts, axis=0) if pts else np.zeros(3))
    idx = _maximin_select(np.array(centroids), k)
    return [keys[i] for i in idx]


def build_condensed_singlechain(in_pdb, out_pdb, dev_to_nconf, max_k, cys_floor=1):
    """Per-residue altloc maximin. Output: single-chain altloc, occupancies pre-normalized to 1/k.
    cys_floor enforces a per-residue minimum k for CYS (clamped to available altlocs).
    """
    header, residues = parse_singlechain_altloc(in_pdb)
    out_lines = list(header)
    res_k_actual = {}  # for stats: actual conformer count kept per residue
    for key in sorted(residues.keys()):
        seqnum, icode, resname = key
        altlocs = residues[key]
        n_have = len(altlocs)
        dev = residue_max_dev(altlocs)
        k_target = min(max_k, n_have, dev_to_nconf(dev))
        if resname == 'CYS':
            k_target = min(n_have, max(k_target, cys_floor))
        if k_target < 1:
            k_target = 1
        kept = select_altlocs_maximin(altlocs, k_target)
        res_k_actual[key] = len(kept)
        occ = 1.0 / len(kept)
        for alt in sorted(kept):
            for line in altlocs[alt]:
                # Set occupancy column 55-60 (idx 54-60) to occ
                new_line = line[:54] + f'{occ:6.2f}' + line[60:]
                out_lines.append(new_line)
    out_lines.append('END\n')
    Path(out_pdb).write_text(''.join(out_lines))
    return res_k_actual


def reoccupy_normalize(in_pdb, out_pdb):
    """Run reoccupy.awk for proper sum=1 per-residue normalization."""
    with open(out_pdb, 'w') as g:
        subprocess.run([str(REOCCUPY), str(in_pdb)], stdout=g, check=True)


def generate_occ_groups_from_script(pdb_path):
    """Run ~/Develop/refmac_occupancy_setup.com to generate refmac occ-group keywords.
    Returns the keyword block as bytes ready to append to refmac kw input.
    """
    import tempfile
    pdb_path = Path(pdb_path).resolve()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # The script writes refmac_opts_occ.txt to cwd AND prints to stdout via tee
        r = subprocess.run([str(OCCSETUP), str(pdb_path)],
                           cwd=td, capture_output=True, check=True)
        return r.stdout


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--singlechain-pdb', required=True,
                    help='Source single-chain altloc PDB (e.g. 1aho/deconform_under20_best_0025.pdb)')
    ap.add_argument('--fobs-mtz',   default=None,
                    help='Pre-built Fobs MTZ. If omitted, build fresh from the clean singlechain bbss.')
    ap.add_argument('--refme',      default='1aho/refme_minRfree.mtz',
                    help='MTZ providing FreeR_flag for fresh Fobs build.')
    ap.add_argument('--outdir',     required=True)
    ap.add_argument('--threshold-set', default='midrich', choices=list(THRESHOLD_SETS.keys()))
    ap.add_argument('--max-k',     type=int, default=22)
    ap.add_argument('--n-rounds',  type=int, default=4)
    ap.add_argument('--cys-floor', type=int, default=1,
                    help='Minimum k for CYS residues (e.g. 3 to keep disulfide flexibility)')
    ap.add_argument('--occ-setup-from-round', type=int, default=3,
                    help='From this round on, use refmac_occupancy_setup.com to generate occ keywords '
                         '(set to 99 to disable)')
    args = ap.parse_args()

    src = Path(args.singlechain_pdb).resolve()
    outdir = Path(args.outdir).resolve() / args.threshold_set
    outdir.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

    print(f'Threshold set: {args.threshold_set}  max_k={args.max_k}')
    print(f'Output dir:    {outdir}')

    # 1. Strip backbone+SS in single-chain altloc layout
    bbss = outdir / 'bbss.pdb'
    strip_to_backbone_singlechain(src, bbss, keep_disulfides=True)
    n_bbss = sum(1 for ln in bbss.read_text().splitlines() if ln.startswith(('ATOM','HETATM')))
    print(f'  bbss: {n_bbss} atoms (single-chain altloc preserved)')

    # 1b. Build Fobs from the clean bbss (no duplicated altlocs)
    if args.fobs_mtz is None:
        fobs = outdir.parent / 'fobs.mtz'
        if not fobs.exists():
            print(f'Building Fobs from clean bbss → {fobs}')
            fobs_tmp = outdir.parent / '_fobs_tmp'; fobs_tmp.mkdir(exist_ok=True)
            built = build_fobs_calc_only(bbss, Path(args.refme).resolve(), fobs_tmp)
            shutil.copy2(built, fobs)
        else:
            print(f'Fobs already exists: {fobs}')
    else:
        fobs = Path(args.fobs_mtz).resolve()

    # 2. Per-residue altloc maximin
    raw = outdir / 'starthere_raw.pdb'
    dev_fn = make_dev_to_nconf(args.threshold_set)
    res_k_actual = build_condensed_singlechain(bbss, raw, dev_fn, args.max_k,
                                                cys_floor=args.cys_floor)
    n_kept = sum(res_k_actual.values())
    n_res  = len(res_k_actual)
    res_k_dist = {}
    for k in res_k_actual.values():
        res_k_dist[str(k)] = res_k_dist.get(str(k), 0) + 1
    n_atoms_raw = sum(1 for ln in raw.read_text().splitlines() if ln.startswith(('ATOM','HETATM')))
    print(f'  condensed: {n_res} residues, {n_kept} altloc slots, {n_atoms_raw} atoms')
    print(f'  res_k distribution: ' + ', '.join(f'{k}×{v}' for k,v in sorted(res_k_dist.items(), key=lambda x: int(x[0]))))

    # 3. reoccupy.awk for renormalization
    starthere = outdir / 'starthere.pdb'
    reoccupy_normalize(raw, starthere)

    # 4. Weight-snap rounds
    print(f'Running {args.n_rounds} weight-snap round(s) (refmac5-newhess)...')
    print(f'  occ-group keywords from refmac_occupancy_setup.com starting at round {args.occ_setup_from_round}')
    xyz = starthere
    rounds = []
    # Save original generate_occ_groups for rounds before the cutoff
    _orig_generate_occ_groups = ef.generate_occ_groups
    for r in range(1, args.n_rounds + 1):
        rd = outdir if args.n_rounds == 1 else (outdir / f'round{r}')
        rd.mkdir(exist_ok=True)
        if r >= args.occ_setup_from_round:
            ec.generate_occ_groups = generate_occ_groups_from_script
            ef.generate_occ_groups = generate_occ_groups_from_script
        else:
            ec.generate_occ_groups = _orig_generate_occ_groups
            ef.generate_occ_groups = _orig_generate_occ_groups
        r_i, rf_i, r_f, rf_f, t, fmtz, fpdb, log = run_weightsnap(xyz, fobs, rd)
        if fmtz: fmtz.rename(rd / 'refmacout.mtz')
        if fpdb: fpdb.rename(rd / 'refmacout.pdb')
        (rd / 'refmac.log').write_text(log)
        xyz = rd / 'refmacout.pdb'
        rounds.append(dict(round=r, r_init=r_i, rf_init=rf_i, r_final=r_f, rf_final=rf_f, elapsed=t))
        print(f'  Round {r}: R_init={r_i:.4f} Rf_init={rf_i:.4f} R_final={r_f:.4f} Rf_final={rf_f:.4f} t={t:.0f}s')

    n_atoms = sum(1 for ln in starthere.read_text().splitlines() if ln.startswith(('ATOM','HETATM')))
    result = dict(threshold_set=args.threshold_set, max_k=args.max_k,
                  n_residues=n_res, sum_k=n_kept, n_atoms=n_atoms,
                  res_k_dist=res_k_dist, n_rounds=args.n_rounds,
                  rounds=rounds)
    (outdir / 'result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
