#!/usr/bin/env ccp4-python
"""
condense_bb_varconf.py — Per-residue varconf condensation on backbone-only gt48.

Like condense_bb.py but: each residue gets its own k_r based on backbone
spread (heavy_atom_max_dev), so well-determined residues use fewer altlocs.
Builds a varconf PDB via build_varconf_pdb, then runs the same refmac
weight-snap schedule.

Refines against the same backbone-only Fc data (1aho/condense_bb_newhess/fobs.mtz).
Goal: hit R ≤ 3% with fewer total atoms than the global-k=8 model.

Usage:
  ccp4-python condense_bb_varconf.py --threshold-set default
  ccp4-python condense_bb_varconf.py --threshold-set lean
"""
import json
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

import explore_1aho_fusion as ef
import explore_condensation as ec
from explore_1aho_fusion import parse_conformers, build_varconf_pdb, heavy_atom_max_dev
from explore_condensation import run_weightsnap

# Use refmac5-newhess
ec.REFMAC5 = Path('/programs/ccp4-8.0/bin/refmac5-newhess')

SCRIPT_DIR = Path(__file__).resolve().parent
REOCCUPY = Path('/home/jamesh/Develop/reoccupy.awk')


def normalize_occupancies_via_reoccupy(in_pdb, out_pdb):
    """Convert multi-chain conformer PDB → single-chain altloc → reoccupy.awk.
    Result: single-chain altloc PDB with per-residue occupancies summing to 1.
    """
    import subprocess, tempfile, os
    in_pdb  = Path(in_pdb)
    out_pdb = Path(out_pdb)
    with tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=False) as f:
        tmp = Path(f.name)
        for line in in_pdb.read_text().splitlines(keepends=True):
            if line.startswith(('ATOM  ', 'HETATM')):
                # set chain (col 22, idx 21) to 'A'; keep altloc (col 17, idx 16) as slot letter
                line = line[:21] + 'A' + line[22:]
            f.write(line)
    try:
        with open(out_pdb, 'w') as g:
            subprocess.run([str(REOCCUPY), str(tmp)], stdout=g, check=True)
    finally:
        os.unlink(tmp)

# Backbone-tuned thresholds (override default dev_to_nconf which is for sidechains)
THRESHOLD_SETS = {
    'default':   [(0.6, 2), (0.8, 4), (1.0, 6), (1.5, 8), (2.5, 12), (99, 16)],
    'lean':      [(0.6, 1), (0.8, 2), (1.0, 4), (1.5, 6), (2.5, 8),  (99, 12)],
    'ultralean': [(0.6, 1), (0.8, 1), (1.0, 2), (1.5, 4), (2.5, 6),  (99, 8)],
    'midrich':   [(0.6, 3), (0.8, 5), (1.0, 7), (1.5, 10),(2.5, 14), (99, 20)],
    'rich':      [(0.6, 4), (0.8, 6), (1.0, 8), (1.5, 12),(2.5, 16), (99, 24)],
}

N_REFINE_ROUNDS = 1  # default; CLI can override

def make_dev_to_nconf(threshold_set):
    table = THRESHOLD_SETS[threshold_set]
    def f(dev):
        for lim, n in table:
            if dev < lim:
                return n
        return table[-1][1]
    return f


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--bb-pdb',     default='1aho/condense_bb_newhess/gt48_bb.pdb')
    ap.add_argument('--fobs-mtz',   default='1aho/condense_bb_newhess/fobs.mtz')
    ap.add_argument('--outdir',     default='1aho/condense_bb_varconf')
    ap.add_argument('--threshold-set', default='default', choices=list(THRESHOLD_SETS.keys()))
    ap.add_argument('--max-k',     type=int, default=16)
    ap.add_argument('--n-rounds',  type=int, default=1, help='Repeat weight-snap N times')
    args = ap.parse_args()

    bb_pdb   = Path(args.bb_pdb).resolve()
    fobs_mtz = Path(args.fobs_mtz).resolve()
    outdir   = Path(args.outdir).resolve() / args.threshold_set
    outdir.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

    # Override dev_to_nconf
    ef.dev_to_nconf = make_dev_to_nconf(args.threshold_set)

    print(f'Threshold set: {args.threshold_set}  max_k={args.max_k}')
    print(f'Output dir:    {outdir}')

    print('Parsing conformers...')
    st_orig, chain_names, conf_data = parse_conformers(bb_pdb)

    raw_pdb = outdir / 'starthere_raw.pdb'
    starthere_pdb = outdir / 'starthere.pdb'
    print('Building varconf PDB...')
    slot_chains, res_k, _ = build_varconf_pdb(
        chain_names, conf_data, st_orig,
        out_pdb=raw_pdb, workdir=outdir,
        max_k=args.max_k, water_pdb=None, limit_o=True,
    )
    print('Normalizing occupancies via reoccupy.awk (single-chain altloc out)...')
    normalize_occupancies_via_reoccupy(raw_pdb, starthere_pdb)

    n_atoms = sum(1 for ln in starthere_pdb.read_text().splitlines()
                  if ln.startswith('ATOM') or ln.startswith('HETATM'))
    n_res = len(res_k)
    sum_k = sum(res_k.values())
    print(f'  total atoms: {n_atoms}')
    print(f'  total per-residue conformer slots: {sum_k} '
          f'(avg {sum_k/n_res:.2f} per residue, vs {args.max_k} flat)')

    print(f'Running {args.n_rounds} weight-snap round(s) (refmac5-newhess)...')
    xyz = starthere_pdb
    rounds = []
    for r in range(1, args.n_rounds + 1):
        round_dir = outdir if args.n_rounds == 1 else (outdir / f'round{r}')
        round_dir.mkdir(exist_ok=True)
        r_i, rf_i, r_f, rf_f, elapsed, final_mtz, final_pdb, log = run_weightsnap(
            xyz, fobs_mtz, round_dir)
        if final_mtz and final_mtz.exists():
            final_mtz.rename(round_dir / 'refmacout.mtz')
        if final_pdb and final_pdb.exists():
            final_pdb.rename(round_dir / 'refmacout.pdb')
        (round_dir / 'refmac.log').write_text(log)
        xyz = round_dir / 'refmacout.pdb'
        rounds.append(dict(round=r, r_init=r_i, rf_init=rf_i,
                           r_final=r_f, rf_final=rf_f, elapsed=elapsed))
        print(f'  Round {r}: R_init={r_i:.4f} Rf_init={rf_i:.4f} '
              f'R_final={r_f:.4f} Rf_final={rf_f:.4f} t={elapsed:.0f}s')
    # Surface the last round's result for the summary line
    elapsed = rounds[-1]['elapsed']

    res_k_summary = {k: sum(1 for v in res_k.values() if v == k)
                     for k in sorted(set(res_k.values()))}
    result = dict(threshold_set=args.threshold_set, max_k=args.max_k,
                  n_residues=n_res, sum_k=sum_k, n_atoms=n_atoms,
                  res_k_dist=res_k_summary, n_rounds=args.n_rounds,
                  r_init=r_i, rf_init=rf_i, r_final=r_f, rf_final=rf_f,
                  rounds=rounds, elapsed=elapsed)
    (outdir / 'result.json').write_text(json.dumps(result, indent=2))
    print(f'Done. R_init={r_i:.4f} Rf_init={rf_i:.4f} '
          f'R_final={r_f:.4f} Rf_final={rf_f:.4f} t={elapsed:.1f}s')
    print(f'res_k distribution: ' +
          ', '.join(f'{k}×{v}' for k,v in res_k_summary.items()))


if __name__ == '__main__':
    main()
