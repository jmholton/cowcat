#!/usr/bin/env ccp4-python
"""Sweep max_k for variable-conformer reference models.

Usage:
  # Submit all max_k values as SLURM jobs:
  ccp4-python make_varconf_sweep.py --submit [--partition lr6] [--account pc_als831]

  # Run a single max_k (called by SLURM workers):
  ccp4-python make_varconf_sweep.py --max-k 16

  # Run all locally (slow):
  ccp4-python make_varconf_sweep.py

Fobs source: 1aho/gt48.mtz (FC_ALL_LS column, proper scale, known phases).
Output: 1aho/varconf_sweep/k{max_k}/starthere.pdb + refmacout.pdb + result.json
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from explore_1aho_fusion import (
    parse_conformers, build_varconf_pdb, save_per_res_sel,
)
from explore_condensation import run_weightsnap
from rebuild_iterate import run_iter_rebuild

PDB    = SCRIPT_DIR / '1aho/gt48.pdb'
WATER  = SCRIPT_DIR / '1aho/gt48_water.pdb'
FOBS   = SCRIPT_DIR / '1aho/refme.mtz'  # FP(=Fgt), Fpart, PHIpart, FreeR_flag
OUTDIR = SCRIPT_DIR / '1aho/varconf_sweep'

MAX_K_VALUES = [3, 8, 16, 32, 48]


def run_one(max_k):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    k_dir = OUTDIR / f'k{max_k}'
    k_dir.mkdir(exist_ok=True)
    starthere = k_dir / 'starthere.pdb'

    print(f'[k={max_k}] Parsing conformers...')
    st_orig, chain_names, conf_data = parse_conformers(str(PDB))
    ref_chain_data = conf_data[chain_names[0]]
    residue_keys = list(ref_chain_data.keys())

    print(f'[k={max_k}] Building varconf PDB...')
    selected, res_k, per_res_sel = build_varconf_pdb(
        chain_names, conf_data, st_orig,
        out_pdb=starthere, workdir=k_dir, max_k=max_k,
        water_pdb=WATER,
    )

    def _dist_str(rk):
        dist = {}
        for v in rk.values():
            dist[v] = dist.get(v, 0) + 1
        return ', '.join(f'{k}×{n}' for k, n in sorted(dist.items()))

    print(f'[k={max_k}] res_k dist: {_dist_str(res_k)}')

    t0 = time.time()
    print(f'[k={max_k}] Round 1: weight-snap refinement...')
    with tempfile.TemporaryDirectory(prefix=f'varconf_k{max_k}_r1_') as td:
        r_i, rf_i, r_f, rf_f, elapsed1, final_mtz, final_pdb, refmac_log = run_weightsnap(
            starthere, FOBS, Path(td))
        if final_pdb and final_pdb.exists():
            shutil.copy2(final_pdb, k_dir / 'refmacout.pdb')
        if final_mtz and final_mtz.exists():
            shutil.copy2(final_mtz, k_dir / 'refmacout.mtz')
        (k_dir / 'refmac_r1.log').write_text(refmac_log)
    print(f'[k={max_k}] R1: R={r_f:.4f} Rf={rf_f:.4f}')

    # Save per_res_sel for standalone debugging with rebuild_iterate.py
    save_per_res_sel(per_res_sel, k_dir / 'per_res_sel.json')

    print(f'[k={max_k}] Iterative rebuild (top-5, NCYC 5 per round)...')
    final_pdb_iter, final_mtz_iter, final_per_res, iter_log = run_iter_rebuild(
        refmacout_pdb  = k_dir / 'refmacout.pdb',
        refmacout_mtz  = k_dir / 'refmacout.mtz',
        per_res_sel    = per_res_sel,
        conf_data      = conf_data,
        chain_names    = chain_names,
        residue_keys   = residue_keys,
        ref_chain_data = ref_chain_data,
        st_orig        = st_orig,
        fobs_mtz       = FOBS,
        workdir        = k_dir / 'iter',
        water_pdb      = WATER,
        max_rounds     = 10,
        top_n          = 5,
        ncyc           = 5,
    )
    save_per_res_sel(final_per_res, k_dir / 'per_res_sel_final.json')
    (k_dir / 'iter_log.json').write_text(json.dumps(iter_log, indent=2))

    r_iter  = iter_log[-1]['r']  if iter_log else r_f
    rf_iter = iter_log[-1]['rf'] if iter_log else rf_f

    elapsed = time.time() - t0
    dist = {}
    for v in res_k.values():
        dist[v] = dist.get(v, 0) + 1
    result = dict(max_k=max_k, r_init=r_i, rf_init=rf_i,
                  r_final=r_f, rf_final=rf_f,
                  r_iter=r_iter, rf_iter=rf_iter,
                  n_iter_rounds=len(iter_log),
                  elapsed=elapsed,
                  res_k_dist={str(k): n for k, n in sorted(dist.items())})
    (k_dir / 'result.json').write_text(json.dumps(result, indent=2))
    print(f'[k={max_k}] R1={r_f:.4f}/{rf_f:.4f}  '
          f'Riter={r_iter:.4f}/{rf_iter:.4f}  '
          f'rounds={len(iter_log)}  t={elapsed:.0f}s')


def submit(partition, account, qos):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for max_k in MAX_K_VALUES:
        job = f'varconf_k{max_k}'
        cmd = (
            f'#!/bin/bash\n'
            f'#SBATCH --job-name={job}\n'
            f'#SBATCH --partition={partition}\n'
            f'#SBATCH --ntasks=1\n'
            f'#SBATCH --cpus-per-task=1\n'
            f'#SBATCH --time=2:00:00\n'
            f'#SBATCH --output={OUTDIR}/k{max_k}/slurm-%j.out\n'
            f'#SBATCH --export=ALL\n'
        )
        if account:
            cmd += f'#SBATCH --account={account}\n'
        if qos:
            cmd += f'#SBATCH --qos={qos}\n'
        cmd += (
            f'cd {SCRIPT_DIR}\n'
            f'source cluster.sh setup_ccp4\n'
            f'ccp4-python make_varconf_sweep.py --max-k {max_k}\n'
        )
        (OUTDIR / f'k{max_k}').mkdir(parents=True, exist_ok=True)
        script = OUTDIR / f'_submit_k{max_k}.sh'
        script.write_text(cmd)
        result = subprocess.run(['sbatch', str(script)], capture_output=True, text=True)
        print(f'k={max_k}: {result.stdout.strip() or result.stderr.strip()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--max-k', type=int)
    ap.add_argument('--partition', default='lr6')
    ap.add_argument('--account', default='pc_als831')
    ap.add_argument('--qos', default='lr_normal')
    args = ap.parse_args()

    if args.submit:
        submit(args.partition, args.account, args.qos)
    elif args.max_k:
        run_one(args.max_k)
    else:
        for max_k in MAX_K_VALUES:
            run_one(max_k)
