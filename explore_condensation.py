#!/usr/bin/env ccp4-python
"""
explore_condensation.py — Sweep max_k conformer count vs refmac R/time.

Runs a weight-snap refinement schedule for each k in K_LEVELS:
  NCYC 10 @ wm=10  →  NCYC 10 @ wm=0.01  →  NCYC 50 @ wm=0.5

Usage:
  # Build fobs.mtz once, then submit one SLURM job per k:
  ccp4-python explore_condensation.py --submit [--partition debug]

  # Collect results after all jobs finish:
  ccp4-python explore_condensation.py --collect

  # Single k level (called by SLURM workers):
  ccp4-python explore_condensation.py --max-k 4 --fobs-mtz outdir/fobs.mtz
"""

import json
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from explore_1aho_fusion import (
    REFMAC5, run,
    parse_conformers, build_fobs_mtz,
    generate_occ_groups, parse_rfactors, load_density_map,
    select_chains_maximin, build_starthere_pdb,
)

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PDB = Path('1aho/gt48.pdb')
DEFAULT_MTZ = Path('1aho/gt48.mtz')

K_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48]


# ── Refmac helpers ────────────────────────────────────────────────────────────

def _run_one(xyzin, xyzout, hklout, fobs_mtz, ncyc, weight_matrix, tmpdir,
             damp=None, fp_col='FP', occ_refine=False):
    """Single refmac run. Returns (R, Rfree, log_text)."""
    if damp is None:
        damp = min(0.5, 0.5 / weight_matrix) if weight_matrix > 1.0 else 0.5
    kw  = f'LABIN FP={fp_col} FPART1=Fpart PHIP1=PHIpart FREE=FreeR_flag\n'.encode()
    kw += b'LABOUT FC=FC PHIC=PHIC FWT=FWT PHWT=PHWT DELFWT=DELFWT PHDELWT=PHDELWT\n'
    kw += f'solvent no\nscpart 1\ndamp {damp:.4f} {damp:.4f}\nmake hout Y\nmake hydr Y\n'.encode()
    kw += f'weight matrix {weight_matrix}\nNCYC {ncyc}\n'.encode()
    if occ_refine:
        kw += generate_occ_groups(xyzin)
    kw += b'END\n'
    log = run(
        [REFMAC5,
         'XYZIN',  xyzin,  'XYZOUT', xyzout,
         'HKLIN',  fobs_mtz, 'HKLOUT', hklout,
         'LIBOUT', tmpdir / '_refmac.lib'],
        input_bytes=kw, cwd=tmpdir, check=False,
    )
    r, rf = parse_rfactors(log)
    return r, rf, log


# Each stage: (ncyc, weight_matrix, occ_refine)
# Settle geometry first (no occ), then refine occupancies at higher weights.
WEIGHT_STAGES = (
    (10, 0.01,  False),
    (10, 0.1,   False),
    (10, 1.0,   False),
    (10, 10.0,  True),
    (10, 0.5,   True),
)


def run_weightsnap(starthere_pdb, fobs_mtz, tmpdir, stages=WEIGHT_STAGES,
                   fp_col='FP'):
    """Multi-stage weight-snap refinement.

    Default: 0.01→0.1→1→10→0.5, building up X-ray weight then settling.
    Returns (r_init, rf_init, r_final, rf_final, elapsed_s, final_mtz, final_pdb).
    """
    t0 = time.time()
    r_init = rf_init = None
    all_logs = []
    xyzin = starthere_pdb
    for si, stage in enumerate(stages):
        ncyc, wm, occ = (stage + (False,))[:3]
        label = f'_stage{si}'
        xyzout = tmpdir / f'{label}.pdb'
        hklout = tmpdir / f'{label}.mtz'
        r, rf, log = _run_one(xyzin, xyzout, hklout, fobs_mtz, ncyc, wm, tmpdir,
                               fp_col=fp_col, occ_refine=occ)
        if not xyzout.exists():
            (tmpdir / f'{label}.log').write_text(log)
            raise RuntimeError(f'refmac stage {si} (wm={wm}) failed; '
                               f'log saved to {tmpdir}/{label}.log\n' + log[-3000:])
        if r_init is None:
            r_init, rf_init = r, rf
        xyzin = xyzout
        all_logs.append(f'\n{"="*60}\n Stage {si}  wm={wm}  ncyc={ncyc}\n{"="*60}\n' + log)
    elapsed = time.time() - t0
    return r_init, rf_init, r, rf, elapsed, (hklout if hklout.exists() else None), (xyzout if xyzout.exists() else None), ''.join(all_logs)


# ── Worker: run one k level ───────────────────────────────────────────────────

def run_one_k(max_k, pdb_path, fobs_mtz, outdir):
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

    print(f'[k={max_k}] Parsing conformers...')
    st_orig, chain_names, conf_data = parse_conformers(pdb_path)

    k_dir = outdir / f'k{max_k}'
    k_dir.mkdir(parents=True, exist_ok=True)
    starthere_pdb = k_dir / 'starthere.pdb'

    print(f'[k={max_k}] Building reduced PDB...')
    n_alt = build_starthere_pdb(chain_names, conf_data, st_orig, max_k,
                                ref_pdb=pdb_path, out_pdb=starthere_pdb,
                                workdir=k_dir)
    print(f'[k={max_k}] Running weight-snap refinement...')
    r_i, rf_i, r_f, rf_f, elapsed, final_mtz, final_pdb, refmac_log = run_weightsnap(
        starthere_pdb, fobs_mtz, k_dir)
    if final_mtz and final_mtz.exists():
        final_mtz.rename(k_dir / 'refmacout.mtz')
    if final_pdb and final_pdb.exists():
        final_pdb.rename(k_dir / 'refmacout.pdb')
    (k_dir / 'refmac.log').write_text(refmac_log)

    result = dict(max_k=max_k, n_alt=n_alt, elapsed=elapsed,
                  r_init=r_i, rf_init=rf_i, r_final=r_f, rf_final=rf_f)
    (k_dir / 'result.json').write_text(json.dumps(result))
    print(f'[k={max_k}] Done. R_init={r_i:.4f} Rf_init={rf_i:.4f} '
          f'R_final={r_f:.4f} Rf_final={rf_f:.4f} t={elapsed:.1f}s')


# ── Submit: build fobs once, then one job per k ───────────────────────────────

def submit(pdb_path, refme_path, outdir, partition, account=None, qos=None):
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)
    fobs_mtz = outdir / 'fobs.mtz'
    if not fobs_mtz.exists():
        print('Building Fobs MTZ...')
        fobs_tmp = outdir / '_fobs_tmp'
        fobs_tmp.mkdir(exist_ok=True)
        built = build_fobs_mtz(pdb_path, refme_path, fobs_tmp)
        shutil.copy2(built, fobs_mtz)
        print(f'  saved → {fobs_mtz}')
    else:
        print(f'Fobs MTZ already exists: {fobs_mtz}')

    me = Path(__file__).resolve()
    for max_k in K_LEVELS:
        script = SCRIPT_DIR / f'_cond_k{max_k}.sh'
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name=cond_k{max_k}',
            '#SBATCH --ntasks=1',
            '#SBATCH --cpus-per-task=1',
            '#SBATCH --mem=8G',
            f'#SBATCH --partition={partition}',
        ]
        if account:
            lines.append(f'#SBATCH --account={account}')
        if qos:
            lines.append(f'#SBATCH --qos={qos}')
        lines += [
            '#SBATCH --export=ALL',
            '',
            'mkdir -p "${CCP4_SCR:-/tmp}"',
            f'cd {SCRIPT_DIR}',
            f'{sys.executable} {me} \\',
            f'  --max-k {max_k} \\',
            f'  --fobs-mtz {fobs_mtz} \\',
            f'  --pdb {pdb_path} \\',
            f'  --outdir {outdir}',
            '',
        ]
        script.write_text('\n'.join(lines))
        script.chmod(0o755)
        r = subprocess.run(['sbatch', str(script)], capture_output=True, text=True)
        print(f'  k={max_k}: {r.stdout.strip() or r.stderr.strip()}')


# ── Collect: print table from saved result JSONs ──────────────────────────────

def collect(outdir):
    rows = []
    for max_k in K_LEVELS:
        p = outdir / f'k{max_k}/result.json'
        if p.exists():
            rows.append(json.loads(p.read_text()))
        else:
            print(f'  k={max_k}: no result.json yet')

    if not rows:
        print('No results found.')
        return

    hdr = (f"{'max_k':>5}  {'n_alt':>5}  {'time_s':>7}  "
           f"{'R_init':>7}  {'Rf_init':>7}  {'R_final':>7}  {'Rf_final':>8}")
    print()
    print(hdr)
    print('-' * len(hdr))
    for d in sorted(rows, key=lambda x: x['max_k']):
        def fmt(v):
            return f'{v:.4f}' if v is not None else '  N/A '
        t_str = f"{d['elapsed']:.1f}" if d.get('elapsed') is not None else 'N/A'
        print(f"{d['max_k']:>5}  {d['n_alt']:>5}  {t_str:>7}  "
              f"{fmt(d.get('r_init')):>7}  {fmt(d.get('rf_init')):>7}  "
              f"{fmt(d.get('r_final')):>7}  {fmt(d.get('rf_final')):>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb',       default=str(DEFAULT_PDB))
    ap.add_argument('--mtz',       default=str(DEFAULT_MTZ))
    ap.add_argument('--outdir',    default='1aho/explore_condensation')
    ap.add_argument('--partition', default='debug')
    ap.add_argument('--account')
    ap.add_argument('--qos')
    # Worker args
    ap.add_argument('--max-k',    type=int, help='Run single k level (worker mode)')
    ap.add_argument('--fobs-mtz', help='Pre-built fobs MTZ (worker mode)')
    # Modes
    ap.add_argument('--submit',  action='store_true', help='Submit one job per k to SLURM')
    ap.add_argument('--collect', action='store_true', help='Print table from saved results')
    args = ap.parse_args()

    pdb_path   = Path(args.pdb).resolve()
    refme_path = Path(args.mtz).resolve()
    outdir     = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.submit:
        submit(pdb_path, refme_path, outdir,
               partition=args.partition, account=args.account, qos=args.qos)

    elif args.max_k is not None:
        fobs_mtz = Path(args.fobs_mtz).resolve()
        run_one_k(args.max_k, pdb_path, fobs_mtz, outdir)

    elif args.collect:
        collect(outdir)

    else:
        # Sequential fallback
        os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)
        fobs_mtz = outdir / 'fobs.mtz'
        if not fobs_mtz.exists():
            with tempfile.TemporaryDirectory(prefix='cond_fobs_') as _td:
                built = build_fobs_mtz(pdb_path, refme_path, Path(_td))
                shutil.copy2(built, fobs_mtz)
        for max_k in K_LEVELS:
            run_one_k(max_k, pdb_path, fobs_mtz, outdir)
        collect(outdir)


if __name__ == '__main__':
    main()
