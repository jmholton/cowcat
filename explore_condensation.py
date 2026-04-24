#!/usr/bin/env ccp4-python
"""
explore_condensation.py — Sweep max_k conformer count vs refmac R/time.

Runs a weight-snap refinement schedule for each k in K_LEVELS:
  NCYC 10 @ wm=10  →  NCYC 10 @ wm=0.01  →  NCYC 50 @ wm=0.5

Prints a table of (max_k, n_altloc_res, time_s, R_init, Rf_init, R_final, Rf_final).
"""

import sys
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from explore_1aho_fusion import (
    REFMAC5, run,
    parse_conformers, build_reduced_pdb, build_fobs_mtz,
    generate_occ_groups, parse_rfactors, load_density_map,
    STRATEGIES,
)

DEFAULT_PDB = Path('1aho/refmacout_minRfree.pdb')
DEFAULT_MTZ = Path('1aho/refme_minRfree.mtz')
# S8 is the best-validated strategy
S8_STRATEGY = next(r for r in STRATEGIES if r[0] == 'S8')
S8 = S8_STRATEGY[2:6]   # (mc_ord, mc_bouq, sc_ord, sc_bouq)
S8_BOUQ_THR   = S8_STRATEGY[6]
S8_MC_BOUQ_THR = S8_STRATEGY[7]

K_LEVELS = [1, 2, 3, 4, 6, 8, 12, 16]


def _run_one(xyzin, xyzout, hklout, fobs_mtz, ncyc, weight_matrix, tmpdir):
    """Single refmac run.  Returns (R, Rfree, log_text, out_pdb_path)."""
    libout = tmpdir / '_refmac.lib'
    occ_kw = generate_occ_groups(xyzin)
    kw  = b'LABIN FP=FP SIGFP=SIGFP FPART1=Fpart PHIP1=PHIpart FREE=FreeR_flag\n'
    kw += b'LABOUT FC=FC PHIC=PHIC FWT=FWT PHWT=PHWT DELFWT=DELFWT PHDELWT=PHDELWT\n'
    kw += b'solvent no\n'
    kw += b'scpart 1\n'
    kw += b'damp 0.5 0.5\n'
    kw += b'make hout Y\n'
    kw += b'make hydr Y\n'
    kw += f'weight matrix {weight_matrix}\n'.encode()
    kw += f'NCYC {ncyc}\n'.encode()
    kw += occ_kw
    kw += b'END\n'
    log = run(
        [REFMAC5,
         'XYZIN',  xyzin,
         'XYZOUT', xyzout,
         'HKLIN',  fobs_mtz,
         'HKLOUT', hklout,
         'LIBOUT', libout],
        input_bytes=kw, cwd=tmpdir, check=False,
    )
    r, rf = parse_rfactors(log)
    return r, rf, log


def run_weightsnap(starthere_pdb, fobs_mtz, tmpdir):
    """Three-stage weight-snap refinement.

    Returns (r_init, rf_init, r_final, rf_final, elapsed_s, final_mtz).
    """
    t0 = time.time()
    pdb_a = tmpdir / '_snap_a.pdb'
    pdb_b = tmpdir / '_snap_b.pdb'
    pdb_c = tmpdir / '_snap_c.pdb'
    mtz_a = tmpdir / '_snap_a.mtz'
    mtz_b = tmpdir / '_snap_b.mtz'
    mtz_c = tmpdir / '_snap_c.mtz'

    r1, rf1, _ = _run_one(starthere_pdb, pdb_a, mtz_a, fobs_mtz, ncyc=10, weight_matrix=10,   tmpdir=tmpdir)
    r2, rf2, _ = _run_one(pdb_a,         pdb_b, mtz_b, fobs_mtz, ncyc=10, weight_matrix=0.01, tmpdir=tmpdir)
    r3, rf3, _ = _run_one(pdb_b,         pdb_c, mtz_c, fobs_mtz, ncyc=50, weight_matrix=0.5,  tmpdir=tmpdir)

    elapsed = time.time() - t0
    final_mtz = mtz_c if mtz_c.exists() else None
    return r1, rf1, r3, rf3, elapsed, final_mtz


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb', default=str(DEFAULT_PDB))
    ap.add_argument('--mtz', default=str(DEFAULT_MTZ))
    ap.add_argument('--outdir', default='1aho/explore_condensation')
    args = ap.parse_args()

    pdb_path   = Path(args.pdb).resolve()
    refme_path = Path(args.mtz).resolve()
    outdir     = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('Conformer-count gradient: max_k sweep with weight-snap refinement')
    print(f'  PDB : {pdb_path}')
    print(f'  MTZ : {refme_path}')
    print(f'  K levels: {K_LEVELS}')
    print('=' * 72)

    with tempfile.TemporaryDirectory(prefix='condensation_fobs_') as _td:
        td_fobs = Path(_td)
        print('\nBuilding Fobs MTZ...')
        fobs_mtz = build_fobs_mtz(pdb_path, refme_path, td_fobs)
        fobs_final = outdir / 'fobs.mtz'
        shutil.copy2(fobs_mtz, fobs_final)

        print('\nParsing 48-conformer PDB...')
        st_orig, chain_names, conf_data = parse_conformers(pdb_path)
        n_res = len(conf_data[chain_names[0]])
        print(f'  {len(chain_names)} chains, {n_res} residues in ref chain')

        print('\nLoading density map...')
        refmac_mtz = pdb_path.parent / 'refmacout_minRfree.mtz'
        density_grid = load_density_map(str(refmac_mtz))

        rows = []
        hdr = (f"{'max_k':>5}  {'n_alt':>5}  {'time_s':>7}  "
               f"{'R_init':>7}  {'Rf_init':>7}  {'R_final':>7}  {'Rf_final':>8}")
        print()
        print(hdr)
        print('-' * len(hdr))

        for max_k in K_LEVELS:
            k_dir = outdir / f'k{max_k}'
            k_dir.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f'condensation_k{max_k}_') as _ktd:
                ktd = Path(_ktd)
                starthere_pdb = k_dir / 'starthere.pdb'
                try:
                    _, n_bouq, n_alt = build_reduced_pdb(
                        st_orig, chain_names, conf_data,
                        strategy=S8,
                        density_grid=density_grid,
                        bouquet_threshold=S8_BOUQ_THR,
                        mc_bouq_threshold=S8_MC_BOUQ_THR,
                        out_pdb=starthere_pdb,
                        tmpdir=ktd,
                        max_k=max_k,
                    )
                except Exception as e:
                    print(f'  k={max_k}: build_reduced_pdb failed: {e}')
                    continue

                try:
                    r_i, rf_i, r_f, rf_f, elapsed, final_mtz = run_weightsnap(
                        starthere_pdb, fobs_final, ktd)
                    if final_mtz and final_mtz.exists():
                        shutil.copy2(final_mtz, k_dir / 'refmacout.mtz')
                except Exception as e:
                    print(f'  k={max_k}: refmac failed: {e}')
                    r_i = rf_i = r_f = rf_f = elapsed = None

            def fmt(v):
                return f'{v:.4f}' if v is not None else '  N/A '

            row = (max_k, n_alt,
                   elapsed if elapsed is not None else float('nan'),
                   r_i, rf_i, r_f, rf_f)
            rows.append(row)
            t_str = f'{elapsed:.1f}' if elapsed is not None else 'N/A'
            print(f"{max_k:>5}  {n_alt:>5}  {t_str:>7}  "
                  f"{fmt(r_i):>7}  {fmt(rf_i):>7}  {fmt(r_f):>7}  {fmt(rf_f):>8}")

        print()
        print('Done.')


if __name__ == '__main__':
    main()
