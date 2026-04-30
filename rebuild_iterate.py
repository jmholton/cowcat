#!/usr/bin/env ccp4-python
"""Iterative rebuild: probe Fo-Fc → top-N outliers → rebuild PDB → NCYC refmac → repeat.

Standalone debuggable script.  Starts from an existing refmacout.pdb/mtz so
you don't need to re-run the full weight-snap from scratch.

Usage:
  ccp4-python rebuild_iterate.py \\
    --pdb  1aho/varconf_sweep/k16/refmacout.pdb \\
    --mtz  1aho/varconf_sweep/k16/refmacout.mtz \\
    --sel  1aho/varconf_sweep/k16/per_res_sel.json \\
    --fobs 1aho/refme.mtz \\
    --outdir 1aho/varconf_sweep/k16/iter \\
    [--gt48  1aho/gt48.pdb]          # default
    [--water 1aho/gt48_water.pdb]    # default if exists
    [--max-rounds 10] [--top-n 5] [--ncyc 5] [--weight 0.5]
    [--neg-thresh -3.0] [--pos-thresh 3.0]

Outputs per round:
  iter/round_NN/rebuilt.pdb    — rebuilt model before refmac
  iter/round_NN/refmacout.pdb  — refined output
  iter/round_NN/refmacout.mtz
  iter/round_NN/refmac.log
  iter/per_res_sel_final.json  — final conformer selection
  iter/iter_log.json           — per-round stats table
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from explore_1aho_fusion import (
    parse_conformers,
    score_density_outliers,
    find_map_peak_candidate,
    find_swap_candidate,
    apply_rebuild_topn,
    _load_slot_res,
    _find_disulfide_pairs,
    save_per_res_sel,
    load_per_res_sel,
    run_refmac_quick,
    parse_rfactors,
)

DEFAULT_GT48  = SCRIPT_DIR / '1aho/gt48.pdb'
DEFAULT_WATER = SCRIPT_DIR / '1aho/gt48_water.pdb'


def run_iter_rebuild(refmacout_pdb, refmacout_mtz, per_res_sel,
                     conf_data, chain_names, residue_keys, ref_chain_data, st_orig,
                     fobs_mtz, workdir, water_pdb=None,
                     max_rounds=10, top_n=5, ncyc=5, weight_matrix=0.5,
                     occ_refine=True, neg_thresh=-3.0, pos_thresh=3.0):
    """Iterative top-N rebuild loop.

    Returns (final_pdb, final_mtz, final_per_res_sel, iter_log).
    iter_log: list of dicts {round, n_prune, n_add, r, rf, elapsed_s}.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    ss_pairs = _find_disulfide_pairs(conf_data, chain_names)
    print(f'  Disulfide pairs: ' +
          ', '.join(f'{a[0]}-{b[0]}' for a, b in sorted(ss_pairs.items()) if a < b))

    current_pdb        = Path(refmacout_pdb)
    current_mtz        = Path(refmacout_mtz)
    current_per_res    = per_res_sel
    orig_per_res       = per_res_sel   # tracks coord ordering of current_pdb
    iter_log           = []

    for round_i in range(max_rounds):
        t0   = time.time()
        rdir = workdir / f'round_{round_i:02d}'
        rdir.mkdir(exist_ok=True)

        print(f'\n[iter {round_i}] Scoring density outliers...')
        candidates, sigma = score_density_outliers(
            current_pdb, current_mtz,
            conf_data, chain_names,
            current_per_res, residue_keys, ref_chain_data,
            neg_thresh=neg_thresh, pos_thresh=pos_thresh,
        )
        n_prune_cands = sum(1 for c in candidates if c['action'] == 'prune')
        n_add_cands   = sum(1 for c in candidates if c['action'] == 'add')
        print(f'[iter {round_i}] sigma={sigma:.4f} e/Å³  '
              f'candidates: {n_prune_cands} prune, {n_add_cands} add')

        # Peak criterion: add conformer nearest to global Fo-Fc maximum.
        peak_cand = find_map_peak_candidate(
            current_mtz, conf_data, chain_names,
            current_per_res, residue_keys, ref_chain_data,
        )
        if peak_cand is not None:
            already = any(
                c['action'] == 'add' and c['rk'] == peak_cand['rk']
                and c['gt48_cn'] == peak_cand['gt48_cn']
                for c in candidates
            )
            if not already:
                candidates.append(peak_cand)
                print(f'[iter {round_i}] Peak cand: res {peak_cand["rk"][0]} '
                      f'{peak_cand["resname"]} gt48:{peak_cand["gt48_cn"]}  '
                      f'peak={peak_cand["peak_sigma"]:.2f}σ')
            else:
                # Peak conformer already caught by regular scoring.
                # Check if its residue has a significantly negative existing slot
                # and if so insert a swap candidate (prune worst + add peak = 1 action).
                swap_cand = find_swap_candidate(
                    current_mtz, current_pdb, peak_cand,
                    current_per_res, neg_thresh=neg_thresh,
                )
                if swap_cand is not None:
                    candidates.insert(0, swap_cand)
                    print(f'[iter {round_i}] Peak cand: res {peak_cand["rk"][0]} '
                          f'{peak_cand["resname"]} gt48:{peak_cand["gt48_cn"]}  '
                          f'peak={peak_cand["peak_sigma"]:.2f}σ  (already in list)'
                          f'  → swap worst slot '
                          f'gt48:{swap_cand["old_gt48_cn"]} '
                          f'min={swap_cand["dmin"]:.2f}σ')
                else:
                    print(f'[iter {round_i}] Peak cand: res {peak_cand["rk"][0]} '
                          f'{peak_cand["resname"]} gt48:{peak_cand["gt48_cn"]}  '
                          f'peak={peak_cand["peak_sigma"]:.2f}σ  (already in list, '
                          f'no negative slot to swap)')
        else:
            print(f'[iter {round_i}] Peak cand: none above 1σ')

        if not candidates:
            print(f'[iter {round_i}] No outliers above threshold — converged.')
            break

        # Show top candidates before applying
        print(f'[iter {round_i}] Top {min(top_n, len(candidates))} (of {len(candidates)}):')
        for c in candidates[:top_n]:
            tag = f'min={c["dmin"]:.2f}σ' if c['action'] == 'prune' else f'max={c["dmax"]:.2f}σ'
            print(f'  {c["action"]:5s}  res {c["rk"][0]:3d} {c["resname"]:3s}  '
                  f'gt48:{c["gt48_cn"]}  excess={c["excess"]:.2f}σ  {tag}')

        slot_res    = _load_slot_res(current_pdb)
        rebuilt_pdb = rdir / 'rebuilt.pdb'
        new_per_res, new_res_k, actions = apply_rebuild_topn(
            candidates, top_n,
            per_res_sel=current_per_res, orig_per_res_sel=orig_per_res,
            slot_res=slot_res,
            residue_keys=residue_keys,
            ref_chain_data=ref_chain_data,
            conf_data=conf_data,
            ss_pairs=ss_pairs,
            st_orig=st_orig,
            out_pdb=rebuilt_pdb,
            water_pdb=water_pdb,
        )

        n_prune = sum(1 for a in actions if a['action'] == 'prune')
        n_add   = sum(1 for a in actions if a['action'] == 'add')

        if not actions:
            print(f'[iter {round_i}] No actions applied — done.')
            break

        print(f'[iter {round_i}] Applied {n_prune} prune, {n_add} add.  '
              f'Running refmac NCYC {ncyc} wm={weight_matrix}...')

        rout_pdb = rdir / 'refmacout.pdb'
        rout_mtz = rdir / 'refmacout.mtz'
        with tempfile.TemporaryDirectory(prefix=f'iter_r{round_i}_') as td:
            r, rf, log, mtz_out, pdb_out = run_refmac_quick(
                rebuilt_pdb, fobs_mtz, ncyc, weight_matrix, Path(td),
                occ_refine=occ_refine,
            )
            if pdb_out and pdb_out.exists():
                shutil.copy2(pdb_out, rout_pdb)
            if mtz_out and mtz_out.exists():
                shutil.copy2(mtz_out, rout_mtz)
            (rdir / 'refmac.log').write_text(log or '')

        elapsed = time.time() - t0
        r_str  = f'{r:.4f}'  if r  is not None else 'N/A'
        rf_str = f'{rf:.4f}' if rf is not None else 'N/A'
        print(f'[iter {round_i}] R={r_str} Rf={rf_str}  t={elapsed:.0f}s')

        iter_log.append({
            'round': round_i, 'n_prune': n_prune, 'n_add': n_add,
            'r': r, 'rf': rf, 'elapsed_s': round(elapsed, 1),
        })

        if not rout_pdb.exists() or not rout_mtz.exists():
            print(f'[iter {round_i}] refmac produced no output — stopping.')
            break

        # Advance state.
        orig_per_res    = new_per_res   # new_per_res defines chain ordering in rout_pdb
        current_per_res = new_per_res
        current_pdb     = rout_pdb
        current_mtz     = rout_mtz

    return current_pdb, current_mtz, current_per_res, iter_log


def _print_table(iter_log):
    if not iter_log:
        return
    hdr = f"{'round':>5}  {'prune':>5}  {'add':>5}  {'R':>7}  {'Rfree':>7}  {'t_s':>6}"
    print()
    print(hdr)
    print('-' * len(hdr))
    for row in iter_log:
        r_s  = f'{row["r"]:.4f}'  if row['r']  is not None else '  N/A '
        rf_s = f'{row["rf"]:.4f}' if row['rf'] is not None else '  N/A '
        print(f'{row["round"]:>5}  {row["n_prune"]:>5}  {row["n_add"]:>5}  '
              f'{r_s:>7}  {rf_s:>7}  {row["elapsed_s"]:>6.0f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pdb',         required=True, help='Input refmacout.pdb')
    ap.add_argument('--mtz',         required=True, help='Input refmacout.mtz (Fo-Fc source)')
    ap.add_argument('--sel',         required=True, help='per_res_sel.json from build step')
    ap.add_argument('--fobs',        required=True, help='Fobs MTZ (FP/Fpart/PHIpart/FreeR_flag)')
    ap.add_argument('--outdir',      required=True, help='Output directory for rounds')
    ap.add_argument('--gt48',        default=str(DEFAULT_GT48))
    ap.add_argument('--water',       default=str(DEFAULT_WATER) if DEFAULT_WATER.exists() else None)
    ap.add_argument('--max-rounds',  type=int,   default=10)
    ap.add_argument('--top-n',       type=int,   default=5)
    ap.add_argument('--ncyc',        type=int,   default=5)
    ap.add_argument('--weight',      type=float, default=0.5)
    ap.add_argument('--neg-thresh',  type=float, default=-3.0)
    ap.add_argument('--pos-thresh',  type=float, default=3.0)
    args = ap.parse_args()

    print(f'Loading gt48 conformers from {args.gt48}...')
    st_orig, chain_names, conf_data = parse_conformers(args.gt48)
    ref_chain_data = conf_data[chain_names[0]]
    residue_keys   = list(ref_chain_data.keys())

    print(f'Loading per_res_sel from {args.sel}...')
    per_res_sel = load_per_res_sel(args.sel)

    water_pdb = args.water if args.water and Path(args.water).exists() else None

    final_pdb, final_mtz, final_per_res, iter_log = run_iter_rebuild(
        refmacout_pdb  = args.pdb,
        refmacout_mtz  = args.mtz,
        per_res_sel    = per_res_sel,
        conf_data      = conf_data,
        chain_names    = chain_names,
        residue_keys   = residue_keys,
        ref_chain_data = ref_chain_data,
        st_orig        = st_orig,
        fobs_mtz       = args.fobs,
        workdir        = args.outdir,
        water_pdb      = water_pdb,
        max_rounds     = args.max_rounds,
        top_n          = args.top_n,
        ncyc           = args.ncyc,
        weight_matrix  = args.weight,
        neg_thresh     = args.neg_thresh,
        pos_thresh     = args.pos_thresh,
    )

    # Save final state
    outdir = Path(args.outdir)
    save_per_res_sel(final_per_res, outdir / 'per_res_sel_final.json')
    (outdir / 'iter_log.json').write_text(json.dumps(iter_log, indent=2))

    _print_table(iter_log)
    print(f'\nFinal PDB: {final_pdb}')
    print(f'Final MTZ: {final_mtz}')
    print(f'Done.  {len(iter_log)} rounds.')


if __name__ == '__main__':
    main()
