#!/usr/bin/env python3
"""Run Untangler on 1AHO varconf structure.

Usage:
    /programs/pytorch/envs/pt/bin/python run_untangler_1aho.py [--pdb 1aho/under20_fitGT48.pdb]

Input PDB must be single-chain altloc format: protein atoms in chain A with
altloc letters (A, B, C, ...), waters in chain z.  under20_fitGT48.pdb has 22
conformers (altlocs A–V) and is directly compatible.

Untangler resolves 'tangled' conformer labels by iteratively swapping altloc
assignments to minimise geometry violations (wE) and crystallographic R factors
via phenix.refine.  No ground-truth reference is used — evaluation is on wE
and Rfree only.

Output PDB is written to untangler/output/<model>_loopEndN.pdb
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
UNTANGLER_DIR = SCRIPT_DIR / 'untangler'
AHO_DIR = SCRIPT_DIR / '1aho'

PHENIX_BIN = Path('/programs/phenix-2.0-5936/phenix_bin')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pdb', default=str(AHO_DIR / 'under20_fitGT48.pdb'),
                        help='Input PDB in single-chain altloc format (default: 1aho/under20_fitGT48.pdb)')
    parser.add_argument('--mtz', default=str(AHO_DIR / 'refme_minRfree.mtz'),
                        help='MTZ with FP/SIGFP/FreeR_flag (default: 1aho/refme_minRfree.mtz)')
    parser.add_argument('--reference', default=None,
                        help='Ground-truth PDB for tangle evaluation (optional)')
    parser.add_argument('--altloc-subset-size', type=int, default=2,
                        help='Number of altlocs considered simultaneously by ILP solver (default: 2)')
    parser.add_argument('--max-runs', type=int, default=5,
                        help='Max optimisation loops (default: 5)')
    parser.add_argument('--desired-score', type=float, default=18.4,
                        help='wE score to stop at (default: 18.4)')
    args = parser.parse_args()

    pdb_src = Path(args.pdb).resolve()
    mtz_src = Path(args.mtz).resolve()
    data_dir = UNTANGLER_DIR / 'data'

    # Copy input files into untangler/data/ with their original names
    pdb_dst = data_dir / pdb_src.name
    mtz_dst = data_dir / mtz_src.name
    shutil.copy2(pdb_src, pdb_dst)
    shutil.copy2(mtz_src, mtz_dst)
    print(f'Input PDB : {pdb_dst.relative_to(UNTANGLER_DIR)}')
    print(f'Input MTZ : {mtz_dst.relative_to(UNTANGLER_DIR)}')

    solution_reference = None
    if args.reference is not None:
        ref_src = Path(args.reference).resolve()
        if ref_src.exists():
            ref_dst = data_dir / ref_src.name
            shutil.copy2(ref_src, ref_dst)
            solution_reference = str(ref_dst.relative_to(UNTANGLER_DIR))
            print(f'Reference : {solution_reference}')

    # Add phenix to PATH (needed for phenix.refine and geometry scoring scripts)
    if PHENIX_BIN.exists():
        os.environ['PATH'] = str(PHENIX_BIN) + ':' + os.environ.get('PATH', '')
    else:
        print(f'WARNING: phenix bin not found at {PHENIX_BIN}', file=sys.stderr)

    # Untangler sets UNTANGLER_WORKING_DIRECTORY = os.path.abspath(os.getcwd()) at
    # module import time, so chdir before importing anything from the package.
    os.chdir(UNTANGLER_DIR)
    sys.path.insert(0, str(UNTANGLER_DIR))

    from untangle import Untangler
    from LinearOptimizer.Input import ConstraintsHandler

    pdb_rel = str(pdb_dst.relative_to(UNTANGLER_DIR))
    mtz_rel = str(mtz_dst.relative_to(UNTANGLER_DIR))

    Untangler(
        default_wc=1,
        endloop_wc=1,
        num_end_loop_refine_cycles=6,
        refine_for_positions_geo_weight=0,
        starting_num_best_swaps_considered=50,
        max_num_best_swaps_considered=50,
        altloc_subset_size=args.altloc_subset_size,
        unrestrained_damp=0,
        num_refine_for_positions_macro_cycles_phenix=1,
        max_bond_changes=99999,
        weight_factors={
            ConstraintsHandler.BondConstraint: 0.1,
            ConstraintsHandler.AngleConstraint: 80,
            ConstraintsHandler.NonbondConstraint: 0.1,
            ConstraintsHandler.ClashConstraint: 1e2,
            ConstraintsHandler.TwoAtomPenalty: 0,
        },
        solution_reference=solution_reference,
    ).run(
        pdb_rel,
        mtz_rel,
        desired_score=args.desired_score,
        max_num_runs=args.max_runs,
    )


if __name__ == '__main__':
    main()
