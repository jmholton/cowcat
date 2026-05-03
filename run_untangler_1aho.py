#!/usr/bin/env python3
"""Run Untangler on 1AHO varconf structure.

Usage:
    python3 run_untangler_1aho.py [--pdb 1aho/varconf_opt6.pdb] [--mtz 1aho/refme_minRfree.mtz]

Untangler resolves 'tangled' conformer labels in multi-conformer structures by
iteratively swapping altloc assignments to minimise geometry violations and
crystallographic R factors. Uses phenix for refinement and geometry scoring.

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
    parser.add_argument('--pdb', default=str(AHO_DIR / 'varconf_opt6.pdb'),
                        help='Input multi-conformer PDB (default: 1aho/varconf_opt6.pdb)')
    parser.add_argument('--mtz', default=str(AHO_DIR / 'refme_minRfree.mtz'),
                        help='MTZ with FP/SIGFP/FreeR_flag (default: 1aho/refme_minRfree.mtz)')
    parser.add_argument('--reference', default=str(AHO_DIR / 'gt48.pdb'),
                        help='Ground-truth PDB for tangle evaluation (optional)')
    parser.add_argument('--no-reference', action='store_true',
                        help='Skip tangle evaluation even if reference file exists')
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

    ref_argv = []
    if not args.no_reference:
        ref_src = Path(args.reference).resolve()
        if ref_src.exists():
            ref_dst = data_dir / ref_src.name
            shutil.copy2(ref_src, ref_dst)
            ref_argv = [str(ref_dst.relative_to(UNTANGLER_DIR))]
            print(f'Reference : {ref_dst.relative_to(UNTANGLER_DIR)}')

    # Add phenix to PATH (needed for refinement + scoring scripts)
    if PHENIX_BIN.exists():
        os.environ['PATH'] = str(PHENIX_BIN) + ':' + os.environ.get('PATH', '')
    else:
        print(f'WARNING: phenix bin not found at {PHENIX_BIN}', file=sys.stderr)

    # Untangler uses UNTANGLER_WORKING_DIRECTORY = os.path.abspath(os.getcwd()) at import
    # time, so we must chdir before any import from the untangler package.
    os.chdir(UNTANGLER_DIR)
    sys.path.insert(0, str(UNTANGLER_DIR))

    sys.argv = [
        'untangle.py',
        str(pdb_dst.relative_to(UNTANGLER_DIR)),
        str(mtz_dst.relative_to(UNTANGLER_DIR)),
    ] + ref_argv

    import untangle
    untangle.main()


if __name__ == '__main__':
    main()
