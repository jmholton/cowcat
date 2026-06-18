#!/usr/bin/env ccp4-python
"""make_maps.py — Convert refmacout.mtz + truth_full.pdb → training sample maps.

Inputs (per sample directory):
  refmacout.mtz   — refmac output (must have FWT/PHWT, DELFWT/PHDELWT, FC_ALL_LS/PHIC_ALL_LS)
  truth_full.pdb  — full multi-conf truth model (H atoms optional; will be added if absent)

Outputs (written to same directory):
  truth.map       — FC/PHIC from gemmi sfcalc on truth_full.pdb
  2fofc.map       — FWT/PHWT from refmacout.mtz
  fofc.map        — DELFWT/PHDELWT from refmacout.mtz
  fc.map          — FC_ALL_LS/PHIC_ALL_LS from refmacout.mtz
  truediff.map    — truth.map − fc.map  (CNN training target)

Usage:
  ccp4-python make_maps.py sample_dir/
  ccp4-python make_maps.py data_dir/           # all sample_NNNNN/ subdirs
  ccp4-python make_maps.py dir1/ dir2/ ...
  ccp4-python make_maps.py data_dir/ --force   # overwrite existing maps
  ccp4-python make_maps.py data_dir/ --workers 8
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np

SAMPLE_RATE = 3.0


def _pdb_has_hydrogens(pdb_path):
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith(('ATOM  ', 'HETATM')) and len(line) >= 78:
            if line[76:78].strip() == 'H':
                return True
    return False


def _mtz_to_ccp4(mtz_path, f_col, phi_col, out_path, sample_rate=SAMPLE_RATE):
    mtz  = gemmi.read_mtz_file(str(mtz_path))
    grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=sample_rate)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(out_path))


def convert_sample(sample_dir, force=False, sample_rate=SAMPLE_RATE):
    """Convert refmacout.mtz + truth_full.pdb → 5 CCP4 maps in sample_dir.

    Returns (ok: bool, message: str).
    """
    sample_dir = Path(sample_dir).resolve()
    mtz_r      = sample_dir / 'refmacout.mtz'
    pdb_truth  = sample_dir / 'truth_full.pdb'
    mtz_t      = sample_dir / 'truth.mtz'

    if not mtz_r.exists():
        return False, f'no refmacout.mtz in {sample_dir}'
    if not pdb_truth.exists():
        return False, f'no truth_full.pdb in {sample_dir}'

    output_maps = ['truth.map', '2fofc.map', 'fofc.map', 'fc.map', 'truediff.map']
    if not force and all((sample_dir / m).exists() for m in output_maps):
        return True, 'skipped (all maps present)'

    # Determine resolution from refmacout.mtz
    dmin = gemmi.read_mtz_file(str(mtz_r)).resolution_high()

    # Build truth.mtz via gemmi sfcalc (add H if absent)
    if force or not mtz_t.exists():
        pdb_in = pdb_truth
        with tempfile.TemporaryDirectory() as tmpd:
            tmpdir = Path(tmpd)
            if not _pdb_has_hydrogens(pdb_in):
                phenix_reduce = shutil.which('phenix.reduce')
                if phenix_reduce:
                    r = subprocess.run(
                        [phenix_reduce, str(pdb_in)],
                        cwd=str(tmpdir),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    if r.returncode in (0, 1):
                        pdb_with_h = tmpdir / '_truth_with_h.pdb'
                        pdb_with_h.write_bytes(r.stdout)
                        pdb_in = pdb_with_h

            r = subprocess.run(
                ['gemmi', 'sfcalc', f'--dmin={dmin:.4f}',
                 f'--to-mtz={mtz_t}', str(pdb_in)],
                cwd=str(tmpdir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if r.returncode != 0 or not mtz_t.exists():
                err = r.stderr.decode(errors='replace')[-600:]
                return False, f'gemmi sfcalc failed: {err}'

    # Convert MTZ columns → CCP4 maps
    try:
        _mtz_to_ccp4(mtz_t, 'FC',        'PHIC',        sample_dir / 'truth.map', sample_rate)
        _mtz_to_ccp4(mtz_r, 'FWT',       'PHWT',        sample_dir / '2fofc.map', sample_rate)
        _mtz_to_ccp4(mtz_r, 'DELFWT',    'PHDELWT',     sample_dir / 'fofc.map',  sample_rate)
        _mtz_to_ccp4(mtz_r, 'FC_ALL_LS', 'PHIC_ALL_LS', sample_dir / 'fc.map',    sample_rate)
    except Exception as e:
        return False, f'map conversion failed: {e}'

    # truediff = truth - fc
    try:
        truth_grid = gemmi.read_ccp4_map(str(sample_dir / 'truth.map')).grid
        fc_grid    = gemmi.read_ccp4_map(str(sample_dir / 'fc.map')).grid
        diff_arr   = np.array(truth_grid, copy=False) - np.array(fc_grid, copy=False)
        diff_grid  = gemmi.FloatGrid(diff_arr.astype(np.float32),
                                     truth_grid.unit_cell, truth_grid.spacegroup)
        diff_ccp4  = gemmi.Ccp4Map()
        diff_ccp4.grid = diff_grid
        diff_ccp4.update_ccp4_header()
        diff_ccp4.write_ccp4_map(str(sample_dir / 'truediff.map'))
    except Exception as e:
        return False, f'truediff.map failed: {e}'

    return True, 'ok'


def _collect_dirs(paths):
    """Expand each path: if it has sample_NNNNN/ subdirs, return those; else return as-is."""
    result = []
    for p in paths:
        p = Path(p)
        subdirs = sorted(p.glob('sample_?????'))
        if subdirs:
            result.extend(subdirs)
        else:
            result.append(p)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dirs', nargs='+', metavar='DIR',
                        help='Sample dir(s) or data dir containing sample_NNNNN/ subdirs')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing maps')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers (default 1)')
    parser.add_argument('--sample-rate', type=float, default=SAMPLE_RATE,
                        help=f'Map oversampling rate (default {SAMPLE_RATE})')
    args = parser.parse_args()

    targets = _collect_dirs(args.dirs)
    if not targets:
        print('No directories found.', file=sys.stderr)
        sys.exit(1)

    print(f'Processing {len(targets)} sample(s) with {args.workers} worker(s)...')

    ok_count = fail_count = skip_count = 0

    def _process(d):
        return d, *convert_sample(d, force=args.force, sample_rate=args.sample_rate)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process, d): d for d in targets}
        for fut in as_completed(futs):
            d, ok, msg = fut.result()
            label = Path(d).name
            if ok:
                if 'skipped' in msg:
                    skip_count += 1
                    print(f'  {label}: {msg}')
                else:
                    ok_count += 1
                    print(f'  {label}: {msg}')
            else:
                fail_count += 1
                print(f'  {label}: FAILED — {msg}', file=sys.stderr)

    total = ok_count + fail_count + skip_count
    print(f'\n{total} processed: {ok_count} ok, {skip_count} skipped, {fail_count} failed')
    if fail_count:
        sys.exit(1)


if __name__ == '__main__':
    main()
