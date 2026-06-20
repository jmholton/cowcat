#!/usr/bin/env python3
"""Measure R-iso between flooded truth.mtz and unflooded gt48 SFs.

For each sample_NNNNN/ in a data directory:
  R_iso = sum|F_flood - F_gt48| / sum|F_gt48|
where F_flood = FC/PHIC from truth.mtz (sfcalc of flooded structure)
  and F_gt48  = Fgt/PHIgt from 1aho/gt48.mtz

Also reads the Rfree from refmac (metadata.json) for comparison.

Usage:
    ccp4-python measure_flood_riso.py data/data_1aho_flood10
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import gemmi

SCRIPT_DIR = Path(__file__).parent
GT48_MTZ   = SCRIPT_DIR / '1aho' / 'gt48.mtz'


def load_sf_dict(mtz_path, f_col, phi_col):
    mtz = gemmi.read_mtz_file(str(mtz_path))
    h  = np.asarray(mtz.column_with_label('H'),     dtype=np.int32)
    k  = np.asarray(mtz.column_with_label('K'),     dtype=np.int32)
    l  = np.asarray(mtz.column_with_label('L'),     dtype=np.int32)
    F  = np.asarray(mtz.column_with_label(f_col),   dtype=np.float64)
    ph = np.asarray(mtz.column_with_label(phi_col), dtype=np.float64)
    valid = np.isfinite(F) & (F > 0)
    return {
        (int(h[i]), int(k[i]), int(l[i])): F[i] * np.exp(1j * np.radians(ph[i]))
        for i in range(len(h)) if valid[i]
    }


def r_iso(sf_a, sf_b):
    """R_iso = sum|F_a - F_b| / sum|F_b| over common HKLs."""
    hkls = sorted(set(sf_a) & set(sf_b))
    if not hkls:
        return float('nan'), 0
    Fa = np.array([sf_a[hkl] for hkl in hkls])
    Fb = np.array([sf_b[hkl] for hkl in hkls])
    return float(np.sum(np.abs(Fa - Fb)) / np.sum(np.abs(Fb))), len(hkls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datadir')
    ap.add_argument('--gt48-mtz', default=str(GT48_MTZ))
    args = ap.parse_args()

    datadir  = Path(args.datadir)
    gt48_mtz = Path(args.gt48_mtz)

    if not gt48_mtz.exists():
        print(f'ERROR: gt48.mtz not found: {gt48_mtz}', file=sys.stderr)
        sys.exit(1)

    gt48_sfs = load_sf_dict(gt48_mtz, 'Fgt', 'PHIgt')
    print(f'gt48 reflections: {len(gt48_sfs)}')

    samples = sorted(datadir.glob('sample_*'))
    if not samples:
        print(f'No sample_* dirs in {datadir}')
        sys.exit(1)

    risos, rfrees, rworks = [], [], []
    print(f'\n{"sample":<14}  {"R_iso":>7}  {"nhkl":>6}  {"R_work":>7}  {"R_free":>7}')
    print('-' * 56)

    for s in samples:
        truth_mtz = s / 'truth.mtz'
        meta_path = s / 'metadata.json'

        if not truth_mtz.exists():
            # truth.map exists but truth.mtz may have been removed; skip quietly
            print(f'{s.name:<14}  (no truth.mtz)')
            continue

        try:
            truth_sfs = load_sf_dict(truth_mtz, 'FC', 'PHIC')
        except Exception as e:
            print(f'{s.name:<14}  ERROR loading truth.mtz: {e}')
            continue

        ri, nhkl = r_iso(truth_sfs, gt48_sfs)
        risos.append(ri)

        rwork = rfree = float('nan')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            rwork = meta.get('rwork', float('nan')) or float('nan')
            rfree = meta.get('rfree', float('nan')) or float('nan')
            if rfree is not None and not np.isnan(rfree):
                rfrees.append(rfree)
            if rwork is not None and not np.isnan(rwork):
                rworks.append(rwork)

        print(f'{s.name:<14}  {ri:7.4f}  {nhkl:6d}  {rwork:7.4f}  {rfree:7.4f}')

    print('-' * 56)
    if risos:
        print(f'{"mean":<14}  {np.mean(risos):7.4f}           '
              f'{np.mean(rworks) if rworks else float("nan"):7.4f}  '
              f'{np.mean(rfrees) if rfrees else float("nan"):7.4f}')
        print(f'{"std":<14}  {np.std(risos):7.4f}           '
              f'{np.std(rworks) if rworks else float("nan"):7.4f}  '
              f'{np.std(rfrees) if rfrees else float("nan"):7.4f}')


if __name__ == '__main__':
    main()
