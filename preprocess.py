#!/usr/bin/env python3
"""
preprocess.py  –  Cache cross-Patterson channel as crossp.npy.

Reads fofc.map and fc.map directly (no gemmi) and writes crossp.npy alongside
the existing .map files.  Run once before training; dataset.py will use the
cached file automatically.

Usage:
    python preprocess.py --data ./data
    python preprocess.py --data ./data --workers 8   # parallel
"""

import argparse
import os
import concurrent.futures
from pathlib import Path

import numpy as np

REQUIRED = {'fofc.map', 'fc.map'}


def _load_map(path):
    """Read a CCP4 map without gemmi using the header-offset trick."""
    with open(path, 'rb') as f:
        nc, nr, ns = np.frombuffer(f.read(12), dtype=np.int32)
    n = int(nc) * int(nr) * int(ns)
    offset = os.path.getsize(path) - 4 * n
    data = np.fromfile(path, dtype=np.float32, count=n, offset=offset)
    return data.reshape(int(ns), int(nr), int(nc))


def _cross_patterson(fofc_arr, fc_arr):
    """IFFT(FFT(FoFc) * conj(FFT(Fc))) — peaks at vectors to missing atoms."""
    return np.fft.irfftn(
        np.fft.rfftn(fofc_arr) * np.conj(np.fft.rfftn(fc_arr)),
        s=fc_arr.shape,
    ).real.astype(np.float32)


def process_sample(sample_dir):
    sample_dir = Path(sample_dir)
    out = sample_dir / 'crossp.npy'

    if out.exists():
        return str(sample_dir), 'skipped'

    missing = [m for m in REQUIRED if not (sample_dir / m).exists()]
    if missing:
        return str(sample_dir), f'missing: {missing}'

    try:
        fofc = _load_map(str(sample_dir / 'fofc.map'))
        fc   = _load_map(str(sample_dir / 'fc.map'))
        np.save(str(out), _cross_patterson(fofc, fc))
        return str(sample_dir), 'ok'
    except Exception as e:
        return str(sample_dir), f'ERROR: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='./data')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    data = Path(args.data)
    sample_dirs = sorted(d for d in data.iterdir()
                         if d.is_dir() and d.name.startswith('sample_'))
    print(f'Processing {len(sample_dirs)} samples with {args.workers} workers ...')

    ok = skipped = errors = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for path, status in ex.map(process_sample, sample_dirs):
            if status == 'ok':
                ok += 1
            elif status == 'skipped':
                skipped += 1
            else:
                errors += 1
                print(f'  {path}: {status}')

    print(f'Done. ok={ok}  skipped={skipped}  errors={errors}')


if __name__ == '__main__':
    main()
