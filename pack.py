#!/usr/bin/env python3
"""
pack.py  –  Pack sample directories into two memory-mappable numpy arrays.

Reads all sample_NNNNN/ directories, computes/loads cross-Patterson,
z-normalises each channel, and writes:
    X.npy  –  float32 (N, 4, D, H, W)  – input channels
    Y.npy  –  float32 (N, 1, D, H, W)  – truth target

After packing, training opens 2 files instead of 6×N files, eliminating
filesystem overhead on large datasets.

Usage:
    python pack.py --data ./data_N80
    python pack.py --data ./data_N80 --workers 8
"""

import argparse
import os
import concurrent.futures
from pathlib import Path

import numpy as np

REQUIRED = {'truth.map', '2fofc.map', 'fofc.map', 'fc.map', 'metadata.json'}


def _load_map(path):
    with open(path, 'rb') as f:
        nc, nr, ns = np.frombuffer(f.read(12), dtype=np.int32)
    n = int(nc) * int(nr) * int(ns)
    offset = os.path.getsize(path) - 4 * n
    data = np.fromfile(path, dtype=np.float32, count=n, offset=offset)
    return data.reshape(int(ns), int(nr), int(nc))


def _cross_patterson(fofc_arr, fc_arr):
    return np.fft.irfftn(
        np.fft.rfftn(fofc_arr) * np.conj(np.fft.rfftn(fc_arr)),
        s=fc_arr.shape,
    ).real.astype(np.float32)


def _znorm(arr):
    std = arr.std()
    if std < 1e-8:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def process_sample(base):
    base = str(base)
    ch0      = _znorm(_load_map(os.path.join(base, '2fofc.map')))
    fofc_raw = _load_map(os.path.join(base, 'fofc.map'))
    fc_raw   = _load_map(os.path.join(base, 'fc.map'))
    ch1 = _znorm(fofc_raw)
    ch2 = _znorm(fc_raw)

    crossp_path = os.path.join(base, 'crossp.npy')
    if os.path.exists(crossp_path):
        ch3 = _znorm(np.load(crossp_path))
    else:
        ch3 = _znorm(_cross_patterson(fofc_raw, fc_raw))

    tgt = _znorm(_load_map(os.path.join(base, 'truth.map')))

    x = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)  # (4,D,H,W)
    y = tgt[np.newaxis].astype(np.float32)                           # (1,D,H,W)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='./data')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    data = Path(args.data)
    sample_dirs = sorted(
        d for d in data.iterdir()
        if d.is_dir() and d.name.startswith('sample_')
        and REQUIRED.issubset({f.name for f in d.iterdir()})
    )
    n = len(sample_dirs)
    print(f'Packing {n} samples from {data} ...')

    # Load first sample to get grid shape
    x0, y0 = process_sample(sample_dirs[0])
    x_shape = (n,) + x0.shape  # (N, 4, D, H, W)
    y_shape = (n,) + y0.shape  # (N, 1, D, H, W)
    print(f'Grid shape: {x0.shape[1:]}  X: {x_shape}  Y: {y_shape}')
    nbytes = (np.prod(x_shape) + np.prod(y_shape)) * 4
    print(f'Output size: {nbytes / 1e9:.2f} GB')

    X = np.lib.format.open_memmap(str(data / 'X.npy'), mode='w+',
                                   dtype=np.float32, shape=x_shape)
    Y = np.lib.format.open_memmap(str(data / 'Y.npy'), mode='w+',
                                   dtype=np.float32, shape=y_shape)
    X[0] = x0
    Y[0] = y0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_sample, d): i
                   for i, d in enumerate(sample_dirs[1:], start=1)}
        done = 1
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            x, y = fut.result()
            X[i] = x
            Y[i] = y
            done += 1
            if done % 500 == 0 or done == n:
                print(f'  {done}/{n}')

    # Flush to disk
    del X, Y
    print('Done.')


if __name__ == '__main__':
    main()
