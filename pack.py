#!/usr/bin/env python3
"""
pack.py  –  Pack sample directories into memory-mappable numpy arrays.

Reads all sample_NNNNN/ directories, computes/loads cross-Patterson,
z-normalises each channel, and writes:
    X.npy  –  float32 (N, 4, D, H, W)  – input channels (znorm'd)
    Y.npy  –  float32 (N, 1, D, H, W)  – znorm'd (truth - Fc) difference map
    S.npy  –  float32 (N,)             – log(std(truth - Fc)) scale factor

After packing, training opens 3 files instead of 5×N files per epoch.

Usage:
    python pack.py --data ./data/data_1aho_s100
    python pack.py --data ./data/data_1aho_s100 --workers 8
    python pack.py --data ./data/data_1aho_s100 --force
"""

import argparse
import os
import concurrent.futures
from pathlib import Path

import numpy as np

REQUIRED = {'truth.map', '2fofc.map', 'fofc.map', 'fc.map'}


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
    """Return (x, y, s) for one sample directory."""
    base = str(base)
    ch0      = _znorm(_load_map(os.path.join(base, '2fofc.map')))
    fofc_raw = _load_map(os.path.join(base, 'fofc.map'))
    fc_raw   = _load_map(os.path.join(base, 'fc.map'))
    ch1 = _znorm(fofc_raw)
    ch2 = _znorm(fc_raw)

    crossp_path = os.path.join(base, 'crossp.npy')
    if os.path.exists(crossp_path):
        _cp = np.load(crossp_path)
        ch3 = _znorm(_cp if _cp.shape == ch0.shape else _cross_patterson(fofc_raw, fc_raw))
    else:
        ch3 = _znorm(_cross_patterson(fofc_raw, fc_raw))

    truth_raw = _load_map(os.path.join(base, 'truth.map'))
    diff_raw  = truth_raw - fc_raw
    log_scale = np.float32(np.log(diff_raw.std() + 1e-8))
    tgt       = _znorm(diff_raw)

    x = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)  # (4,D,H,W)
    y = tgt[np.newaxis].astype(np.float32)                           # (1,D,H,W)
    return x, y, log_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    required=True,
                        help='Directory containing sample_NNNNN/ subdirs')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--force',   action='store_true',
                        help='Overwrite existing packs')
    args = parser.parse_args()

    data = Path(args.data)
    sample_dirs = sorted(
        d for d in data.iterdir()
        if d.is_dir() and d.name.startswith('sample_')
        and REQUIRED.issubset({f.name for f in d.iterdir()})
    )
    n = len(sample_dirs)
    if n == 0:
        print('No valid samples found.')
        return
    print(f'Packing {n} samples from {data} ...')

    x_path = data / 'X.npy'
    if x_path.exists() and not args.force:
        print(f'Pack already exists. Use --force to overwrite.')
        return

    x0, y0, _ = process_sample(sample_dirs[0])
    x_shape = (n,) + x0.shape   # (N, 4, D, H, W)
    y_shape = (n,) + y0.shape   # (N, 1, D, H, W)
    nbytes = (np.prod(x_shape) + np.prod(y_shape) + n) * 4
    print(f'Grid: {x0.shape[1:]}  Output: {nbytes / 1e9:.2f} GB')

    X = np.lib.format.open_memmap(str(data / 'X.npy'), mode='w+',
                                   dtype=np.float32, shape=x_shape)
    Y = np.lib.format.open_memmap(str(data / 'Y.npy'), mode='w+',
                                   dtype=np.float32, shape=y_shape)
    S = np.lib.format.open_memmap(str(data / 'S.npy'), mode='w+',
                                   dtype=np.float32, shape=(n,))
    X[0] = x0
    Y[0] = y0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_sample, d): i
                   for i, d in enumerate(sample_dirs[1:], start=1)}
        done = 1
        errors = 0
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                x, y, s = fut.result()
                X[i] = x
                Y[i] = y
                S[i] = s
                done += 1
            except Exception as e:
                errors += 1
                print(f'  ERROR {sample_dirs[i].name}: {e}')
            if (done + errors) % 100 == 0 or (done + errors) == n:
                print(f'  {done + errors}/{n}', flush=True)

    del X, Y, S
    print(f'Done. ok={done}  errors={errors}')


if __name__ == '__main__':
    main()
