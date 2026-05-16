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
    parser.add_argument('--workers', type=int, default=8,
                        help='(ignored — packing is now sequential to bound memory)')
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

    x0, y0, s0 = process_sample(sample_dirs[0])
    x_shape = (n,) + x0.shape   # (N, 4, D, H, W)
    y_shape = (n,) + y0.shape   # (N, 1, D, H, W)
    nbytes = (np.prod(x_shape) + np.prod(y_shape) + n) * 4
    print(f'Grid: {x0.shape[1:]}  Output: {nbytes / 1e9:.2f} GB')

    # Write using regular sequential file I/O — one sample at a time — so
    # written pages are immediately evictable and never counted against the
    # process cgroup memory limit the way mmap dirty pages are.
    def _open_npy(path, dtype, shape):
        """Open a .npy file for sequential writing, emit header, return file."""
        f = open(str(path), 'wb')
        np.lib.format.write_array_header_2_0(f, np.lib.format.header_data_from_array_1_0(
            np.empty(shape, dtype=dtype)))
        return f

    fx = _open_npy(data / 'X.npy', np.float32, tuple(x_shape))
    fy = _open_npy(data / 'Y.npy', np.float32, tuple(y_shape))
    fs = _open_npy(data / 'S.npy', np.float32, (n,))

    done = errors = 0
    s_buf = np.zeros(n, dtype=np.float32)
    cached = {0: (x0, y0, s0)}  # reuse sample 0 already computed for shape
    for i, d in enumerate(sample_dirs):
        try:
            x, y, s = cached.pop(i) if i in cached else process_sample(d)
            fx.write(x.tobytes()); fy.write(y.tobytes()); s_buf[i] = s
            done += 1
        except Exception as e:
            fx.write(np.zeros(x_shape[1:], dtype=np.float32).tobytes())
            fy.write(np.zeros(y_shape[1:], dtype=np.float32).tobytes())
            errors += 1
            print(f'  ERROR {d.name}: {e}')
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f'  {i + 1}/{n}', flush=True)

    fs.write(s_buf.tobytes())
    fx.close(); fy.close(); fs.close()

    for label, path in [('X', data / 'X.npy'), ('Y', data / 'Y.npy'), ('S', data / 'S.npy')]:
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(f'{label}.npy is empty after packing: {path}')
    print(f'Done. ok={done}  errors={errors}')


if __name__ == '__main__':
    main()
