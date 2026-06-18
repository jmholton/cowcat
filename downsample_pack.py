#!/usr/bin/env python3
"""Downsample packed X.npy/Y.npy by striding in spatial dims (no averaging).

Usage:
    python downsample_pack.py --data data/data_simple_v2_s0 --stride 2
    python downsample_pack.py --data data/data_simple_v2_s0 --stride 2 --outdir data/data_simple_v2_s0_ds2
"""

import argparse
import os
import numpy as np
from pathlib import Path


def _open_npy(path, dtype, shape):
    f = open(str(path), 'wb')
    np.lib.format.write_array_header_2_0(
        f, np.lib.format.header_data_from_array_1_0(np.empty(shape, dtype=dtype)))
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   required=True)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--outdir', default=None,
                        help='Output dir (default: <data>_ds<stride>)')
    args = parser.parse_args()

    src = Path(args.data)
    dst = Path(args.outdir) if args.outdir else src.parent / (src.name + f'_ds{args.stride}')
    dst.mkdir(parents=True, exist_ok=True)

    X = np.load(src / 'X.npy', mmap_mode='r')
    Y = np.load(src / 'Y.npy', mmap_mode='r')
    S = np.load(src / 'S.npy', mmap_mode='r')

    n = len(X)
    s = args.stride
    x0 = X[0, :, ::s, ::s, ::s]
    y0 = Y[0, :, ::s, ::s, ::s]
    x_shape = (n,) + x0.shape
    y_shape = (n,) + y0.shape

    print(f'Input:  {X.shape[2:]}  →  Output: {x0.shape[1:]}  (stride={s})')
    print(f'Writing to {dst} ...')

    fx = _open_npy(dst / 'X.npy', np.float32, x_shape)
    fy = _open_npy(dst / 'Y.npy', np.float32, y_shape)

    for i in range(n):
        fx.write(X[i, :, ::s, ::s, ::s].tobytes())
        fy.write(Y[i, :, ::s, ::s, ::s].tobytes())
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f'  {i + 1}/{n}', flush=True)

    fx.close(); fy.close()

    # Copy S unchanged
    np.save(str(dst / 'S.npy'), S)

    for label, p in [('X', dst/'X.npy'), ('Y', dst/'Y.npy'), ('S', dst/'S.npy')]:
        if os.path.getsize(p) == 0:
            raise RuntimeError(f'{label}.npy is empty: {p}')
    print(f'Done. Grid {X.shape[2:]} → {x0.shape[1:]}')


if __name__ == '__main__':
    main()
