#!/usr/bin/env python3
"""
pack.py  –  Pack sample directories into memory-mappable numpy arrays.

Reads all sample_NNNNN/ directories, computes/loads cross-Patterson,
and writes:
    X.npy  –  float32 (N, 4, D, H, W)  – input channels in e/Å³ (ch0-2 raw;
                                           ch3 cross-Patterson signed-sqrt)
    Y.npy  –  float32 (N, 1, D, H, W)  – (truth - Fc) difference map in e/Å³
    S.npy  –  float32 (N,)             – std(truth - Fc) in e/Å³

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


def _signed_sqrt(arr):
    """sign(x) * sqrt(|x|) — compresses dynamic range while preserving sign."""
    return np.sign(arr) * np.sqrt(np.abs(arr))


def _unit_ratio_deconv(fofc_arr, fc_arr):
    """Unit-ratio deconvolution of Fo-Fc by Fc.

    For each reciprocal-space frequency h, compute the complex ratio
    r(h) = rfft(fofc)(h) / rfft(fc)(h), then weight by
    W(h) = min(|r|, 1/|r|) -- bounded [0,1], maximum when |fofc|=|fc|.
    The output is irfftn(W * r/|r|): phase difference weighted by how
    similar the two amplitudes are at each frequency.

    This suppresses Harker cross-terms (|fc|>>|fofc| -> W->0) and avoids
    pure-deconvolution blowup (|fc|->0 -> ratio->inf -> W->0 too).
    """
    eps = 1e-6
    F_fofc = np.fft.rfftn(fofc_arr)
    F_fc   = np.fft.rfftn(fc_arr)
    ratio  = F_fofc / (F_fc + eps)
    amp    = np.abs(ratio)
    W      = np.where(amp > 1.0, 1.0 / (amp + eps), amp)
    unit_phase = ratio / (amp + eps)
    result = np.fft.irfftn(W * unit_phase, s=fc_arr.shape)
    return result.real.astype(np.float32)


def _softsign_deconv(fofc_arr, fc_arr):
    """Softsign Fc-deconvolution: ratio / (1 + |ratio|).

    Bounded (-1, 1), monotonic, zero at perfect fit (ratio=0), only saturates
    at the high end. Unlike Möbius, the zero-crossing is at ratio=0 (not at
    unit ratio), so well-fitted reflections map near 0 and large discrepancies
    approach ±1.
    """
    eps = 1e-6
    F_fofc = np.fft.rfftn(fofc_arr)
    F_fc   = np.fft.rfftn(fc_arr)
    ratio  = F_fofc / (F_fc + eps)
    amp    = np.abs(ratio)
    return np.fft.irfftn(ratio / (1.0 + amp), s=fc_arr.shape).real.astype(np.float32)


def _mobius_deconv(fofc_arr, fc_arr):
    """Möbius-bounded Fc-deconvolution of Fo-Fc.

    ratio = rfft(fofc) / rfft(fc)  — real (Fc phases cancel), may be large.
    Mapped through (amp-1)/(amp+1) * sign, which is bounded [-1,1], monotonic,
    and zero at unit ratio. Unlike unit-ratio, this is injective: ratio=0.5 and
    ratio=2.0 give distinct outputs (-1/3 and +1/3 respectively).
    """
    eps = 1e-6
    F_fofc = np.fft.rfftn(fofc_arr)
    F_fc   = np.fft.rfftn(fc_arr)
    ratio      = F_fofc / (F_fc + eps)
    amp        = np.abs(ratio)
    f          = (amp - 1.0) / (amp + 1.0)   # Möbius: [-1, 1], monotonic
    unit_phase = ratio / (amp + eps)          # ±1 (real, since ratio is real)
    return np.fft.irfftn(f * unit_phase, s=fc_arr.shape).real.astype(np.float32)


def process_sample(base, crossp_transform='signed_sqrt'):
    """Return (x, y, s) for one sample directory.

    x: (4,D,H,W) float32 — ch0-2 in raw e/Å³; ch3 cross-Patterson (see crossp_transform)
    y: (1,D,H,W) float32 — truth−Fc difference map in e/Å³
    s: float32            — std(truth−Fc) in e/Å³

    crossp_transform: 'signed_sqrt' (default), 'raw', or 'unitratio'
    """
    base = str(base)
    ch0      = _load_map(os.path.join(base, '2fofc.map'))
    fofc_raw = _load_map(os.path.join(base, 'fofc.map'))
    fc_raw   = _load_map(os.path.join(base, 'fc.map'))
    ch1 = fofc_raw
    ch2 = fc_raw

    if crossp_transform == 'softsign':
        ch3 = _softsign_deconv(fofc_raw, fc_raw)
    elif crossp_transform == 'mobius':
        ch3 = _mobius_deconv(fofc_raw, fc_raw)
    elif crossp_transform == 'unitratio':
        ch3 = _unit_ratio_deconv(fofc_raw, fc_raw)
    else:
        crossp_path = os.path.join(base, 'crossp.npy')
        if os.path.exists(crossp_path):
            _cp = np.load(crossp_path)
            cp_raw = _cp if _cp.shape == ch0.shape else _cross_patterson(fofc_raw, fc_raw)
        else:
            cp_raw = _cross_patterson(fofc_raw, fc_raw)
        ch3 = _signed_sqrt(cp_raw) if crossp_transform == 'signed_sqrt' else cp_raw.astype(np.float32)

    truth_raw = _load_map(os.path.join(base, 'truth.map'))
    diff_raw  = truth_raw - fc_raw
    scale     = np.float32(diff_raw.std())
    tgt       = diff_raw

    x = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)  # (4,D,H,W)
    y = tgt[np.newaxis].astype(np.float32)                           # (1,D,H,W)
    return x, y, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    required=True, action='append',
                        help='Directory containing sample_NNNNN/ subdirs (repeatable)')
    parser.add_argument('--workers', type=int, default=8,
                        help='(ignored — packing is now sequential to bound memory)')
    parser.add_argument('--force',   action='store_true',
                        help='Overwrite existing packs')
    parser.add_argument('--crossp-raw', action='store_true',
                        help='Store raw cross-Patterson in ch3 (default: signed-sqrt)')
    parser.add_argument('--crossp-unitratio', action='store_true',
                        help='Store unit-ratio deconvolution in ch3 instead of cross-Patterson')
    parser.add_argument('--softsign', action='store_true',
                        help='Store softsign Fc-deconvolution in ch3: ratio/(1+|ratio|), '
                             'bounded (-1,1), zero at perfect fit')
    parser.add_argument('--mobius', action='store_true',
                        help='Store Möbius-bounded Fc-deconvolution in ch3 (monotonic, bounded [-1,1])')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (default: same as --data)')
    args = parser.parse_args()

    if args.softsign:
        crossp_transform = 'softsign'
    elif args.mobius:
        crossp_transform = 'mobius'
    elif args.crossp_unitratio:
        crossp_transform = 'unitratio'
    elif args.crossp_raw:
        crossp_transform = 'raw'
    else:
        crossp_transform = 'signed_sqrt'
    data_dirs = [Path(d) for d in args.data]
    if args.outdir:
        out = Path(args.outdir)
    elif len(data_dirs) == 1:
        out = data_dirs[0]
    else:
        print('Error: --outdir required when multiple --data dirs are given.')
        return
    out.mkdir(parents=True, exist_ok=True)
    sample_dirs = []
    for data in data_dirs:
        sample_dirs.extend(sorted(
            d for d in data.iterdir()
            if d.is_dir() and d.name.startswith('sample_')
            and REQUIRED.issubset({f.name for f in d.iterdir()})
        ))
    n = len(sample_dirs)
    if n == 0:
        print('No valid samples found.')
        return
    src_str = ', '.join(str(d) for d in data_dirs)
    print(f'Packing {n} samples from [{src_str}]  crossp={crossp_transform}  →  {out}')

    x_path = out / 'X.npy'
    if x_path.exists() and not args.force:
        print(f'Pack already exists. Use --force to overwrite.')
        return

    x0, y0, s0 = process_sample(sample_dirs[0], crossp_transform)
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
        header = {'descr': np.lib.format.dtype_to_descr(np.dtype(dtype)),
                  'fortran_order': False,
                  'shape': tuple(shape)}
        np.lib.format.write_array_header_2_0(f, header)
        return f

    fx = _open_npy(out / 'X.npy', np.float32, tuple(x_shape))
    fy = _open_npy(out / 'Y.npy', np.float32, tuple(y_shape))
    fs = _open_npy(out / 'S.npy', np.float32, (n,))

    done = errors = 0
    s_buf = np.zeros(n, dtype=np.float32)
    cached = {0: (x0, y0, s0)}  # reuse sample 0 already computed for shape
    for i, d in enumerate(sample_dirs):
        try:
            x, y, s = cached.pop(i) if i in cached else process_sample(d, crossp_transform)
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

    for label, path in [('X', out / 'X.npy'), ('Y', out / 'Y.npy'), ('S', out / 'S.npy')]:
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(f'{label}.npy is empty after packing: {path}')
    print(f'Done. ok={done}  errors={errors}')


if __name__ == '__main__':
    main()
