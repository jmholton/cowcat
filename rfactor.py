#!/usr/bin/env ccp4-python
"""
rfactor.py — Compute crystallographic R factors from CCP4 map files vs a refmac MTZ.

FreeR_flag convention (CCP4 uniqueify): flag == 0 → FREE set (~5%); flag != 0 → WORK.
Reflections with NaN Fo are withheld (missing/never-collected); reported separately.

Usage:
    ccp4-python rfactor.py \
        --mtz     sample_00000/refmacout.mtz \
        --fc      sample_00000/fc.map \
        --pred    sample_00000/predicted.map \
        --truth   sample_00000/truth.map
"""

import argparse
import numpy as np
import gemmi


def _load_ccp4_array(path):
    ccp4 = gemmi.read_ccp4_map(path, setup=True)
    return np.array(ccp4.grid, dtype=np.float32), ccp4.grid.unit_cell.volume


def _map_to_sf(arr, cell_volume):
    """Full 3D FFT → structure factor amplitudes at all Miller indices."""
    ft = np.fft.fftn(arr.astype(np.float64)) * (cell_volume / arr.size)
    return ft.astype(np.complex64)


def _extract(ft, H, K, L):
    """Extract |F| from FFT array at Miller indices (negative H,K,L wrap via modulo)."""
    nx, ny, nz = ft.shape
    hi = np.asarray(H, dtype=np.int32) % nx
    ki = np.asarray(K, dtype=np.int32) % ny
    li = np.asarray(L, dtype=np.int32) % nz
    return np.abs(ft[hi, ki, li]).astype(np.float32)


def _scale_kb(Fo, Fc, s2, n_cycles=4):
    """F-space nonlinear LS fit of k and B (SC(1)=1 fixed).

    Model: Fc_scaled = k · exp(-B/4 · s²) · Fc
    Minimises Σ(Fo − Fc_scaled)² via scipy LM.  Iterated n_cycles times.
    Returns (k, B).
    """
    from scipy.optimize import least_squares as _lsq
    good = (Fo > 0) & (Fc > 0) & np.isfinite(Fo) & np.isfinite(Fc)
    if good.sum() < 10:
        return 1.0, 0.0
    r, t, ss = Fo[good].astype(np.float64), Fc[good].astype(np.float64), s2[good].astype(np.float64)
    k0 = float(np.dot(r, t) / np.dot(t, t))
    params = np.array([np.log(max(k0, 1e-10)), 0.0])

    def _scale_fn(p):
        return np.exp(p[0]) * np.exp(-p[1] / 4.0 * ss)

    for _ in range(n_cycles):
        try:
            res = _lsq(lambda p: r - _scale_fn(p) * t, params, method='lm', max_nfev=200)
            params = res.x
        except Exception:
            break

    return float(np.exp(params[0])), float(params[1])


def _apply_scale(Fc, k, B, s2):
    return k * Fc * np.exp(-B / 4 * s2)


def _rfactor(Fo, Fc_scaled):
    num = float(np.sum(np.abs(Fo - Fc_scaled)))
    den = float(np.sum(np.abs(Fo)))
    return num / den if den > 0 else float('nan')


def _report(label, Fc_dict, k_dict, masks, names):
    parts = [f'{label:14s}']
    for name, mask in zip(names, masks):
        Fo_m = masks[mask]['Fo']
        Fc_m = Fc_dict[masks[mask]['fc_key']][mask]
        k    = k_dict[masks[mask]['k_key']]
        r    = _rfactor(Fo_m, Fc_m, k)
        parts.append(f'R_{name}={r:.4f} (n={mask.sum()})')
    return '  '.join(parts)


def main():
    ap = argparse.ArgumentParser(description='R factors from maps vs refmac MTZ.')
    ap.add_argument('--mtz',   required=True)
    ap.add_argument('--fc',    required=True)
    ap.add_argument('--pred',  required=True)
    ap.add_argument('--truth', default=None)
    args = ap.parse_args()

    mtz  = gemmi.read_mtz_file(args.mtz)
    H    = np.asarray(mtz.column_with_label('H'),          dtype=np.int32)
    K    = np.asarray(mtz.column_with_label('K'),          dtype=np.int32)
    L    = np.asarray(mtz.column_with_label('L'),          dtype=np.int32)
    Fo   = np.asarray(mtz.column_with_label('F'),          dtype=np.float32)
    free = np.asarray(mtz.column_with_label('FreeR_flag'), dtype=np.float32)

    # s² = 1/d² for each reflection
    cell = mtz.cell
    s2 = np.array([cell.calculate_1_d2([h, k, l])
                   for h, k, l in zip(H.tolist(), K.tolist(), L.tolist())],
                  dtype=np.float32)

    # CCP4 uniqueify convention: flag==0 → FREE, flag!=0 → WORK
    obs    = np.isfinite(Fo) & (Fo > 0)   # measured reflections
    miss   = ~obs                           # withheld/unmeasured
    is_free = (free == 0)
    work   = obs & ~is_free
    free_m = obs &  is_free

    n_tot  = len(H)
    n_obs  = int(obs.sum())
    n_work = int(work.sum())
    n_free = int(free_m.sum())
    n_miss = int(miss.sum())
    print(f'Reflections: {n_tot} total  {n_obs} observed ({n_work} work / {n_free} free)  {n_miss} withheld')

    fc_arr,   vol = _load_ccp4_array(args.fc)
    pred_arr, _   = _load_ccp4_array(args.pred)   # total predicted density (fc + diff)

    ft_fc   = _map_to_sf(fc_arr,   vol)
    ft_pred = _map_to_sf(pred_arr, vol)

    Fc_fc   = _extract(ft_fc,   H, K, L)
    Fc_pred = _extract(ft_pred, H, K, L)

    sources = {'fc': Fc_fc, 'pred': Fc_pred}
    if args.truth:
        truth_arr, _ = _load_ccp4_array(args.truth)
        ft_truth = _map_to_sf(truth_arr, vol)
        sources['truth'] = _extract(ft_truth, H, K, L)

    print()
    header = f'{"":14s}  {"R_work":>8s}  {"R_free":>8s}  {"R_miss":>8s}  {"k":>7s}  {"B":>7s}'
    print(header)
    print('-' * len(header))

    for label, Fc in sources.items():
        k, B = _scale_kb(Fo[work], Fc[work], s2[work])
        Fc_sw = _apply_scale(Fc[work],   k, B, s2[work])
        Fc_sf = _apply_scale(Fc[free_m], k, B, s2[free_m])
        r_work = _rfactor(Fo[work],   Fc_sw)
        r_free = _rfactor(Fo[free_m], Fc_sf)
        if n_miss > 0 and 'truth' in sources:
            Fo_miss_proxy = sources['truth'][miss]
            Fc_sm = _apply_scale(Fc[miss], k, B, s2[miss])
            r_miss = _rfactor(Fo_miss_proxy, Fc_sm)
        else:
            r_miss = float('nan')
        miss_str = f'{r_miss:.4f}' if np.isfinite(r_miss) else '  n/a '
        print(f'{label:14s}  {r_work:.4f}    {r_free:.4f}    {miss_str}    {k:.4f}  {B:+.2f}')


if __name__ == '__main__':
    main()
