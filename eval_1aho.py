"""
eval_1aho.py — in-training Rfree-vs-real-1aho diagnostic.

Loaded by train.py to report Rfree on a held-out real-data sample (typically
1aho_test/) every epoch as a generalisation check that the val-set MSE alone
cannot give.

Usage from train.py:
    ctx   = setup_1aho_eval('1aho_test')              # rank 0, at startup
    rfree = eval_rfree(raw_model, ctx, device)        # rank 0, end of each epoch

Returns None from setup_1aho_eval if anything fails (ccp4-python not on PATH,
maps missing, ...). The training continues without the column.

The ccp4-python subprocess at startup extracts FP/H/K/L/FreeR_flag/s² from
the MTZ into a numpy cache; per-epoch work is pure pytorch+numpy.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


# ── Map I/O / input construction ─────────────────────────────────────────────

def _load_ccp4_map(path):
    """Read a CCP4 map header-agnostically. Returns (NS, NR, NC) float32."""
    with open(path, 'rb') as f:
        nc, nr, ns = np.frombuffer(f.read(12), dtype=np.int32)
    n      = int(nc) * int(nr) * int(ns)
    offset = os.path.getsize(path) - 4 * n
    data   = np.fromfile(path, dtype=np.float32, count=n, offset=offset)
    return data.reshape(int(ns), int(nr), int(nc))


def _cross_patterson(fofc, fc):
    """Same operator as pack.py / infer.py: irfft( rfft(fofc) · conj(rfft(fc)) )."""
    return np.fft.irfftn(
        np.fft.rfftn(fofc) * np.conj(np.fft.rfftn(fc)),
        s=fc.shape).real.astype(np.float32)


def _signed_sqrt(arr):
    return np.sign(arr) * np.sqrt(np.abs(arr))


def _unit_ratio_deconv(fofc_arr, fc_arr):
    """Unit-ratio deconvolution — matches pack.py / infer.py implementation."""
    eps = 1e-6
    F_fofc = np.fft.rfftn(fofc_arr)
    F_fc   = np.fft.rfftn(fc_arr)
    ratio  = F_fofc / (F_fc + eps)
    amp    = np.abs(ratio)
    W      = np.where(amp > 1.0, 1.0 / (amp + eps), amp)
    unit_phase = ratio / (amp + eps)
    result = np.fft.irfftn(W * unit_phase, s=fc_arr.shape)
    return result.real.astype(np.float32)


def _build_input(twofofc, fofc, fc, crossp_raw=False, crossp_unitratio=False):
    """4-channel input tensor.

    ch3 encoding must match the pack.py variant used during training:
      default           -- signed-sqrt cross-Patterson (*_ssqrt packs)
      crossp_raw=True   -- raw cross-Patterson (*_rawcrossp packs)
      crossp_unitratio  -- unit-ratio deconvolution (*_unitratio packs)
    """
    if crossp_unitratio:
        ch3 = _unit_ratio_deconv(fofc, fc)
    else:
        crossp = _cross_patterson(fofc, fc)
        ch3 = crossp if crossp_raw else _signed_sqrt(crossp)
    x   = np.stack([twofofc, fofc, fc, ch3], axis=0).astype(np.float32)
    return torch.from_numpy(x[np.newaxis])   # (1, 4, D, H, W)


# ── MTZ extraction (one-shot subprocess to ccp4-python) ──────────────────────

def _read_mtz_via_ccp4python(mtz_path, fo_label, free_label):
    """Extract H/K/L/Fo/free/s²/cell from MTZ. Returns dict or None on failure."""
    cache  = f'/tmp/_1aho_mtz_{os.getpid()}.npz'
    script = (
        'import gemmi, numpy as np; '
        f'mtz = gemmi.read_mtz_file({str(mtz_path)!r}); '
        'H = np.asarray(mtz.column_with_label("H"), dtype=np.int32); '
        'K = np.asarray(mtz.column_with_label("K"), dtype=np.int32); '
        'L = np.asarray(mtz.column_with_label("L"), dtype=np.int32); '
        f'Fo = np.asarray(mtz.column_with_label({fo_label!r}), dtype=np.float32); '
        f'free = np.asarray(mtz.column_with_label({free_label!r}), dtype=np.float32); '
        'cell = mtz.cell; '
        's2 = np.array([cell.calculate_1_d2([int(h), int(k), int(l)]) '
        '               for h, k, l in zip(H, K, L)], dtype=np.float32); '
        f'np.savez({cache!r}, H=H, K=K, L=L, Fo=Fo, free=free, s2=s2, '
        '          cell_a=cell.a, cell_b=cell.b, cell_c=cell.c)'
    )
    try:
        subprocess.run(['ccp4-python', '-c', script], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        msg = e.stderr.decode(errors='replace')[-400:] if hasattr(e, 'stderr') and e.stderr else str(e)
        print(f'  eval_1aho: ccp4-python MTZ extraction failed: {msg}', file=sys.stderr)
        return None
    d = np.load(cache)
    return {k: np.array(d[k]) for k in d.files}


# ── F-space k+B scaling (same as rfactor.py) ─────────────────────────────────

def _scale_kb(Fo, Fc, s2, n_cycles=4):
    from scipy.optimize import least_squares
    good = (Fo > 0) & (Fc > 0) & np.isfinite(Fo) & np.isfinite(Fc)
    if good.sum() < 10:
        return 1.0, 0.0
    r  = Fo[good].astype(np.float64)
    t  = Fc[good].astype(np.float64)
    ss = s2[good].astype(np.float64)
    k0 = float(np.dot(r, t) / np.dot(t, t))
    params = np.array([np.log(max(k0, 1e-10)), 0.0])
    for _ in range(n_cycles):
        try:
            res = least_squares(
                lambda p: r - np.exp(p[0]) * np.exp(-p[1] / 4.0 * ss) * t,
                params, method='lm', max_nfev=200)
            params = res.x
        except Exception:
            break
    return float(np.exp(params[0])), float(params[1])


# ── Public API ───────────────────────────────────────────────────────────────

def setup_1aho_eval(eval_dir, fo_label='FP', free_label='FreeR_flag',
                    mtz_name='refmacout_minRfree.mtz', crossp_raw=False,
                    crossp_unitratio=False):
    """Build the context dict consumed by eval_rfree. Returns None on failure.

    Loads twofofc/fofc/fc maps + extracts MTZ data. ccp4-python must be on PATH
    (for the MTZ read). Everything per-epoch is then pure pytorch+numpy.
    """
    d = Path(eval_dir)
    needed = [d / '2fofc.map', d / 'fofc.map', d / 'fc.map', d / mtz_name]
    for p in needed:
        if not p.exists():
            print(f'  eval_1aho: missing {p}; disabled', file=sys.stderr)
            return None

    twofofc = _load_ccp4_map(d / '2fofc.map')
    fofc    = _load_ccp4_map(d / 'fofc.map')
    fc      = _load_ccp4_map(d / 'fc.map')
    x       = _build_input(twofofc, fofc, fc, crossp_raw=crossp_raw,
                           crossp_unitratio=crossp_unitratio)

    mtz = _read_mtz_via_ccp4python(d / mtz_name, fo_label, free_label)
    if mtz is None:
        return None

    Fo, free = mtz['Fo'], mtz['free']
    obs      = np.isfinite(Fo) & (Fo > 0)
    is_free  = (free == 0)            # CCP4 uniqueify: flag==0 → FREE
    ctx = dict(
        x=x, fofc=fofc, fc=fc,
        H=mtz['H'].astype(np.int32),
        K=mtz['K'].astype(np.int32),
        L=mtz['L'].astype(np.int32),
        Fo=Fo, s2=mtz['s2'],
        work=obs & ~is_free, free_m=obs & is_free,
        cell_a=float(mtz['cell_a']), cell_b=float(mtz['cell_b']),
        cell_c=float(mtz['cell_c']),
    )
    print(f'  eval_1aho: enabled on {d}  '
          f'({ctx["work"].sum()} work / {ctx["free_m"].sum()} free reflections)')
    return ctx


def eval_rfree(model, ctx, device):
    """Return R_free of the model's predicted total map against ctx['Fo'].

    Applies the same amplitude rescale infer.py uses (demean + RMS-match to fofc)
    so the model's output is at a sensible scale before the k+B fit.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out  = model(ctx['x'].to(device))
            pred = (out[0] if isinstance(out, (tuple, list)) else out)[0, 0].cpu().numpy()
    finally:
        if was_training:
            model.train()

    fofc  = ctx['fofc']
    pred  = pred - pred.mean()
    std_p = float(pred.std())
    if std_p > 0:
        pred = pred * (float(fofc.std()) / std_p)
    pred_total = pred + ctx['fc']

    vol = ctx['cell_a'] * ctx['cell_b'] * ctx['cell_c']
    ft  = np.fft.fftn(pred_total.astype(np.float64)) * (vol / pred_total.size)
    # Raw CCP4 maps load as (NS, NR, NC) = (Z, Y, X) for the standard
    # MAPC=1, MAPR=2, MAPS=3 axis convention. Miller H/K/L correspond to
    # frequencies along (X, Y, Z) → index ft[L, K, H], not ft[H, K, L].
    nz_g, ny_g, nx_g = ft.shape
    Fc = np.abs(ft[ctx['L'] % nz_g, ctx['K'] % ny_g, ctx['H'] % nx_g]).astype(np.float32)

    work, free_m = ctx['work'], ctx['free_m']
    k, B = _scale_kb(ctx['Fo'][work], Fc[work], ctx['s2'][work])
    Fc_s = k * Fc[free_m] * np.exp(-B / 4 * ctx['s2'][free_m])
    den  = float(np.sum(np.abs(ctx['Fo'][free_m])))
    return float(np.sum(np.abs(ctx['Fo'][free_m] - Fc_s))) / max(den, 1e-12)
