"""
infer_multisample.py — run a checkpoint over N synthetic samples and
report per-sample R_work / R_free / R_miss against each sample's
`refmacout.mtz`, plus aggregate statistics.

Each sample dir is expected to contain: 2fofc.map, fofc.map, fc.map,
truth.map, refmacout.mtz. Output map is NOT written (memory-only).

R_miss uses |F| computed from truth.map (FFT) as the "Fo proxy" for
withheld/unmeasured reflections, scaled with the same k+B fit derived
from work reflections — same convention as rfactor.py.

ccp4-python is subprocessed once at startup to extract every MTZ's
H/K/L/Fo/free/s² into an .npz cache (under /tmp/_mtz_cache_*.npz)
then the per-sample loop is pure pytorch+numpy.

Usage:
    python infer_multisample.py \
        --checkpoint checkpoints_v4_ssqrt_pretrain/best.pt \
        --data-dir data/data_protein_v4_s0 \
        --n 100
    # add --crossp-raw for *_rawcrossp-trained models
    # add --val-only to use only the last 20% (the val split)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Pull in the same map I/O + input-build helpers as eval_1aho/infer.
from eval_1aho import (_load_ccp4_map, _cross_patterson, _signed_sqrt,
                       _unit_ratio_deconv, _build_input, _scale_kb)


# ── MTZ batch extraction ──────────────────────────────────────────────────────

def extract_all_mtzs(sample_paths, cache_dir, fo_label='F', free_label='FreeR_flag'):
    """Subprocess to ccp4-python: extract H/K/L/Fo/free/s² for each MTZ → per-sample npz.

    Different samples may have slightly different reflection counts (refmac
    can drop reflections), so each is cached separately.
    Returns a list of paths to the per-sample npz files.
    """
    samples_str = ','.join(str(p) for p in sample_paths)
    script = f'''
import gemmi, numpy as np, sys, os

samples = "{samples_str}".split(",")
cache_dir = "{cache_dir}"
for i, s in enumerate(samples):
    m = gemmi.read_mtz_file(os.path.join(s, "refmacout.mtz"))
    H = np.asarray(m.column_with_label("H"), dtype=np.int32)
    K = np.asarray(m.column_with_label("K"), dtype=np.int32)
    L = np.asarray(m.column_with_label("L"), dtype=np.int32)
    Fo = np.asarray(m.column_with_label("{fo_label}"), dtype=np.float32)
    free = np.asarray(m.column_with_label("{free_label}"), dtype=np.float32)
    cell = m.cell
    s2 = np.array([cell.calculate_1_d2([int(h), int(k), int(l)])
                   for h, k, l in zip(H, K, L)], dtype=np.float32)
    np.savez(os.path.join(cache_dir, f"sample_{{i:05d}}.npz"),
             H=H, K=K, L=L, Fo=Fo, free=free, s2=s2,
             cell_a=cell.a, cell_b=cell.b, cell_c=cell.c)
    if (i + 1) % 25 == 0:
        sys.stderr.write(f"  mtz {{i+1}}/{{len(samples)}}\\n")
'''
    print(f'Extracting {len(sample_paths)} MTZs via ccp4-python ...', flush=True)
    t0 = time.time()
    r = subprocess.run(['ccp4-python', '-c', script], stderr=subprocess.PIPE)
    if r.returncode != 0:
        print('ccp4-python failed:', file=sys.stderr)
        print(r.stderr.decode(errors='replace'), file=sys.stderr)
        sys.exit(1)
    print(f'  done in {time.time()-t0:.1f}s', flush=True)
    return [Path(cache_dir) / f'sample_{i:05d}.npz' for i in range(len(sample_paths))]


# ── F-extraction (matches rfactor.py and eval_1aho.py axis convention) ────────

def map_to_F(arr, vol, H, K, L):
    """FFT a real-space map and extract |F| at Miller indices.

    Maps load as (NS, NR, NC) = (Z, Y, X) for MAPC=1/MAPR=2/MAPS=3, so
    index ft[L % nz, K % ny, H % nx].
    """
    ft = np.fft.fftn(arr.astype(np.float64)) * (vol / arr.size)
    nz, ny, nx = ft.shape
    return np.abs(ft[L % nz, K % ny, H % nx]).astype(np.float32)


def r_factor(Fo, Fc_scaled):
    num = float(np.sum(np.abs(Fo - Fc_scaled)))
    den = float(np.sum(np.abs(Fo)))
    return num / den if den > 0 else float('nan')


def apply_scale(Fc, k, B, s2):
    return k * Fc * np.exp(-B / 4 * s2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-dir',   required=True,
                   help='Directory containing sample_NNNNN/ subdirs')
    p.add_argument('--n',          type=int, default=100, help='Number of samples to evaluate')
    p.add_argument('--start',      type=int, default=None,
                   help='Starting sample index (default: last 20%% = val split if --val-only, else 0)')
    p.add_argument('--val-only',   action='store_true',
                   help='Use only the last 20%% of samples (the dataset val split)')
    p.add_argument('--base-features', type=int, default=32)
    p.add_argument('--crossp-raw', action='store_true',
                   help='Feed raw cross-Patterson at ch3 (required for *_rawcrossp models)')
    p.add_argument('--crossp-unitratio', action='store_true',
                   help='Feed unit-ratio deconvolution at ch3 (required for *_unitratio models)')
    p.add_argument('--fo-label',   default='F')
    p.add_argument('--free-label', default='FreeR_flag')
    p.add_argument('--no-scale',   action='store_true',
                   help='Skip the demean+RMS-match rescale (match infer.py --no-scale)')
    p.add_argument('--output',     default=None, help='Per-sample CSV output path')
    args = p.parse_args()

    # ── Locate sample directories ────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    all_samples = sorted(d for d in data_dir.glob('sample_*') if d.is_dir())
    if args.val_only:
        n_total = len(all_samples)
        start = n_total - int(n_total * 0.2)
        all_samples = all_samples[start:]
    elif args.start is not None:
        all_samples = all_samples[args.start:]
    # filter to those with all required files
    required = ('2fofc.map', 'fofc.map', 'fc.map', 'truth.map', 'refmacout.mtz')
    samples = [s for s in all_samples if all((s / f).exists() for f in required)]
    samples = samples[:args.n]
    if not samples:
        print('No usable samples found.', file=sys.stderr)
        sys.exit(1)
    print(f'Found {len(samples)} usable samples in {data_dir}')

    # ── Pre-extract all MTZs ─────────────────────────────────────────────────
    import tempfile
    cache_dir = tempfile.mkdtemp(prefix=f'_mtz_cache_{os.getpid()}_')
    cache_paths = extract_all_mtzs(samples, cache_dir,
                                    fo_label=args.fo_label, free_label=args.free_label)

    # ── Load model ──────────────────────────────────────────────────────────
    from model import UNet3D
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = UNet3D(in_channels=4, out_channels=1,
                   base_features=args.base_features).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded {args.checkpoint}  ep={ckpt.get("epoch","?")}  best_val={ckpt.get("best_val",float("nan")):.5f}')
    if args.crossp_unitratio:
        enc_label = 'unit-ratio deconvolution'
    elif args.crossp_raw:
        enc_label = 'raw'
    else:
        enc_label = 'signed-sqrt'
    print(f'  cross-Patterson encoding: {enc_label}')

    # ── Per-sample loop ─────────────────────────────────────────────────────
    rows = []
    fc_rows = []   # paired baseline (fc alone, no model)
    t0 = time.time()
    for i, sample in enumerate(samples):
        twofofc = _load_ccp4_map(sample / '2fofc.map')
        fofc    = _load_ccp4_map(sample / 'fofc.map')
        fc      = _load_ccp4_map(sample / 'fc.map')
        truth   = _load_ccp4_map(sample / 'truth.map')

        # Build input matching the model's training encoding.
        x = _build_input(twofofc, fofc, fc,
                         crossp_raw=args.crossp_raw,
                         crossp_unitratio=args.crossp_unitratio).to(device)

        with torch.no_grad():
            out = model(x)
            pred = (out[0] if isinstance(out, (tuple, list)) else out)[0, 0].cpu().numpy()

        # Demean + RMS-match-to-fofc rescale (matches infer.py default)
        if not args.no_scale:
            pred = pred - pred.mean()
            std_p = float(pred.std())
            if std_p > 0:
                pred = pred * (float(fofc.std()) / std_p)

        pred_total = pred + fc

        # Load per-sample MTZ data
        sd = np.load(cache_paths[i])
        H, K, L, Fo, free, s2 = sd['H'], sd['K'], sd['L'], sd['Fo'], sd['free'], sd['s2']
        vol = float(sd['cell_a'] * sd['cell_b'] * sd['cell_c'])
        is_free = (free == 0)
        obs = np.isfinite(Fo) & (Fo > 0)
        miss = ~obs
        work = obs & ~is_free
        free_m = obs & is_free

        Fc_pred  = map_to_F(pred_total, vol, H, K, L)
        Fc_fc    = map_to_F(fc,         vol, H, K, L)
        Fc_truth = map_to_F(truth,      vol, H, K, L)

        # Fit k+B on WORK reflections, apply to free + miss
        kp, Bp = _scale_kb(Fo[work], Fc_pred[work], s2[work])
        kc, Bc = _scale_kb(Fo[work], Fc_fc[work],   s2[work])

        Fc_pred_w  = apply_scale(Fc_pred[work],   kp, Bp, s2[work])
        Fc_pred_f  = apply_scale(Fc_pred[free_m], kp, Bp, s2[free_m])
        Fc_pred_m  = apply_scale(Fc_pred[miss],   kp, Bp, s2[miss])

        Fc_fc_w    = apply_scale(Fc_fc[work],   kc, Bc, s2[work])
        Fc_fc_f    = apply_scale(Fc_fc[free_m], kc, Bc, s2[free_m])
        Fc_fc_m    = apply_scale(Fc_fc[miss],   kc, Bc, s2[miss])

        r_pred = dict(
            R_work=r_factor(Fo[work],  Fc_pred_w),
            R_free=r_factor(Fo[free_m], Fc_pred_f),
            # R_miss: use FFT(truth) at the missing Miller indices as the "Fo proxy"
            R_miss=r_factor(Fc_truth[miss], Fc_pred_m),
            k=kp, B=Bp,
        )
        r_fc = dict(
            R_work=r_factor(Fo[work],  Fc_fc_w),
            R_free=r_factor(Fo[free_m], Fc_fc_f),
            R_miss=r_factor(Fc_truth[miss], Fc_fc_m),
            k=kc, B=Bc,
        )
        # Also CC vs true_diff = truth - fc
        true_diff = truth - fc
        cc = float(np.corrcoef(pred.ravel(), true_diff.ravel())[0, 1])

        rows.append(dict(sample=sample.name, n_free=int(free_m.sum()), n_miss=int(miss.sum()),
                          CC_truediff=cc, **{f'pred_{k}': v for k, v in r_pred.items()}))
        fc_rows.append({f'fc_{k}': v for k, v in r_fc.items()})

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            print(f'  [{i+1}/{len(samples)}]  pred R_free={r_pred["R_free"]:.4f}  '
                  f'fc R_free={r_fc["R_free"]:.4f}  CC={cc:.3f}  '
                  f'elapsed={elapsed:.0f}s  eta={eta:.0f}s', flush=True)

    # ── Aggregate ───────────────────────────────────────────────────────────
    def agg(values):
        arr = np.asarray([v for v in values if np.isfinite(v)])
        if len(arr) == 0:
            return float('nan'), float('nan'), float('nan'), float('nan')
        return float(arr.mean()), float(arr.std()), float(np.median(arr)), len(arr)

    print()
    print('=' * 80)
    print(f'Aggregated over {len(rows)} samples:')
    print(f'{"metric":24s}  {"mean":>10s} ± {"std":>7s}    {"median":>9s}  {"n":>4s}')
    print('-' * 80)
    for key, label in [
            ('R_work', 'pred R_work'), ('R_free', 'pred R_free'), ('R_miss', 'pred R_miss'),
            ('R_work', 'fc   R_work'), ('R_free', 'fc   R_free'), ('R_miss', 'fc   R_miss'),
            ('CC_truediff', 'CC(pred,true_diff)')]:
        if label.startswith('pred '):
            vals = [r[f'pred_{key}'] for r in rows]
        elif label.startswith('fc '):
            vals = [r[f'fc_{key}'] for r in fc_rows]
        else:
            vals = [r[key] for r in rows]
        m, s, med, n = agg(vals)
        print(f'{label:24s}  {m:>10.4f} ± {s:>7.4f}    {med:>9.4f}  {n:>4d}')

    # Per-sample delta R_free
    deltas = np.array([rows[i]['pred_R_free'] - fc_rows[i]['fc_R_free'] for i in range(len(rows))])
    print()
    print(f'ΔR_free (pred - fc):  mean={deltas.mean():+.4f}  '
          f'fraction better={np.mean(deltas<0):.2%}  '
          f'median={np.median(deltas):+.4f}')

    deltas_miss = np.array([rows[i]['pred_R_miss'] - fc_rows[i]['fc_R_miss'] for i in range(len(rows))])
    print(f'ΔR_miss (pred - fc):  mean={deltas_miss.mean():+.4f}  '
          f'fraction better={np.mean(deltas_miss<0):.2%}  '
          f'median={np.median(deltas_miss):+.4f}')

    if args.output:
        import csv
        all_rows = [{**r, **fc_rows[i]} for i, r in enumerate(rows)]
        with open(args.output, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
        print(f'\nPer-sample CSV: {args.output}')

    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
