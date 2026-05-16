#!/usr/bin/env python3
"""
infer.py — Run the trained UNet3D on CCP4 map files and write a predicted map.

Usage:
    /programs/pytorch/envs/pt/bin/python infer.py \
        --checkpoint checkpoints_protein_occ25_scratch/best.pt \
        --2fofc 2fofc.map --fofc fofc.map --fc fc.map \
        --output predicted.map

The model is fully convolutional and accepts any grid size — no resampling
needed as long as the input maps are at the same voxel spacing used during
training (0.667 Å/vox for the default 40 Å / 60-voxel P1 cell).  For large
grids that exceed GPU memory, use --tile to process in 60³ patches with
half-patch overlap and cosine blending.
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

import torch


# ── CCP4 map I/O ─────────────────────────────────────────────────────────────

def _load_map(path):
    """Read a CCP4 map file without gemmi.  Returns array shaped (NS, NR, NC)."""
    with open(path, 'rb') as f:
        nc, nr, ns = np.frombuffer(f.read(12), dtype=np.int32)
    n = int(nc) * int(nr) * int(ns)
    offset = os.path.getsize(path) - 4 * n
    data = np.fromfile(path, dtype=np.float32, count=n, offset=offset)
    return data.reshape(int(ns), int(nr), int(nc))


def _write_map(path, arr, template_path):
    """Write arr as a CCP4 map, reusing the header bytes from template_path."""
    with open(template_path, 'rb') as f:
        nc, nr, ns = np.frombuffer(f.read(12), dtype=np.int32)
    n = int(nc) * int(nr) * int(ns)
    header_len = os.path.getsize(template_path) - 4 * n
    with open(template_path, 'rb') as f:
        header = f.read(header_len)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(arr.ravel().astype(np.float32).tobytes())
    print(f'Written: {path}')


# ── Signal processing helpers ─────────────────────────────────────────────────

def _signed_sqrt(arr):
    return np.sign(arr) * np.sqrt(np.abs(arr))


def _cross_patterson(fofc_arr, fc_arr):
    """Cross-correlation of Fo-Fc and Fc maps (no origin peak)."""
    return np.fft.irfftn(
        np.fft.rfftn(fofc_arr) * np.conj(np.fft.rfftn(fc_arr)),
        s=fc_arr.shape,
    ).real.astype(np.float32)


def _build_input(twofofc, fofc, fc):
    """Stack four channels and return a (1, 4, D, H, W) float32 tensor.

    ch0-2: raw e/Å³; ch3: sign(crossp)*sqrt(|crossp|)
    """
    ch3 = _signed_sqrt(_cross_patterson(fofc, fc))
    x = np.stack([twofofc, fofc, fc, ch3], axis=0)
    return torch.from_numpy(x[np.newaxis].astype(np.float32))  # (1,4,D,H,W)


# ── Inference: whole-map or tiled ────────────────────────────────────────────

def _infer_whole(model, x, device):
    """Run model on the full volume at once. Returns pred_map as numpy array."""
    with torch.no_grad():
        out = model(x.to(device))
        mean = out[0] if isinstance(out, (tuple, list)) else out
    return mean[0, 0].cpu().numpy()


def _infer_tiled(model, x, device, patch=60, overlap=30):
    """Patch-based inference with cosine-blending at boundaries.

    Returns (mean_map, uncertainty_map, log_scale) as numpy arrays / scalar.
    mean and uncertainty are both blended with a Hanning window.
    log_scale is averaged across patches (global prediction).
    """
    _, _, D, H, W = x.shape
    stride = patch - overlap
    out_mean = np.zeros((D, H, W), dtype=np.float64)
    out_var  = np.zeros((D, H, W), dtype=np.float64)
    weight   = np.zeros((D, H, W), dtype=np.float64)
    scales   = []

    window_1d = np.hanning(patch).astype(np.float64)
    win3d = window_1d[:, None, None] * window_1d[None, :, None] * window_1d[None, None, :]

    def _starts(size):
        pts = list(range(0, size - patch, stride))
        pts.append(size - patch)
        return sorted(set(pts))

    for iz in _starts(D):
        for iy in _starts(H):
            for ix in _starts(W):
                chunk = x[:, :, iz:iz+patch, iy:iy+patch, ix:ix+patch]
                with torch.no_grad():
                    out = model(chunk.to(device))
                    m = (out[0] if isinstance(out, (tuple, list)) else out)[0, 0].cpu().numpy()
                out_mean[iz:iz+patch, iy:iy+patch, ix:ix+patch] += m * win3d
                weight  [iz:iz+patch, iy:iy+patch, ix:ix+patch] += win3d

    weight = np.where(weight < 1e-12, 1.0, weight)
    return (out_mean / weight).astype(np.float32)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run UNet3D inference on CCP4 map files.'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Model checkpoint (best.pt or latest.pt)')
    parser.add_argument('--2fofc', dest='fofc2', required=True,
                        help='2Fo-Fc map file (CCP4)')
    parser.add_argument('--fofc', required=True,
                        help='Fo-Fc difference map (CCP4)')
    parser.add_argument('--fc', required=True,
                        help='Fc map (CCP4)')
    parser.add_argument('--output', default='predicted_diff.map',
                        help='Output path for predicted difference map (default: predicted_diff.map)')
    parser.add_argument('--base-features', type=int, default=32,
                        help='U-Net base channel count; must match training (default: 32)')
    parser.add_argument('--tile', action='store_true',
                        help='Use tiled inference for grids that exceed GPU memory')
    parser.add_argument('--patch', type=int, default=60,
                        help='Patch size for tiled inference (default: 60)')
    parser.add_argument('--overlap', type=int, default=30,
                        help='Overlap between adjacent patches (default: 30)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA is available')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')

    # ── Load maps ─────────────────────────────────────────────────────────────
    twofofc = _load_map(args.fofc2)
    fofc    = _load_map(args.fofc)
    fc      = _load_map(args.fc)
    print(f'Grid shape: {twofofc.shape}')

    x = _build_input(twofofc, fofc, fc)

    # ── Load model ────────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from model import UNet3D
    model = UNet3D(in_channels=4, out_channels=1,
                   base_features=args.base_features).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    best_val = ckpt.get('best_val', float('inf'))
    epoch    = ckpt.get('epoch', '?')
    print(f'Loaded checkpoint  epoch={epoch}  best_val={best_val:.5f}')

    # ── Inference ─────────────────────────────────────────────────────────────
    if args.tile:
        print(f'Tiled inference  patch={args.patch}  overlap={args.overlap}')
        mean_map = _infer_tiled(model, x, device, patch=args.patch, overlap=args.overlap)
    else:
        mean_map = _infer_whole(model, x, device)

    print(f'pred range:  [{mean_map.min():.4f}, {mean_map.max():.4f}]  mean={mean_map.mean():.4f}  std={mean_map.std():.4f}  (e/Å³)')

    # ── Write output maps ─────────────────────────────────────────────────────
    _write_map(args.output, mean_map, args.fofc2)
    pred_total = mean_map + fc
    out_dir    = Path(args.output).parent
    pred_map_path = out_dir / 'predicted.map'
    _write_map(str(pred_map_path), pred_total, args.fofc2)

    # ── Evaluate vs truth when available ─────────────────────────────────────
    truth_path = out_dir / 'truth.map'
    if truth_path.exists():
        truth_raw = _load_map(str(truth_path))
        true_diff = truth_raw - fc          # raw e/Å³, same units as pred
        rmsd_true = float(np.sqrt(np.mean(true_diff ** 2)))
        mse  = float(np.mean((mean_map - true_diff) ** 2))
        rmse = float(np.sqrt(mse))
        cc   = float(np.corrcoef(mean_map.ravel(), true_diff.ravel())[0, 1])
        cc_fofc = float(np.corrcoef(fofc.ravel(), true_diff.ravel())[0, 1])
        print(f'true_diff range: [{true_diff.min():.4f}, {true_diff.max():.4f}]  rmsd={rmsd_true:.4f} e/Å³')
        print(f'RMSE(pred, true_diff) = {rmse:.4f} e/Å³  ({rmse/rmsd_true:.4f} × rmsd_true)')
        print(f'CC(pred,   true_diff) = {cc:.4f}')
        print(f'CC(fofc,   true_diff) = {cc_fofc:.4f}')
        # R factor in map space: Σ|pred - true| / Σ|true|
        _write_map(str(out_dir / 'true_diff.map'), true_diff, args.fofc2)

    # ── Crystallographic R factors (requires ccp4-python / gemmi) ────────────
    mtz_path = out_dir / 'refmacout.mtz'
    if mtz_path.exists():
        import subprocess, shutil, sys as _sys
        _sys.stdout.flush()
        ccp4py = shutil.which('ccp4-python') or 'ccp4-python'
        rfactor_script = Path(__file__).parent / 'rfactor.py'
        cmd = [ccp4py, str(rfactor_script),
               '--mtz',  str(mtz_path),
               '--fc',   str(out_dir / 'fc.map'),
               '--pred', str(pred_map_path)]
        if truth_path.exists():
            cmd += ['--truth', str(truth_path)]
        print()
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f'rfactor.py exited {result.returncode}')


if __name__ == '__main__':
    main()
