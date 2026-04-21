#!/usr/bin/env python3
"""
infer_dataset.py — Run a trained model on every sample in a dataset directory.

Writes predicted_diff.map and predicted_uncertainty.map into each sample dir.
Prints a calibration summary: how well predicted sigma correlates with actual error.

Usage:
    /programs/pytorch/envs/pt/bin/python infer_dataset.py \
        --checkpoint checkpoints_diff_n1del/best.pt \
        --data data_n10_N1del_hydr_n1000 \
        [--split val|train|all]  (default: val)
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from model import UNet3D
from dataset import ElectronDensityDataset, make_splits, _load_map, _znorm


def _write_map(path, arr, template_path):
    with open(template_path, 'rb') as f:
        nc, nr, ns = np.frombuffer(f.read(12), dtype=np.int32)
    n = int(nc) * int(nr) * int(ns)
    header_len = os.path.getsize(template_path) - 4 * n
    with open(template_path, 'rb') as f:
        header = f.read(header_len)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(arr.ravel().astype(np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data',       required=True)
    parser.add_argument('--split',      default='val', choices=['val', 'train', 'all'])
    parser.add_argument('--base-features', type=int, default=32)
    parser.add_argument('--cpu',        action='store_true')
    parser.add_argument('--suffix',     default='',
                        help='Suffix for output map filenames, e.g. "_r2"')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = UNet3D(in_channels=4, out_channels=2,
                   base_features=args.base_features).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded {args.checkpoint}  epoch={ckpt.get("epoch","?")}  '
          f'best_val={ckpt.get("best_val", float("inf")):.5f}')

    # Select sample IDs according to split
    full_ds = ElectronDensityDataset(args.data)
    all_ids = full_ds.sample_ids
    n = len(all_ids)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_val = max(1, int(n * 0.2))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    if args.split == 'val':
        sample_ids = [all_ids[i] for i in val_idx]
    elif args.split == 'train':
        sample_ids = [all_ids[i] for i in train_idx]
    else:
        sample_ids = all_ids

    print(f'Running inference on {len(sample_ids)} {args.split} samples')

    # Per-sample calibration accumulators
    all_actual_err = []
    all_pred_sigma = []
    all_pred_scale = []
    all_true_scale = []

    for sid in sample_ids:
        base = Path(args.data) / sid

        fofc_raw = _load_map(str(base / 'fofc.map'))
        fc_raw   = _load_map(str(base / 'fc.map'))
        twofofc  = _load_map(str(base / '2fofc.map'))
        truth    = _load_map(str(base / 'truth.map'))

        # Cross-Patterson
        crossp_path = base / 'crossp.npy'
        if crossp_path.exists():
            ch3 = _znorm(np.load(str(crossp_path)))
        else:
            from dataset import _cross_patterson
            ch3 = _znorm(_cross_patterson(fofc_raw, fc_raw))

        x = np.stack([_znorm(twofofc), _znorm(fofc_raw), _znorm(fc_raw), ch3], axis=0)
        x_t = torch.from_numpy(x[np.newaxis].astype(np.float32)).to(device)

        with torch.no_grad():
            mean_map, log_var_map, log_scale = model(x_t)

        mean_np    = mean_map[0, 0].cpu().numpy()
        log_var_np = log_var_map[0, 0].cpu().numpy()
        sigma_np   = np.exp(0.5 * log_var_np)
        ls         = log_scale[0].item()

        # Ground-truth difference (z-normalised same way as training)
        diff_raw = truth - fc_raw
        diff_std = float(diff_raw.std())
        true_log_scale = float(np.log(diff_std + 1e-8))
        target = (diff_raw - diff_raw.mean()) / (diff_std + 1e-8)

        actual_err = np.abs(mean_np - target)

        all_actual_err.append(actual_err.ravel())
        all_pred_sigma.append(sigma_np.ravel())
        all_pred_scale.append(ls)
        all_true_scale.append(true_log_scale)

        # Write output maps (reuse fc.map header)
        template = str(base / 'fc.map')
        _write_map(str(base / f'predicted_diff{args.suffix}.map'),        mean_np,   template)
        _write_map(str(base / f'predicted_uncertainty{args.suffix}.map'), sigma_np,  template)

    # ── Calibration summary ───────────────────────────────────────────────────
    all_actual_err = np.concatenate(all_actual_err)
    all_pred_sigma = np.concatenate(all_pred_sigma)

    # Bin by predicted sigma, report mean actual error per bin
    print('\n--- Calibration (predicted σ vs actual |error|) ---')
    bins = np.percentile(all_pred_sigma, [0, 20, 40, 60, 80, 100])
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (all_pred_sigma >= lo) & (all_pred_sigma < hi)
        if mask.sum() == 0:
            continue
        print(f'  pred_sigma [{lo:.3f}, {hi:.3f})  '
              f'mean_actual_err= {all_actual_err[mask].mean():.4f}  '
              f'n= {mask.sum()}')

    corr = np.corrcoef(all_pred_sigma, all_actual_err)[0, 1]
    print(f'\nPearson r(pred_sigma, actual_err) = {corr:.4f}')

    # Scale calibration
    all_pred_scale = np.array(all_pred_scale)
    all_true_scale = np.array(all_true_scale)
    scale_corr = np.corrcoef(all_pred_scale, all_true_scale)[0, 1]
    scale_bias = float((all_pred_scale - all_true_scale).mean())
    print(f'log_scale: r= {scale_corr:.4f}  bias= {scale_bias:+.4f} '
          f'(pred - true, nats)')


if __name__ == '__main__':
    main()
