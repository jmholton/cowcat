#!/usr/bin/env python3
"""
pack_with_pred.py — extend a packed dataset with net1's predicted_diff.

Given a net1 checkpoint and a directory that already contains
X.npy / Y.npy / S.npy, run net1 inference on every sample, apply the same
amplitude rescale that infer.py uses (demean + RMS-match to fofc), and write
P.npy (N, 1, D, H, W) aligned to X.

To train net2 with calibrated σ, point this at a dataset net1 has NEVER
seen — e.g. a fresh v4 seed range that wasn't in net1's train/val split.
Predictions are then genuinely out-of-fold; net2 won't learn net1's
training-set overfit as "low error."

Usage:
    /programs/pytorch/envs/pt/bin/python pack_with_pred.py \\
        --checkpoint checkpoints_protein_v4_4gpu_rc/best.pt \\
        --data data/data_protein_v4_s1000_rawcrossp
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from model import UNet3D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',    required=True)
    ap.add_argument('--data',          required=True,
                    help='Directory with X.npy/Y.npy/S.npy; P.npy is written here')
    ap.add_argument('--base-features', type=int, default=32,
                    help='Must match net1 (default: 32)')
    ap.add_argument('--no-scale',      action='store_true',
                    help='Skip the demean + RMS-match-to-fofc rescale')
    args = ap.parse_args()

    data = Path(args.data)
    X = np.load(data / 'X.npy', mmap_mode='r')   # (N, 4, D, H, W)
    N, C = X.shape[0], X.shape[1]
    print(f'Packing predictions for {N} samples in {data}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = UNet3D(in_channels=C, out_channels=1,
                    base_features=args.base_features).to(device)
    ckpt   = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint epoch={ckpt.get("epoch","?")} '
          f'best_val={ckpt.get("best_val",float("nan")):.5f}')

    P = np.lib.format.open_memmap(
        data / 'P.npy', mode='w+', dtype=np.float32,
        shape=(N, 1) + X.shape[2:])

    with torch.no_grad():
        for i in range(N):
            x_np = np.array(X[i])
            x    = torch.from_numpy(x_np).to(device).unsqueeze(0)
            out  = model(x)
            pred = (out[0] if isinstance(out, (tuple, list)) else out)[0, 0].cpu().numpy()

            if not args.no_scale:
                # Match infer.py: demean, RMS-match to fofc (input channel 1).
                fofc  = x_np[1]
                pred  = pred - pred.mean()
                std_p = float(pred.std())
                if std_p > 0:
                    pred = pred * (float(fofc.std()) / std_p)

            P[i, 0] = pred.astype(np.float32)
            if (i + 1) % 50 == 0:
                P.flush()
                print(f'  {i+1}/{N}')
    P.flush()
    print(f'Wrote {data/"P.npy"}')


if __name__ == '__main__':
    main()
