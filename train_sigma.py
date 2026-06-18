#!/usr/bin/env python3
"""
train_sigma.py — train net2 to predict per-voxel σ for net1's predictions.

Inputs (5 channels) : 2Fo-Fc, Fo-Fc, Fc, cross-Patterson, net1.predicted_diff
Output (1 channel)  : log_var per voxel (clamped to [-10, +10])

Loss: Gaussian NLL on the residual P − Y, where P is net1's (precomputed,
out-of-fold) prediction and Y is the truth difference map:

    L = mean( 0.5 · exp(-log_var) · (P − Y)²  +  0.5 · log_var )

This is calibrated by construction: optimum is exp(log_var) = (P−Y)², so
sigma(x) ≈ |P(x) − Y(x)| in expectation.

For calibrated σ, train on a dataset net1 has NEVER seen — see
pack_with_pred.py. Otherwise net2 learns net1's training-set overfit and
outputs falsely-low σ on novel inputs.

DDP plumbing (LOCAL_RANK init, DistributedSampler, DDP wrapper, all_reduce
in the loss accumulator) is omitted here — copy from train.py if/when needed.
"""
import argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PackedDatasetWithP
from model import UNet3D, count_parameters


def split_indices(n, val_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    return idx[n_val:], idx[:n_val]


def run_epoch(model, loader, device, optimizer=None, log_var_clamp=10.0):
    training = optimizer is not None
    model.train(training)
    total_nll = 0.0
    total_n   = 0
    with torch.set_grad_enabled(training):
        if training:
            optimizer.zero_grad()
        for x, p, y, _s in loader:
            x, p, y = x.to(device), p.to(device), y.to(device)
            inp     = torch.cat([x, p], dim=1)              # (B, 5, D, H, W)
            out     = model(inp)
            # UNet3D returns (mean, log_var, log_scale); we treat its mean head
            # as our log_var output (the model architecture is what matters).
            log_var = out[0] if isinstance(out, (tuple, list)) else out
            log_var = torch.clamp(log_var, min=-log_var_clamp, max=log_var_clamp)

            residual_sq = (p - y) ** 2
            nll = 0.5 * (torch.exp(-log_var) * residual_sq + log_var).mean()

            if training:
                nll.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n = x.size(0)
            total_n   += n
            total_nll += nll.item() * n

    return total_nll / total_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',          required=True,
                    help='Dataset directory with X/Y/S/P .npy (out-of-fold for net1)')
    ap.add_argument('--outdir',        default='./checkpoints_sigma')
    ap.add_argument('--epochs',        type=int,   default=100)
    ap.add_argument('--batch-size',    type=int,   default=1)
    ap.add_argument('--lr',            type=float, default=3e-4)
    ap.add_argument('--val-frac',      type=float, default=0.2)
    ap.add_argument('--base-features', type=int,   default=32)
    ap.add_argument('--workers',       type=int,   default=2)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Path(args.data)

    n_total = len(np.load(data / 'X.npy', mmap_mode='r'))
    tr_idx, va_idx = split_indices(n_total, args.val_frac)
    train_ds = PackedDatasetWithP(data/'X.npy', data/'Y.npy', data/'S.npy', data/'P.npy', tr_idx)
    val_ds   = PackedDatasetWithP(data/'X.npy', data/'Y.npy', data/'S.npy', data/'P.npy', va_idx)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    model = UNet3D(in_channels=5, out_channels=1, base_features=args.base_features).to(device)
    print(f'Parameters: {count_parameters(model):,}')

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr / 100)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')
    for epoch in range(args.epochs):
        t0     = time.time()
        tr_nll = run_epoch(model, train_ld, device, optim)
        va_nll = run_epoch(model, val_ld,   device, optimizer=None)
        sched.step()
        print(f'epoch {epoch:04d}  train_NLL= {tr_nll:.5f}  val_NLL= {va_nll:.5f}  '
              f'lr= {sched.get_last_lr()[0]:.2e}  t= {time.time()-t0:.1f}s')

        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optim.state_dict(), scheduler=sched.state_dict(),
                    best_val=best_val)
        torch.save(ckpt, out / 'latest.pt')
        if va_nll < best_val:
            best_val = va_nll
            torch.save(ckpt, out / 'best.pt')
            print(f'  ↳ new best val={best_val:.5f}')


if __name__ == '__main__':
    main()
