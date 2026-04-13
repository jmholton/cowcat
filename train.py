#!/usr/bin/env python3
"""
train.py  –  Train the 3D U-Net to reconstruct ground-truth electron density.

Usage (GPU node via SLURM):
    srun --partition=gpu --gres=gpu:1 \\
        /programs/pytorch/envs/pt/bin/python train.py \\
        --data ./data --epochs 100 --batch-size 2

Loss
----
Peak-weighted MSE: L = mean( (1 + α|y|) · (ŷ - y)² )
Upweights errors at real density peaks, where the ghost-peak problem matters most.
"""

import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import make_splits
from model import UNet3D, count_parameters


# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════

def peak_weighted_mse(pred, target, alpha=0.5):
    """MSE weighted by absolute ground-truth density magnitude."""
    weights = 1.0 + alpha * target.abs()
    return (weights * (pred - target) ** 2).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Training / validation loops
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, device, optimizer=None):
    """Single train or eval pass. Pass optimizer=None for eval."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = peak_weighted_mse(pred, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train 3D U-Net for electron density reconstruction.'
    )
    parser.add_argument('--data',        default='./data',
                        help='Root data directory (default: ./data)')
    parser.add_argument('--outdir',      default='./checkpoints',
                        help='Where to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--epochs',      type=int, default=100)
    parser.add_argument('--batch-size',  type=int, default=2)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--val-frac',    type=float, default=0.2,
                        help='Fraction of data held out for validation')
    parser.add_argument('--base-features', type=int, default=32,
                        help='U-Net base channel count (default: 32)')
    parser.add_argument('--alpha',       type=float, default=0.5,
                        help='Peak-weight coefficient in loss (default: 0.5)')
    parser.add_argument('--workers',     type=int, default=2,
                        help='DataLoader worker processes (default: 2)')
    parser.add_argument('--resume',      default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--pretrain',    default=None,
                        help='Path to checkpoint to load model weights only '
                             '(optimizer/scheduler reset — for curriculum transfer)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = make_splits(args.data, val_fraction=args.val_frac)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=(device.type == 'cuda'))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(in_channels=4, out_channels=1,
                   base_features=args.base_features).to(device)
    print(f'Parameters: {count_parameters(model):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    start_epoch = 0
    best_val    = float('inf')
    log         = []

    # ── Resume (full state) ───────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        log         = ckpt.get('log', [])
        print(f'Resumed from epoch {ckpt["epoch"]}  best_val={best_val:.5f}')

    # ── Pretrain (weights only, optimizer/scheduler reset) ────────────────────
    elif args.pretrain and os.path.exists(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'Loaded pretrained weights (best_val={ckpt.get("best_val", float("inf")):.5f}), '
              f'optimizer reset')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss   = run_epoch(model, val_loader,   device, optimizer=None)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f'epoch {epoch:04d}  '
              f'train= {train_loss:.5f}  val= {val_loss:.5f}  '
              f'lr= {lr_now:.2e}  t= {elapsed:.1f}s')

        entry = dict(epoch=epoch, train=round(train_loss, 6),
                     val=round(val_loss, 6), lr=lr_now)
        log.append(entry)

        # Save latest checkpoint
        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    best_val=best_val, log=log)
        torch.save(ckpt, outdir / 'latest.pt')

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, outdir / 'best.pt')
            print(f'  ↳ new best val={best_val:.5f}')

        # Write training log as JSON for easy inspection
        (outdir / 'log.json').write_text(json.dumps(log, indent=2))

    print(f'Done. Best val loss: {best_val:.5f}')
    print(f'Best checkpoint: {outdir / "best.pt"}')


if __name__ == '__main__':
    main()
