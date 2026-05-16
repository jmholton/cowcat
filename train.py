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
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import make_splits, make_splits_multi
from model import UNet3D, count_parameters


# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Training / validation loops
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, device, optimizer=None, scale_weight=1.0, accum_steps=1):
    """Single train or eval pass. Pass optimizer=None for eval.

    accum_steps: accumulate gradients over this many batches before stepping,
    simulating a larger effective batch size without extra GPU memory.
    """
    training = optimizer is not None
    model.train(training)
    total_mse  = 0.0
    total_yss  = 0.0  # sum of mean(y²) * n, for dataset-level RMSD

    with torch.set_grad_enabled(training):
        if training:
            optimizer.zero_grad()
        for step, (x, y, s) in enumerate(loader):
            x, y, s = x.to(device), y.to(device), s.to(device)
            pred_map, pred_log_var, pred_log_scale = model(x)
            loss = F.mse_loss(pred_map, y)

            if training:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            n = x.size(0)
            total_mse += loss.item() * n
            with torch.no_grad():
                total_yss += (y ** 2).mean().item() * n

    n_total = len(loader.dataset)
    mse  = total_mse / n_total
    rmsd = (total_yss / n_total) ** 0.5   # RMS of true map (e/Å³)
    return mse, mse / rmsd if rmsd > 0 else float('nan')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train 3D U-Net for electron density reconstruction.'
    )
    parser.add_argument('--data',        default=['./data'], nargs='+',
                        help='One or more data directories (space-separated)')
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
    parser.add_argument('--scale-weight', type=float, default=1.0,
                        help='Weight on scale-prediction MSE loss (default: 1.0)')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1). Effective '
                             'batch size = batch-size * accum-steps.')
    args = parser.parse_args()

    # ── Log file (train_<suffix>.log beside the script) ───────────────────────
    _suffix  = Path(args.outdir).name.replace('checkpoints_', '', 1)
    _logpath = Path(__file__).parent / f'train_{_suffix}.log'

    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams: s.write(data)
        def flush(self):
            for s in self.streams: s.flush()

    _lf = open(_logpath, 'a')
    sys.stdout = _Tee(sys.__stdout__, _lf)
    sys.stderr = _Tee(sys.__stderr__, _lf)
    print(f'Logging to {_logpath}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    print(f'Device: {device}  GPUs: {n_gpus}')

    # ── Data ──────────────────────────────────────────────────────────────────
    import time as _time
    print('Data files:')
    for d in args.data:
        for fname in ('X.npy', 'Y.npy', 'S.npy'):
            p = os.path.join(d, fname)
            if os.path.exists(p):
                mtime = _time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(os.path.getmtime(p)))
                size  = os.path.getsize(p)
                print(f'  {p}  {size:>12,d} bytes  mtime={mtime}')
            else:
                print(f'  {p}  MISSING')
    train_ds, val_ds = make_splits_multi(args.data, val_fraction=args.val_frac)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    # pin_memory causes deadlock with DataParallel + forked workers
    pin = (device.type == 'cuda') and n_gpus <= 1
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────────
    in_channels = 4
    model = UNet3D(in_channels=in_channels, out_channels=1,
                   base_features=args.base_features).to(device)
    raw_model = model
    if n_gpus > 1:
        model = nn.DataParallel(model)
        args.batch_size *= n_gpus
    print(f'Parameters: {count_parameters(raw_model):,}  batch_size: {args.batch_size}  '
          f'accum_steps: {args.accum_steps}  eff_batch: {args.batch_size * args.accum_steps}  '
          f'in_channels: {in_channels}')

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
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        log         = ckpt.get('log', [])
        print(f'Resumed from epoch {ckpt["epoch"]}  best_val={best_val:.5f}')

    # ── Pretrain (weights only, optimizer/scheduler reset) ────────────────────
    elif args.pretrain and os.path.exists(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location=device)
        sd   = ckpt['model']
        raw_model.load_state_dict(sd)
        print(f'Loaded pretrained weights (best_val={ckpt.get("best_val", float("inf")):.5f}), '
              f'optimizer reset')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_mse, train_ratio = run_epoch(model, train_loader, device, optimizer,      args.scale_weight, args.accum_steps)
        val_mse,   val_ratio   = run_epoch(model, val_loader,   device, optimizer=None, scale_weight=args.scale_weight)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f'epoch {epoch:04d}  '
              f'train= {train_mse:.5f}  val= {val_mse:.5f}  '
              f'ratio= {val_ratio:.4f}  '
              f'lr= {lr_now:.2e}  t= {elapsed:.1f}s')

        entry = dict(epoch=epoch, train=round(train_mse, 6),
                     val=round(val_mse, 6), ratio=round(val_ratio, 4), lr=lr_now)
        log.append(entry)

        # Save latest checkpoint
        ckpt = dict(epoch=epoch, model=raw_model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    best_val=best_val, log=log)
        torch.save(ckpt, outdir / 'latest.pt')

        # Save best checkpoint
        if val_mse < best_val:
            best_val = val_mse
            torch.save(ckpt, outdir / 'best.pt')
            print(f'  ↳ new best val={best_val:.5f}')

        # Write training log as JSON for easy inspection
        (outdir / 'log.json').write_text(json.dumps(log, indent=2))

    print(f'Done. Best val MSE: {best_val:.5f}')
    print(f'Best checkpoint: {outdir / "best.pt"}')


if __name__ == '__main__':
    main()
