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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import make_splits, make_splits_multi
from model import UNet3D, count_parameters


# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Training / validation loops
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, device, optimizer=None, alpha=0.5, accum_steps=1,
              world_size=1):
    """Single train or eval pass. Pass optimizer=None for eval.

    alpha: peak-weight coefficient. Loss is mean((1 + alpha·|y|)·(ŷ-y)²) —
    errors at real density peaks (large |y|) are upweighted so the model does
    not smear amplitude away from sparse high-magnitude features. The reported
    metric is still plain MSE, so it stays comparable across alpha values.
    accum_steps: accumulate gradients over this many batches before stepping,
    simulating a larger effective batch size without extra GPU memory.
    world_size: number of DDP ranks; results are all-reduced across ranks.
    """
    training = optimizer is not None
    model.train(training)
    total_mse  = 0.0  # plain MSE, for the reported metric
    total_yss  = 0.0  # sum of mean(y²) * n, for dataset-level RMSD
    total_n    = 0    # local sample count (DistributedSampler shards the dataset)

    with torch.set_grad_enabled(training):
        if training:
            optimizer.zero_grad()
        for step, (x, y, s) in enumerate(loader):
            x, y, s = x.to(device), y.to(device), s.to(device)
            pred_map, pred_log_var, pred_log_scale = model(x)
            # Peak-weighted MSE drives backprop; plain MSE is logged as the metric.
            sq   = (pred_map - y) ** 2
            loss = ((1.0 + alpha * y.abs()) * sq).mean()
            mse  = sq.mean()

            if training:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            n = x.size(0)
            total_n   += n
            total_mse += mse.item() * n
            with torch.no_grad():
                truth = y + x[:, 2:3]   # fc is input channel 2
                total_yss += (truth ** 2).mean().item() * n

    if world_size > 1:
        t = torch.tensor([total_mse, total_yss, float(total_n)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_mse, total_yss, total_n = t[0].item(), t[1].item(), int(t[2].item())

    mse  = total_mse / total_n
    rmsd = (total_yss / total_n) ** 0.5   # RMS of truth map (e/Å³)
    Rrms = mse**0.5 / rmsd if rmsd > 0 else float('nan')
    return mse, Rrms, rmsd


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
                        help='Peak-weight coefficient in loss: '
                             'mean((1+alpha*|y|)*(pred-y)^2) (default: 0.5)')
    parser.add_argument('--workers',     type=int, default=2,
                        help='DataLoader worker processes (default: 2)')
    parser.add_argument('--resume',      default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--pretrain',    default=None,
                        help='Path to checkpoint to load model weights only '
                             '(optimizer/scheduler reset — for curriculum transfer)')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1). Effective '
                             'batch size = batch-size * accum-steps.')
    parser.add_argument('--eval-1aho-dir', default=None,
                        help='Directory with 1aho_test maps + refmac MTZ; if set, '
                             'reports Rfree against this real-data sample each '
                             'epoch as a generalisation check')
    parser.add_argument('--eval-1aho-fo-label',   default='FP')
    parser.add_argument('--eval-1aho-free-label', default='FreeR_flag')
    parser.add_argument('--eval-1aho-mtz-name',   default='refmacout_minRfree.mtz')
    parser.add_argument('--eval-1aho-crossp',     default='raw',
                        choices=['raw', 'signed_sqrt'],
                        help='Cross-Patterson transform for 1aho eval input '
                             '(raw=rawcrossp datasets; signed_sqrt=ssqrt datasets)')
    args = parser.parse_args()

    # ── DDP setup (torchrun sets LOCAL_RANK / WORLD_SIZE) ─────────────────────
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    is_ddp     = local_rank >= 0
    if is_ddp:
        dist.init_process_group('nccl')
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        rank       = 0
        world_size = 1
        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_rank0 = (rank == 0)
    n_gpus   = torch.cuda.device_count() if device.type == 'cuda' else 0

    # ── Log file (train_<suffix>.log beside the script) — rank 0 only ─────────
    if is_rank0:
        _suffix  = Path(args.outdir).name.replace('checkpoints_', '', 1)
        _logpath = Path(__file__).parent / f'train_{_suffix}.log'

        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, data):
                for s in self.streams: s.write(data)
            def flush(self):
                for s in self.streams: s.flush()

        if _logpath.exists():
            _logpath.rename(_logpath.with_suffix('.log.bak'))
        _lf = open(_logpath, 'w')
        sys.stdout = _Tee(sys.__stdout__, _lf)
        sys.stderr = _Tee(sys.__stderr__, _lf)
        print(f'Logging to {_logpath}')
        print(f'Device: {device}  GPUs: {n_gpus}  DDP: {is_ddp}  world_size: {world_size}')

    # ── Data ──────────────────────────────────────────────────────────────────
    if is_rank0:
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
    if is_rank0:
        print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
        train_loader  = DataLoader(train_ds, batch_size=args.batch_size,
                                   sampler=train_sampler, num_workers=args.workers, pin_memory=True)
        val_loader    = DataLoader(val_ds,   batch_size=args.batch_size,
                                   sampler=val_sampler,   num_workers=args.workers, pin_memory=True)
    else:
        # pin_memory causes deadlock with DataParallel + forked workers
        pin          = (device.type == 'cuda') and n_gpus <= 1
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=args.workers, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.workers, pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────────
    in_channels = 4
    model = UNet3D(in_channels=in_channels, out_channels=1,
                   base_features=args.base_features).to(device)
    raw_model = model
    if is_ddp:
        # find_unused_parameters=True: pred_log_var/pred_log_scale heads have no gradient under MSE loss
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    elif n_gpus > 1:
        model = nn.DataParallel(model)
        args.batch_size *= n_gpus
    if is_rank0:
        print(f'Parameters: {count_parameters(raw_model):,}  batch_size: {args.batch_size}  '
              f'accum_steps: {args.accum_steps}  eff_batch: {args.batch_size * args.accum_steps * world_size}  '
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
        if is_rank0:
            print(f'Resumed from epoch {ckpt["epoch"]}  best_val={best_val:.5f}')

    # ── Pretrain (weights only, optimizer/scheduler reset) ────────────────────
    elif args.pretrain and os.path.exists(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location=device)
        sd   = ckpt['model']
        missing, unexpected = raw_model.load_state_dict(sd, strict=False)
        if is_rank0:
            print(f'Loaded pretrained weights (best_val={ckpt.get("best_val", float("inf")):.5f}), '
                  f'optimizer reset')
            if missing:
                print(f'  missing keys (random init): {len(missing)}')
            if unexpected:
                print(f'  unexpected keys (ignored, e.g. old BN): {len(unexpected)}')

    outdir = Path(args.outdir)
    if is_rank0:
        outdir.mkdir(parents=True, exist_ok=True)

    # ── Optional Rfree-on-real-1aho diagnostic ────────────────────────────────
    eval_ctx = None
    if is_rank0 and args.eval_1aho_dir:
        from eval_1aho import setup_1aho_eval, eval_rfree
        eval_ctx = setup_1aho_eval(
            args.eval_1aho_dir,
            fo_label=args.eval_1aho_fo_label,
            free_label=args.eval_1aho_free_label,
            mtz_name=args.eval_1aho_mtz_name,
            crossp_transform=args.eval_1aho_crossp,
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        t0 = time.time()

        train_mse, train_Rrms, _        = run_epoch(model, train_loader, device, optimizer,
                                                    args.alpha, args.accum_steps, world_size)
        val_mse,   val_Rrms,   val_rmsd = run_epoch(model, val_loader,   device, optimizer=None,
                                                    alpha=args.alpha, world_size=world_size)
        scheduler.step()

        # Rfree-on-real-1aho diagnostic (rank 0 only, after val)
        rfree_1aho = None
        if eval_ctx is not None:
            try:
                rfree_1aho = eval_rfree(raw_model, eval_ctx, device)
            except Exception as e:
                print(f'  eval_1aho.eval_rfree failed: {e}; disabling', flush=True)
                eval_ctx = None

        if is_rank0:
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]

            # Print val rmsd_truth once (it's a fixed dataset property, not per-epoch info).
            if epoch == start_epoch:
                print(f'  val rmsd_truth = {val_rmsd:.4f} e/Å³ (RMS of truth map; fixed)')

            rfree_str = f'Rfree_1aho= {rfree_1aho:.4f}  ' if rfree_1aho is not None else ''
            print(f'epoch {epoch:04d}  '
                  f'train= {train_mse:.5f}  val= {val_mse:.5f}  '
                  f'Rrms= {val_Rrms:.4f}  '
                  f'{rfree_str}'
                  f'lr= {lr_now:.2e}  t= {elapsed:.1f}s')

            entry = dict(epoch=epoch, train=round(train_mse, 6),
                         val=round(val_mse, 6), Rrms=round(val_Rrms, 4), lr=lr_now)
            if rfree_1aho is not None:
                entry['Rfree_1aho'] = round(rfree_1aho, 4)
            log.append(entry)

            ckpt = dict(epoch=epoch, model=raw_model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        best_val=best_val, log=log)
            torch.save(ckpt, outdir / 'latest.pt')

            if val_mse < best_val:
                best_val = val_mse
                torch.save(ckpt, outdir / 'best.pt')
                print(f'  ↳ new best val={best_val:.5f}')

            (outdir / 'log.json').write_text(json.dumps(log, indent=2))

    if is_rank0:
        print(f'Done. Best val MSE: {best_val:.5f}')
        print(f'Best checkpoint: {outdir / "best.pt"}')

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
