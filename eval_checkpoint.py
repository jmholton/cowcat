#!/usr/bin/env python3
"""Evaluate one or more checkpoints on a fixed val split."""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import make_splits_multi
from model import UNet3D
from train import heteroscedastic_nll


def eval_val(checkpoint_path, data_dirs, val_frac=0.2, batch_size=2, workers=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(checkpoint_path, map_location=device)
    base_features = ckpt.get('base_features', 32)
    model = UNet3D(in_channels=4, out_channels=2, base_features=base_features).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    _, val_ds = make_splits_multi(data_dirs, val_fraction=val_frac, seed=42)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=(device.type == 'cuda'))

    total = 0.0
    with torch.no_grad():
        for x, y, s in val_loader:
            x, y, s = x.to(device), y.to(device), s.to(device)
            pred_map, pred_log_var, pred_log_scale = model(x)
            loss = heteroscedastic_nll(pred_map, pred_log_var, y) + F.mse_loss(pred_log_scale, s)
            total += loss.item() * x.size(0)

    return total / len(val_ds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--val-frac', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    _, val_ds = make_splits_multi(args.data, val_fraction=args.val_frac, seed=42)
    print(f'Val samples: {len(val_ds)}  (data: {args.data})')

    for ckpt_path in args.checkpoints:
        val_loss = eval_val(ckpt_path, args.data, args.val_frac, args.batch_size, args.workers)
        print(f'{ckpt_path:60s}  val= {val_loss:.5f}')


if __name__ == '__main__':
    main()
