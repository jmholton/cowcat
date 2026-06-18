#!/usr/bin/env python3
"""Evaluate a checkpoint on each dataset separately and report NLL loss."""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import make_splits_multi
from model import UNet3D
from train import heteroscedastic_nll


def eval_dataset(model, data_dirs, device, batch_size=1, workers=4):
    _, val_ds = make_splits_multi(data_dirs, val_fraction=0.2)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=False)
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y, s in loader:
            x, y, s = x.to(device), y.to(device), s.to(device)
            pred_map, pred_log_var, pred_log_scale = model(x)
            loss = heteroscedastic_nll(pred_map, pred_log_var, y) + \
                   F.mse_loss(pred_log_scale, s)
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / n if n > 0 else float('nan'), n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = UNet3D(in_channels=4, base_features=32).to(device)
    sd = ckpt.get('model', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(f'Loaded {args.checkpoint}  best_val={ckpt.get("best_val","?"):.5f}  ep={ckpt.get("epoch","?")}')
    print()

    datasets = {
        'simple_1aho_s0':       ['data/data_simple_1aho_s0'],
        'protein_v2_s0_new':    ['data/data_protein_v2_s0_new'],
        'protein_v2_old (s0)':  ['data/data_protein_v2_s0'],
        'protein_v2_old (all)': [f'data/data_protein_v2_s{s}'
                                 for s in [0,1000,2000,3000,4000,5000,6000,7000,8000,9000]],
    }

    for name, dirs in datasets.items():
        loss, n = eval_dataset(model, dirs, device, workers=args.workers)
        print(f'{name:30s}  val_nll= {loss:+.5f}  n={n}')


if __name__ == '__main__':
    main()
