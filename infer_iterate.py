#!/usr/bin/env python3
"""
Inference-time iteration for electron density reconstruction.

Runs the model N times on a sample, updating the Fc (and optionally 2Fo-Fc
and Fo-Fc) maps each iteration using the CNN's previous prediction as the
new "calculated" density.

Usage:
    python infer_iterate.py --checkpoint checkpoints_n10_N1del_altconf3refmac/best.pt \
                            --sample-dir data_n10_N1altconf3_refmac_n1000/sample_00042 \
                            --n-iter 5

    # Run on multiple random validation samples:
    python infer_iterate.py --checkpoint checkpoints_n10_N1del_altconf3refmac/best.pt \
                            --data-dir data_n10_N1altconf3_refmac_n1000 \
                            --n-samples 20 --n-iter 5
"""
import argparse
import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import UNet3D
from dataset import _load_map, _znorm, _cross_patterson


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet3D(in_channels=4, out_channels=1, base_features=32)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Single forward pass
# ---------------------------------------------------------------------------

def run_inference(model, ch0, ch1, ch2, ch3, device):
    """Run one forward pass.  All inputs are z-normalised (60,60,60) np arrays.
    Returns z-normalised prediction (60,60,60)."""
    x = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
    x = torch.from_numpy(x[np.newaxis]).to(device)   # (1, 4, D, H, W)
    with torch.no_grad():
        y = model(x)
    return y.squeeze().cpu().numpy()                  # (60, 60, 60)


# ---------------------------------------------------------------------------
# Map update helpers
# ---------------------------------------------------------------------------

def rebuild_maps_fc_only(truth_raw, cnn_znorm):
    """Replace only the Fc channel; keep original 2Fo-Fc and Fo-Fc."""
    # Scale CNN output to approximate physical units of truth map
    scale = truth_raw.std() if truth_raw.std() > 1e-8 else 1.0
    fc_new = cnn_znorm * scale + truth_raw.mean()
    return None, None, fc_new.astype(np.float32)   # sentinel None = keep original


def rebuild_maps_fft(truth_raw, cnn_znorm):
    """Rebuild all three maps via Fourier synthesis using CNN prediction as Fc.

    Fo structure factors: FFT of truth.map  (in this simulation Fo = FC_truth)
    Fc structure factors: FFT of scaled CNN prediction

    2Fo-Fc = IFFT(2|Fo|·exp(i·φ_c) − Fc(hkl))
    Fo-Fc  = IFFT(  |Fo|·exp(i·φ_c) − Fc(hkl))
    """
    truth_std  = truth_raw.std()  if truth_raw.std()  > 1e-8 else 1.0
    truth_mean = truth_raw.mean()

    # Scale CNN output to physical units
    fc_phys = cnn_znorm * truth_std + truth_mean

    Fo_hkl = np.fft.fftn(truth_raw)
    Fc_hkl = np.fft.fftn(fc_phys)

    Fc_amp = np.abs(Fc_hkl)
    safe   = np.where(Fc_amp > 1e-10, Fc_amp, 1e-10)
    exp_phi_c = Fc_hkl / safe          # unit phasors

    Fo_amp = np.abs(Fo_hkl)
    map_2fofc = np.fft.ifftn(2 * Fo_amp * exp_phi_c - Fc_hkl).real
    map_fofc  = np.fft.ifftn(    Fo_amp * exp_phi_c - Fc_hkl).real

    return (map_2fofc.astype(np.float32),
            map_fofc.astype(np.float32),
            fc_phys.astype(np.float32))


# ---------------------------------------------------------------------------
# Per-sample iteration
# ---------------------------------------------------------------------------

def iterate_sample(model, sample_dir, n_iter, mode, device):
    """Run inference iteration on one sample.  Returns list of MSE values."""
    truth_raw   = _load_map(os.path.join(sample_dir, 'truth.map'))
    fofc_raw    = _load_map(os.path.join(sample_dir, 'fofc.map'))
    fc_raw      = _load_map(os.path.join(sample_dir, 'fc.map'))
    twofofc_raw = _load_map(os.path.join(sample_dir, '2fofc.map'))

    truth_znorm = _znorm(truth_raw)

    cur_2fofc = twofofc_raw
    cur_fofc  = fofc_raw
    cur_fc    = fc_raw

    losses = []

    for i in range(n_iter + 1):
        ch0 = _znorm(cur_2fofc)
        ch1 = _znorm(cur_fofc)
        ch2 = _znorm(cur_fc)
        ch3 = _znorm(_cross_patterson(cur_fofc, cur_fc))

        pred = run_inference(model, ch0, ch1, ch2, ch3, device)

        # MSE in z-score space (same metric as training loss)
        mse = float(np.mean((pred - truth_znorm) ** 2))
        losses.append(mse)

        if i == n_iter:
            break

        # Update maps
        if mode == 'fc-only':
            _, _, fc_new = rebuild_maps_fc_only(truth_raw, pred)
            cur_fc = fc_new
            # cur_2fofc and cur_fofc unchanged
        else:  # full-fft
            new_2fofc, new_fofc, fc_new = rebuild_maps_fft(truth_raw, pred)
            cur_2fofc = new_2fofc
            cur_fofc  = new_fofc
            cur_fc    = fc_new

    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Inference-time iteration for electron density CNN')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (best.pt)')
    # Single sample
    parser.add_argument('--sample-dir', default=None,
                        help='Path to a single sample directory')
    # Multiple samples
    parser.add_argument('--data-dir', default=None,
                        help='Data directory; iterate over --n-samples random samples')
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    # Iteration control
    parser.add_argument('--n-iter', type=int, default=5,
                        help='Number of inference iterations (0 = single pass)')
    parser.add_argument('--mode', choices=['fc-only', 'full-fft'], default='full-fft',
                        help='fc-only: replace Fc channel only; '
                             'full-fft: rebuild 2Fo-Fc and Fo-Fc via FFT (default)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading model from {args.checkpoint} …")
    model = load_model(args.checkpoint, device)
    print(f"Mode: {args.mode},  iterations: {args.n_iter},  device: {device}\n")

    # Collect sample dirs
    if args.sample_dir:
        sample_dirs = [args.sample_dir]
    elif args.data_dir:
        random.seed(args.seed)
        all_dirs = sorted([
            os.path.join(args.data_dir, d)
            for d in os.listdir(args.data_dir)
            if d.startswith('sample_')
            and os.path.isdir(os.path.join(args.data_dir, d))
            and {'truth.map', '2fofc.map', 'fofc.map', 'fc.map'}.issubset(
                set(os.listdir(os.path.join(args.data_dir, d))))
        ])
        k = min(args.n_samples, len(all_dirs))
        sample_dirs = random.sample(all_dirs, k)
    else:
        parser.error('Provide --sample-dir or --data-dir')

    # Run
    all_losses = []
    for sd in sample_dirs:
        name = os.path.basename(sd)
        losses = iterate_sample(model, sd, args.n_iter, args.mode, device)
        all_losses.append(losses)
        parts = '  '.join(f'{v:.5f}' for v in losses)
        arrow = '↓' if losses[-1] < losses[0] else ('↑' if losses[-1] > losses[0] else '=')
        print(f'{name}:  {parts}  {arrow}')

    # Summary
    arr = np.array(all_losses)   # (n_samples, n_iter+1)
    print()
    print('Iteration means:')
    for i, mean_loss in enumerate(arr.mean(axis=0)):
        tag = ' ← start' if i == 0 else (f' ← iter {i}')
        print(f'  iter {i}: {mean_loss:.6f}{tag}')

    deltas = arr[:, -1] - arr[:, 0]
    n_improved = int((deltas < 0).sum())
    print(f'\n{n_improved}/{len(sample_dirs)} samples improved after {args.n_iter} iterations')
    print(f'Mean Δ MSE: {deltas.mean():+.6f}  (negative = better)')


if __name__ == '__main__':
    main()
