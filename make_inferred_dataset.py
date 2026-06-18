#!/usr/bin/env python3
"""
make_inferred_dataset.py  –  Build a new dataset where the 2Fo-Fc, Fo-Fc, and Fc
maps are replaced by maps computed from a Stage-1 CNN prediction.

For each input sample:
  1. Load original maps (2fofc, fofc, fc, truth)
  2. Run Stage-1 inference → predicted density (z-score space)
  3. Scale prediction to physical units using truth statistics
  4. Rebuild 2Fo-Fc and Fo-Fc via FFT (|Ftrue| amplitudes + CNN phases)
  5. Save updated maps to output dir; truth.map copied unchanged

The resulting dataset can be trained on with train.py exactly as normal.

Usage:
    python make_inferred_dataset.py \\
        --checkpoint checkpoints_n10_N1del_allclust5refmac/best.pt \\
        --input-dirs data_n10_N1del_n1000 data_n10000_allclust_altconf5_n1000 \\
        --output-dir data_inferred_allclust5 \\
        --device cuda
"""

import argparse
import os
import sys
import json
import shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import UNet3D
from dataset import _load_map, _znorm, _cross_patterson


# ---------------------------------------------------------------------------
# CCP4 map I/O
# ---------------------------------------------------------------------------

def write_ccp4_map(data, template_path, out_path):
    """Write a float32 (D,H,W) array as a CCP4 map.

    Copies the 1024-byte standard header plus any extended header (NSYMBT bytes)
    verbatim from template_path, then appends the new data section.
    Cell, spacegroup, and grid dimensions are all correct since the output grid
    is the same shape as the template.
    """
    with open(template_path, 'rb') as f:
        header = f.read(1024)
        nsymbt = int(np.frombuffer(header[92:96], dtype=np.int32)[0])
        extended = f.read(nsymbt) if nsymbt > 0 else b''
    with open(out_path, 'wb') as f:
        f.write(header)
        f.write(extended)
        f.write(data.astype(np.float32).tobytes())


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet3D(in_channels=4, out_channels=1, base_features=32)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.to(device)
    return model


def run_inference(model, ch0, ch1, ch2, ch3, device):
    x = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
    x = torch.from_numpy(x[np.newaxis]).to(device)
    with torch.no_grad():
        y = model(x)
    return y.squeeze().cpu().numpy()


def rebuild_maps_fft(truth_raw, cnn_znorm):
    """Rebuild 2Fo-Fc, Fo-Fc, Fc from a CNN prediction.

    Uses |Ftrue| amplitudes with phases from the CNN prediction.
    Fc = CNN prediction scaled to physical units.
    """
    truth_std  = truth_raw.std()  if truth_raw.std()  > 1e-8 else 1.0
    truth_mean = truth_raw.mean()

    fc_phys = cnn_znorm * truth_std + truth_mean

    Fo_hkl = np.fft.fftn(truth_raw)
    Fc_hkl = np.fft.fftn(fc_phys)

    Fc_amp    = np.abs(Fc_hkl)
    safe      = np.where(Fc_amp > 1e-10, Fc_amp, 1e-10)
    exp_phi_c = Fc_hkl / safe

    Fo_amp    = np.abs(Fo_hkl)
    map_2fofc = np.fft.ifftn(2 * Fo_amp * exp_phi_c - Fc_hkl).real
    map_fofc  = np.fft.ifftn(    Fo_amp * exp_phi_c - Fc_hkl).real

    return (map_2fofc.astype(np.float32),
            map_fofc.astype(np.float32),
            fc_phys.astype(np.float32))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate inferred-map dataset from Stage-1 CNN predictions.')
    parser.add_argument('--checkpoint',  required=True,
                        help='Stage-1 model checkpoint (best.pt)')
    parser.add_argument('--input-dirs',  nargs='+', required=True,
                        help='Source dataset directories')
    parser.add_argument('--output-dir',  required=True,
                        help='Output dataset directory')
    parser.add_argument('--device',      default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f'Loading model from {args.checkpoint} ...')
    model = load_model(args.checkpoint, device)
    print(f'Device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect sample dirs from all inputs, sorted for reproducibility
    sample_dirs = []
    for d in args.input_dirs:
        for name in sorted(os.listdir(d)):
            sd = os.path.join(d, name)
            if (name.startswith('sample_') and os.path.isdir(sd) and
                    {'truth.map', '2fofc.map', 'fofc.map', 'fc.map'}.issubset(
                        set(os.listdir(sd)))):
                sample_dirs.append((d, name, sd))

    print(f'Processing {len(sample_dirs)} samples → {args.output_dir}')

    for i, (src_dir, name, sd) in enumerate(sample_dirs):
        out_sd = os.path.join(args.output_dir, f'sample_{i:05d}')
        os.makedirs(out_sd, exist_ok=True)

        truth_raw   = _load_map(os.path.join(sd, 'truth.map'))
        twofofc_raw = _load_map(os.path.join(sd, '2fofc.map'))
        fofc_raw    = _load_map(os.path.join(sd, 'fofc.map'))
        fc_raw      = _load_map(os.path.join(sd, 'fc.map'))

        ch0 = _znorm(twofofc_raw)
        ch1 = _znorm(fofc_raw)
        ch2 = _znorm(fc_raw)
        ch3 = _znorm(_cross_patterson(fofc_raw, fc_raw))

        pred = run_inference(model, ch0, ch1, ch2, ch3, device)

        new_2fofc, new_fofc, new_fc = rebuild_maps_fft(truth_raw, pred)

        template = os.path.join(sd, 'truth.map')
        shutil.copy(template, os.path.join(out_sd, 'truth.map'))
        write_ccp4_map(new_2fofc, template, os.path.join(out_sd, '2fofc.map'))
        write_ccp4_map(new_fofc,  template, os.path.join(out_sd, 'fofc.map'))
        write_ccp4_map(new_fc,    template, os.path.join(out_sd, 'fc.map'))

        # Copy metadata.json (required by ElectronDensityDataset)
        src_meta = os.path.join(sd, 'metadata.json')
        if os.path.exists(src_meta):
            shutil.copy(src_meta, os.path.join(out_sd, 'metadata.json'))

        # Provenance
        with open(os.path.join(out_sd, 'provenance.json'), 'w') as f:
            json.dump({'source_dir': src_dir, 'source_name': name,
                       'checkpoint': args.checkpoint}, f)

        if (i + 1) % 100 == 0 or i == 0:
            print(f'  {i+1}/{len(sample_dirs)}')

    print('Done.')


if __name__ == '__main__':
    main()
