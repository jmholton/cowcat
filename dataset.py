"""
dataset.py  –  PyTorch Dataset for electron density map pairs.

Each sample directory contains:
    truth.map   – ground-truth Fc density (from truth_full.pdb)
    2fofc.map   – 2Fo-Fc from refmac       (input channel 0)
    fofc.map    – Fo-Fc difference map     (input channel 1)
    fc.map      – FC_ALL_LS from refmac    (input channel 2)

Target: truth.map − fc.map  (ideal Fo-Fc: truth density minus partial-model Fc).
Positive peaks mark missing density; negative peaks mark spurious centroid density.
Maps are z-score normalised per-channel per-sample before returning.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split


REQUIRED = {'truth.map', '2fofc.map', 'fofc.map', 'fc.map', 'metadata.json'}


def _load_map(path):
    """Read a CCP4 map without gemmi.

    The first three int32s of the header are NC, NR, NS (grid dimensions).
    The float32 data occupies exactly NC*NR*NS*4 bytes at the end of the file.
    offset = filesize - 4*NC*NR*NS
    """
    with open(path, 'rb') as f:
        nc, nr, ns = np.frombuffer(f.read(12), dtype=np.int32)
    n = int(nc) * int(nr) * int(ns)
    offset = os.path.getsize(path) - 4 * n
    data = np.fromfile(path, dtype=np.float32, count=n, offset=offset)
    return data.reshape(int(ns), int(nr), int(nc))


def _cross_patterson(fofc_arr, fc_arr):
    """Cross-correlation of FoFc and Fc maps: IFFT(FFT(FoFc) * conj(FFT(Fc))).
    Peaks at vectors from modeled atoms to missing atoms — directly encodes
    where the ghost peaks in FoFc point relative to the Fc density.
    No origin-peak problem since it is a cross- not auto-correlation."""
    return np.fft.irfftn(
        np.fft.rfftn(fofc_arr) * np.conj(np.fft.rfftn(fc_arr)),
        s=fc_arr.shape,
    ).real.astype(np.float32)


def _znorm(arr):
    """Z-score normalise; return as-is if std ≈ 0."""
    std = arr.std()
    if std < 1e-8:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


class ElectronDensityDataset(Dataset):
    def __init__(self, data_dir, sample_ids=None):
        """
        Args:
            data_dir:   root directory containing sample_NNNNN/ subdirectories
            sample_ids: optional list of subdirectory names; scans data_dir if None
        """
        self.data_dir = data_dir
        if sample_ids is None:
            self.sample_ids = sorted([
                d for d in os.listdir(data_dir)
                if d.startswith('sample_')
                and os.path.isdir(os.path.join(data_dir, d))
                and REQUIRED.issubset(set(os.listdir(os.path.join(data_dir, d))))
            ])
        else:
            self.sample_ids = list(sample_ids)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        base = os.path.join(self.data_dir, self.sample_ids[idx])

        ch0      = _znorm(_load_map(os.path.join(base, '2fofc.map')))
        fofc_raw = _load_map(os.path.join(base, 'fofc.map'))
        fc_raw   = _load_map(os.path.join(base, 'fc.map'))
        ch1 = _znorm(fofc_raw)
        ch2 = _znorm(fc_raw)

        crossp_path = os.path.join(base, 'crossp.npy')
        if os.path.exists(crossp_path):
            ch3 = _znorm(np.load(crossp_path))
        else:
            ch3 = _znorm(_cross_patterson(fofc_raw, fc_raw))

        truth_raw = _load_map(os.path.join(base, 'truth.map'))
        diff_raw  = truth_raw - fc_raw
        diff_std  = float(diff_raw.std())
        log_scale = np.log(diff_std + 1e-8)
        tgt = _znorm(diff_raw)

        x = np.stack([ch0, ch1, ch2, ch3], axis=0)  # (4, D, H, W)

        # Random axis flips — each of the 3 spatial axes flipped independently
        # gives 8 equally valid orientations (P1 has no symmetry constraints).
        for spatial_axis in (0, 1, 2):
            if np.random.rand() < 0.5:
                x   = np.flip(x,   axis=spatial_axis + 1)  # x has leading channel dim
                tgt = np.flip(tgt, axis=spatial_axis)       # tgt is (D,H,W)

        x   = torch.from_numpy(x.copy())
        y   = torch.from_numpy(tgt[np.newaxis].copy())      # (1, D, H, W)
        s   = torch.tensor(log_scale, dtype=torch.float32)  # scalar

        # Random periodic translation — exact for P1 maps (wrap-around is correct BC).
        # ch3 (cross-Patterson) is translation-invariant: P(M(·-d), N(·-d)) = P(M,N).
        # Only roll channels 0-2; ch3 stays at its natural (unshifted) state.
        shifts = [int(np.random.randint(0, d)) for d in x.shape[1:]]
        x = torch.cat([torch.roll(x[:3], shifts, dims=[1, 2, 3]), x[3:]], dim=0)
        y = torch.roll(y, shifts, dims=[1, 2, 3])

        return x, y, s


class PackedDataset(Dataset):
    """Fast dataset backed by pre-packed X.npy / Y.npy memory-mapped arrays.

    Created by pack.py. Opens 2 files instead of 6×N files — eliminates
    filesystem overhead on large datasets (10k+ samples).
    Augmentation (random axis flips) is still applied on the fly.
    """
    def __init__(self, x_path, y_path, indices=None):
        self.X = np.load(x_path, mmap_mode='r')  # (N, 4, D, H, W)
        self.Y = np.load(y_path, mmap_mode='r')  # (N, 1, D, H, W)
        self.indices = indices if indices is not None else np.arange(len(self.X))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = np.array(self.X[i])   # copy out of mmap → writable
        y = np.array(self.Y[i])   # (1, D, H, W)

        for spatial_axis in (0, 1, 2):
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=spatial_axis + 1)
                y = np.flip(y, axis=spatial_axis + 1)

        x = torch.from_numpy(x.copy())
        y = torch.from_numpy(y.copy())

        shifts = [int(np.random.randint(0, d)) for d in x.shape[1:]]
        x = torch.cat([torch.roll(x[:3], shifts, dims=[1, 2, 3]), x[3:]], dim=0)
        y = torch.roll(y, shifts, dims=[1, 2, 3])

        return x, y


def make_splits(data_dir, val_fraction=0.2, seed=42):
    """Return (train_dataset, val_dataset) with a reproducible random split.

    Uses PackedDataset (X.npy/Y.npy) if available, else ElectronDensityDataset.
    """
    x_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'Y.npy')
    if os.path.exists(x_path) and os.path.exists(y_path):
        n = len(np.load(x_path, mmap_mode='r'))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        return (PackedDataset(x_path, y_path, idx[n_val:]),
                PackedDataset(x_path, y_path, idx[:n_val]))

    full = ElectronDensityDataset(data_dir)
    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    return random_split(full, [n_train, n_val],
                        generator=torch.Generator().manual_seed(seed))


def make_splits_multi(data_dirs, val_fraction=0.2, seed=42):
    """Like make_splits but accepts a list of data directories.

    Concatenates all directories into one pool, then does a single train/val split.
    Falls back to make_splits for a single directory (supports PackedDataset).
    """
    if len(data_dirs) == 1:
        return make_splits(data_dirs[0], val_fraction=val_fraction, seed=seed)
    datasets = [ElectronDensityDataset(d) for d in data_dirs]
    combined = ConcatDataset(datasets)
    n = len(combined)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    return random_split(combined, [n_train, n_val],
                        generator=torch.Generator().manual_seed(seed))
