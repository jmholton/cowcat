# CowCat — CNN Electron Density Reconstruction

A 3D U-Net that reconstructs ground-truth electron density from phased maps containing systematic errors due to alternate conformations. Given the 2Fo-Fc, Fo-Fc, and Fc maps from a partial model (single centroid atom where the truth has a disordered cluster), the network recovers the true multi-Gaussian difference density.

## Architecture

- **Model**: 3D U-Net, 4 input channels, 1 output, 5.84M parameters
- **Input channels**: 2Fo-Fc, Fo-Fc, Fc, cross-Patterson (IFFT(FoFc · conj(Fc)))
- **Target**: `znorm(truth_density − Fc)` — the ideal difference map
- **Loss**: heteroscedastic Gaussian NLL with per-voxel uncertainty
- **Cell**: P1, 40×40×40 Å, 60×60×60 voxels (d_min = 2.0 Å)

## Data Generation

Each pipeline generates `sample_NNNNN/` directories containing `2fofc.map`, `fofc.map`, `fc.map`, `truth.map`, and `metadata.json`.

**1AHO system** (`generate_1aho.py`) — current pipeline:
```bash
# Submit one batch of 1000 samples per directory (Einsteinium/Lawrencium)
ccp4-python generate_1aho.py --submit --nsamples 1000 --seed 100 \
    --outdir data/data_1aho_s100 \
    --partition lr6 --account pc_als831 --qos lr_normal

# Repeat with different seeds for more data
# data/data_1aho_s200  (--seed 200)
# data/data_1aho_s300  (--seed 300)  ...
```

**Protein backbone** (`generate_protein.py`) — random 20-residue sequences with altloc:
```bash
ccp4-python generate_protein.py --submit --nsamples 1000 \
    --outdir data/data_protein_n1000 --partition debug --max-array 300
```

Keep ≤1000 samples per directory to avoid filesystem metadata slowdown.

## Packing (optional but recommended for large datasets)

Converts per-sample `.map` files into three memory-mapped arrays, reducing per-epoch file opens from `5×N` to `3`:

```bash
python3 pack.py --data data/data_1aho_s100 --workers 8
# writes: data/data_1aho_s100/X.npy  (N, 4, 60, 60, 60)
#         data/data_1aho_s100/Y.npy  (N, 1, 60, 60, 60)
#         data/data_1aho_s100/S.npy  (N,)  log-scale factors
```

`dataset.py` detects and uses packs automatically; unpacked directories still work.

## Training

**Einsteinium (SLURM batch):**
```bash
sbatch train.sh \
    --data data/data_1aho_s100 data/data_1aho_s200 \
    --outdir checkpoints_1aho \
    --epochs 200 --resume checkpoints_1aho/latest.pt
```

**Original cluster (interactive):**
```bash
srun --partition gpu --gres=gpu:1 \
    /programs/pytorch/envs/pt/bin/python train.py \
    --data data/data_protein_n1000 --epochs 200
```

Checkpoints saved to `<outdir>/best.pt` and `<outdir>/latest.pt`. Training log at `<outdir>/log.json`.

## Key Files

| File | Purpose |
|------|---------|
| `model.py` | UNet3D definition |
| `train.py` | Training loop |
| `dataset.py` | Dataset classes and split utilities |
| `pack.py` | Pack sample dirs → mmap arrays |
| `preprocess.py` | Pre-cache cross-Patterson (`crossp.npy`) per sample |
| `generate_1aho.py` | 1AHO data pipeline (current) |
| `generate_protein.py` | Random-protein data pipeline |
| `cluster.sh` | Cluster environment setup (sourced by SLURM scripts) |
| `train.sh` | SLURM batch script for training |

## Requirements

- CCP4 (with `ccp4-python`, `gemmi`, `refmac5`, `uniqueify`) — for data generation
- PyTorch ≥ 2.3 — for training (`ml/pytorch/2.3.1-py3.11.7-mf` on Einsteinium)
- Phenix — for `phenix.geometry_minimization` (protein pipeline only)
