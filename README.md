# CowCat — CNN Electron Density Reconstruction

A 3D U-Net + FNO network that reconstructs ground-truth electron density from
phased maps containing systematic errors due to alternate conformations. Given
the 2Fo-Fc, Fo-Fc, Fc, and a ch3 deconvolution channel from a partial model
(centroid atom where truth has a disordered cluster), the network recovers the
true multi-Gaussian difference density.

## Current Status (2026-06)

The network consistently improves on synthetic validation loss and on most
synthetic test samples. On real 1AHO data, one run (`fno_lr3e4_4gpu_rc/best.pt`)
has beaten the fc-alone R_free baseline (0.1083 vs fc=0.1097). Current
no-BN runs sit ~1–2% above baseline; closing this gap is the active focus.

**Real-1AHO R_free leaderboard** (fc baseline = 0.1097):

| checkpoint | R_free | arch |
|---|---|---|
| `fno_lr3e4_4gpu_rc/best.pt` | **0.1083** | FNO+BN, lr=3e-4, ep27 |
| `fno_noBN_4gpu_rc` | 0.1192 | FNO no-BN, current runs |
| fc baseline | 0.1097 | — |

Key finding: lower synthetic val loss does not guarantee lower real-data R_free.
The network is prone to overfitting the synthetic peak distribution and losing
generalisation. BN appears to regularise usefully despite its theoretical
drawbacks at batch=1.

## Architecture

- **Model**: UNet3D + parallel FNO branch, 6.50M params, no BatchNorm
- **Input (4 channels)**: 2Fo-Fc, Fo-Fc, Fc, ch3 (deconvolution encoding)
- **ch3 options** (`pack.py` flag → `train.py` flag):
  - `--mobius` — Möbius-bounded Fc-deconvolution, bounded [-1,1], monotonic *(current best)*
  - `--crossp-unitratio` — unit-ratio deconvolution (non-injective at unity)
  - `--crossp-raw` — raw cross-Patterson (spiky in P2₁2₁2₁)
  - *(default)* — signed-sqrt cross-Patterson
- **Target**: `truth − fc` difference map in e/Å³
- **Loss**: peak-weighted MSE `mean((1 + 0.5·|y|)·(pred−y)²)`
- **Grid**: 144×128×96 for 45.9×40.7×30.1 Å / d_min=0.965 Å (sample_rate=3.0)
- **Cell**: P 2₁ 2₁ 2₁ — maps cover full unit cell (4 ASU copies)

## Data Pipelines

Each pipeline writes `sample_NNNNN/` dirs: `2fofc.map fofc.map fc.map truth.map metadata.json`.

**Protein backbone** (`generate_protein.py`) — 64-residue random sequences,
5 altloc conformers, flood waters, P 2₁ 2₁ 2₁:
```bash
ccp4-python generate_protein.py --submit --nsamples 1000 \
    --outdir data/data_protein_v4_s0 \
    --n-altlocs 5 --n-flood 5000 --flood-occ 0.08 --seed 0 \
    --partition lr6 --account pc_als831 --qos lr_normal
```

**Simple O-atoms** (`generate_simple.py`) — 20 random O atoms, 1 missing,
fast baseline for encoding experiments:
```bash
ccp4-python generate_simple.py --submit --nsamples 1000 \
    --outdir data/data_simple_b10 --b-range 10 10 \
    --partition lr6 --account pc_als831 --qos lr_normal

# With signed missing-atom occupancy (negative occ → negative Fo-Fc peak):
ccp4-python generate_simple.py --submit --nsamples 1000 \
    --outdir data/data_simple_negocc --truth-occ-range -1.0 1.0 \
    --partition lr6 --account pc_als831 --qos lr_normal
```

Keep ≤1000 samples per directory (Lustre metadata limit).

## Packing

Converts per-sample `.map` files into three memory-mapped arrays:
```bash
# Choose ch3 encoding to match your training flag:
python3 pack.py --data data/data_protein_v4_s0 --mobius --workers 8
# writes: X.npy (N,4,D,H,W)  Y.npy (N,1,D,H,W)  S.npy (N,)
# output dir suffix: *_mobius
```

## Training

```bash
# Voltron (1 GPU):
train.csh outdir=checkpoints_my_run mobius=1 \
    data/data_protein_v4_s0_mobius data/data_simple_b10_mobius

# Lawrencium (4×A40):
train.csh outdir=checkpoints_my_run nGPUs=4 mobius=1 accum_steps=4 \
    data/data_protein_v4_s0_mobius
```

Flags: `accum_steps`, `lr`, `lr_min`, `epochs`, `base_features`, `crossp_unitratio`, `mobius`.
Checkpoints: `best.pt` (best val), `best_rfree.pt` (best Rfree_1aho), `latest.pt`.

## Net2 — Per-voxel σ predictor

After training net1, train a second network on held-out data to predict
per-voxel uncertainty of net1's residual (Gaussian NLL):

```bash
# 1. Pack fresh (net1-unseen) samples with net1's predictions:
python3 pack_with_pred.py \
    --checkpoint checkpoints_my_run/best.pt \
    --data data/data_protein_v4_s1000_mobius

# 2. Train net2:
python3 train_sigma.py \
    --data data/data_protein_v4_s1000_mobius \
    --outdir checkpoints_sigma_my_run \
    --epochs 200
```

Net2 must be trained on data net1 has **never seen** — otherwise σ will be
falsely low where net1 overfits. See `train_sigma.py` for details.

## Inference

```bash
# Whole-map inference (recommended):
infer.csh checkpoint=checkpoints_my_run/best.pt mobius=1

# rfactor.py (F-space R/Rfree) expects the TOTAL map (pred + fc):
ccp4-python rfactor.py \
    --mtz refmacout.mtz --fc fc.map --pred predicted.map --fo-label FP
```

## Key Files

| File | Purpose |
|------|---------|
| `model.py` | UNet3D + FNO branch |
| `train.py` | Training loop (DDP via torchrun) |
| `train_sigma.py` | Net2: per-voxel σ predictor (Gaussian NLL on net1 residual) |
| `dataset.py` | Dataset classes (PackedDataset, PackedDatasetWithP) |
| `pack.py` | Pack sample dirs → mmap arrays (--mobius / --crossp-unitratio / --crossp-raw) |
| `pack_with_pred.py` | Add net1 predictions (P.npy) to a packed dataset for net2 training |
| `infer.py` | Whole-map or tiled inference → predicted_diff.map + predicted.map |
| `infer_multisample.py` | Batch inference + R-factor stats over N synthetic samples |
| `eval_1aho.py` | In-training Rfree diagnostic on real 1AHO data |
| `rfactor.py` | F-space k+B scaling + R/Rfree vs MTZ |
| `generate_protein.py` | Random-protein pipeline (current primary) |
| `generate_simple.py` | Simple O-atom pipeline (fast encoding experiments) |
| `generate_1aho.py` | 1AHO-specific jiggle pipeline |
| `train.csh` / `infer.csh` | tcsh wrappers with cluster defaults |
| `cluster.sh` | Cluster environment setup (Einsteinium/Lawrencium) |

## Requirements

- CCP4 (`ccp4-python`, `gemmi`, `refmac5`, `uniqueify`) — data generation
- PyTorch ≥ 2.3 — training (`ml/pytorch/2.3.1-py3.11.7-mf` on Lawrencium)
- Phenix — `phenix.geometry_minimization` (protein pipeline only)
