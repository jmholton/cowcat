# CowCat — Convolutional Neural Network (CNN) Electron Density Reconstruction

![CowCat](cowcat.png)

This image — from Dr. Cowtan's
[A Tail of Two Cats](http://www.ysbl.york.ac.uk/~cowtan/fourier/coeff.html) —
inspired this project. It shows a cat reconstructed using observed Fourier
magnitudes from a complete cat image combined with phases from a *Manx* (tailless)
cat, producing the crystallographic analogue of a 2Fo-Fc map. The wispy cloud
above the ear is a ghost peak of the missing tail: the correct density is present
in the magnitudes but the wrong phases displace it to the wrong location. The
CNN in this project learns to correct exactly this kind of error — recovering
missing or misplaced density from the residual signal in the 2Fo-Fc and Fo-Fc
map coefficients.

A 3D U-Net + Fourier Neural Operator (FNO) network that reconstructs ground-truth
electron density from phased maps containing systematic errors due to alternate
conformations. Given the 2Fo-Fc, Fo-Fc, Fc, and a deconvolution channel (ch3) from
a partial model (centroid atom where truth has a disordered cluster), the network
recovers the true multi-Gaussian difference density.

## Current Status (2026-06)

The network consistently improves on synthetic validation loss and on most synthetic
test samples. On real 1AHO data, one run (`fno_lr3e4_4gpu_rc/best.pt`) has beaten
the Fc-alone R_free baseline. Current runs sit ~0.002–0.010 above that best; closing
this gap is the active focus.

Key finding: lower synthetic validation loss does not guarantee lower real-data R_free.
The network overfits the synthetic peak distribution and loses generalisation. Batch
Normalization (BN) appears to regularise usefully despite its theoretical drawbacks at
batch size 1. The epoch with the worst validation spike is often the best real-data
checkpoint (anti-correlation between synthetic val and real-data Rfree).

**Real-1AHO R_free leaderboard** (Fc baseline = 0.1097):

| checkpoint | R_free | arch / notes |
|---|---|---|
| **`fno_lr3e4_4gpu_rc/best.pt`** | **0.1083** | FNO+BN, lr=3×10⁻⁴, ep 27 — only run to beat Fc |
| mobius run `best_rfree.pt` | 0.1121 | FNO no-BN, Möbius ch3, ep 12 |
| ssqrt+lr3e4 warm-start (LR, ep 1) | 0.1173 | FNO no-BN, ssqrt ch3, warm-started from lr3e4 |
| `fno_4gpu_rc/best.pt` | 0.1138 | FNO+BN, lr=10⁻³, ep 40 |
| `fno_noBN_4gpu_rc` | 0.1192 | FNO no-BN, ssqrt ch3 |
| Fc baseline | 0.1097 | — |

## Architecture

- **Model**: UNet3D + parallel FNO branch, 6.50M parameters, no BN in conv blocks
- **Input (4 channels)**: 2Fo-Fc, Fo-Fc, Fc, ch3 (Fc-deconvolution encoding, see below)
- **ch3 encoding** — must match between `pack.py`, `train.py`, `infer.py`:
  - `--softsign` — `ratio/(1+|ratio|)`, zero at perfect fit (ratio=0), monotonic, bounded (−1,1) *(recommended)*
  - `--mobius` — Möbius transform `(|r|−1)/(|r|+1)·sign`, bounded [−1,1], injective
  - `--crossp-unitratio` — unit-ratio `min(|r|,1/|r|)·sign`, bounded [−1,1] but non-injective
  - `--crossp-raw` — raw cross-Patterson, unnormalised (spiky in P 2₁ 2₁ 2₁)
  - *(default)* — signed-sqrt cross-Patterson
  - In all cases, `ratio = FFT(Fo-Fc) / FFT(Fc)` (real-valued — Fc phases cancel)
- **Target**: `truth − Fc` difference map in e/Å³
- **Loss**: peak-weighted Mean Squared Error (MSE): `mean((1 + 0.5·|y|)·(pred−y)²)`
- **Grid**: 144×128×96 for 45.9×40.7×30.1 Å, d_min=0.965 Å, sample_rate=3.0
- **Cell**: P 2₁ 2₁ 2₁ — maps cover full unit cell (4 copies of the Asymmetric Unit (ASU))

## Data Pipelines

Each pipeline writes `sample_NNNNN/` directories containing:
`2fofc.map fofc.map fc.map truth.map metadata.json`.

**Protein backbone** (`generate_protein.py`) — 64-residue random sequences,
5 alternate-location (altloc) conformers, flood waters, P 2₁ 2₁ 2₁:
```bash
ccp4-python generate_protein.py --submit --nsamples 1000 \
    --outdir data/data_protein_v4_s0 \
    --n-altlocs 5 --n-flood 5000 --flood-occ 0.08 --seed 0 \
    --partition lr6 --account pc_als831 --qos lr_normal
```

**Simple O-atoms** (`generate_simple.py`) — 20 random oxygen atoms, 1 missing,
fast baseline for encoding experiments:
```bash
ccp4-python generate_simple.py --submit --nsamples 1000 \
    --outdir data/data_simple_b10 --b-range 10 10 \
    --partition lr6 --account pc_als831 --qos lr_normal

# Signed missing-atom occupancy (negative occupancy → negative Fo-Fc peak):
ccp4-python generate_simple.py --submit --nsamples 1000 \
    --outdir data/data_simple_negocc --truth-occ-range -1.0 1.0 \
    --partition lr6 --account pc_als831 --qos lr_normal
```

Keep ≤1000 samples per directory (Lustre metadata limit).

## Packing

Converts per-sample `.map` files into three memory-mapped (mmap) NumPy arrays:
```bash
# Choose ch3 encoding to match your training flag:
python3 pack.py --data data/data_protein_v4_s0 --softsign --workers 8
# Writes: X.npy (N,4,D,H,W)  Y.npy (N,1,D,H,W)  S.npy (N,)
# Output directory suffix: *_softsign
```

`--vary-flood` requires `--n-flood N` (any N > 0) to fire; without it, no flood waters
are generated. `--flood-occ` is ignored when `--vary-flood` is active (occupancy is
computed from the calibration line internally).

## Training

```bash
# Voltron (1 GPU):
train.csh outdir=checkpoints_my_run softsign=1 \
    data/data_protein_v4_s0_softsign data/data_simple_b10_softsign

# Lawrencium (4×A40, Distributed Data Parallel (DDP) via torchrun):
train.csh outdir=checkpoints_my_run nGPUs=4 softsign=1 accum_steps=4 \
    data/data_protein_v4_s0_softsign
```

Key flags: `accum_steps`, `lr`, `lr_min`, `epochs`, `base_features`,
`mobius`, `softsign`, `crossp_unitratio`.

Checkpoints saved: `best.pt` (best validation loss), `best_rfree.pt` (best
real-1AHO Rfree), `latest.pt` (most recent epoch).

## Net2 — Per-voxel σ predictor

After training net1, train a second 5-channel UNet3D on held-out data to predict
per-voxel uncertainty of net1's residual using Gaussian Negative Log-Likelihood (NLL)
loss:

```bash
# 1. Pack fresh (net1-unseen) samples with net1's predictions appended:
python3 pack_with_pred.py \
    --checkpoint checkpoints_my_run/best.pt \
    --data data/data_protein_v4_s1000_softsign

# 2. Train net2:
python3 train_sigma.py \
    --data data/data_protein_v4_s1000_softsign \
    --outdir checkpoints_sigma_my_run \
    --epochs 200
```

Net2 must be trained on data net1 has **never seen** — otherwise σ will be
falsely low where net1 overfits.

## Inference

```bash
# Whole-map inference (recommended; same path as in-training eval):
infer.csh checkpoint=checkpoints_my_run/best.pt softsign=1

# rfactor.py (F-space R/Rfree) — expects the TOTAL map (pred + Fc),
# i.e. predicted.map, NOT the predicted diff map:
python3 rfactor.py \
    --mtz refmacout.mtz --fc fc.map --pred predicted.map --fo-label FP
```

## Key Files

| File | Purpose |
|------|---------|
| `model.py` | UNet3D + parallel FNO branch |
| `train.py` | Training loop (DDP via torchrun, peak-weighted MSE loss) |
| `train_sigma.py` | Net2: per-voxel σ predictor (Gaussian NLL on net1 residual) |
| `dataset.py` | Dataset classes (`PackedDataset`, `PackedDatasetWithP`) |
| `pack.py` | Pack sample dirs → mmap arrays (`--softsign` / `--mobius` / `--crossp-unitratio` / `--crossp-raw`) |
| `pack_with_pred.py` | Append net1 predictions (`P.npy`) to a packed dataset for net2 training |
| `infer.py` | Whole-map inference → `predicted_diff.map` + `predicted.map` (pred + Fc) |
| `infer_multisample.py` | Batch inference + R-factor stats over N synthetic samples |
| `eval_1aho.py` | In-training Rfree diagnostic on real 1AHO data |
| `rfactor.py` | F-space Levenberg-Marquardt (LM) k+B scaling + R/Rfree vs MTZ |
| `generate_protein.py` | Random-protein pipeline (current primary) |
| `generate_simple.py` | Simple O-atom pipeline (fast encoding experiments) |
| `generate_1aho.py` | 1AHO-specific jiggle pipeline |
| `train.csh` / `infer.csh` | tcsh wrappers with cluster defaults |
| `cluster.sh` | Cluster environment setup (Einsteinium/Lawrencium) |

## Requirements

- CCP4 (`ccp4-python`, gemmi, refmac5, uniqueify) — data generation only
- PyTorch ≥ 2.3 — training (`ml/pytorch/2.3.1-py3.11.7-mf` on Lawrencium)
- Phenix — `phenix.geometry_minimization` (protein pipeline only)
