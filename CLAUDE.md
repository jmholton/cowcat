# Claude CNN — Electron Density Reconstruction

## Project Goal

Train a 3D U-Net to reconstruct ground-truth electron density (Fo) from phased maps that contain systematic errors. The current focus is on **alternate conformations**: the partial model has a single centroid atom where the truth has a disordered cluster; refmac refines the centroid's occupancy/position against the true multi-conf density. The CNN learns to recover the true multi-Gaussian density from the blurred/biased 2Fo-Fc, Fo-Fc, and Fc maps.

---

## Key Files

| File | Purpose |
|------|---------|
| `generate_protein.py` | **Primary data pipeline** — random protein backbone + altloc conformers + flood waters + refmac |
| `generate_simple.py` | Simple O-atom pipeline — N random O atoms in configurable cell/SG, 1 missing, refmac |
| `generate_1aho.py` | 1AHO-specific pipeline — jiggle real protein, flood waters, sfcalc, refmac, CCP4 maps |
| `generate_data.py` | Oldest pipeline — random O atoms, atom deletion, no refmac (kept for reference) |
| `model.py` | UNet3D: 4 input channels, 1 output, base_features=32, circular padding; **U-Net + parallel FNO branch** (factorized full-band F-FNO), no BN, 6.50M params |
| `train.py` | Training loop: peak-weighted MSE / heteroscedastic NLL, AdamW, CosineAnnealingLR, DDP via torchrun, --accum-steps, --eval-1aho-dir per-epoch real-data Rfree diagnostic |
| `eval_1aho.py` | In-training Rfree diagnostic on a held-out real-1aho sample; loaded by train.py at startup (axis convention: `ft[L,K,H]` for CCP4 (NS,NR,NC) maps) |
| `infer.py` | Inference on CCP4 map triples; writes `<output>` (predicted **diff**) and `predicted.map` (predicted **total** = pred+fc); --no-scale skips the demean+RMS-match-to-fofc rescale; auto-runs inline rfactor if `refmacout.mtz` present |
| `rfactor.py` | F-space LM k+B scaling + R/Rfree against MTZ; **`--pred` expects the total map (pred+fc), not the diff** |
| `dataset.py` | `ElectronDensityDataset`, `PackedDataset`, `PackedDatasetWithP` (for net2), `make_splits` / `make_splits_multi` |
| `pack.py` | Packs `sample_NNNNN/` dirs into `X.npy`/`Y.npy`/`S.npy`; ch3 encoding: `--mobius` (default new), `--crossp-unitratio`, `--crossp-raw`, or signed-sqrt |
| `pack_with_pred.py` | Runs net1 inference over a packed dataset and appends `P.npy` (net1 predictions) for net2 training |
| `train_sigma.py` | Trains net2 (5-channel UNet3D) to predict per-voxel log-variance of net1's residual via Gaussian NLL |
| `infer_multisample.py` | Batch inference + R-factor stats over N synthetic samples |
| `generate.csh` / `pack.csh` / `train.csh` / `infer.csh` | tcsh wrappers with sane v4 defaults |
| `jigglepdb.awk` | Displaces atom positions to generate alternate conformers |
| `explore_1aho_fusion.py` | Conformer scoring, rebuild, refmac utilities for the 1AHO iterative pipeline |
| `swapscan_varconf.py` | Chain-letter swap optimisation: random/geo-targeted trials, refmac NCYC=50, wE scoring |
| `swapscan_to_samples.py` | Convert swapscan `refmacout.mtz` files → `sample_NNNNN/` dirs for UNet3D training |
| `rebuild_iterate.py` | Standalone iterative rebuild loop: score Fo-Fc outliers → rebuild → refmac → repeat |
| `run_untangler_1aho.py` | Wrapper to run Untangler ILP on 1AHO varconf structure via SLURM |
| `condense_bb.py` | Backbone(+SS)-only maximin condensation sweep (gt48 → various k) |
| `condense_bb_varconf.py` | Per-residue varconf condensation on a multi-chain conformer model |
| `condense_singlechain.py` | Per-residue altloc maximin on a single-chain altloc PDB |

---

## Architecture

- **Map channels (input)**: 2Fo-Fc (FWT/PHWT), Fo-Fc (DELFWT/PHDELWT), Fc (FC_ALL_LS/PHIC_ALL_LS), ch3 deconvolution encoding (see options below)
- **ch3 encoding** — must match between `pack.py`, `train.py`, `infer.py`, `eval_1aho.py`:
  - `--mobius` → `(amp−1)/(amp+1)·sign(ratio)` where `ratio = FFT(fofc)/FFT(fc)` — bounded [−1,1], **injective** (monotonic), Fc phases cancel so ratio is real. Suffix `*_mobius`. *(current recommended)*
  - `--crossp-unitratio` → `min(|r|,1/|r|)·sign` — bounded [−1,1] but **non-injective** (ratio=0.5 and 2.0 give same output). Suffix `*_unitratio`.
  - `--crossp-raw` → raw cross-Patterson `IFFT(FFT(fofc)·conj(FFT(fc)))`. Suffix `*_rawcrossp`.
  - *(default)* → signed-sqrt cross-Patterson. Suffix `*_ssqrt`.
- **Target (Y)**: `znorm(truth.map − fc.map)` for older runs; plain `truth − fc` for current MSE-based runs
- **Scale (S)**: `std(truth − fc)` in e/Å³ — stored in S.npy; predicted by `scale_head` (currently unused in loss, requires `find_unused_parameters=True` in DDP)
- **Model**: UNet3D + parallel FNO branch, base_features=32, circular padding; **no BN** in `_ConvBlock` (Conv → ReLU only); fully convolutional (handles any grid size); 6,503,716 params
- **FNO branch**: factorized full-band 3D spectral conv (F-FNO style) running parallel to the U-Net; rfftn → per-axis per-frequency channel mixing → irfftn; covers the entire spectrum at O(C²·m) per axis (no low-mode truncation). Sum into the head with the U-Net mean prediction
- **Loss**: peak-weighted MSE `mean((1 + α|y|)·(pred-y)²)` with α=0.5 — current default; plain MSE returned as metric for Rrms comparability
- **Grid**: 60×60×60 for 40×40×40 Å / dmin=2.0; 144×128×96 for 45.9×40.7×30.1 Å / dmin=0.965 (sample_rate=3.0)
- **Note**: CCP4 maps cover the full P1 unit cell regardless of space group (4 ASU copies for P 21 21 21)

---

## Python Interpreters

**Critical — do not mix these up. Rules apply on both clusters:**

- `ccp4-python` — use for `generate_1aho.py`, `generate_protein.py`, and any script importing `gemmi`. Gemmi is only installed here.
- PyTorch python — use for `train.py`, `pack.py`, `dataset.py`, anything using torch.
- Never use the system `python` for project code.

**Original cluster** (`/programs/` paths visible):
- CCP4 python: `ccp4-python` (on PATH after `source /global/home/groups-sw/ac_als831/ccp4-X/bin/ccp4.setup-sh`)
- PyTorch python: `/programs/pytorch/envs/pt/bin/python`
- System `python` is Python 2. The symlink `./python3` points to the base pytorch python (no torch) — use the full `envs/pt` path for training.

**Einsteinium / Lawrencium** (`cluster.sh` handles both):
- CCP4 python: `ccp4-python` via `setup_ccp4` (sources `/global/home/groups-sw/ac_als831/ccp4-9/bin/ccp4.setup-sh`)
- PyTorch python: `python3` after `setup_pytorch`:
  ```bash
  source /etc/profile.d/modules.sh
  export MODULEPATH=$MODULEPATH:/global/software/rocky-8.x86_64/modfiles/Core
  module load --force ml/pytorch/2.3.1-py3.11.7-mf
  ```

---

## SLURM Cluster

Always pass `--export=ALL` to SLURM array jobs so CCP4 environment variables propagate.
`CCP4_SCR` directory must exist on compute nodes: `os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)`.

**Original cluster:**
- **Data generation**: `--partition debug`, `--workers 300` (cluster ~400 cores total)
- **Training**: `--partition gpu --gres=gpu:1`

```bash
ccp4-python generate_protein.py --submit --nsamples 1000 \
    --outdir data_n10_N1altconf3_refmac_n1000 --partition debug --max-array 300

srun --partition gpu --gres=gpu:1 \
    /programs/pytorch/envs/pt/bin/python train.py \
    --data data_n10_N1altconf3_refmac_n1000 --epochs 200
```

**Einsteinium / Lawrencium:**
- **Account**: `pc_als831`; **GPU QOS**: `es_normal` (min 16 CPUs); **CPU QOS**: `lr_normal`
- **GPU partition**: `es1` (4× A40 per node); **CPU partition**: `lr6`
- **MaxArraySize**: 1001 — submit ≤1000 samples per array job, one directory per batch
- Use `cd "$SLURM_SUBMIT_DIR"` before sourcing `cluster.sh` — SLURM runs from a spool directory.

```bash
# Generate: one 1000-sample batch per directory (avoids large-dir filesystem slowdown)
ccp4-python generate_1aho.py --submit --nsamples 1000 --seed 100 \
    --outdir data/data_1aho_s100 --partition lr6 --account pc_als831 --qos lr_normal

# Pack (once per dir, before training)
python3 pack.py --data data/data_1aho_s100 --workers 8

# Train
sbatch train.sh --data data/data_1aho_s100 data/data_1aho_s200 \
    --outdir checkpoints_1aho --epochs 200
```

---

## generate_protein.py Pipeline (Current — v2)

Each sample runs in an isolated `tempfile.mkdtemp()` directory.

1. Random 64-residue AA sequence (natural UniProt frequencies)
2. Build backbone via `build_n2c.awk` (Ramachandran-sampled phi/psi)
3. Build side chains via `build_side.awk` (Ponder-Richards chi rotamers)
4. Add random water molecules; set CRYST1 P 21 21 21 45.9×40.7×30.1; centre in box; randomise B factors
5. `phenix.geometry_minimization` → `minimized.pdb`
6. Self-refine B factors: refmac NCYC=20 against own SFs → realistic correlated B factors
7. `jigglepdb.awk` × N seeds (byB mode, shift_scale=0.5, through-bond correlated) → N conformers, each independently minimized
8. Combine into single-chain altloc format: **chain A** (protein, altlocs A–E, occ=1/N each), **chain S** (waters, altlocs A–E, occ=1/N each) → `multiconf.pdb` / `truth_full.pdb`
9. Inject flood waters (chain F, ±occ) into `truth_full.pdb` to simulate ghost solvent peaks
10. `_sfcalc_with_bulksolv(truth_full.pdb)` → `truth.mtz` (includes H via phenix.reduce, cavenv bulk solvent)
11. Build `refme.mtz` (F=|FC|, SIGF=0.02·|FC|), `uniqueify`, simulate missing/never-collected reflections
12. `step8_build_mixed_model`: reads N altloc chains → collapses to starthere.pdb:
    - All residues kept as altlocs A–E at occ=1/N (ALTLOC_DIST_THRESHOLD=0)
    - max_confs=min(N,3) — so with N=5, only 3 altlocs in starthere (2 conformers "missing")
    - Waters → chain S with 5-way altlocs at occ=1/N
    - H atoms stripped (refmac MAKE HYDR A adds them back)
    - `refmac_occupancy_setup.com` generates per-residue occ groups
13. `step9_refmac`: 2 rounds of NCYC=20, occupancy refinement via `refmac_occupancy_setup.com`
14. `step10_convert_maps`: truth.map, 2fofc.map, fofc.map, fc.map, **truediff.map** (= truth−fc, the training target)

**Key design decisions:**
- `truth_full.pdb` uses single-chain altloc format matching `refmacout.pdb` (both chain A + chain S) for direct visual comparison in Coot/PyMOL
- Flood waters (chain F) are injected AFTER all geommin steps to avoid false clash detection
- Waters in starthere.pdb start at occ=1/N so refmac can refine toward true occupancy
- `--flood-avoid-fullocc` flag is accepted but currently has no effect (legacy; `_generate_flood_waters` always avoids all existing atoms)

**v2 generation parameters (data_protein_v2_s*):**
```bash
ccp4-python generate_protein.py --submit --nsamples 1000 \
    --outdir data/data_protein_v2_s0 \
    --cell 45.9 40.7 30.1 --dmin 0.965 --spacegroup "P 21 21 21" \
    --nresidues 64 --nwaters 30 \
    --n-altlocs 5 --n-flood 5000 --flood-occ 0.08 \
    --altloc-swaps-per-res 5 --seed 0 \
    --partition lr6 --account pc_als831 --qos lr_normal
# flood-occ 0.08 → Rfree ≈ 10.7% from flood contribution alone
# altloc-swaps-per-res 5 = Poisson(5) random pairwise altloc label swaps per residue
# no --weight-matrix (refmac auto weighting)
```

---

## Key Tool Paths

```python
BUILD_N2C  = Path('/home/jamesh/projects/git/build_pdb/build_n2c.awk')
BUILD_SIDE = Path('/home/jamesh/projects/git/build_pdb/build_side.awk')
PHENIX_GM  = Path('/programs/phenix-2.0-5936/phenix_bin/phenix.geometry_minimization')
REFMAC5    = Path('/programs/ccp4-8.0/bin/refmac5')
JIGGLEPDB  = SCRIPT_DIR / 'jigglepdb.awk'
UNIQUEIFY  = 'uniqueify'   # on PATH via CCP4 environment
```

---

## Current Training Status (as of 2026-05-23)

Current loss is **peak-weighted MSE** (α=0.5). Plain MSE returned as metric so `Rrms = sqrt(val_MSE) / sqrt(mean(truth²))` stays comparable across runs.

### Real-1aho R_free leaderboard (vs `1aho_test/refmacout_minRfree.mtz`, fc baseline = 0.1097)

| checkpoint | R_work | R_free | best_val | ep | arch |
|---|---|---|---|---|---|
| **`fno_lr3e4_4gpu_rc/best.pt`** | **0.0687** | **0.1083** | 0.0133 | 27 | FNO+BN, lr=3e-4 — **only model to beat fc** |
| `fno_4gpu_rc/best.pt` | 0.1010 | 0.1138 | 0.0099 | 40 | FNO+BN, lr=1e-3 |
| `fno_noBN_4gpu_rc/latest.pt` | 0.0909 | 0.1192 | 0.0044 | 50 | FNO no-BN (current run) |
| `fno_acc4_4gpu_rc/best.pt` | 0.0835 | 0.1247 | 0.0047 | 89 | FNO+BN, accum 4 |
| fc baseline | 0.0840 | 0.1097 | — | — | — |

**Key finding (2026-05-23):** lower synthetic val loss → higher real-1aho R_free. `fno_lr3e4` with best_val=0.0133 beats fc on real data; the no-BN run with best_val=0.0044 sits ~1% above fc. We're overfitting the synthetic peak distribution and losing real-data generalization. Either BN regularizes usefully, or `fno_lr3e4` simply caught the sweet spot (ep 27) before overfitting.

**ch3 encoding progress (2026-06):** unit-ratio deconvolution (`--crossp-unitratio`) outperforms signed-sqrt cross-Patterson (urat3 Rfree_1aho=0.1128 vs ssqrt ~0.1154). Unit-ratio is a Fc-deconvolution in reciprocal space — physically more direct than the cross-Patterson for localising missing atoms. However it is **non-injective**: ratio=0.5 and ratio=2.0 give the same ch3 value. Replaced by `--mobius` encoding which is identical in physical interpretation but injective (monotonic Möbius transform), bounded [-1,1].

### eval pipeline validation

The in-training Rfree_1aho column (printed by `eval_1aho.py` in `train.py`) and external `infer.csh` + `rfactor.py` agree to ~0.003–0.004 R_free. By default `infer.csh` runs whole-map inference (no `--tile`) — same as `eval_1aho.py` — so this isn't a patching artefact. Remaining gap likely comes from differences in how the amplitude rescale interacts with the saved-and-reloaded vs in-memory map values, and from `eval_1aho.py` skipping the demean step. Use eval_1aho for relative comparison across epochs; use infer.csh for the canonical reportable number.

### Datasets available (packed)

Four ch3 encodings exist; suffix indicates which was used at pack time. `--mobius` is the current recommended encoding (monotonic, injective, bounded [-1,1]).

| Path | N | Grid | Notes |
|------|---|------|-------|
| `data/data_protein_v4_s0_rawcrossp` | 1000 | 144×128×96 | protein v4 (Wilson B match to 1aho, parallel reduce+sfcalc, per-conf geommin) |
| `data/data_simple_b10_rawcrossp` | 1000 | 144×128×96 | 20 O-atoms B=10 (sharp peak baseline) |
| `data/data_protein_v4_s0_ssqrt` | 1000 | 144×128×96 | same source as rawcrossp, signed-sqrt ch3 |
| `data/data_simple_b10_ssqrt` | 1000 | 144×128×96 | same source as rawcrossp, signed-sqrt ch3 |
| `data/data_1aho_n1000` | 991 | 144×128×96 | 1AHO n=1000 (older NLL training set) |
| `data/data_simple_v2_s0` | 1000 | 144×128×96 | 20 O-atoms, P 21 21 21 |

`data/data_protein_v2_s0` is **corrupted** — 962 dirs but 519 X.npy rows zero-filled (pack.py ran before generation finished); regenerate before reuse. See gotchas.

### generate_simple.py — signed missing-atom occupancy

`--truth-occ-range MIN MAX` draws the missing atom's occupancy from `Uniform[MIN, MAX]`; all present atoms remain at occ=1.0. Negative occ produces a **negative** Fo-Fc peak at the missing atom position (model has density truth doesn't). The partial model always uses occ=1.0 for present atoms, so refmac works normally — refmac never sees the missing atom.

```bash
# Half negative, half positive peaks (uniform across [-1, 1]):
ccp4-python generate_simple.py --submit --nsamples 1000 \
    --outdir data/data_simple_negocc_s0 --truth-occ-range -1.0 1.0 \
    --partition lr6 --account pc_als831 --qos lr_normal
```

### Net2 — per-voxel σ predictor

Trains a second 5-channel UNet3D to predict per-voxel log-variance of net1's error using Gaussian NLL loss. **Must be trained on data net1 has never seen** — otherwise σ is falsely low where net1 overfits.

```bash
# 1. Pack a fresh batch of samples with net1's predictions (creates P.npy):
python3 pack_with_pred.py \
    --checkpoint checkpoints_my_run/best.pt \
    --data data/data_protein_v4_s1000_mobius   # net1-unseen samples

# 2. Train net2 (single GPU, no DDP needed for ~4000 samples):
python3 train_sigma.py \
    --data data/data_protein_v4_s1000_mobius \
    --outdir checkpoints_sigma_my_run \
    --epochs 200 --lr 3e-4
```

Prior run `logs_stage2_n10000_allclust5.txt` reached val_NLL=0.07852 at epoch 10 before SLURM timeout.

### Architectural changes (2026-05-22 → 2026-05-23)

- **FNO branch added** to `UNet3D` (`SpectralConv3d` + `FNOBlock3d`). Factorized full-band F-FNO: per-axis weights at fixed `ref_len=96` control points, linearly interpolated to runtime axis length. Covers entire spectrum O(C²·m); no low-mode truncation. Stored real, viewed as complex in forward (DDP-safe).
- **BN removed** from `_ConvBlock`. With batch=1 per rank and no SyncBN, BN's per-rank running stats add noise and lock in wrong eval-mode values. Also at odds with the task (absolute-scale electron density predictions should pass through unmolested). Result: synthetic val drops further, but real-1aho generalization regresses — see leaderboard.
- **DDP via torchrun** in `train.py` (`LOCAL_RANK`, NCCL, DistributedSampler, `find_unused_parameters=True`). `train.csh` submits 4-GPU jobs; `train1.csh` is the 1-GPU variant.
- **Amplitude rescale** in `infer.py`: demean + RMS-match-to-fofc (`k = fofc.std() / pred.std()`). Disabled with `--no-scale`.
- **eval_1aho axis fix**: CCP4 maps load as (NS, NR, NC) = (Z, Y, X); index `ft[L%nz, K%ny, H%nx]` not `ft[H, K, L]`. Without this, in-training R_free was reading 0.67 instead of ~0.12.

### Loading legacy FNO+BN checkpoints

Current `model.py` has BN-less `_ConvBlock` (44 weight keys + FNO). To load checkpoints from before BN was removed (100 keys, `bias=False` Conv3d + BatchNorm3d), temporarily restore BN:

```python
self.net = nn.Sequential(
    nn.Conv3d(in_ch, out_ch, 3, padding=1, padding_mode='circular', bias=False),
    nn.BatchNorm3d(out_ch),
    nn.ReLU(inplace=True),
    nn.Conv3d(out_ch, out_ch, 3, padding=1, padding_mode='circular', bias=False),
    nn.BatchNorm3d(out_ch),
    nn.ReLU(inplace=True),
)
```

Revert before resuming training. Pre-FNO checkpoints (88 keys, no FNO branch) additionally need the FNO branch removed — easier to just re-train from scratch.

The Lawrencium `train.py` uses **`strict=False`** when loading `--pretrain`, which silently drops mismatched keys: a 100-key FNO+BN checkpoint loaded into the no-BN model gives `missing keys (random init): 21  unexpected keys (ignored): 77`. The 23 conv weights and the entire FNO branch transfer, but BN's per-channel scale is lost — ep 0 train loss spikes (~120) while the optimizer re-fits the missing pieces. The Voltron `train.py` uses `strict=True` and will error in the same situation. Backport `strict=False` if you want to warm-start across architecture changes here too.

### Ongoing runs (as of 2026-05-23 evening)

| job | host | data | warm-start | ep | best_val | Rfree_1aho |
|---|---|---|---|---|---|---|
| `26568133` train_v4_b10_fno_noBN_4gpu_rc | Voltron 4×TITAN V | rawcrossp | none (scratch) | 118 | 0.00384 (ep 116) | 0.1203 |
| `26569602` train_v4_b10_fno_noBN_ssqrt_2gpu_rc | Voltron 2×TITAN V | ssqrt | `fno_noBN_4gpu_rc/best.pt` | 12 | 0.00378 (ep 12) | 0.1210 |
| Lawrencium `train_v4_ssqrt_pretrain` | Lawrencium 1×A40 | ssqrt | `fno_lr3e4_4gpu_rc/best.pt` (strict=False) | 2 | 0.00451 (ep 2) | **0.1173** (ep 1) |

Lawrencium's `lr3e4`-warm-start + ssqrt is the most promising of the three so far — Rfree_1aho 0.1173 after just 1 epoch, second-best real-data result behind lr3e4 itself (0.1083). The other two runs are stuck at the ~0.121 plateau characteristic of from-scratch noBN training.

### GPU hardware

| host | GPUs | per-sample wall (training, eff_batch=16) |
|---|---|---|
| Voltron | 4× **TITAN V** (Volta GV100, 12 GB, 2017) | ~0.85 s |
| Lawrencium es1 | 4× **A40** (Ampere, 48 GB, 2020) | ~0.49 s |

A40 is ~1.7× faster per card and has 4× the memory. The earlier `base_features=64` OOM on TITAN V is a 12 GB-VRAM limit, not architectural — bf64 should fit on A40 with room to spare.

Cross-cluster wall-clock comparisons must be normalised by GPU count: Lawrencium 1×A40 at 785 s/ep is *faster per card* than Voltron 4×TITAN V at 341 s/ep — same total compute per epoch, the difference is just how many cards are sharing it.

## 1AHO Iterative Rebuild (varconf_sweep)

`explore_1aho_fusion.py` + `rebuild_iterate.py` implement a post-weight-snap rebuild loop for the 1AHO model: score Fo-Fc density outliers, apply top-N prune/add actions on gt48 conformers, run refmac NCYC=5, repeat.

**k16_old results** (`1aho/varconf_sweep/k16_old/`):
- Starting R=0.046/Rf=0.050 (16-slot weight-snap)
- After 10 rounds: R=0.039/Rf=0.042
- Peak candidate criterion (atom-position sampling of Fo-Fc at absent conformers) added but was always redundant with the regular scorer — peak was always already in the top-N list
- CYS26/CYS48 disulfide never appeared as a candidate; disulfide residues have low per-atom Fo-Fc signal

**Key implementation notes:**
- `run_refmac_quick` must resolve input paths to absolute before calling refmac (refmac runs in a tmpdir via `cwd=`, so relative paths fail)
- `find_map_peak_candidate` samples the DELFWT map at absent conformer atom positions (not the global map maximum, which lands in symmetry copies in P2₁2₁2₁)
- SS-coupled residues (CYS16/36, CYS22/46, CYS26/48, CYS12/63) are added as pairs

---

## 1AHO Varconf Optimisation (swapscan_varconf.py)

`swapscan_varconf.py` randomly trials chain-letter swap moves on a varconf PDB, runs refmac NCYC=50, and measures Rfree + rmsd_e (RMSD between |FC_ALL_LS| and LS-scaled |Fgt|) + wE (molprobify geometry score, lower=better).

**Optimisation trajectory** (each built on the previous, starting from gt48.pdb / varconf_opt3.pdb):

| File | Description | rmsd_e | Rf | wE |
|------|-------------|--------|-----|-----|
| `1aho/varconf_opt1.pdb` | Baseline (gt48 condensed to k=32) | 2.3522 | — | — |
| `1aho/varconf_opt2.pdb` | sc H→L ARG62 | 2.3522 | — | — |
| `1aho/varconf_opt3.pdb` | +2 more swaps | 2.3334 | 0.0398 | — |
| `1aho/varconf_opt4.pdb` | spr_0p12 t189 (8 swaps, from swapscan_spr_fine) | 2.3187 | 0.0398 | 161.933 |
| `1aho/varconf_opt5.pdb` | targeted t13: GLU32+LYS50+PHE15+VAL10+TYR42+ARG18 | 2.3395 | 0.0398 | 155.495 |
| `1aho/varconf_opt6.pdb` | swapscan_opt5 spr_0p06 t28: LYS2+ARG18+THR27+GLU24 | 2.3431 | 0.0395 | 154.349 |
| `1aho/varconf_opt7.pdb` | targeted: HIS64 sc A→L + ARG62 sc A→O | — | 0.0373 | 148.928 |
| `1aho/varconf_opt8.pdb` | geo-submit opt7_geo: LYS30 sc I→J | — | — | ~144.6 |
| `1aho/varconf_opt9.pdb` | geo-submit opt8_geo: ASP8 sc C→O | — | — | ~144.0 |
| `1aho/varconf_opt10.pdb` | geo-submit opt9_geo: ASP8 pep N→P | — | — | 142.921 |
| `1aho/varconf_opt11.pdb` | geo-submit opt10_geo: GLU32 sc E→F | — | — | 142.375 |
| `1aho/varconf_opt12.pdb` | geo-submit opt11_geo: GLU24 pep D→L | — | — | 142.062 |
| `1aho/varconf_opt13.pdb` | geo-submit opt12_geo: CYS63 sc B→E | — | — | 141.521 |
| `1aho/varconf_opt14.pdb` | targeted opt12: CYS63sc+ASP9pep simultaneous | — | — | 141.303 |
| `1aho/varconf_opt15.pdb` | geo-submit opt13_geo: ASP9 pep Q→S (sequential better than simultaneous) | — | — | 141.153 |

**Scoring metrics:**
- **rmsd_e**: lower = better agreement with ground-truth structure factors
- **wE (weighted energy)**: `molprobify_runme.com` output — sum of weighted phenix geometry deviations (ramalyze, rotalyze, omegalyze, cbetadev, etc.). Lower = better geometry. Started at ~161.5, now ~143.5.
- `molprobify_runme.com` is at `~/Develop/molprobify_runme.com`; takes ~5–10 min per run on the 10101-atom 23-chain varconf structure. Runs inline in every trial (no separate rescore needed).

**Standard workflow per round (Lawrencium):**
```bash
# 1. Submit sweep from current opt PDB (4 spr values × 250 = 1000 jobs)
ccp4-python swapscan_varconf.py --submit \
    --pdb 1aho/varconf_optN.pdb --fobs 1aho/refme.mtz --truth 1aho/gt48.mtz \
    --outdir 1aho/swapscan_optN --sweep-spr 0.063,0.094,0.125,0.188 \
    --n-trials 250 --ncyc 50 \
    --partition lr6 --account pc_als831 --qos lr_normal

# 2. Collate when done — prints ΔRf table, wE table, AND compatible combos
ccp4-python swapscan_varconf.py --collate --outdir 1aho/swapscan_optN

# 3. Save winner as next opt PDB
cp 1aho/swapscan_optN/spr_0pXX/trial_NNNNN/swap.pdb 1aho/varconf_opt{N+1}.pdb

# 4. Update targeted_submit() in swapscan_varconf.py:
#    - set base_pdb to varconf_opt{N+1}.pdb
#    - replace building blocks + combos with groups from combo-finder output
# Then submit:
ccp4-python swapscan_varconf.py --targeted-submit --outdir 1aho/swapscan_targeted_opt{N+1} \
    --partition lr6 --account pc_als831 --qos lr_normal

# 5. Collate targeted results; save best as next opt PDB; repeat
```

**Geo-targeted workflow (--geo-submit, used from opt7 onward):**

Runs `phenix.ramalyze` + `phenix.rotalyze` on the base PDB, restricts the swap catalog to residues with OUTLIER geometry, and submits only those (~2,700–3,900 trials vs 11,218 exhaustive). Yields one best-ΔwE swap per round. Uses `refmacout.pdb` from the winning trial as the next round's base PDB.

```bash
# Submit geo-targeted scan from current opt PDB
ccp4-python swapscan_varconf.py --geo-submit \
    --pdb 1aho/varconf_optN.pdb --fobs 1aho/refme.mtz --truth 1aho/gt48.mtz \
    --outdir 1aho/swapscan_optN_geo --ncyc 50 \
    --partition lr6 --account pc_als831 --qos lr_normal

# Collate when done — winner is the trial with best ΔwE
ccp4-python swapscan_varconf.py --collate --outdir 1aho/swapscan_optN_geo

# Save winner as next opt PDB (use refmacout.pdb, not swap.pdb)
cp 1aho/swapscan_optN_geo/trial_NNNNN/refmacout.pdb 1aho/varconf_opt{N+1}.pdb

# Repeat
```

- `--include-allowed` flag also adds Ramachandran Allowed residues to the scan target
- Outlier count decreases each round as geometry improves (13 → 10 → 10 → 9 over opt7–opt10)

**Compatible-combo detection** (automatic in `--collate`):
- After each per-subdir collation, `find_compatible_combos()` scans the top-20 wE trials
- Two trials are compatible if no swap residue in one is within 1 sequence position of any in the other
- Skipped automatically if trials have >8 swaps (too complex to interpret)
- Prints top-5 compatible pairs and top-3 compatible triples ranked by combined ΔwE
- Use the printed groups to populate `targeted_submit()` for the next targeted round

**Other CLI modes:**
```bash
# Rescore old runs that predate molprobify integration
ccp4-python swapscan_varconf.py --rescore-submit --outdir 1aho/swapscan_spr_fine \
    --partition lr6 --account pc_als831 --qos lr_normal
```

**Lawrencium notes:**
- Partition `lr6`, account `pc_als831`, QOS `lr_normal`; no special time limit needed (lr_normal default is generous)
- molprobify calls phenix tools — verify phenix is on PATH after `setup_ccp4` on compute nodes before first run
- MaxArraySize 1001: our 251-trial batches fit in a single array job with no splitting needed
- Home directory (`~/Develop/molprobify_runme.com`) is NFS-mounted and accessible from compute nodes

**Gotchas:**
- wE and rmsd_e are in tension: high-spr trials improve wE but hurt rmsd_e/Rf. Low-spr (0.063–0.094) gives best balance.
- `targeted_submit()` has hardcoded base PDB and building blocks — update both when advancing to a new opt PDB.
- spr=2× previous optimal swaps is a good rule of thumb for the sweep range upper bound.
- `--geo-submit` winner: copy `refmacout.pdb` (not `swap.pdb`) as the next opt PDB — refmacout.pdb is the fully refined structure after the swap.
- CCP4 is already set up in the shell environment on Lawrencium — no need to re-source setup scripts before running swapscan commands.

---

## Backbone Conformer Condensation

Three scripts implement increasingly sophisticated conformer-count reduction on 1AHO models, all refining a backbone+disulfide-only model against Fc data computed from the same backbone+SS atoms.

| script | input format | conformer selection | starting atoms |
|--------|--------------|--------------------|----------------|
| `condense_bb.py` | multi-chain (e.g. gt48.pdb, 48 chains) | global maximin → flat k chains | ~18,700 (gt48 backbone) |
| `condense_bb_varconf.py` | multi-chain | per-residue maximin via `build_varconf_pdb` (uses chain A as residue/atom template) | as input |
| `condense_singlechain.py` | single-chain altloc (e.g. deconform output) | per-residue altloc maximin (no template, no duplicates) | as input |

**Pipeline (`condense_singlechain.py`, the current best):**

1. Strip input PDB to backbone + cysteine CB/SG (preserves disulfide network) → `bbss.pdb`
2. Compute Fobs MTZ as `FP = |Fc(bbss)|`, `Fpart = 0`, `FreeR_flag` from `1aho/refme_minRfree.mtz` (the actual diffraction free-R split)
3. For each residue: heavy-atom max-deviation across altlocs → look up target k from a threshold table → maximin-pick that many altlocs
4. `reoccupy.awk` (`~/Develop/`) renormalizes per-residue occupancies to sum=1
5. 4 rounds of refmac5-newhess weight-snap (NCYC 10 each at wm 0.01→0.1→1→10→0.5)
6. Rounds ≥3 generate occupancy-group refmac keywords via `~/Develop/refmac_occupancy_setup.com` (instead of the built-in `generate_occ_groups`)

**Threshold sets** are defined in `condense_singlechain.py:THRESHOLD_SETS` (top of the file). To list available names from the CLI: `ccp4-python condense_singlechain.py --help` shows them under `--threshold-set`. Each set maps heavy-atom max-deviation (Å) to per-residue k:

| set | (dev, k) brackets | bottom k |
|---|---|---|
| `floor1lean` | (0.5,1) (0.8,2) (1.2,3) (1.8,5) (2.5,7) (99,10) | 1 |
| `floor1` | (0.4,1) (0.6,2) (0.8,3) (1.2,5) (2.0,8) (99,12) | 1 |
| `floor2` | (0.6,2) (0.8,3) (1.2,5) (2.0,8) (99,12) | 2 |
| `default` | (0.6,2) (0.8,4) (1.0,6) (1.5,8) (2.5,12) (99,16) | 2 |
| `lean` | (0.6,1) (0.8,2) (1.0,4) (1.5,6) (2.5,8) (99,12) | 1 |
| `midrich` | (0.6,3) (0.8,5) (1.0,7) (1.5,10) (2.5,14) (99,20) | 3 |
| `rich` | (0.6,4) (0.8,6) (1.0,8) (1.5,12) (2.5,16) (99,24) | 4 |

**Best results from `1aho/deconform_under20_best_0025.pdb` (22-altloc deconform output)** — backbone+SS, 4 weight-snap rounds, `--cys-floor 3`:

| set | atoms | R rd 4 | Rfree rd 4 |
|---|---|---|---|
| floor1lean | 945 | 6.69% | 6.82% |
| floor1 | 1281 | 5.57% | 5.80% |
| **floor2** | **1381** | **4.40%** | **4.71%** |
| midrich | 1966 | 3.16% | 3.37% |

`floor2` is the sweet spot: under 5% Rfree budget, ~30% smaller atom count than `midrich`.

### floor2 recipe (Lawrencium)

```bash
# On Lawrencium (lr6 partition, pc_als831 account, lr_normal QOS).
# Source code already includes refmac5-newhess override and reoccupy/occ-setup wiring.

cd /path/to/claude_CNN
source cluster.sh && setup_ccp4

# Verify required external tools are reachable from the compute node
ls ~/Develop/reoccupy.awk ~/Develop/refmac_occupancy_setup.com
which refmac5-newhess  # path is hardcoded in condense_singlechain.py:
                       # /programs/ccp4-8.0/bin/refmac5-newhess — adjust if cluster differs

sbatch --partition=lr6 --account=pc_als831 --qos=lr_normal \
       --ntasks=1 --cpus-per-task=1 --mem=8G --export=ALL \
       --job-name=floor2 --output=floor2_%j.log \
       --wrap="cd \$SLURM_SUBMIT_DIR && \
               ccp4-python condense_singlechain.py \
                 --threshold-set floor2 --n-rounds 4 --cys-floor 3 \
                 --singlechain-pdb 1aho/deconform_under20_best_0025.pdb \
                 --outdir 1aho/condense_singlechain"

# Single ~10-minute job. Result lands in 1aho/condense_singlechain/floor2/result.json
ccp4-python -c "import json; d=json.load(open('1aho/condense_singlechain/floor2/result.json')); print(d['rounds'][-1])"
```

**Notes for the Lawrencium counterpart:**
- The hardcoded refmac5-newhess path (`/programs/ccp4-8.0/bin/refmac5-newhess`) in `condense_singlechain.py` works on the original cluster only. On Lawrencium, point it at the equivalent newhess binary or revert to plain refmac5 in `ccp4-9` if newhess is unavailable.
- `~/Develop/reoccupy.awk` and `~/Develop/refmac_occupancy_setup.com` are NFS-mounted from the user's home, so they should be available on Lawrencium compute nodes.
- The `--cys-floor 3` keeps the disulfide-bonded cysteines at ≥3 conformers, which buys ~0.2–0.7 % Rfree at +60–100 atom cost — worth it for SS-rich structures like 1AHO (4 disulfides).
- Threshold sets are defined inline in `condense_singlechain.py:THRESHOLD_SETS`; add new ones there.

---

## Untangler (ILP Conformer Label Optimisation)

`untangler/` is a git submodule (branch `2_conformer_challenge_solution` of github.com/Phoelionix/Untangler). It resolves tangled altloc assignments by ILP, scoring via wE (phenix geometry) + Rfree from phenix.refine.

**Run:**
```bash
sbatch slurm_untangler_1aho.sh --pdb 1aho/conf3norm_fitGT48.pdb \
    --altloc-subset-size 3 --max-runs 5
```

**Input requirements (critical):**
- Single-chain altloc format: all protein atoms in chain A with altloc letters (A, B, C, ...)
- **Uniform altloc coverage**: every disordered residue must have ALL altloc letters present — no partial sets (e.g., B+C without A). Fill gaps by copying the available altloc line with the altloc column changed.
- **Aromatic side chains required**: `relabel_ring` crashes on backbone-only structures. Use full-atom PDB.
- `altloc_subset_size` must equal the number of conformers, or the highest-letter conformer is left unpaired.

**`1aho/conf3norm_fitGT48.pdb`** (working 3-conformer input):
- 2865 atoms = 955 atoms × 3 altlocs (A, B, C), fully normalized coverage
- Derived from `under20_fitGT48.pdb` altlocs A/B/C; missing altlocs filled by copying nearest available

**Submodule patches** (applied directly in `untangler/`, not committable from parent):
- `LinearOptimizer/ConstraintsHandler.py` lines 641–644: changed duplicate clash-distance `assert` to a warning (f-string bug accessed wrong dict key during assert message formatting)
- `LinearOptimizer/Solver.py` lines ~1144 and ~1316: changed `if lp_problem.sol_status==LpStatusInfeasible` → `if lp_problem.sol_status != 1` — PuLP returns `sol_status=0` (no solution found) for infeasible next-best problems, not `-1` (infeasible), so the original guard was never triggered
- `untangle.py` line ~1503: replaced `shutil.rmtree(tmp_refine_subdir)` with `subprocess.run(['rm', '-rf', tmp_refine_subdir], check=False)` — `shutil.rmtree` raises `OSError: ENOTEMPTY` on NFS after unlinking files (NFS directory-cache race); `rm -rf` is more robust

**SLURM routing:** `Untangler.refine_shell_file` is set to `untangler/Refinement/Refine_slurm.sh`, which submits each phenix.refine call to the `refmac` partition via `sbatch --parsable`, polls until the job exits, and cancels the SLURM job via `trap "scancel $SLURM_JOB_ID" EXIT` if the wrapper is killed (e.g. by Python's subprocess timeout). This prevents zombie jobs from piling up in the refmac queue.

**Skip cross-conformer subprocess calls:** set BOTH `NonbondConstraint: 0` AND `ClashConstraint: 0` in `weight_factors`. Setting only one still runs both cross-conformer scripts (see `skip_nonbonds` logic in ConstraintsHandler.py).

**Performance note:** `GenerateHoltonData.sh` → `untangle_score_weighted.csh` → `phenix.molprobity` runs single-threaded on the full model every ILP loop. With 22 conformers (~10,000 atoms) this is hours per call; the 3-conformer model (~955 atoms) is fast.

**For Untangler to make swaps**, the input must be deliberately tangled (conformer labels not yet optimized). A well-refined structure scores 0 high-tension connections each ILP loop and makes no moves (`"moves": {}` in every `xLO-toFlip_*.json`). Score improvements in that case come entirely from the phenix.refine cycles Untangler runs between ILP iterations, not from any relabelling.

**Completed run on `conf3norm_fitGT48.pdb`** (5 loops, `--altloc-subset-size 3`):
- No altloc swaps in any loop — input was already optimally labelled
- Rfree improved 18.62 → 18.51% purely from refinement; wE 87.7 → 84.0
- Final output: `untangler/output/conf3norm_fitGT48_loopEnd4.pdb` (same conformer assignments, slightly better geometry)

---

## Backbone Conformer Condensation

Three scripts implement increasingly sophisticated conformer-count reduction on 1AHO models, all refining a backbone+disulfide-only model against Fc data computed from the same backbone+SS atoms.

| script | input format | conformer selection | starting atoms |
|--------|--------------|--------------------|----------------|
| `condense_bb.py` | multi-chain (e.g. gt48.pdb, 48 chains) | global maximin → flat k chains | ~18,700 (gt48 backbone) |
| `condense_bb_varconf.py` | multi-chain | per-residue maximin via `build_varconf_pdb` (uses chain A as residue/atom template) | as input |
| `condense_singlechain.py` | single-chain altloc (e.g. deconform output) | per-residue altloc maximin (no template, no duplicates) | as input |

**Pipeline (`condense_singlechain.py`, the current best):**

1. Strip input PDB to backbone + cysteine CB/SG (preserves disulfide network) → `bbss.pdb`
2. Compute Fobs MTZ as `FP = |Fc(bbss)|`, `Fpart = 0`, `FreeR_flag` from `1aho/refme_minRfree.mtz` (the actual diffraction free-R split)
3. For each residue: heavy-atom max-deviation across altlocs → look up target k from a threshold table → maximin-pick that many altlocs
4. `reoccupy.awk` (`~/Develop/`) renormalizes per-residue occupancies to sum=1
5. 4 rounds of refmac5-newhess weight-snap (NCYC 10 each at wm 0.01→0.1→1→10→0.5)
6. Rounds ≥3 generate occupancy-group refmac keywords via `~/Develop/refmac_occupancy_setup.com` (instead of the built-in `generate_occ_groups`)

**Threshold sets** are defined in `condense_singlechain.py:THRESHOLD_SETS` (top of the file). To list available names from the CLI: `ccp4-python condense_singlechain.py --help` shows them under `--threshold-set`. Each set maps heavy-atom max-deviation (Å) to per-residue k:

| set | (dev, k) brackets | bottom k |
|---|---|---|
| `floor1lean` | (0.5,1) (0.8,2) (1.2,3) (1.8,5) (2.5,7) (99,10) | 1 |
| `floor1` | (0.4,1) (0.6,2) (0.8,3) (1.2,5) (2.0,8) (99,12) | 1 |
| `floor2` | (0.6,2) (0.8,3) (1.2,5) (2.0,8) (99,12) | 2 |
| `default` | (0.6,2) (0.8,4) (1.0,6) (1.5,8) (2.5,12) (99,16) | 2 |
| `lean` | (0.6,1) (0.8,2) (1.0,4) (1.5,6) (2.5,8) (99,12) | 1 |
| `midrich` | (0.6,3) (0.8,5) (1.0,7) (1.5,10) (2.5,14) (99,20) | 3 |
| `rich` | (0.6,4) (0.8,6) (1.0,8) (1.5,12) (2.5,16) (99,24) | 4 |

**Best results from `1aho/deconform_under20_best_0025.pdb` (22-altloc deconform output)** — backbone+SS, 4 weight-snap rounds, `--cys-floor 3`:

| set | atoms | R rd 4 | Rfree rd 4 |
|---|---|---|---|
| floor1lean | 945 | 6.69% | 6.82% |
| floor1 | 1281 | 5.57% | 5.80% |
| **floor2** | **1381** | **4.40%** | **4.71%** |
| midrich | 1966 | 3.16% | 3.37% |

`floor2` is the sweet spot: under 5% Rfree budget, ~30% smaller atom count than `midrich`.

### floor2 recipe (Lawrencium)

```bash
# On Lawrencium (lr6 partition, pc_als831 account, lr_normal QOS).
# Source code already includes refmac5-newhess override and reoccupy/occ-setup wiring.

cd /path/to/claude_CNN
source cluster.sh && setup_ccp4

# Verify required external tools are reachable from the compute node
ls ~/Develop/reoccupy.awk ~/Develop/refmac_occupancy_setup.com
which refmac5-newhess  # path is hardcoded in condense_singlechain.py:
                       # /programs/ccp4-8.0/bin/refmac5-newhess — adjust if cluster differs

sbatch --partition=lr6 --account=pc_als831 --qos=lr_normal \
       --ntasks=1 --cpus-per-task=1 --mem=8G --export=ALL \
       --job-name=floor2 --output=floor2_%j.log \
       --wrap="cd \$SLURM_SUBMIT_DIR && \
               ccp4-python condense_singlechain.py \
                 --threshold-set floor2 --n-rounds 4 --cys-floor 3 \
                 --singlechain-pdb 1aho/deconform_under20_best_0025.pdb \
                 --outdir 1aho/condense_singlechain"

# Single ~10-minute job. Result lands in 1aho/condense_singlechain/floor2/result.json
ccp4-python -c "import json; d=json.load(open('1aho/condense_singlechain/floor2/result.json')); print(d['rounds'][-1])"
```

**Notes for the Lawrencium counterpart:**
- The hardcoded refmac5-newhess path (`/programs/ccp4-8.0/bin/refmac5-newhess`) in `condense_singlechain.py` works on the original cluster only. On Lawrencium, point it at the equivalent newhess binary or revert to plain refmac5 in `ccp4-9` if newhess is unavailable.
- `~/Develop/reoccupy.awk` and `~/Develop/refmac_occupancy_setup.com` are NFS-mounted from the user's home, so they should be available on Lawrencium compute nodes.
- The `--cys-floor 3` keeps the disulfide-bonded cysteines at ≥3 conformers, which buys ~0.2–0.7 % Rfree at +60–100 atom cost — worth it for SS-rich structures like 1AHO (4 disulfides).
- Threshold sets are defined inline in `condense_singlechain.py:THRESHOLD_SETS`; add new ones there.

---

## Untangler (ILP Conformer Label Optimisation)

`untangler/` is a git submodule (branch `2_conformer_challenge_solution` of github.com/Phoelionix/Untangler). It resolves tangled altloc assignments by ILP, scoring via wE (phenix geometry) + Rfree from phenix.refine.

**Run:**
```bash
sbatch slurm_untangler_1aho.sh --pdb 1aho/conf3norm_fitGT48.pdb \
    --altloc-subset-size 3 --max-runs 5
```

**Input requirements (critical):**
- Single-chain altloc format: all protein atoms in chain A with altloc letters (A, B, C, ...)
- **Uniform altloc coverage**: every disordered residue must have ALL altloc letters present — no partial sets (e.g., B+C without A). Fill gaps by copying the available altloc line with the altloc column changed.
- **Aromatic side chains required**: `relabel_ring` crashes on backbone-only structures. Use full-atom PDB.
- `altloc_subset_size` must equal the number of conformers, or the highest-letter conformer is left unpaired.

**`1aho/conf3norm_fitGT48.pdb`** (working 3-conformer input):
- 2865 atoms = 955 atoms × 3 altlocs (A, B, C), fully normalized coverage
- Derived from `under20_fitGT48.pdb` altlocs A/B/C; missing altlocs filled by copying nearest available

**Submodule patches** (applied directly in `untangler/`, not committable from parent):
- `LinearOptimizer/ConstraintsHandler.py` lines 641–644: changed duplicate clash-distance `assert` to a warning (f-string bug accessed wrong dict key during assert message formatting)
- `LinearOptimizer/Solver.py` lines ~1144 and ~1316: changed `if lp_problem.sol_status==LpStatusInfeasible` → `if lp_problem.sol_status != 1` — PuLP returns `sol_status=0` (no solution found) for infeasible next-best problems, not `-1` (infeasible), so the original guard was never triggered
- `untangle.py` line ~1503: replaced `shutil.rmtree(tmp_refine_subdir)` with `subprocess.run(['rm', '-rf', tmp_refine_subdir], check=False)` — `shutil.rmtree` raises `OSError: ENOTEMPTY` on NFS after unlinking files (NFS directory-cache race); `rm -rf` is more robust

**SLURM routing:** `Untangler.refine_shell_file` is set to `untangler/Refinement/Refine_slurm.sh`, which submits each phenix.refine call to the `refmac` partition via `sbatch --parsable`, polls until the job exits, and cancels the SLURM job via `trap "scancel $SLURM_JOB_ID" EXIT` if the wrapper is killed (e.g. by Python's subprocess timeout). This prevents zombie jobs from piling up in the refmac queue.

**Skip cross-conformer subprocess calls:** set BOTH `NonbondConstraint: 0` AND `ClashConstraint: 0` in `weight_factors`. Setting only one still runs both cross-conformer scripts (see `skip_nonbonds` logic in ConstraintsHandler.py).

**Performance note:** `GenerateHoltonData.sh` → `untangle_score_weighted.csh` → `phenix.molprobity` runs single-threaded on the full model every ILP loop. With 22 conformers (~10,000 atoms) this is hours per call; the 3-conformer model (~955 atoms) is fast.

**For Untangler to make swaps**, the input must be deliberately tangled (conformer labels not yet optimized). A well-refined structure scores 0 high-tension connections each ILP loop and makes no moves (`"moves": {}` in every `xLO-toFlip_*.json`). Score improvements in that case come entirely from the phenix.refine cycles Untangler runs between ILP iterations, not from any relabelling.

**Completed run on `conf3norm_fitGT48.pdb`** (5 loops, `--altloc-subset-size 3`):
- No altloc swaps in any loop — input was already optimally labelled
- Rfree improved 18.62 → 18.51% purely from refinement; wE 87.7 → 84.0
- Final output: `untangler/output/conf3norm_fitGT48_loopEnd4.pdb` (same conformer assignments, slightly better geometry)

---

## Known Gotchas

- **pack.py target**: Y.npy stores `znorm(truth - fc)` (the difference map), not `znorm(truth)`. S.npy stores `log(std(truth - fc))`. Matches `ElectronDensityDataset` exactly. pack.py never reads `metadata.json`.
- **CCP4 map header size**: gemmi writes 1344-byte headers for P 21 21 21 (1024 standard + 320 symmetry records). `_load_map` uses `offset = filesize - 4*n` so it's header-size agnostic and always correct.
- **Map grid vs metadata grid_shape**: gemmi's `transform_f_phi_to_map` rounds up to FFT-friendly numbers (e.g. 144×128×96), not the `round(cell/dmin*3)` estimate stored in metadata. Read grid from map header, not metadata.
- **Full unit cell in maps**: `transform_f_phi_to_map` returns the full P1 unit cell regardless of space group. For P 21 21 21, the network sees 4 ASU copies of the protein.
- **Cross-Patterson spikiness in P 21 21 21**: the full-cell map has 4 ASU copies → cross-Patterson accumulates Harker cross-terms → much spikier than P1. May cause extreme X.npy values and training instability.
- **Heteroscedastic NLL overconfidence collapse**: with batch_size=1, the model quickly learns to set log_var→-3 (clamp floor) on training samples. Val loss explodes when it's wrong with high confidence. Fix: use `--accum-steps 8` to simulate larger batch.
- **--vary-flood overrides --n-flood**: when `--vary-flood` is set, n_flood is drawn randomly from log-uniform [FLOOD_NF_MIN=700, FLOOD_NF_MAX=4000] ignoring --n-flood. Use `--n-flood N --flood-occ O` without `--vary-flood` for deterministic flood parameters.
- **--flood-avoid-fullocc is a no-op**: accepted by argparse and passed through but never used in `_generate_flood_waters` (which always avoids all existing atoms).
- **SLURM bash arrays**: bash array indexing `${arr[$ID]}` fails silently on compute nodes. Use Python to decode task ID: `N=$(python3 -c "print([200,500,1000][$SLURM_ARRAY_TASK_ID])")`.
- **Data generation directories**: keep ≤1000 samples per directory to avoid Lustre metadata slowdown.
- **DataParallel + pin_memory deadlock**: `pin_memory=True` with DataParallel and forked DataLoader workers causes a hang. `train.py` disables pin_memory when `n_gpus > 1`.
- **Einsteinium module loading**: `module load --force ml/pytorch/2.3.1-py3.11.7-mf` requires `Core` in MODULEPATH and the `--force` flag; CUDA/cuDNN warnings on login nodes are harmless.
- **refmac cwd vs relative paths**: always resolve XYZIN and HKLIN to absolute paths; refmac runs with `cwd=tmpdir` so relative paths fail.
- **cavenv SYMM**: must be integer space group number (e.g. 19 for P 21 21 21), not HM string. Use `gemmi.find_spacegroup_by_name(sg).number`.
- **P2₁2₁2₁ map global peak in symmetry copies**: `transform_f_phi_to_map` returns the full unit cell (4 ASU copies). The global maximum is often in a non-ASU region. When searching for density near protein atoms, sample the map at known atom positions rather than finding the global peak and searching nearby.
- **shutil.rmtree ENOTEMPTY on NFS**: Python's `shutil.rmtree` calls `os.rmdir` after unlinking all files; on NFS the server may still report the directory non-empty due to caching, raising `OSError: [Errno 39]`. Use `subprocess.run(['rm', '-rf', path])` instead for cleanup of refinement tmp dirs.
- **Flood water occupancy**: `flood_occ = FLOOD_LINE_K / sqrt(n_flood)` is always positive. The `--flood-occ` CLI flag is unchecked — passing a negative value would propagate to gemmi and write negative occupancies. Normal code paths (including `--vary-flood`) can never produce negative occ.
- **pack.py zero-fills failed samples**: `open_memmap` with `mode='w+'` initialises to zeros. If `process_sample` raises, that row stays zero — no error propagates to the caller. Check that all rows are non-zero after packing: `(np.abs(X).reshape(N,-1).max(1) == 0).sum()` gives the failure count. Failed samples must be regenerated and repacked; the safest fix is to repack with `--force` once all map files are present.
- **data_protein_v2_s0 corruption**: 962 sample dirs but only 443 have non-zero rows in X.npy (519 zero-filled). Root cause: map files were still being flushed (or pack.py ran before generation completed). The 443 valid samples yield only ~354 train / 89 val — too few to avoid overfitting a 5.84M-param network. Regenerate and repack.
- **generate_sample() local multi-sample path bug (fixed 2026-05-15)**: the `--nsamples N` code path (without `--submit`/`--sample-id`) passed `shift_scale` in the `vary_flood` positional slot, then also passed `vary_flood` as a keyword → `TypeError: multiple values for argument 'vary_flood'`. The SLURM array path (`--sample-id`) was unaffected. Fixed by using a `_kw` dict for all keyword args.
- **Space group domain gap for CNN inference on real 1AHO data**: Training uses synthetic P1 40×40×40 Å cells (one molecule, no symmetry). 1AHO is P2₁2₁2₁ with 4 ASU copies per unit cell. Applying the trained CNN to real 1AHO maps means each 60×60×60 patch sees portions of 2–3 symmetry-related molecules simultaneously — a pattern the network was never trained on. The cross-Patterson channel implicitly encodes inter-ASU vectors but the CNN has no way to exploit that. Expect degraded performance if ever applied to non-P1 experimental data.
- **`rfactor.py --pred` expects the TOTAL map (pred + fc), not the diff**: `infer.py` writes two maps: `<output>` is the predicted *difference*; `predicted.map` (alongside it) is the predicted *total*. Passing the diff to `rfactor.py --pred` gives nonsense (k≈6, R≈0.5) because rfactor.py FFTs `--pred` directly and treats |F| as the total amplitude. Always pass `predicted.map` (or anything that is pred + fc).
- **Real-1aho MTZ column labels**: `1aho_test/refmacout_minRfree.mtz` and `1aho/refme*.mtz` use Fo column **`FP`**, not the default `F`. Pass `--fo-label FP` to `rfactor.py` (and `fo_label='FP'` to `eval_1aho.setup_1aho_eval`). Forgetting this gives 0 matched reflections and an `IndexError` deep in `_scale_kb`.
- **eval_1aho.py Miller-index axis convention**: CCP4 maps load with shape (NS, NR, NC) = (Z, Y, X) for the standard MAPC=1/MAPR=2/MAPS=3 axis order. Miller H/K/L correspond to X, Y, Z frequencies. Index `ft[L % nz, K % ny, H % nx]`, **not** `ft[H, K, L]`. The wrong order gives Rfree ≈ 0.67 (uncorrelated noise) instead of ~0.12.
- **Stale `predicted.map` from a failed inference**: `infer.py` writes `predicted.map` next to `<output>`. If `model.load_state_dict()` raises (architecture mismatch), the prior `predicted.map` from a different checkpoint stays on disk. A subsequent `rfactor.py --pred predicted.map` will silently score that stale map. Either delete `predicted.map` between runs or check infer.py's exit status.
- **`best.pt` `best_val` field is stale by one update**: in `train.py`, the checkpoint dict is constructed *before* `best_val` is updated when a new best is found, so `best.pt`'s stored `best_val` reflects the *previous* best, not the one being saved. Model weights are correct; the metadata field is off. Trust the printed log line, not the field.
- **Loading legacy FNO+BN checkpoints (100 keys) into current model.py (44 keys)**: current `_ConvBlock` has no BN. Legacy checkpoints used `Conv3d(..., bias=False) + BatchNorm3d`. Re-adding BN without setting `bias=False` produces the wrong key set (the extra `.bias` keys appear and the BN weights miss). See "Loading legacy FNO+BN checkpoints" under Training Status.
- **DDP + `find_unused_parameters=True` is required**: `scale_head` (params 30–31, the Linear weight/bias) has no gradient under MSE loss because `pred_log_scale` is never used in the loss. Without the flag, DDP hangs waiting for an all-reduce that never fires. If a prior run printed "did not find any unused parameters" that was a false positive — removing the flag crashes training (confirmed slurm-26602773).
- **torchrun port collision when two DDP jobs land on the same node**: torchrun's default rendezvous port is `29500`. If SLURM places a second multi-GPU job on a node already running another torchrun-launched job, both try to bind the same port → `Address already in use` on the second. Pass `--master-port=<unique>` to torchrun (e.g. 29550 / 29551) for any concurrent runs, or use `--master-port=0` to pick freely. `train.csh`/`train1.csh` currently hard-code the default port — fix when running multiple jobs on the same node.
- **Voltron vs Lawrencium GPU comparisons need GPU-count normalisation**: Voltron = TITAN V (12 GB Volta, 2017), Lawrencium es1 = A40 (48 GB Ampere, 2020). A40 is ~1.7× faster per card. Wall-clock per epoch is therefore misleading across clusters: 1× A40 ≈ 4× TITAN V for the same epoch time. Use s-per-sample-per-GPU = (wall × n_gpus) / 1600 to compare.
- **train.py strict-load asymmetry across clusters**: Voltron `train.py` does `load_state_dict(sd)` (strict=True), Lawrencium does `load_state_dict(sd, strict=False)`. The latter silently accepts mismatched key sets — useful for warm-starting across architecture changes (e.g. loading a FNO+BN checkpoint into the no-BN model). Backport if you want this on Voltron too.
