# MRE FEM Pipeline

Generates 50,000 paired synthetic MRE training samples (complex displacement field u → shear modulus map G) using a 2D finite-difference Helmholtz solver, distributed across a SLURM cluster (NCSA Delta). Trains a Fourier Neural Operator (FNO) for the inverse mapping, with a Direct Inversion (DI) baseline for head-to-head comparison.

## Pipeline versions

| Version | Frequencies | Excitation | Damping ξ | Stiffness range | Output |
|---|---|---|---|---|---|
| **v2** | 60 Hz only (2 channels) | 1–10 random source patches on boundary | random ∈ [0.02, 0.20] | [0.8, 45] kPa | `data/mre_v2_50000.h5` |
| **v3** *(current)* | 60 Hz only (2 channels) | 1–10 random source patches on boundary | random ∈ [0.005, 0.70] | [0.5, 15] kPa, piecewise-smooth background | `data/mre_v3_50000.h5` |

v3 fully aligns with Scott/Murphy 2020 (ILI): stiffness range [0.5, 15] kPa, damping spanning the full ILI range, and piecewise-smooth heterogeneous backgrounds (vs v2's uniform backgrounds).

## Setup

```bash
conda env create -f environment.yml
conda activate mre_pipeline
```

## Quick test

```bash
pytest tests/ -v
```

All 5 tests should pass in under 30 seconds.

## Running on Delta

### 1. Set your allocation account

Edit `slurm/gen_array.sbatch`, `slurm/merge.sbatch`, and `slurm/train.sbatch` to set `--account`.

### 2. Generate the synthetic dataset

```bash
bash scripts/submit_all.sh
```

100-task SLURM array followed by a merge job. Produces `data/mre_v3_50000.h5` (~2.3 GB).

### 3. Train the FNO

```bash
sbatch slurm/train.sbatch
```

Trains for 100 epochs on a single A100 (~25 minutes). Saves `runs/phase0_v3/best.pt` plus periodic checkpoints at epochs 25/50/75/100.

### 4. Run head-to-head evaluation

```bash
python scripts/eval_compare.py 2>&1 | tee runs/phase0_v3/eval_results.txt
```

Compares FNO vs Direct Inversion (DI) on 5,000 held-out validation samples. Reports mean RL², mean SSIM, and Pearson R at the inclusion-centre voxel — the metric directly comparable to Scott/Murphy 2020 Table 1.

### 5. Visualise predictions

```bash
python scripts/plot_gt_vs_pred.py    # 6-sample GT/prediction grid with units
```

## Phase 0 v3 results

| Metric | FNO v3 | DI baseline | ILI (published, Scott/Murphy 2020) |
|---|---|---|---|
| Mean RL² | **0.221** | 0.462 | — |
| Mean SSIM | **0.644** | 0.235 | — |
| Pearson R (inclusion centre) | **0.840** | 0.626 | 0.940 |

FNO outperforms DI across all metrics. v3's ILI-aligned distribution raises the DI baseline R from 0.067 (v2) to 0.626 — now in the same ballpark as ILI's published DI = 0.685, confirming v3 is a comparable benchmark. ILI's R = 0.940 uses 5 M training examples and a 3D footprint-optimised CNN.

## Output file

`data/mre_v3_50000.h5` contains:

| Dataset | Shape | Description |
|---|---|---|
| `/X` | (50000, 2, 64, 64) float32 | Re(u), Im(u) at 60 Hz |
| `/Y` | (50000, 64, 64) float32 | Shear modulus G [Pa] |
| `/damping` | (50000,) float32 | Per-sample damping ξ |
| `/n_src` | (50000,) int32 | Per-sample source-patch count (1–10) |
| `/snr_db` | (50000,) float32 | Per-sample SNR in dB |

Attributes record physical parameters: freq, dx, rho, damping range, G ranges, channel labels.

## Repository layout

```
mre_pipeline/
├── src/
│   ├── fem_solver.py          # 2D Helmholtz FD solver (multi-source capable)
│   ├── phantom.py             # Random elliptical-inclusion stiffness maps
│   ├── dataset.py             # Single-frequency 2-channel pair generator
│   ├── fno_model.py           # 2D Fourier Neural Operator
│   ├── direct_inversion.py    # Algebraic DI baseline (Romano-style)
│   └── train_fno.py           # Training loop with cosine LR + Helmholtz reg
├── scripts/
│   ├── generate_chunk.py      # Per-task array entry point
│   ├── merge_chunks.py        # Concatenates 100 chunks into one HDF5
│   ├── check_dataset.py       # Integrity assertions on the dataset
│   ├── eval_compare.py        # FNO vs DI head-to-head, ILI-comparable metrics
│   ├── plot_gt_vs_pred.py     # Per-sample GT/prediction figure with units
│   ├── problem_setup_figure.py# 4-panel BC + material + displacement schematic
│   └── build_briefing_pptx.py # Generates EHMAN_BRIEFING.pptx
├── slurm/
│   ├── gen_array.sbatch       # 100-task generation array
│   ├── merge.sbatch           # Merges chunks (depends on gen)
│   └── train.sbatch           # GPU training on Delta
└── tests/
    └── test_solver.py
```
