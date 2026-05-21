# Figure 10 — 2D FNO/SIREN on BBIR UDel

First real-data supervised training in the foundation-model pipeline.

## Inputs

- `/projects/bfid/sobh/data/bbir_udel_2d_slices.h5` — 17,535 axial slices
  from 82 subjects × 3 frequencies (30/50/70 Hz), produced by
  `bbir_ingest.py` from `/projects/bfid/sobh/BBIR/U01_UDEL_v5a-All`.
- 2D operators (`fno2d_brain.py`):
  - `FNO2dBrain(width=48, modes=16, n_blocks=4)` — 4.7 M params
  - `SIREN2dBrain(width=48, kernel=7, n_blocks=4)` — ~600 k params

## Why this is not done on CPU

Per-batch forward+backward for the 2D operators at 80×80 resolution
takes several seconds on 16 CPU threads, even at small widths. A single
epoch over 1,500 preloaded slices at batch 16 takes 3–4 minutes; a full
8-epoch fine-tune on the full 14k-slice training split would take
hours. This is a GPU job.

## How to run on Delta

The conda env name is `mri_mrf_pytorch_env` (already on Delta). Example
SLURM submission:

```bash
#!/bin/bash
#SBATCH --job-name=fig10
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bfid-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig10_%j.out

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate mri_mrf_pytorch_env
cd /u/sobh/Eman_Richard/paper/figures

python figure10_bbir_train.py \
    --arch fno --width 48 --modes 16 --n_blocks 4 \
    --epochs 30 --batch 32 --n_train 14000
```

Expected wall clock: ~30 min on one A100 for the FNO; ~15 min for SIREN.

## What the code paths are

| File | Purpose |
|---|---|
| `bbir_ingest.py` | NIfTI → HDF5. Run once, reused across experiments. |
| `bbir_inspect.py` | Sanity-check the HDF5 (sample slices per frequency). |
| `bbir_dataset.py` | `BBIRSliceDataset` (PyTorch), subject-wise split helper, masked relative-L² loss. |
| `fno2d_brain.py`  | `FNO2dBrain`, `SIREN2dBrain`, `helmholtz_residual_2d`. |
| `figure10_bbir_train.py` | Training loop: subject-disjoint split, in-memory preload, masked RL² supervised loss against NLI labels, val+test eval, prediction-grid figure. |

## Status

- ✅ HDF5 produced and verified (308 MB, 82 subjects, 17,535 slices, 6 acquisitions skipped).
- ✅ Dataset + architectures + training script tested end-to-end on CPU
  (preload + forward + backward verified; only the wall clock prevents
  full epochs from finishing).
- ⏳ GPU run pending — submit the SLURM script above to actually
  produce `fig10_bbir_fno2d.pt`, `fig10_bbir_history.json`,
  `fig10_bbir_predict.png`, and the masked-RL² test number.

## Next experiments (after Fig 10 GPU run)

1. **Self-supervised fine-tune on WUSTL.** WUSTL has displacement but no
   NLI labels (`/projects/bfid/sobh/BBIR/U01_WUSTL_v5a-All`, 100
   subjects, 5 frequencies). Start from the Fig 10 checkpoint, use the
   `helmholtz_residual_2d` loss only (interior pixels, BC-agnostic), no
   ground truth required. Tests whether SSL on real unlabelled data
   closes the synthetic-to-real gap further.
2. **Multi-frequency conditioning.** Add the driver frequency as a
   third input channel (Hz/100). Train across 30/50/70 Hz UDel + 20/30/
   50/70/90 Hz WUSTL in one model.
3. **Strain head on HJF.** HJF has full strain tensors (Exx, …, Ezz).
   Add a strain output head and pretrain on synthetic + HJF strain
   labels; transfer back to UDel for joint G + strain inversion.
