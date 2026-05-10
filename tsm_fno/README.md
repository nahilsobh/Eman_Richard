# TSM-FNO — dual-head Fourier Neural Operator for Tissue Strain Mapping

Dual-output FNO that predicts both the shear-modulus field `G(x,y)` and the perilesional latent strain `ε_latent(x,y)` from displacement-field MRE inputs. Trained on synthetic phantoms with explicit Lamé pre-stress fields around pressurised inclusions, so the network can distinguish *expanding* lesions (visible stiffening ring) from *static* ones.

## Layout

```
tsm_fno/
├── src/
│   ├── phantom/    geometry.py, acoustoelastic.py
│   ├── solver/     helmholtz_fd.py, hodge.py
│   ├── data/       generate_tsm.py, dataset.py
│   ├── model/      spectral_conv.py, fourier_block.py, fno_tsm.py, losses.py
│   └── eval/       metrics.py, visualize.py
├── scripts/        generate_chunk.py, merge_chunks.py, train.py, evaluate.py, submit_all.sh
├── slurm/          gen_tsm.sbatch, merge_tsm.sbatch, train_tsm.sbatch, eval_tsm.sbatch
├── tests/          test_geometry, test_acoustoelastic, test_hodge, test_fno_tsm, test_losses
└── README.md
```

## Quick test

```bash
python -m pytest tests/ -v          # 25 tests, all pass in ~5 s
python scripts/generate_chunk.py --task_id 0 --chunk_size 10
python scripts/train.py --data_path data/chunks/tsm_chunk_0000.h5 \
    --run_dir runs/smoke --epochs 3 --batch_size 2 --num_workers 0
python scripts/evaluate.py --checkpoint runs/smoke/best.pt \
    --data_path data/chunks/tsm_chunk_0000.h5 --save_dir runs/smoke/results
```

## Full pipeline on Delta

```bash
bash scripts/submit_all.sh
# Submits gen_array → merge → train → eval as a dependency chain.
```

Total wall clock ≈ 1 hour (10 min generation, 40 min training, 5 min evaluation).

## Inputs

`X` is float32 of shape `(4, 80, 80)`:

| Channel | Content |
|---|---|
| 0, 1 | Re/Im of u at 80 Hz |
| 2 | Lamé pre-stress prior Δσ / G_bg (analytical, dimensionless) |
| 3 | Distance from lesion surface, normalised to [0, 1] |

Channels 2 and 3 are the *physics-informed* inputs: they tell the FNO where the lesion is and what the analytical pre-stress field would look like, dramatically reducing the learning problem.

## Outputs

| Output | Shape | Range | Meaning |
|---|---|---|---|
| `G` | (B, 80, 80) | [800, 79000] Pa | shear modulus |
| `ε` | (B, 80, 80) | [0, 3] | latent strain |
| `A` | (B,) | < -2.0 | acoustoelastic constant |

## Pass criteria

- `val_ring_RL² < 0.10` at convergence
- `val_AUC > 0.90` for expanding-vs-control classification
- `A_relative_err < 0.20` averaged across the val set
- 6-panel benchmark figure shows clear ring in `ε_pred` for expanding cases and `ε ≈ 0` for controls

## Phantom physics

```
σ_rr(r) = -p · (a_eq/r)²        (radial, compressive)
σ_θθ(r) = +p · (a_eq/r)²        (circumferential, tensile)
Δσ(r)  = 2 p · (a_eq/r)²        (deviatoric, ≥ 0 in shell)

G_eff(x,y)    = G_bg + A · Δσ(x,y)        with A > 0  (stiffening ring)
ε_latent(x,y) = clip(Δσ / G_bg, 0, ε_max)
```

For control samples (`p = 0`), Δσ ≡ 0 and ε_latent ≡ 0 — the model must produce a near-zero ε output.

## Composition of the 50,000-sample dataset

| Mode | Probability | What it tests |
|---|---|---|
| Expanding | 0.60 | Active ring detection |
| Control (`p = 0`) | 0.20 | Negative-control discrimination |
| High-contrast inclusion (`G_lesion / G_bg > 10`) | 0.10 | Edge-sharpness |
| Near-circular (`a ≈ b`) | 0.10 | Easier geometry baseline |
