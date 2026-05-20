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

## In vivo validation plan

Synthetic FEM training establishes the operator; real data closes the credibility gap. Validation proceeds in four phases of increasing domain difficulty.

### Phase A — Physical phantom (smallest sim-to-real gap)

**Goal**: confirm the model transfers from FEM-synthetic displacement to real wave data acquired on identical phantom geometry.

- **Data**: gelatin or agar phantom with a single stiff inclusion, scanned on the 3 T MRE driver at 80 Hz (matched to `FREQ_1`). Acquire 5–10 inclusion sizes / contrasts.
- **Ground truth**: NLI inversion (gold standard) and DI baseline computed on the same wave fields.
- **Success criterion**:
  - `G_pred` Pearson r ≥ 0.85 vs NLI in inclusion ROI
  - `ε_pred ≈ 0` everywhere (static phantom — no expansion)
- **Risk**: pre-processing mismatch (units, normalisation, curl computation). Mitigate by porting the synthetic pipeline's preprocessing exactly.

### Phase B — Open benchmark dataset

**Goal**: stratified comparison against published methods on standardised acquisitions.

- **Data**: Wang et al. 2025 *Scientific Data* MRE benchmark (phantom + healthy liver + healthy brain, 3 T, multiple frequencies, 5 inversion algorithms supplied).
- **Comparators**: TWENN, LFE, DI, MERSA, MICRo.
- **Success criterion**:
  - `G_pred` ranks within ±1 position of TWENN on whole-field RMSE
  - `ε_pred ≈ 0` (all subjects are healthy — no expanding lesions expected)
- **Risk**: frequency mismatch (their data is at 30/50/60 Hz; our model is trained at 80 Hz). Mitigate by retraining on 60 Hz or adding frequency conditioning (see Phase D).

### Phase C — Retrospective active-lesion data

**Goal**: validate the strain head and expansion classifier on the clinical population they were designed for.

- **Data**: retrospective MS-lesion MRE cohort with paired gadolinium-enhancement labels (active vs stable). Yin et al. 2026 TSM cohort if accessible; otherwise an institutional MS+MRE cohort.
- **Ground truth**: gadolinium enhancement as the active-lesion proxy; expert ROI annotation for ring localisation.
- **Success criterion**:
  - Expansion classifier AUC ≥ 0.80 on held-out subjects (subject-wise split)
  - `ε_pred` ring co-localises with gadolinium enhancement on ≥ 70% of active lesions
- **Risk**: synthetic phantom anatomy is too simple. Mitigate via test-time fine-tuning on a small labelled subset, or by retraining with anatomically-realistic backgrounds segmented from real T1.

### Phase D — Multi-frequency, multi-site prospective

**Goal**: production-grade generalisation across scanners, frequencies, and tissue.

- **Architectural additions**:
  - Frequency-conditioning input channel (Hz/100), trained jointly on 30/50/60/80 Hz synthetic data — borrowed from SPADE-oNLI's multi-frequency strategy.
  - Curl-of-displacement input transform to suppress longitudinal-wave contamination at acquisition time.
- **Data**: prospective acquisition across ≥ 2 sites, ≥ 30 patients with suspected active lesions.
- **Success criterion**: AUC ≥ 0.85 retained on the external site (no site-specific retraining).
- **Risk**: scanner-specific noise distributions, motion artefacts. Mitigate by augmenting synthetic training with measured noise profiles per scanner.

### Cross-phase tooling needed

| Tooling | Purpose | Status |
|---|---|---|
| Real-data ingestion script | DICOM/HDF5 → model input tensor with matched preprocessing | not started |
| NLI / DI reference inverters | Per-sample gold-standard for Phases A–B | DI exists in `mre_pipeline`; NLI needed |
| Test-time fine-tuning script | Few-shot adaptation on a labelled subset | not started |
| Frequency-conditioned model variant | Multi-frequency training and inference (Phase D) | not started |
| Wall-clock benchmark vs NLI | Reproduce oNLI-style speedup claim | not started |
