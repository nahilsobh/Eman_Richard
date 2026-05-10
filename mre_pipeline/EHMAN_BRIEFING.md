# MRE Stiffness Reconstruction with a Learned Inverse Operator
**Phase 0 briefing — May 6, 2026**

## What we are testing
Whether a trained function approximator can replace direct Helmholtz inversion as the displacement→stiffness step in MRE — with the goal of being **more robust to noise** and **less sensitive to boundary effects** than analytical inversion (MDEV, LFE, direct inversion).

## What we built (Phase 0)
A complete simulation-and-training pipeline:

1. **Forward simulator** — 2D finite-difference solver for the same time-harmonic shear wave equation used in MRE: ∇·(G* ∇u) + ρω²u = 0, with G* = G(1 + iξ) and ξ = 0.05 damping.
2. **Phantom generator** — 50,000 cases with random elliptical inclusions:
   - Background: 800–3,000 Pa (soft tissue range)
   - Inclusions: 5,000–45,000 Pa (lesion range)
   - Realistic complex Gaussian noise added at 15–30 dB SNR
3. **Two driving frequencies per case** — 60 Hz and 120 Hz (multi-frequency, same principle as MDEV).
4. **Learned inverse** — a function approximator trained on 45,000 cases, validated on 5,000 unseen cases.

## Why this is methodologically clean
- Ground-truth G is **exact** — we generated it. No segmentation, no inversion-of-an-inversion.
- The network sees only the noisy displacement field — same information available in real MRE.
- Inputs are physically realistic: complex u at two frequencies, with SNR matching real acquisitions.

## What we know so far (data validated)

| Check | Result |
|-------|--------|
| 50,000 paired samples generated | ✓ |
| No NaN/Inf, no degenerate cases | ✓ |
| Stiffness diversity (per-sample G std = 4,569 Pa) | ✓ |
| Re/Im channels independent (corr = −0.02) | ✓ |
| Wave physics visually correct at 60 + 120 Hz | ✓ |

Training in progress on NCSA Delta (1× A100 GPU, ~30 min). Results expected today.

## What Phase 1 measures
Once trained, the model is run **zero-shot on BioQIC phantom displacement fields** — no fine-tuning. The gap between Phase 0 validation accuracy and BioQIC accuracy is the **sim-to-real gap** — the headline result for the methods paper.

## The ask
We need raw displacement volumes from the BioQIC phantom acquisitions — **not stiffness maps**, since we generate our own predictions. The scientific question is *how far off are we before any fine-tuning?* — a measurement, not a fitting exercise.

## Anticipated questions

| Question | Answer |
|----------|--------|
| How does it "know" what stiffness is? | It doesn't reason about it. It learns the statistical mapping from 50k examples. Same idea as a radiologist learning to recognize patterns. |
| What if the inclusion isn't in the training distribution? | That is exactly what Phase 1 tests. BioQIC geometry is different. |
| Why not just use direct inversion? | Not replacing it — testing whether the learned model is more robust to noise and missing data. Direct inversion remains the baseline. |
| What's the failure mode? | Domain gap. If real MRE has noise structure our simulation doesn't capture, accuracy drops. That gap is the publishable measurement. |

---

**Figures to show:**
- `data/spot_check.png` — 5 random training samples (wave + stiffness)
- `data/sample_viz.png` — single sample with all 4 input channels + ground truth
