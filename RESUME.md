# Resume — Ehman Visit (May/June 2026)

**Last verified:** 2026-09-06
**Session UUID:** `bfa97a0c-5c81-4fde-9347-03cc1bf354a4.jsonl`
(under `~/.claude/projects/-u-sobh-Eman-Richard/`)

## To resume

```bash
cd /u/sobh/Eman_Richard
claude
# then /resume  →  pick the entry titled "Resume Ehman visit"
```

## State snapshot (all intact as of 2026-09-06)

| Thread | Location | Status |
|---|---|---|
| Phase-0 briefing + FD-Helmholtz FNO | `mre_pipeline/` | 5/5 tests pass. `runs/phase0_v3/best.pt` = epoch 47, val_rl²=0.221, val_ssim=0.643. Slice-by-slice R = 0.9653 (ILI ref 0.940). |
| Dual-head TSM-FNO (Nature draft, 2D) | `tsm_fno/` | **58/58 tests pass** (32 2D + 12 3D + 7 SLS-viscoelastic + 8 anisotropy+KV-viscosity). Smoke: `pytest tests/ -v` (~13 s). |
| Nature manuscript | `paper/main.tex` → `main.pdf` | Built cleanly May 25 (tectonic). 1162 lines. |
| Last thing sent to Mayo | `paper/Reply_to_Eman_at_Mayo_Query.pdf` | May 25, 2026 |

## 3D solver + spherical balloon demo (added 2026-09-06, commit `1b2bb66`)

3D forward-modeling companion to the 2D open-top work. Files:

- `tsm_fno/src/solver/helmholtz_fd_3d.py` — 7-point stencil, `top_free`
  ghost-mirror on the i=0 slab, `bottom_plate_driver_sources_3d` (disk
  driver on i=N-1), `direct_inversion_3d` for a training-free G baseline.
- `tsm_fno/src/phantom/geometry_3d.py` — `SphericalBalloon` with
  Lamé pre-stress `(3/2) p (a/r)^3` outside, uniform p inside.
- `tsm_fno/scripts/paper_phantom_demo_3d.py` — 32³ grid, dx=3 mm,
  60 Hz, 6-state inflation. ~90 s wall clock.
- `tsm_fno/tests/test_helmholtz_3d.py` — 8 tests (BC regression,
  top-slab stencil residual → machine precision, driver geometry,
  spherical volume, DI recovery of uniform G).

**Headline result (`results/paper_demo_3d/summary.txt`):**

| Pressure | 2D FNO G_ring | 3D DI G_ring | Ratio |
|---|---|---|---|
| 0 Pa | 3,323 | **2,517** (matches G_bg=2500 to 0.7%) | 0.76 |
| 1 kPa | 2,846 | 7,437 | 2.6× |
| 3 kPa | 8,421 | 20,105 | 2.4× |
| 7 kPa | 17,311 | 44,608 | 2.6× |

The 2D FNO **systematically under-predicts perilesional stiffening by
~2.5×** vs 3D physics. That's the sim-dimensionality gap — a real
finding for the paper Methods. Above 5 kPa the balloon (r=39 mm) fills
most of the 9.6 cm FOV, so those numbers are approximate.

To re-run:
```bash
cd /u/sobh/Eman_Richard/tsm_fno
python scripts/paper_phantom_demo_3d.py             # inflation only (~90 s)
python scripts/paper_phantom_demo_3d.py --deflation # inflation + deflation (~110 s)
```

**Inflation + deflation:** running with `--deflation` retraces peak→baseline
after the inflation branch. The forward model is **memoryless** (linear or
hyperelastic, neither has viscoelastic memory), so deflation numbers are
**bit-exact** to inflation numbers at matched pressure — reversibility
check reports max |ΔG_ring| = 0.00 Pa across all 5 shared pressures.
`hysteresis_curve.png` is a Yin-style Figure 6 lookalike with the two
curves overlaid. Real gel shows slight hysteresis from viscoelastic
creep between scan pauses; modeling that needs a Kelvin-Voigt G*(ω)
with a time-domain memory kernel — out of scope for this demo.

**Hyperelastic strain-stiffening** (added 2026-09-06): `--stiffening-exponent m`
switches the acoustoelastic law from linear
`G_eff = G_base + A·Δσ` (m=1, Phantom 1 flavor) to power-law
`G_eff = G_base · (1 + A·Δσ/G_base)^m` for m > 1 (Phantom 2 / cellulose-
reinforced flavor). At m=2, G_ring at 1 kPa is 68 % higher than the
linear case (12.5 vs 7.4 kPa) and rises super-linearly until the
G_max_pa=500 kPa clip saturates it around 3–7 kPa. Results live in
`results/paper_demo_3d_hyper2/`; the linear baseline stays in
`results/paper_demo_3d/`.

**A_coeff calibrated to Yin Fig 6** (2026-09-06): the paper's training-time
range `A ∈ [2, 8]` was chosen for phantom diversity, not gel realism.
A quick single-state sweep (peak inflation, 250 mL) showed A ≈ 0.20 gives
G_ring = 4.28 kPa matching Yin Phantom 1's 4.4 kPa; A ≈ 0.30 gives 5.07
matching Phantom 2's 5.15. **A_COEFF default in both 3D demo scripts is
now 0.20** (physically honest gel value); paper's [2, 8] range remains
valid for the 2D FNO's training distribution.

Post-calibration match to Yin Figure 6:

| Volume | Yin P1 TSM | Ours (A=0.20, m=1) | Yin P2 TSM | Ours (A=0.20, m=2) |
|---|---|---|---|---|
| 50 mL | 3.5 kPa | 2.5 | 3.5 kPa | 2.5 |
| 100 mL | 3.8 | 2.7 | 3.9 | 2.9 |
| 200 mL | 4.2 | 3.2 | 4.9 | 4.1 |
| 250 mL (peak, m=2) | 4.4 | 4.2 ✓ | 5.15 | 6.85 (+33%) |
| **250 mL (peak, m=1.5 rerun)** | — | — | **5.15** | **5.43 ✓ (+5%)** |

**m=1.5 for Phantom 2 lands the peak to 5 %**, same match quality as
Phantom 1. Results in `results/paper_demo_3d_hyper1.5/`. Baseline offset
(ours 2.5 vs Yin 3.5) is because Yin's TSM has a MIP-upward-bias vs true
G_bg=2.7 kPa; Yin explicitly notes this in the paper Discussion.

**Amplitude-thresholded TSM MIP** (added 2026-09-06): `--amp-threshold 0.15`
in `paper_phantom_demo_3d_tsm.py` matches Yin's semi-automatic gate
(mask G_DI where |u| < 15 % of per-direction peak before combining).
Marginal improvement — TSM/conv ratio 1.15 → 1.29 (Yin: 1.57). Remaining
gap is our 6 face-normal directions vs Yin's 20-direction 3D DF set;
adding more directions is the next lever (costs 3–4× more compute per
state). As a result μ_conv also now recovers a physical value:
2.4 kPa (was 300 Pa pre-calibration — the DI amplitude-thresholding
artifact self-heals once A is realistic).

**Anisotropic TSM pipeline** (added 2026-09-06): the biggest addition —
this is what Yin's TSM signal actually measures. `stress_tensor_sphere`
returns the full Cauchy σ_ij(x) field (radial compression σ_rr = -p(a/r)³,
tangential tension σ_θθ = σ_φφ = +½p(a/r)³). `effective_G_for_direction`
computes direction-dependent apparent stiffness
`G_eff(k̂, x) = G_base · (1 + A·k̂·σ·k̂/G_base)^m`. Radial-propagation
directions soften the ring (compressive), tangential-propagation
directions stiffen it (tensile).

`scripts/paper_phantom_demo_3d_tsm.py` sweeps 6 face-normal directions,
runs a per-direction DI, then combines two ways:
- μ_conv = amplitude-weighted mean (Yin's conventional inversion)
- μ_TSM  = voxelwise max across directions (Yin's TSM MIP)

At p = 3 kPa peak (from `results/paper_demo_3d_tsm/summary.txt`):

| Field | Ring mean [Pa] |
|---|---|
| G_true (isotropic acoustoelastic) | 21,408 |
| μ_conv (amp-weighted mean of 6 dirs) | 300 |
| μ_TSM (MIP of 6 dirs) | **9,458** |
| **TSM / conv ratio** | **31.6×** |

The ring exists ONLY in μ_TSM — μ_conv averages the tangential-stiffening
signal away. That 30+× amplification is Yin's signature; μ_conv going
to 300 Pa (well below G_bg=2500) is a DI amplitude-thresholding artifact
we haven't modeled yet.

**Frequency-dependent Kelvin-Voigt damping** (added 2026-09-06): the solver
now takes an optional `viscosity` parameter η [Pa·s], adding
`+iωη` to G* — so damping grows linearly with ω, matching real gel.
At η = 2 Pa·s the field attenuation across the cube is: 30 Hz → 0.233,
60 Hz → 0.006, 80 Hz → 0.000 (essentially killed by 80 Hz).

**Viscoelastic SLS hysteresis** (added 2026-09-06): `--viscoelastic-tau τ`
(seconds) + `--scan-pause dt` (default 45 s) applies a standard-linear-solid
relaxation to the applied-pressure schedule. The effective pressure driving
the pre-stress field follows
`p_eff[k] = p_new + (p_eff[k-1] − p_new)·exp(-Δt/τ)`, so on inflation the
gel lags below applied and on deflation it lags above applied — producing
the two separated curves Yin reports.

`src/phantom/viscoelastic.py` holds the analytical SLS closed form
(no ODE integration). At τ=60 s and pause=45 s (linear m=1):

| Applied | Inflation G_ring | Deflation G_ring | Δ (kPa) |
|---|---|---|---|
| 0 | 2.5 | 6.2 | +3.7 (gel not fully relaxed) |
| 1 kPa | 5.2 | 10.6 | +5.4 |
| 2 kPa | 9.2 | 15.2 | +6.0 |
| 3 kPa | 15.0 | 23.7 | **+8.7** (max hysteresis) |
| 5 kPa | 18.8 | 23.2 | +4.4 |

Results: `results/paper_demo_3d_visc60/` (linear + visc),
`results/paper_demo_3d_hyper2_visc60/` (hyperelastic + visc).
Note the wave-scale damping ξ handles frequency-domain losses
separately; this SLS handles quasi-static creep between scans.

**Next step to consider:** train a `FNO_TSM_3D` on a 3D dataset (5k
volumes). Needs GPU time on Delta; scaffold in `helmholtz_fd_3d.py`
+ `geometry_3d.py` is ready.

## Bottom-plate driver (added 2026-09-06, commit `cf0a6ab`)

Second 2D test axis: coherent piston-plate source on the bottom edge,
switchable via `paper_phantom_demo.py --driver bottom`. Combined with
`--top-free` gives the full 2×2 (BC × driver) grid; results split
across `results/paper_demo{,_freetop,_bottomdrive,_freetop_bottomdrive}/`.

**Findings:** driver location dominates BC as a source of variation.
Baseline G_ring shifts +700 Pa (27 %) when switching random→bottom
driver; free-top only nudges it by 3–8 %. ε_ring stays within 2 %
across all four combos — a positive OOD-robustness result for the FNO.

## Open-top boundary condition (added 2026-09-06, commit `63ba5f8`)

Yin gelatin-cylinder physics needs a traction-free top, not the clamped
Dirichlet u=0 that was in the original solver. Added a `top_free` flag to
`helmholtz_solve` / `helmholtz_eshelby_solve` / `solve_two_frequencies`
(ghost-mirror Neumann stencil at row 0, non-corner) and a `skip_top`
flag to `random_sources`. `scripts/paper_phantom_demo.py` now accepts
`--top-free`; results split between `results/paper_demo/` (clamped) and
`results/paper_demo_freetop/` (open top).

Comparison at the current N=80, dx=3 mm balloon geometry:

- Perilesional strain: <2 % shift across all six inflation states
- Ring stiffness: +43 to +1,004 Pa systematic drift (free-top slightly stiffer)
- Saturation plateau at ε = 3.0 above 5 kPa: unchanged (training-cap artifact)
- Model was trained on clamped-top → this is an OOD BC test; the fact that
  qualitative behavior survives is itself a robustness result.

New tests in `tsm_fno/tests/test_helmholtz_bc.py` cover: default BC unchanged
(regression guard), soft-reflection sanity (top-row RMS jumps by >10× under
free-top), row-0 stencil residual at machine precision, source-skipping,
and source-override still honored under free-top.

To re-run:
```bash
cd /u/sobh/Eman_Richard/tsm_fno
python scripts/paper_phantom_demo.py            # clamped
python scripts/paper_phantom_demo.py --top-free # open top
```

## Environment

Declared envs `mre_pipeline` / `tsm_fno` don't exist locally — use **`mri_mrf_pytorch_env`** (has torch 2.5.1+cu121, scipy, h5py, skimage, sklearn, pytest, matplotlib).

```bash
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate mri_mrf_pytorch_env
```

`tectonic` not on PATH; `pdflatex` at `~/.local/bin/pdflatex` if the paper needs a rebuild.

## Git

`main`, clean, up to date with `origin/main`. Last commit: `1b2bb66` (2026-09-06, `tsm_fno: 3D FD Helmholtz solver + spherical balloon demo`).
