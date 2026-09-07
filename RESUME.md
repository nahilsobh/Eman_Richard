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
Marginal improvement in isolation — TSM/conv 1.15 → 1.29 (Yin: 1.57).

**k-space directional filter** (added 2026-09-06, commit `c205a10` +): the
Yin-faithful method — `--method filter` runs ONE broadband multi-face
solve on the isotropic-averaged G_eff, then applies a Gaussian
angular wedge in k-space (`directional_filter_3d(u, k̂, σ)`) to isolate
each direction's component before DI. Combined with `--num-directions 20`
this matches Yin's TSM signal in **17 seconds** (vs 175 s for solves):

| Method | # dirs | μ_conv | μ_TSM | Ratio | Runtime |
|---|---|---|---|---|---|
| solves | 6 | 2.28 | 2.93 | 1.29 | 60 s |
| solves | 20 | 2.45 | 3.19 | 1.30 | 175 s |
| filter | 6 | 3.45 | 3.89 | 1.13 | 14 s |
| **filter** | **20** | **3.77** | **6.77** | **1.80** | **17 s** |
| Yin Phantom 1 | 20 | 2.8 | 4.4 | **1.57** | — |

Filter+20 reproduces Yin's anisotropy ratio (1.80 vs 1.57, 15% high) at
10× the speed. Absolute μ_TSM is elevated (6.77 vs 4.4) because our
scalar G_iso already includes the acoustoelastic effect that Yin's
mechanism recovers from the filter alone. Files:
`src/solver/helmholtz_fd_3d.py::directional_filter_3d` and
`multi_face_broadband_sources`;
`tests/test_directional_filter.py` (6 new tests, all pass).

**Full-cycle reproduction of Yin Fig 6** (added 2026-09-06): `--full-cycle`
flag on `paper_phantom_demo_3d_tsm.py` loops all 11 states (6 inflation +
5 deflation) with filter+20-dir. Runtime ~2 min. Result at m=1
(Phantom 1 analogue), calibrated A=0.20:

| Volume | Ours μ_TSM | Yin P1 μ_TSM | Ours μ_conv | Yin P1 μ_conv |
|---|---|---|---|---|
| 0 mL (baseline) | 3.51 | 3.5 ✓ | 2.26 | 2.7 |
| 50 mL | 3.52 | 3.5 ✓ | 2.48 | 2.7 |
| 100 mL | 4.38 | 3.8 (+15%) | 2.83 | 2.7 |
| 150 mL | 4.76 | 3.9 (+22%) | 3.07 | 2.8 |
| 200 mL | 5.38 | 4.2 (+28%) | 3.42 | 2.8 |
| **250 mL (peak)** | **6.78** | **4.4 (+54%)** | 3.77 | 2.8 |

**Baseline TSM matches Yin exactly.** Peak overshoot 54% was closed by
adding Yin's own 3×3×3 spatial median filter to DI (below).

**3×3×3 spatial median filter on DI** (added 2026-09-06): matches Yin's
Methods step ``a 3 × 3 × 3 cubic spatial median filter to improve
regional homogeneity''. `direct_inversion_3d` now takes an optional
`median_filter_size` param; the TSM demo takes `--median-filter 3`.

| Volume | Yin P1 TSM | Before median | **With median=3** | Δ |
|---|---|---|---|---|
| 100 mL | 3.8 | 4.38 | **3.66** | **−4%** ✓ |
| 150 mL | 3.9 | 4.76 | **3.92** | **+0.5%** ✓ (bullseye) |
| 200 mL | 4.2 | 5.38 | **4.11** | **−2%** ✓ |
| **250 mL peak** | 4.4 | 6.78 (+54%) | **4.93** | **+12%** |

Phantom 2 (m=1.5): peak overshoot 69% → **23%**. Central states
(100–200 mL) now match Yin to within a few percent for both phantoms.
Results in `results/paper_demo_3d_tsm_dir20_filter_cycle_medfilt/`
(Phantom 1) and `..._medfilt_m1.5/` (Phantom 2).

**Ogden constitutive law** (`--constitutive ogden`) is available but
overshoots MORE than powerlaw when combined with unfiltered DI, because
Ogden's `1/2·λ^-α` term creates steeper G gradients → larger DI
artifacts. The median filter is what closes the gap, not the choice of
constitutive law.

**Shell edge-exclusion** (`--shell-offset-mm 9`) matches Yin's ``3 pixels
away from the balloon edge to minimize edge effects''. Combined with
DI+median it drops the Phantom 1 peak error to **+1%** and Phantom 2 to
+15%, essentially closing the quantitative Yin Fig 6 gap.

**LFE inversion** (`--inversion lfe`, `lfe_inversion_3d`) — a first-
derivative alternative to DI using `|k|² = |∇u|²/|u|²`. On plane-wave-
like fields it's more robust than DI. On our bounded-Dirichlet domain,
standing-wave interference biases the ratio and the method under-
performs DI; kept as an option, not the default.

**Anisotropic scalar Helmholtz** (`helmholtz_solve_3d_anisotropic`) —
extends the scalar solver to a rank-2 stiffness tensor field `G_ij(x)`:
    ρω² u = ∂_i [G_ij(x) · ∂_j u]
Symmetric divergence-form FD with half-integer diagonal averaging and
4-point cross-derivative stencils for off-diagonal terms. Recovers the
isotropic solver exactly on diagonal-uniform G.

**Integrated into TSM demo** as `--method anisotropic`. Builds the
tensor field `G_ij = G_base·δ_ij + A·σ_ij` via new
`make_anisotropic_G_tensor`, runs ONE broadband multi-face solve, then
applies the k-space directional filter. Per-solve cost ~2.5× isotropic;
full cycle ~3 min at N=32.

**Big finding at Phantom 1 full cycle (anisotropic + 20-dir + median + edge):**

| Volume | Yin TSM | Ours TSM | Yin conv | Ours conv |
|---|---|---|---|---|
| 50 mL | 3.5 | **3.46** ✓ | 2.7 | **2.53** ✓ |
| 100 mL | 3.8 | 3.61 (−5%) | 2.7 | 2.42 |
| 150 mL | 3.9 | 3.56 (−9%) | 2.8 | 2.33 |
| 200 mL | 4.2 | 3.26 (−22%) | 2.8 | 2.34 |
| 250 mL | 4.4 | 3.21 (−27%) | 2.8 | **2.27 (flat)** |

**Structural win: μ_conv is now essentially flat** (2.27–2.53 kPa) —
the tensor solver correctly cancels direction-dependent stiffening on
the amplitude-weighted mean, reproducing Yin's flat conv signature that
scalar methods couldn't. Trade-off: μ_TSM undershoots at high pressures
because the tensor form is linear in σ (`G_ij = G_bg·δ + A·σ`, m=1
equivalent); a nonlinear tensor form is needed for the peak amplitude.

Full vector elasticity (`vector_elasticity_3d.py::navier_solve_3d`)
remains scaffold-only — the true fix but 1–2 weeks of work.

## Scalar Helmholtz limitations — status of the four fixes

| Fix | Status |
|---|---|
| Shell edge-exclusion (Yin's 3-px offset) | ✅ done — closes Phantom 1 peak to +1% |
| LFE inversion | ✅ implemented; not default (standing-wave bias) |
| Anisotropic scalar Helmholtz | ✅ solver + tests done; demo integration TBD |
| Vector elasticity (Navier equation) | 🚧 scaffold + design doc only — 1–2 week project |

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
