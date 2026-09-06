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
| Dual-head TSM-FNO (Nature draft, 2D) | `tsm_fno/` | 32/32 2D tests + 8/8 3D tests pass = **40/40 total**. Smoke: `pytest tests/ -v` (~6 s). |
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
python scripts/paper_phantom_demo_3d.py
```

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
