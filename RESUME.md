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
| Dual-head TSM-FNO (Nature draft) | `tsm_fno/` | 30/30 tests pass (was 25 before open-top BC). Smoke: `pytest tests/ -v` (~3 s). |
| Nature manuscript | `paper/main.tex` → `main.pdf` | Built cleanly May 25 (tectonic). 1162 lines. |
| Last thing sent to Mayo | `paper/Reply_to_Eman_at_Mayo_Query.pdf` | May 25, 2026 |

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

`main`, clean, up to date with `origin/main`. Last commit: `63ba5f8` (2026-09-06, `tsm_fno: add open-top (Neumann) BC and re-run balloon demo`).
