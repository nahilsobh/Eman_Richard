# Resume — Ehman Visit (May/June 2026)

**Last verified:** 2026-09-06
**Session UUID:** `bfa97a0c-5c81-4fde-9347-03cc1bf354a4.jsonl`
(under `~/.claude/projects/-u-sobh-Eman-Richard/`)

## To resume

```bash
cd /u/sobh/Eman_Richard
claude
# then /resume  →  pick the entry whose first prompt is "status so far"
```

## State snapshot (all intact as of 2026-09-06)

| Thread | Location | Status |
|---|---|---|
| Phase-0 briefing + FD-Helmholtz FNO | `mre_pipeline/` | 5/5 tests pass. `runs/phase0_v3/best.pt` = epoch 47, val_rl²=0.221, val_ssim=0.643. Slice-by-slice R = 0.9653 (ILI ref 0.940). |
| Dual-head TSM-FNO (Nature draft) | `tsm_fno/` | 25/25 tests pass. Smoke: `pytest tests/ -v` (~3.5 s). |
| Nature manuscript | `paper/main.tex` → `main.pdf` | Built cleanly May 25 (tectonic). 1162 lines. |
| Last thing sent to Mayo | `paper/Reply_to_Eman_at_Mayo_Query.pdf` | May 25, 2026 |

## Environment

Declared envs `mre_pipeline` / `tsm_fno` don't exist locally — use **`mri_mrf_pytorch_env`** (has torch 2.5.1+cu121, scipy, h5py, skimage, sklearn, pytest, matplotlib).

```bash
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate mri_mrf_pytorch_env
```

`tectonic` not on PATH; `pdflatex` at `~/.local/bin/pdflatex` if the paper needs a rebuild.

## Git

`main`, clean, up to date with `origin/main`. Last commit: `1f1e7d1` (2026-08-29, `gitignore: ignore *.tar.gz`).
