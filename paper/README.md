# Nature paper draft — TSM-FNO

Manuscript draft positioning the dual-head TSM-FNO as a physics-informed
Fourier neural operator for joint stiffness and perilesional-strain recovery
from MR elastography.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Single-file Nature-format manuscript (Abstract, Main, Methods, Figure legends) |
| `refs.bib` | Bibliography curated from the May 2026 MRE+ML literature survey |
| `figures/` | Display items (placeholders — populate before submission) |

## Build

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Produces `main.pdf`. Requires the `naturemag.bst` style (ships with most
TeXLive distributions); fall back to `unsrtnat` if unavailable by editing
the `\bibliographystyle` line.

## Status

- **Main text**: drafted (~3000 words). Anchored by the 1D Helmholtz +
  transfer-matrix derivation, motivates the strain-output gap, presents the
  architecture and the Phase A phantom result.
- **Methods**: drafted (~3000 words). Includes:
  - 1D Helmholtz forward + transfer matrix + heterogeneous eigenstrain inversion
  - Quasi-static limit showing the perilesional ring requires d ≥ 2
  - 2D Helmholtz + Eshelby form + Lamé pre-stress
  - Spectral convolution / FNO architecture
  - Six-term composite loss
  - Training and Phase A validation protocols
- **Figures**: legends drafted, panels not yet generated. Six display items:
  1. Problem setup + 1D analytical anchor
  2. TSM-FNO architecture schematic
  3. Acoustoelastic phantom physics
  4. Phase A phantom validation panel
  5. Synthetic cohort performance (RL², ROC, A scatter)
  6. Comparison to operator-learning literature
- **Bibliography**: 21 entries, ~75% from primary 2023–2026 literature.

## Outstanding work before submission

| Item | Owner | Notes |
|---|---|---|
| Generate Figure 1c (1D analytical vs FD solver) | code | Script to add: `paper/figures/figure1_1d.py` |
| Generate Figure 2 architecture schematic | manual | TikZ or external diagram |
| Generate Figure 3 acoustoelastic panel | code | Reuses `tsm_fno/scripts/paper_phantom_demo.py` |
| Generate Figure 4 from Phase A run | code | `tsm_fno/results/phantom_phaseA/phantom_panel.png` already produced |
| Generate Figure 5 from held-out cohort | code | Wraps `tsm_fno/scripts/evaluate.py` outputs |
| Generate Figure 6 comparison table | manual | Numbers from oNLI / TWENN / DIME / DI papers |
| Run Phase B–D in vivo validation | data | See in-vivo validation plan in `tsm_fno/README.md` |
| Author list + affiliations | manual | Currently lists Murphy-Ragoza Richard + Sobh; confirm |
| Funding + acknowledgements section | manual | Add before submission |
| Word count audit | manual | Currently main ~3000, Methods ~3000 — within Nature limits |
