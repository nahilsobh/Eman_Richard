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

Use [tectonic](https://tectonic-typesetting.github.io/) — self-contained,
fetches packages on demand, handles bibliography in one command:

```bash
cd paper
tectonic main.tex          # produces main.pdf
```

If using a traditional TeX Live distribution instead:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The Nature bibliography style `naturemag.bst` ships with TeX Live and is
auto-fetched by tectonic.

### Known harmless warnings

- `lineno.sty:296: Invalid UTF-8` — cosmetic, in the line-numbering package.
- `TeX rerun seems needed, but stopping at 6 passes` — `natbib` + tectonic
  cycling on `.bbl`; output is still correct.
- Drop `\usepackage{lineno}` and `\linenumbers` to silence both for the
  submission build (line numbers are draft-only).

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
- **Figures**: legends drafted, four of six panels generated. Six display items:
  1. **(generated)** Brain-scale 1D forward problem: continuous G(x), analytical Euler solution, FD overlay, second-order convergence to N=1024.
  2. **(generated)** 1D FNO inversion: training/val loss curve + four held-out G(x) predictions (Re and Im).
  3. TSM-FNO architecture schematic *(TikZ — manual)*
  4. Acoustoelastic phantom physics *(code — `tsm_fno/scripts/paper_phantom_demo.py`)*
  5. **(generated)** Phase A phantom validation panel (`tsm_fno/results/phantom_phaseA/phantom_panel.png`).
  6. **(generated)** Synthetic cohort performance (RL² stratified, ring RL² vs pressure, headline metrics).
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
