# FEniCS finite-strain FEM validation at 3 mm

## What this run confirms

Vol 100 mL FEM ring stiffness matches the linear-FDM option-3 baseline
(from `paper_ogden_option3_container_fem/`) to within 1 %:

| Vol | FEM 3 mm (G_P1) | Linear FDM (G_P1) | Δ |
|-----|-----------------|-------------------|---|
| 50 mL | 2.50 | 2.50 | 0 % |
| **100 mL** | **2.73** | **2.71** | **+0.7 %** |

Same agreement for P2 (4.93 vs 4.87 kPa, +1.2 %).

The linear-FDM small-strain approximation used in `paper_ogden_option3_container_fem`
is therefore validated by full nonlinear neo-Hookean FEM up to Vol 100 mL
(surface strain ~26 %).

## What broke

Vol 150 mL crashed at step 1 iter 1 with MUMPS error 76 (NaN in factor).
Root cause: near-incompressible penalty (κ = 100 μ) + full Newton makes
the tangent stiffness matrix ill-conditioned at large deformation
(surface displacement > ~4 mm), and MUMPS partial pivoting fails.

Attempts that did NOT fix it:
- SuperLU / SUPERLU_DIST (same NaN)
- CG + GAMG iterative (very slow at this DOF count)
- Softer κ = 10 μ (extends stable range to Vol 100 mL, still crashes at 150)
- Cell-exclusion (balloon interior removed from integral) — created singular matrix
- Lamé warm start (produced same NaN, larger initial residual)
- Modified Newton with fixed tangent + backtracking line search (line search fails)

## What would fix it (deferred)

Proper mixed **u–p Taylor–Hood formulation** with Lagrange multiplier
for exact incompressibility, in the spirit of Simo & Taylor (1985). This
avoids the volumetric-penalty ill-conditioning entirely by treating the
pressure as an independent field. Estimated 1–2 weeks of code + validation.

## Files

- `results.json` — machine-readable results (Vol 50 & 100 mL)
- `summary.txt` — human-readable table
- Companion scripts: `tsm_fno/scripts/paper_ogden_fenics_fem_quarter.py`,
  `run_fenics_fem_quarter.sbatch`, `paper_ogden_fenics_mnewton.py`
