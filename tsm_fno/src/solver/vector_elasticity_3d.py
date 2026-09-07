"""3D vector elasticity solver — SCAFFOLD ONLY.

This module is a placeholder for a full Navier-equation vector-elasticity
solver, the correct physical model for MRE shear waves in a pre-stressed
anisotropic medium. It's intentionally *not implemented yet* — this file
holds the design contract and a stub so future work has a clean landing.

Physics
-------
For a linear elastic solid with density ρ, displacement 3-vector u_i(x),
and elasticity tensor C_ijkl(x), the time-harmonic wave equation is

    ρ ω² u_i(x) = ∂_j [ C_ijkl(x) · ∂_k u_l(x) ]                   (Navier)

For isotropic linear elasticity, C_ijkl = λ · δ_ij·δ_kl + μ · (δ_ik·δ_jl + δ_il·δ_jk).
Under acoustoelastic coupling to a static pre-stress field σ_mn(x), the
effective elasticity tensor acquires a direction-dependent correction

    C_ijkl(x) = C_ijkl^0 + M_ijklmn · σ_mn(x)                     (small-strain)

where M is the sixth-rank Murnaghan / Landau tensor for the base material.
This is what gives real MRE the direction-dependent shear speed that Yin's
directional filter + MIP surfaces — and what our scalar Helmholtz cannot
reproduce because it collapses u to a single component.

Discretisation plan
-------------------
- Displacement u lives on a Cartesian grid with 3 components per voxel.
  Sparse assembly: N³ voxels × 3 components = 3 N³ unknowns.
- 7-point stencil per component per equation (i.e. 3 equations at each
  voxel, each coupling to 6 neighbours × 3 components + itself × 3
  components). Roughly 63 non-zeros per row → ~63 × 3 N³ nnz total,
  about 10× the scalar case.
- Boundary conditions: Dirichlet on all faces except optionally top-free
  (traction-free Neumann on the top slab — needs the traction tensor
  σ_ij · n_j, not just u).
- Source terms: prescribed u at driver nodes (all 3 components or shear-
  polarised set).
- Solve: SuperLU direct up to N ~ 40; iterative (BiCGSTAB + ILU) for
  larger. Complex-valued sparse solve is well-supported in scipy.sparse.

Estimated implementation effort
-------------------------------
- Full tensor assembly with acoustoelastic coupling: 3–5 days
- Traction-free top-face BC (much harder than Dirichlet u=0 or Neumann-∂u/∂z=0
  for scalar): 1–2 days
- Test suite (analytic plane-wave modes, force-balance, energy check):
  2–3 days
- Integration with existing balloon-phantom + TSM demo:  1–2 days
- Validation vs the scalar demo's numbers and vs Yin Fig 6: 1–2 days

Total: ~1–2 weeks solid work.

What this would fix
-------------------
1. μ_conv would go flat (as Yin observes) because direction-averaging of a
   true vector wave cancels the anisotropic components of C_ijkl.
2. μ_TSM would recover the correct ring stiffness without needing the
   heuristic median-filter + edge-exclusion + amplitude-thresholding
   we currently rely on.
3. The 12–15% peak overshoot we still see would go away.
4. Correctly handles wave polarisation, needed for HGO / fibered tissue.

References
----------
- Guo J. et al. (2015). Multi-frequency MRE with a Voigt viscoelastic model.
- Egle D., Bray D. (1976). Acoustoelastic theory in Murnaghan's third-order
  invariants (M-tensor definition).
- Murnaghan F.D. (1951). Finite Deformation of an Elastic Solid.
"""
from __future__ import annotations

import numpy as np


def navier_solve_3d(
    C: np.ndarray,           # (N,N,N,3,3,3,3) elasticity tensor
    freq: float,
    rho: float = 1000.0,
    dx: float = 0.003,
    sources: list[tuple[int, int, int, int, complex]] | None = None,
    top_free: bool = False,
) -> np.ndarray:
    """Solve the 3D Navier vector-wave equation. **NOT IMPLEMENTED.**

    Parameters (once implemented)
    -----------------------------
    C : (N, N, N, 3, 3, 3, 3) elasticity tensor field [Pa].
    freq, rho, dx : as scalar case.
    sources : list of (i, j, k, component, complex_amplitude) tuples.
        ``component ∈ {0, 1, 2}`` selects the displacement component
        driven at that node.
    top_free : traction-free (Neumann σ_ij·n_j = 0) top face when True.

    Returns
    -------
    u : (N, N, N, 3) complex ndarray — the vector displacement field.
    """
    raise NotImplementedError(
        "navier_solve_3d is a placeholder — vector elasticity is a "
        "multi-week project. See module docstring for the design contract "
        "and effort estimate."
    )
