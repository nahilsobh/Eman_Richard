"""Closed-form static equilibrium for a 1D pressurized inclusion.

Physical setting: quasi-static, periodic domain x ∈ [0, 2L].
Equilibrium: dσ/dx = 0 → σ = σ_bar (constant everywhere).
Periodic BC: ∫_0^{2L} ε dx = 0.

CLOSED FORM:
    σ_bar = -(eps0 * E_lesion * E_bg * 2a) / (E_bg*2a + E_lesion*(2L-2a))
           = -(eps0 * E_lesion * E_bg) / [E_bg + (L/a - 1)*E_lesion]

In 1D infinite limit (L >> a): σ_bar → 0.
This proves the perilesional stiffening ring requires 2D/3D geometry.
"""
from __future__ import annotations

import numpy as np


def static_solution(N: int, dx: float, a: float, L: float,
                    E_bg: float, E_lesion: float, eps0: float
                    ) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form quasi-static solution on periodic domain.

    Parameters
    ----------
    N        : number of grid points
    dx       : grid spacing [m]
    a        : lesion half-width [m]
    L        : domain half-length [m]  (total domain = 2L)
    E_bg     : background Young's modulus [Pa]
    E_lesion : lesion Young's modulus [Pa]
    eps0     : eigenstrain magnitude inside lesion (0 for control)

    Returns
    -------
    sigma_bar : float — constant stress [Pa]
    u_static  : (N,) float64 — displacement field
    E_field   : (N,) float64 — spatial modulus field
    """
    x = np.arange(N) * dx           # [0, dx, 2dx, ..., (N-1)*dx]
    x_c = L                          # lesion center at x = L

    lesion_mask = np.abs(x - x_c) <= a
    E_field = np.where(lesion_mask, float(E_lesion), float(E_bg))

    # Closed-form σ_bar
    C_compliance = 2.0 * a / E_lesion + (2.0 * L - 2.0 * a) / E_bg
    E_star_integral = 2.0 * a * eps0
    sigma_bar = -E_star_integral / C_compliance if abs(C_compliance) > 1e-30 else 0.0

    # Displacement by piecewise integration from x=L (u(L)=0 by symmetry)
    # du/dx = sigma_bar/E(x) + eps0*lesion_mask(x)
    strain_field = sigma_bar / E_field + eps0 * lesion_mask.astype(float)

    # Integrate strain from center outward, then fix periodicity
    # Cumulative integral with midpoint at index corresponding to x_c
    i_c = int(round(x_c / dx))
    # right of center
    u = np.zeros(N)
    for i in range(i_c, N - 1):
        u[i + 1] = u[i] + strain_field[i] * dx
    # left of center
    for i in range(i_c, 0, -1):
        u[i - 1] = u[i] - strain_field[i - 1] * dx

    # Remove constant offset so mean displacement = 0 (periodic convention)
    u -= u.mean()

    return float(sigma_bar), u, E_field


def sigma_bar_formula(a: float, L: float, E_bg: float,
                       E_lesion: float, eps0: float) -> float:
    """Direct closed-form σ_bar without building the full field."""
    C = 2.0 * a / E_lesion + (2.0 * L - 2.0 * a) / E_bg
    return -(2.0 * a * eps0) / C
