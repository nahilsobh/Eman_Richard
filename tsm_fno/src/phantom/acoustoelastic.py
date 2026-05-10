"""Acoustoelastic effective stiffness field.

The pre-stress field around an expanding lesion modulates the local shear
modulus through the acoustoelastic coupling

    G_eff(x, y) = G_bg(x, y) + A_coeff · Δσ(x, y)

with our sign convention Δσ > 0 in the perilesional shell (tension) and
A_coeff > 0 — so the TSM stiffening ring is reproduced (Yin et al. 2026:
tangential shear-wave propagation reads HIGHER stiffness perpendicular
to a compression axis).
"""
from __future__ import annotations

import numpy as np

from .geometry import LesionGeometry, compute_lame_field

G_MIN_PA = 200.0
G_MAX_PA = 80000.0
# EPS_MAX bounds the realised latent-strain field. The Lamé solution
# reaches ε > 0.15 within a few a_eq of the lesion at our pressure range
# (500–8000 Pa, G_bg ~ 1500 Pa). We use 3.0 so the field has spatial
# structure to learn from; the FNO output head clamps via the same constant.
EPS_MAX = 3.0


def make_intrinsic_G(N: int,
                     dx: float,
                     geometry: LesionGeometry,
                     G_bg: float,
                     G_lesion: float) -> np.ndarray:
    """Intrinsic shear modulus G(x,y) — lesion contrast only, no acoustoelastic.

    Use this with ``helmholtz_eshelby_solve``; the acoustoelastic effect is
    captured separately via the eigenstrain ε* = A·Δσ/G_bg.
    """
    G = np.full((N, N), float(G_bg), dtype=np.float64)
    G[geometry.mask(N)] = float(G_lesion)
    return np.clip(G, G_MIN_PA, G_MAX_PA)


def make_effective_G(N: int,
                     dx: float,
                     geometry: LesionGeometry,
                     G_bg: float,
                     G_lesion: float,
                     A_coeff: float) -> np.ndarray:
    """Effective shear modulus G_eff(x,y) [Pa] seen by the wave solver."""
    G = np.full((N, N), float(G_bg), dtype=np.float64)
    G[geometry.mask(N)] = float(G_lesion)
    lame = compute_lame_field(geometry, N, dx, G_background=1.0)
    # lame here has units of Pa (Δσ); A_coeff is dimensionless; convert to Pa
    # via G_bg in the convention G_eff = G_bg + A_coeff · (Δσ / G_bg) · G_bg
    G += float(A_coeff) * lame
    return np.clip(G, G_MIN_PA, G_MAX_PA)


def make_latent_strain(N: int, dx: float, geometry: LesionGeometry,
                        G_bg: float) -> np.ndarray:
    """Ground-truth latent strain ε_latent = Δσ / G_bg (dimensionless).

    Returned with positive sign — softplus model head outputs ≥ 0.
    Zero everywhere for a non-expanding control case (p = 0).
    """
    lame = compute_lame_field(geometry, N, dx, G_background=float(G_bg))
    return np.clip(lame, 0.0, EPS_MAX).astype(np.float64)


def sample_phantom_params(rng: np.random.Generator,
                           bias: str | None = None) -> dict:
    """Draw a random parameter set for one phantom.

    bias:
      None              — default mix
      'high_contrast'   — G_lesion / G_bg > 10
      'near_isotropic'  — circular lesion (a ≈ b) handled by caller
      'control'         — pressure forced to 0
    """
    G_bg = float(rng.uniform(800.0, 3000.0))
    if bias == 'high_contrast':
        G_lesion = float(rng.uniform(10.0 * G_bg, 15.0 * G_bg))
        G_lesion = min(G_lesion, G_MAX_PA - 1000.0)
    else:
        G_lesion = float(rng.uniform(3000.0, 15000.0))
    A_coeff = float(rng.uniform(2.0, 8.0))
    return {
        "G_bg": G_bg,
        "G_lesion": G_lesion,
        "A_coeff": A_coeff,
    }
