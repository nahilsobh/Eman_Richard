"""3D spherical-balloon phantom for the 3D Helmholtz demo.

Analogue of ``geometry.py`` + ``acoustoelastic.py``, restricted to the
spherical case needed for the paper demo. Provides the intrinsic and
acoustoelastic-effective stiffness fields, plus the perilesional shell.

The Lamé pre-stress around a pressurised spherical inclusion in an
infinite matrix is
    σ_rr(r) = -p · (a/r)^3
    σ_θθ(r) = σ_φφ(r) = +½ p · (a/r)^3
    Δσ(r)   = σ_θθ − σ_rr = (3/2) p · (a/r)^3    (outside, r ≥ a)
    Δσ      = p                                    (inside, uniform)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


G_MIN_PA = 200.0
G_MAX_PA = 80000.0


@dataclass
class SphericalBalloon:
    center: tuple[float, float, float]   # (i, j, k) in voxels
    radius_vx: float
    pressure: float                       # Pa

    def _coords(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ii, jj, kk = np.mgrid[0:N, 0:N, 0:N].astype(np.float64)
        return (ii - self.center[0],
                jj - self.center[1],
                kk - self.center[2])

    def mask(self, N: int) -> np.ndarray:
        di, dj, dk = self._coords(N)
        return (di ** 2 + dj ** 2 + dk ** 2) <= self.radius_vx ** 2

    def distance_field(self, N: int, dx: float) -> np.ndarray:
        outside = ~self.mask(N)
        d = distance_transform_edt(outside).astype(np.float64)
        return d * dx


def lame_field_sphere(balloon: SphericalBalloon, N: int) -> np.ndarray:
    """Lamé deviatoric pre-stress Δσ(x) for a pressurised sphere."""
    if balloon.pressure == 0.0:
        return np.zeros((N, N, N), dtype=np.float64)

    di, dj, dk = balloon._coords(N)
    r = np.sqrt(di ** 2 + dj ** 2 + dk ** 2)
    a = balloon.radius_vx
    inside = balloon.mask(N)
    safe_r = np.maximum(r, a)
    out_decay = 1.5 * balloon.pressure * (a / safe_r) ** 3
    field = np.where(inside, balloon.pressure, out_decay)
    return field.astype(np.float64)


def make_effective_G_3d(N: int, balloon: SphericalBalloon,
                         G_bg: float, G_lesion: float,
                         A_coeff: float) -> np.ndarray:
    """G_eff(x) = G_bg + A_coeff · Δσ(x)/G_bg · G_bg — same convention as 2D."""
    G = np.full((N, N, N), float(G_bg), dtype=np.float64)
    G[balloon.mask(N)] = float(G_lesion)
    G += float(A_coeff) * lame_field_sphere(balloon, N)
    return np.clip(G, G_MIN_PA, G_MAX_PA)


def perilesional_shell_3d(lesion_mask: np.ndarray, shell_mm: float,
                          dx: float) -> np.ndarray:
    """Perilesional shell — voxels within `shell_mm` of the lesion boundary."""
    outside = ~lesion_mask
    dist_vox = distance_transform_edt(outside).astype(np.float64)
    dist_mm = dist_vox * dx * 1000.0
    return outside & (dist_mm > 0.0) & (dist_mm <= shell_mm)
