"""Lesion geometry, Lamé pre-stress field, and perilesional shell.

A lesion is modelled as a rotated ellipse pressurised at p Pa. The Lamé
solution for an ellipsoidal pressurised inclusion in an isotropic matrix
gives the surrounding pre-stress field, characterised by the deviatoric
stress Δσ = σ_θθ − σ_rr that drives acoustoelastic stiffening of the
perilesional shell.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class LesionGeometry:
    """Random pressurised elliptical lesion."""

    center: tuple[float, float]
    semi_axes: tuple[float, float]
    angle: float
    pressure: float

    @classmethod
    def random(cls, N: int, rng: np.random.Generator,
               control_prob: float = 0.20) -> "LesionGeometry":
        cx = float(rng.uniform(N / 3.0, 2.0 * N / 3.0))
        cy = float(rng.uniform(N / 3.0, 2.0 * N / 3.0))
        a = float(rng.uniform(6.0, N / 5.0))
        b = float(rng.uniform(0.6 * a, a))
        angle = float(rng.uniform(0.0, np.pi))
        if rng.random() < control_prob:
            p = 0.0
        else:
            p = float(rng.uniform(500.0, 8000.0))
        return cls(center=(cx, cy), semi_axes=(a, b), angle=angle, pressure=p)

    def _rotated_coords(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        cy, cx = self.center[1], self.center[0]
        yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
        dy = yy - cy
        dx = xx - cx
        c, s = np.cos(self.angle), np.sin(self.angle)
        x_rot = dx * c + dy * s
        y_rot = -dx * s + dy * c
        return x_rot, y_rot

    def mask(self, N: int) -> np.ndarray:
        a, b = self.semi_axes
        x_rot, y_rot = self._rotated_coords(N)
        return ((x_rot / a) ** 2 + (y_rot / b) ** 2) <= 1.0

    def boundary_normals(self, N: int) -> np.ndarray:
        """Outward unit normals at every voxel, computed from the distance
        transform of the lesion's complement. Inside the lesion, normals
        are zero (no defined outward direction)."""
        m = self.mask(N)
        outside = ~m
        # distance from each outside voxel to the nearest lesion voxel
        dist = distance_transform_edt(outside).astype(np.float64)
        gy, gx = np.gradient(dist)
        norm = np.sqrt(gx ** 2 + gy ** 2) + 1e-12
        nx = gx / norm
        ny = gy / norm
        normals = np.stack([nx, ny], axis=-1)
        normals[m] = 0.0
        return normals.astype(np.float64)

    def distance_field(self, N: int, dx: float) -> np.ndarray:
        """Distance from the lesion surface in metres. Zero inside the lesion."""
        outside = ~self.mask(N)
        d = distance_transform_edt(outside).astype(np.float64)
        return d * dx

    def equivalent_radius(self, dx: float) -> float:
        a, b = self.semi_axes
        return float(np.sqrt(a * b) * dx)


def compute_lame_field(geometry: LesionGeometry, N: int, dx: float,
                       G_background: float = 1.0) -> np.ndarray:
    """Lamé deviatoric pre-stress Δσ = σ_θθ − σ_rr, normalised by G_background.

    For a circular pressurised inclusion of equivalent radius a_eq:
        σ_rr(r)  = -p · (a_eq / r)^2     (compressive)
        σ_θθ(r)  = +p · (a_eq / r)^2     (tensile)
        Δσ(r)    = 2 p (a_eq / r)^2      (outside)
        Δσ       = p                     (inside, uniform Lamé)

    Returns dimensionless field (units of strain) scaled by 1 / G_background.
    """
    if geometry.pressure == 0.0:
        return np.zeros((N, N), dtype=np.float64)

    a_eq_vox = float(np.sqrt(geometry.semi_axes[0] * geometry.semi_axes[1]))
    cy, cx = geometry.center[1], geometry.center[0]
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    out = np.empty((N, N), dtype=np.float64)
    inside = geometry.mask(N)
    # Outside: smooth (a_eq / r)^2 decay, with floor at r = a_eq voxels
    safe_r = np.maximum(r, a_eq_vox)
    out_decay = 2.0 * geometry.pressure * (a_eq_vox / safe_r) ** 2
    inside_val = float(geometry.pressure)
    out[inside] = inside_val
    out[~inside] = out_decay[~inside]
    return out / float(G_background)


def perilesional_shell(lesion_mask: np.ndarray, shell_mm: float,
                        dx: float) -> np.ndarray:
    """Boolean annular shell immediately outside the lesion.

    shell_mm: shell thickness in millimetres (default ~5 mm).
    dx:       voxel pitch in metres.
    """
    outside = ~lesion_mask
    dist_vox = distance_transform_edt(outside).astype(np.float64)
    dist_mm = dist_vox * dx * 1000.0
    shell = outside & (dist_mm > 0.0) & (dist_mm <= shell_mm)
    return shell
