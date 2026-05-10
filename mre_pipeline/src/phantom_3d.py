"""3D phantom generator for slice-by-slice 2D inversion experiments.

ILI-aligned distribution (matches Phase 0 v3):
  - Piecewise-smooth 3D background (smoothed Gaussian noise, anisotropic)
  - Optional 3D ellipsoidal inclusion (50% probability)
  - Stiffness range [0.5, 15] kPa with per-sample max from U(0, 14.5) kPa
"""
import numpy as np
from scipy.ndimage import gaussian_filter

G_FLOOR_PA  = 500.0
G_CEIL_PA   = 15000.0
G_RANGE_MAX = 14500.0
SMOOTH_SIGMA_MIN = 1.0
SMOOTH_SIGMA_MAX = 4.0
INCLUSION_PROB = 0.5


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _smooth_random_field_3d(N: int, rng: np.random.Generator) -> np.ndarray:
    """Smoothed 3D Gaussian noise scaled to a random sub-range, clipped to [floor, ceil]."""
    noise = rng.standard_normal((N, N, N))
    sigmas = (rng.uniform(SMOOTH_SIGMA_MIN, SMOOTH_SIGMA_MAX),
              rng.uniform(SMOOTH_SIGMA_MIN, SMOOTH_SIGMA_MAX),
              rng.uniform(SMOOTH_SIGMA_MIN, SMOOTH_SIGMA_MAX))
    field = gaussian_filter(noise, sigma=sigmas)
    f_min, f_max = float(field.min()), float(field.max())
    if f_max - f_min < 1e-12:
        field = np.full_like(field, 0.5)
    else:
        field = (field - f_min) / (f_max - f_min)
    range_max = float(rng.uniform(0.0, G_RANGE_MAX))
    field = field * range_max
    return np.clip(field, G_FLOOR_PA, G_CEIL_PA).astype(np.float64)


def generate_phantom_3d(
    N: int = 64,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Return (N, N, N) ILI-style phantom: smooth background + optional smooth inclusion."""
    if rng is None:
        rng = np.random.default_rng()

    G = _smooth_random_field_3d(N, rng)

    if rng.random() < INCLUSION_PROB:
        cz, cy, cx = rng.uniform(N / 4, 3 * N / 4, size=3)
        a_max = N / 6.0
        a = rng.uniform(min(4.0, a_max * 0.5), a_max)
        b = rng.uniform(0.6 * a, a)
        c = rng.uniform(0.6 * a, a)
        R = random_rotation_matrix(rng)

        zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
        pts = np.stack([zz - cz, yy - cy, xx - cx], axis=-1)
        pts_rot = pts @ R
        z_p, y_p, x_p = pts_rot[..., 0], pts_rot[..., 1], pts_rot[..., 2]
        mask = (x_p / a) ** 2 + (y_p / b) ** 2 + (z_p / c) ** 2 <= 1.0

        inclusion_field = _smooth_random_field_3d(N, rng)
        G = np.where(mask, inclusion_field, G)

    return np.clip(G, G_FLOOR_PA, G_CEIL_PA)
