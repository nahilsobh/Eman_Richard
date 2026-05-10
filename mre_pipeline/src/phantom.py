"""Phantom generators.

ILI-style (Scott/Murphy 2020): piecewise-smooth backgrounds from anisotropic
Gaussian-smoothed noise fields, with an optional ellipsoidal inclusion in
50% of cases (the inclusion itself is another smooth random field).

Stiffness range follows ILI:
    Per-sample range max ~ U(0, 14.5) kPa, absolute clip to [0.5, 15] kPa.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

G_FLOOR_PA  = 500.0       # absolute clip
G_CEIL_PA   = 15000.0
G_RANGE_MAX = 14500.0     # per-sample max draw from U(0, this)
SMOOTH_SIGMA_MIN = 1.0    # voxels (matches ILI spec σ ∈ 1–4)
SMOOTH_SIGMA_MAX = 4.0
INCLUSION_PROB = 0.5      # ILI inserts an inclusion in half of simulations


def _smooth_random_field(N: int, rng: np.random.Generator) -> np.ndarray:
    """Smoothed Gaussian noise scaled into [0, G_max] Pa, then clipped to absolute range.

    The per-sample max is drawn from U(0, G_RANGE_MAX) so individual phantoms
    have varying dynamic range — matches ILI's 'range selected from U(0, 14.5) kPa'.
    """
    noise = rng.standard_normal((N, N))
    sigma_y = rng.uniform(SMOOTH_SIGMA_MIN, SMOOTH_SIGMA_MAX)
    sigma_x = rng.uniform(SMOOTH_SIGMA_MIN, SMOOTH_SIGMA_MAX)
    field = gaussian_filter(noise, sigma=(sigma_y, sigma_x))

    # Normalise to [0, 1] then scale to [0, range_max]
    f_min, f_max = float(field.min()), float(field.max())
    if f_max - f_min < 1e-12:
        field = np.full_like(field, 0.5)
    else:
        field = (field - f_min) / (f_max - f_min)
    range_max = float(rng.uniform(0.0, G_RANGE_MAX))
    field = field * range_max
    return np.clip(field, G_FLOOR_PA, G_CEIL_PA).astype(np.float64)


def _ellipse_mask(N: int, rng: np.random.Generator) -> np.ndarray:
    """Random rotated-ellipse mask covering a moderate region of an N×N grid."""
    cy = rng.uniform(N / 4, 3 * N / 4)
    cx = rng.uniform(N / 4, 3 * N / 4)
    a_max = N / 6.0
    a_min = min(4.0, a_max * 0.5)
    a = rng.uniform(a_min, a_max)
    b = rng.uniform(0.6 * a, a)
    theta = rng.uniform(0, np.pi)

    yy, xx = np.mgrid[0:N, 0:N]
    dy = yy - cy
    dx = xx - cx
    x_rot =  dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
    return (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0


def generate_phantom(N: int = 64, rng: np.random.Generator = None) -> np.ndarray:
    """ILI-style 2D stiffness map [Pa]: smooth background + optional smooth inclusion."""
    if rng is None:
        rng = np.random.default_rng()

    G = _smooth_random_field(N, rng)

    if rng.random() < INCLUSION_PROB:
        inclusion_field = _smooth_random_field(N, rng)
        mask = _ellipse_mask(N, rng)
        G = np.where(mask, inclusion_field, G)

    return np.clip(G, G_FLOOR_PA, G_CEIL_PA)
