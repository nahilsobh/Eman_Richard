"""TSM training-pair generator.

Returns one (X, Y, meta) tuple where:
    X: float32 (4, N, N)
       0,1: Re/Im of u at 80 Hz
       2:   Lamé pre-stress prior Δσ / G_bg (analytical, dimensionless)
       3:   Distance from lesion surface (normalised to [0, 1])
    Y: dict
       'G':       float32 (N, N) — effective stiffness [Pa]
       'epsilon': float32 (N, N) — latent strain (dimensionless)
       'ring':    bool     (N, N) — perilesional shell mask (training weight)
    meta: dict with sample-level scalars (pressure, A, G_bg, …)

Composition (drawn per call):
    60% expanding (p drawn from non-zero range)
    20% control   (p = 0; eps_true ≡ 0 everywhere)
    10% high-contrast (G_lesion / G_bg > 10)
    10% near-circular (a ≈ b)
"""
from __future__ import annotations

import numpy as np

from ..phantom.geometry import LesionGeometry, perilesional_shell
from ..phantom.acoustoelastic import (make_effective_G, make_intrinsic_G,
                                        make_latent_strain,
                                        sample_phantom_params, EPS_MAX)
from ..solver.helmholtz_fd import helmholtz_eshelby_solve, random_sources

RHO = 1000.0
DX  = 0.003        # 3 mm/voxel → matches paper: 80×80 matrix, 240 mm FOV (Yin et al. 2026)
FREQ_1 = 80.0      # 80 Hz pneumatic driver (Yin et al. 2026)
DAMPING = 0.05
SHELL_MM = 5.0
N_DEFAULT = 80     # 80×80 grid → 240 mm FOV, exact paper matrix size

# Composition probabilities (must sum to 1)
P_EXPANDING     = 0.60
P_CONTROL       = 0.20
P_HIGH_CONTRAST = 0.10
P_NEAR_CIRCULAR = 0.10


def _add_complex_noise(u: np.ndarray, snr_db: float,
                        rng: np.random.Generator) -> np.ndarray:
    sig_power = float(np.mean(np.abs(u) ** 2))
    if sig_power < 1e-30:
        return u
    snr_lin = 10 ** (snr_db / 10.0)
    sigma = float(np.sqrt(sig_power / snr_lin / 2.0))
    return u + sigma * (rng.standard_normal(u.shape)
                        + 1j * rng.standard_normal(u.shape))


def _sample_geometry(N: int, rng: np.random.Generator,
                      mode: str) -> LesionGeometry:
    cx = float(rng.uniform(N / 3.0, 2.0 * N / 3.0))
    cy = float(rng.uniform(N / 3.0, 2.0 * N / 3.0))
    a  = float(rng.uniform(6.0, N / 5.0))
    if mode == "near_circular":
        b = float(rng.uniform(0.95 * a, a))
    else:
        b = float(rng.uniform(0.6 * a, a))
    angle = float(rng.uniform(0.0, np.pi))
    if mode == "control":
        p = 0.0
    else:
        p = float(rng.uniform(500.0, 8000.0))
    return LesionGeometry(center=(cx, cy), semi_axes=(a, b),
                           angle=angle, pressure=p)


def _draw_mode(rng: np.random.Generator) -> str:
    r = rng.random()
    if r < P_EXPANDING:
        return "expanding"
    if r < P_EXPANDING + P_CONTROL:
        return "control"
    if r < P_EXPANDING + P_CONTROL + P_HIGH_CONTRAST:
        return "high_contrast"
    return "near_circular"


def make_tsm_pair(N: int = N_DEFAULT, dx: float = DX,
                   rng: np.random.Generator = None
                   ) -> tuple[np.ndarray, dict, dict]:
    if rng is None:
        rng = np.random.default_rng()

    mode = _draw_mode(rng)
    geom = _sample_geometry(N, rng, mode)

    bias = "high_contrast" if mode == "high_contrast" else None
    params = sample_phantom_params(rng, bias=bias)
    G_bg = params["G_bg"]
    G_lesion = params["G_lesion"]
    A_coeff = params["A_coeff"]

    G_int = make_intrinsic_G(N, dx, geom, G_bg, G_lesion)
    eps   = make_latent_strain(N, dx, geom, G_bg)
    ring  = perilesional_shell(geom.mask(N), shell_mm=SHELL_MM, dx=dx)

    # eps_star = A_coeff · (Δσ/G_bg); make_latent_strain returns Δσ/G_bg already
    eps_star = A_coeff * eps   # scalar eigenstrain magnitude field

    # Solve at 80 Hz with Eshelby form: ∇·[G*(∇u − ε̄*)] + ρω²u = 0
    sources = random_sources(N, rng)
    # geom.center = (cx_col, cy_row); solver expects (cy_row, cx_col)
    u80 = helmholtz_eshelby_solve(
        G_int, eps_star, center=(geom.center[1], geom.center[0]),
        freq=FREQ_1, rho=RHO, dx=dx, damping=DAMPING, sources=sources,
    )

    # Keep G_eff for backward compat in Y dict (now = G_int, A separate)
    G_eff = G_int

    snr = float(rng.uniform(15.0, 30.0))
    u80 = _add_complex_noise(u80, snr, rng)

    # Per-sample global scaling so the wave channel magnitude ≤ 1
    u_max = float(np.max(np.abs(u80)))
    if u_max > 0:
        u80 = u80 / u_max

    # Lamé prior channel: dimensionless Δσ / G_bg, clipped to [0, EPS_MAX]
    lame_prior = eps.copy()

    # Distance channel normalised to [0, 1]
    dist = geom.distance_field(N, dx)
    if dist.max() > 0:
        dist_norm = dist / dist.max()
    else:
        dist_norm = dist

    X = np.stack([
        u80.real, u80.imag,
        lame_prior, dist_norm,
    ], axis=0).astype(np.float32)  # shape (4, N, N)

    Y = {
        "G":       G_eff.astype(np.float32),
        "epsilon": eps.astype(np.float32),
        "ring":    ring.astype(np.bool_),
    }

    meta = {
        "p":            float(geom.pressure),
        "A":            float(A_coeff),
        "G_bg":         float(G_bg),
        "G_lesion":     float(G_lesion),
        "a_eq_m":       float(geom.equivalent_radius(dx)),
        "is_expanding": bool(geom.pressure > 0),
        "snr_db":       float(snr),
        "n_sources":    int(len(sources)),
        "mode":         mode,
    }
    return X, Y, meta
