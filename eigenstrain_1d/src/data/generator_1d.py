"""1D training pair generator for eigenstrain inversion.

Each pair: (X, Y, meta) where
    X  : float32 (5, N)  — wave channels + Lamé prior
    Y  : dict            — ground truth targets
    meta: dict           — sample-level scalars
"""
from __future__ import annotations

import numpy as np

from ..analytics.static_solution import sigma_bar_formula
from ..analytics.inversion_formula import inversion_formula
from ..solver.forward_1d import helmholtz_solve_1d

RHO   = 1000.0
FREQ1 = 60.0
FREQ2 = 120.0
L     = 0.10     # domain half-length [m]; total domain = 2L = 0.20 m
N_DEFAULT = 256
DX    = 2 * L / N_DEFAULT


def make_1d_pair(
    N: int = N_DEFAULT,
    dx: float = DX,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """Generate one (X, Y, meta) training pair.

    Returns
    -------
    X    : float32 (5, N)
    Y    : dict with 'eps_star_true', 'eps_star_analytic', and scalar keys
    meta : dict with sample-level scalars
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: sample parameters
    E_bg     = float(rng.uniform(800.0, 3000.0))
    E_lesion = float(rng.uniform(3000.0, 15000.0))
    a        = float(rng.uniform(0.005, 0.020))
    eps0     = float(rng.uniform(0.005, 0.08)) if rng.random() > 0.2 else 0.0
    A_coeff  = float(rng.uniform(-8.0, -2.0))
    ell      = float(rng.uniform(0.003, 0.012))
    snr_db   = float(rng.uniform(15.0, 30.0))
    source_phase_1 = float(rng.uniform(0, 2*np.pi))
    source_phase_2 = float(rng.uniform(0, 2*np.pi))
    source_amp = float(rng.uniform(0.5, 2.0))

    # Step 2: build E(x) field and lesion mask
    x = np.arange(N, dtype=float) * dx
    x_c = L   # lesion center
    lesion_mask = np.abs(x - x_c) <= a
    E_field = np.where(lesion_mask, E_lesion, E_bg)

    # Step 3: static solution
    sigma_bar = sigma_bar_formula(a, L, E_bg, E_lesion, eps0)

    # Step 4: acoustoelastic E_eff
    sigma_static_field = sigma_bar * np.ones(N)
    E_eff = E_field + A_coeff * sigma_static_field
    E_eff = np.clip(E_eff, 1.0, None)

    # Step 5: solve wave equation at two frequencies
    # Step 6: ground truth eigenstrain (box function) — needed for wave generation
    eps_star_true = np.where(lesion_mask, eps0, 0.0)

    # For analytical inversion: field driven by eigenstrain ONLY (no Gaussian)
    # Inversion formula assumes u comes entirely from eigenstrain in homogeneous E
    u_60_inv  = helmholtz_solve_1d(N, dx, E_eff, RHO, FREQ1,
                                    source_amplitude=0.0,
                                    snr_db=None, rng=None, normalize=False,
                                    eps_star=eps_star_true)
    u_120_inv = helmholtz_solve_1d(N, dx, E_eff, RHO, FREQ2,
                                    source_amplitude=0.0,
                                    snr_db=None, rng=None, normalize=False,
                                    eps_star=eps_star_true)

    # For FNO training input: Gaussian source + eigenstrain, with noise
    u_60_raw  = helmholtz_solve_1d(N, dx, E_eff, RHO, FREQ1,
                                    source_phase=source_phase_1,
                                    source_amplitude=source_amp,
                                    snr_db=None, rng=None, normalize=False,
                                    eps_star=eps_star_true)
    u_120_raw = helmholtz_solve_1d(N, dx, E_eff, RHO, FREQ2,
                                    source_phase=source_phase_2,
                                    source_amplitude=source_amp,
                                    snr_db=None, rng=None, normalize=False,
                                    eps_star=eps_star_true)

    # Normalize and add noise for FNO input
    u60_max  = max(float(np.max(np.abs(u_60_raw))),  1e-30)
    u120_max = max(float(np.max(np.abs(u_120_raw))), 1e-30)
    if rng is None:
        rng = np.random.default_rng()
    u_60  = u_60_raw  / u60_max
    u_120 = u_120_raw / u120_max
    if snr_db is not None:
        for u_ref, field in [(u_60, None), (u_120, None)]:
            pass  # inline below
        snr_lin = 10 ** (snr_db / 10.0)
        sig60  = float(np.mean(np.abs(u_60)**2))
        sig120 = float(np.mean(np.abs(u_120)**2))
        u_60  = u_60  + np.sqrt(sig60  / snr_lin / 2) * (
            rng.standard_normal(N) + 1j*rng.standard_normal(N))
        u_120 = u_120 + np.sqrt(sig120 / snr_lin / 2) * (
            rng.standard_normal(N) + 1j*rng.standard_normal(N))

    # Step 7: analytical inversion — use E_eff(x) for exact heterogeneous formula
    eps_a60  = inversion_formula(u_60_inv,  dx, E_eff, RHO, FREQ1, ell)
    eps_a120 = inversion_formula(u_120_inv, dx, E_eff, RHO, FREQ2, ell)
    eps_star_analytic = (eps_a60 + eps_a120) / 2.0

    # Step 8: Lamé prior channel
    # 1D: sigma_bar inside lesion, decaying as 1/|x - x_c| outside
    r = np.abs(x - x_c)
    r_safe = np.where(r < dx, dx, r)
    lame_prior = np.where(lesion_mask, sigma_bar, sigma_bar * (a / r_safe))
    # Clip and normalise for network input
    lame_prior_norm = np.clip(lame_prior / (abs(E_bg) + 1.0), -3.0, 3.0)

    X = np.stack([
        u_60.real,
        u_60.imag,
        u_120.real,
        u_120.imag,
        lame_prior_norm,
    ], axis=0).astype(np.float32)

    Y = {
        "eps_star_true":     eps_star_true.astype(np.float32),
        "eps_star_analytic": eps_star_analytic.astype(np.float32),
        "sigma_bar":         float(sigma_bar),
        "A_coeff":           float(A_coeff),
        "E_bg":              float(E_bg),
        "E_lesion":          float(E_lesion),
        "eps0":              float(eps0),
        "ell":               float(ell),
        "snr_db":            float(snr_db),
        "is_expanding":      bool(eps0 > 0),
    }

    meta = Y.copy()
    return X, Y, meta
