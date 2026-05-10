"""Viscoelastic relaxation analytics for serial MRE.

Prony series relaxation modulus:
    G(t) = G_inf + G1·exp(-t/τ1) + G2·exp(-t/τ2)

The acoustoelastically-detectable pre-stress decays with the same relaxation:
    σ(t) = σ_0 · G(t)/G(0)

Wave speed perturbation:
    δc(t)/c0 = A_coeff · σ(t) / (2·E_bg)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def G_relaxation(t: float | np.ndarray, G_inf: float, G1: float, tau1: float,
                 G2: float = 0.0, tau2: float | None = None) -> float | np.ndarray:
    """Prony series relaxation modulus G(t) [Pa]."""
    G = G_inf + G1 * np.exp(-np.asarray(t, dtype=float) / tau1)
    if G2 > 0 and tau2 is not None:
        G = G + G2 * np.exp(-np.asarray(t, dtype=float) / tau2)
    return G


def delta_c_over_c0(
    t: float | np.ndarray,
    sigma_0: float,
    E_bg: float,
    A_coeff: float,
    G_inf: float,
    G1: float,
    tau1: float,
    G2: float = 0.0,
    tau2: float | None = None,
) -> float | np.ndarray:
    """Relative wave speed perturbation δc(t)/c0."""
    G0 = G_relaxation(0.0, G_inf, G1, tau1, G2, tau2)
    Gt = G_relaxation(t, G_inf, G1, tau1, G2, tau2)
    sigma_t = sigma_0 * Gt / G0
    return A_coeff * sigma_t / (2.0 * E_bg)


def estimate_tau_analytical(
    delta_c_t1: float,
    delta_c_t2: float,
    dt_days: float,
) -> float | None:
    """Single-exponential τ estimation from two delta_c measurements.

    τ = -(t2-t1) / ln(δc(t2)/δc(t1))

    Returns None if ratio is non-positive (unphysical).
    """
    if delta_c_t1 == 0 or delta_c_t2 / delta_c_t1 <= 0:
        return None
    return -dt_days / np.log(delta_c_t2 / delta_c_t1)


def estimate_tau_two_exp(
    delta_c_times: np.ndarray,
    delta_c_values: np.ndarray,
    G_inf_frac: float = 0.7,
) -> dict:
    """Two-exponential τ estimation by nonlinear least squares.

    Returns dict with tau1, tau2, G1_frac, G2_frac.
    Assumes G(t) = G_inf + G1*exp(-t/τ1) + G2*exp(-t/τ2) normalized to G(0)=1.
    """
    t = np.asarray(delta_c_times, dtype=float)
    y = np.asarray(delta_c_values, dtype=float)
    y_norm = y / (y[0] + 1e-30)   # normalize to 1 at t=0

    def model(t, tau1, tau2, g1_frac):
        g2_frac = 1.0 - G_inf_frac - g1_frac
        g2_frac = max(g2_frac, 0.0)
        return G_inf_frac + g1_frac * np.exp(-t/tau1) + g2_frac * np.exp(-t/tau2)

    try:
        p0 = [2.0, 15.0, 0.2]
        bounds = ([0.1, 1.0, 0.0], [10.0, 60.0, 0.3 - G_inf_frac])
        popt, _ = curve_fit(model, t, y_norm, p0=p0, bounds=bounds, maxfev=5000)
        tau1, tau2, g1_frac = popt
        g2_frac = 1.0 - G_inf_frac - g1_frac
        return {"tau1": tau1, "tau2": tau2,
                "G1_frac": g1_frac, "G2_frac": max(g2_frac, 0.0)}
    except Exception:
        return {"tau1": float("nan"), "tau2": float("nan"),
                "G1_frac": float("nan"), "G2_frac": float("nan")}
