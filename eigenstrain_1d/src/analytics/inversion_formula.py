"""Eigenstrain inversion formulas — homogeneous and heterogeneous E.

Homogeneous (k-space), for uniform E:
    Wave eq. in Fourier space:  -ρω²Û = -k²EÛ - ikEε̂*
    → ε̂*(k) = ikÛ(k) · [1 - ρω²/(k²·E)]
    Nonlocal kernel α̂ = 1/(1+k²ℓ²):
    → ε̂*(k) = ikÛ(k) · [1 - ρω²(1+k²ℓ²)/(k²·E)]
    DC (k=0): set to zero — mean ε* not recoverable from waves.

Heterogeneous (real-space), exact for known E(x):
    Wave eq.: dσ/dx = -ρω²u,  σ = E_c(x)[du/dx + ε*(x)]
    Integrate: σ(x) = σ₀ - ρω²∫₀ˣ u dx'
    Invert:    ε*(x) = σ(x)/E_c(x) - du/dx
    Constraint mean(ε*)=0 → σ₀ = ρω² mean(∫u/E_c) / mean(1/E_c)
"""
from __future__ import annotations

import numpy as np
from numpy.fft import fft, ifft, fftfreq


# ── heterogeneous (real-space, exact) ────────────────────────────────────────

def inversion_formula_heterogeneous(
    u: np.ndarray,
    dx: float,
    E_field: np.ndarray,
    rho: float,
    freq: float,
    ell: float = 0.0,
    damping: float = 0.05,
) -> np.ndarray:
    """Real-space eigenstrain inversion for heterogeneous E(x).

    Uses the half-point stress form of momentum conservation, matching the
    forward solver's half-point FD stencil exactly:

        σ_{j+½} = E_{j+½}[(u_{j+1}-u_j)/dx + ε*_{j+½}]
        (σ_{j+½} - σ_{j-½})/dx = -ρω²u_j          (momentum balance)

    Cumulative sum of momentum gives σ at half-points; then:
        ε*_{j+½} = σ_{j+½}/E_{j+½} - (u_{j+1}-u_j)/dx

    Interpolate to cell centres and remove mean (DC not recoverable).

    Parameters
    ----------
    u       : (N,) complex — measured displacement at one frequency
    dx      : grid spacing [m]
    E_field : (N,) float — spatial Young's modulus [Pa]
    rho     : density [kg/m³]
    freq    : drive frequency [Hz]
    ell     : Gaussian smoothing length [m] (regularisation; 0 = none)
    damping : loss tangent ξ; E_c = E(1+iξ) prevents blow-up near resonance

    Returns
    -------
    eps_star : (N,) float64
    """
    N = len(u)
    omega = 2.0 * np.pi * freq
    Ec = E_field.astype(complex) * (1.0 + 1j * damping)

    # Half-point modulus (matches forward solver arithmetic averaging)
    E_half = 0.5 * (Ec + np.roll(Ec, -1))   # E_{j+½}, periodic

    # Forward strain at half-points: (u_{j+1} - u_j) / dx
    du_fwd = (np.roll(u, -1) - u) / dx

    # Cumulative momentum integral: σ_{j+½} = σ_{-½} - ρω²·dx·Σ_{k=0}^{j} u_k
    # Determine σ_{-½} from zero-mean ε* at cell centres (DC constraint)
    #   ε*_{j+½} = σ_{j+½}/E_{j+½} - du_fwd_j
    #   mean cell-centre ε* ≈ mean half-point ε* = 0
    #   → σ_{-½}·mean(1/E_half) = ρω²·dx·mean(cumsum_u / E_half) + mean(du_fwd)
    cumsum_u = np.cumsum(u) - u[0]          # Σ_{k=0}^{j} u_k, starting at 0

    inv_Eh = 1.0 / E_half
    rhs = (rho * omega**2 * dx * np.mean(cumsum_u * inv_Eh)
           + np.mean(du_fwd))
    sigma_m_half = rhs / np.mean(inv_Eh)    # σ_{-½}

    sigma_half = sigma_m_half - rho * omega**2 * dx * cumsum_u

    eps_half = sigma_half * inv_Eh - du_fwd   # ε* at half-points

    # Interpolate half-point ε* to cell centres: (ε*_{j-½} + ε*_{j+½})/2
    eps_cell = 0.5 * (eps_half + np.roll(eps_half, 1))
    eps_star = eps_cell.real

    if ell > 0.0:
        from scipy.ndimage import gaussian_filter1d
        eps_star = gaussian_filter1d(eps_star, sigma=ell / dx)

    return eps_star


# ── homogeneous (k-space) ─────────────────────────────────────────────────────

def inversion_formula(
    u_hat_array: np.ndarray,
    dx: float,
    E: float | np.ndarray,
    rho: float,
    freq: float,
    ell: float = 0.0,
    damping: float = 0.05,
) -> np.ndarray:
    """Recover ε*(x) from measured displacement û(x).

    If E is a 1-D array the exact heterogeneous real-space formula is used.
    If E is a scalar the classic k-space formula (homogeneous E) is used.

    Parameters
    ----------
    u_hat_array : (N,) complex
    dx          : grid spacing [m]
    E           : scalar [Pa] or (N,) array [Pa]
    rho         : density [kg/m³]
    freq        : drive frequency [Hz]
    ell         : nonlocal / smoothing length [m]
    damping     : loss tangent ξ

    Returns
    -------
    eps_star : (N,) float64
    """
    if isinstance(E, np.ndarray) and E.ndim == 1:
        return inversion_formula_heterogeneous(u_hat_array, dx, E, rho, freq, ell, damping)

    # --- homogeneous k-space path ---
    N = len(u_hat_array)
    omega = 2.0 * np.pi * freq
    k = fftfreq(N, d=dx) * 2.0 * np.pi

    U_hat   = fft(u_hat_array)
    eps_hat = 1j * k * U_hat

    alpha_hat    = 1.0 / (1.0 + k**2 * ell**2)
    alpha_hat[0] = 1.0

    Ec = float(E) * (1.0 + 1j * damping)
    k2    = k**2.0 + 0j
    k2[0] = 1.0

    correction    = 1.0 - rho * omega**2 / (k2 * alpha_hat * Ec)
    correction[0] = 0.0

    return ifft(eps_hat * correction).real


def inversion_formula_batch(
    u_batch: np.ndarray,
    dx: float,
    E: float,
    rho: float,
    freq: float,
    ell: float = 0.0,
) -> np.ndarray:
    """Vectorized homogeneous inversion over a batch (B, N) complex array."""
    single = u_batch.ndim == 1
    if single:
        u_batch = u_batch[np.newaxis]

    B, N = u_batch.shape
    omega = 2.0 * np.pi * freq
    k = fftfreq(N, d=dx) * 2.0 * np.pi

    U_hat   = np.fft.fft(u_batch, axis=-1)
    eps_hat = 1j * k[np.newaxis] * U_hat

    alpha_hat    = 1.0 / (1.0 + k**2 * ell**2)
    alpha_hat[0] = 1.0
    k2    = k**2; k2[0] = 1.0

    correction    = 1.0 - rho * omega**2 / (k2 * alpha_hat * E)
    correction[0] = 0.0

    eps_star = np.fft.ifft(eps_hat * correction[np.newaxis], axis=-1).real
    return eps_star[0] if single else eps_star
