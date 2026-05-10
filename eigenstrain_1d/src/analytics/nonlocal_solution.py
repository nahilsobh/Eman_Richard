"""Nonlocal elasticity: exponential kernel and Fourier stress computation.

Kernel: α(r) = (1/2ℓ)·exp(-|r|/ℓ)
Fourier transform: α̂(k) = 1/(1 + k²ℓ²)   (exact, Lorentzian)

Key finding: for constant σ (static equilibrium), the nonlocal kernel
does not alter the static solution. The nonlocal effect enters only in
the dynamic (wave) problem where stress varies in space.
"""
from __future__ import annotations

import numpy as np
from numpy.fft import fft, ifft, fftfreq


def nonlocal_kernel_fourier(k: np.ndarray, ell: float) -> np.ndarray:
    """Fourier transform of exponential nonlocal kernel α(r) = (1/2ℓ)exp(-|r|/ℓ).

    α̂(k) = 1 / (1 + k²ℓ²)
    """
    return 1.0 / (1.0 + k**2 * ell**2)


def nonlocal_kernel_direct(x: np.ndarray, ell: float) -> np.ndarray:
    """Real-space nonlocal kernel α(x) = (1/2ℓ)·exp(-|x|/ℓ)."""
    return (1.0 / (2.0 * ell)) * np.exp(-np.abs(x) / ell)


def nonlocal_stress_fourier(
    eps_field: np.ndarray,
    eps_star_field: np.ndarray,
    E: float,
    ell: float,
    dx: float,
) -> np.ndarray:
    """Compute nonlocal stress field σ(x) = α * [E·(ε - ε*)] via Fourier convolution.

    For homogeneous E:
        σ̂(k) = α̂(k) · E · (ε̂(k) - ε̂*(k))

    Parameters
    ----------
    eps_field      : (N,) total strain ε(x)
    eps_star_field : (N,) eigenstrain ε*(x)
    E              : Young's modulus [Pa]
    ell            : nonlocal length scale [m]
    dx             : grid spacing [m]

    Returns
    -------
    sigma : (N,) float64 — nonlocal stress field
    """
    N = len(eps_field)
    k = fftfreq(N, d=dx) * 2.0 * np.pi
    alpha_hat = nonlocal_kernel_fourier(k, ell)

    elastic_strain = eps_field - eps_star_field
    sigma_hat = alpha_hat * E * fft(elastic_strain)
    return ifft(sigma_hat).real


def verify_kernel_fourier(N: int, dx: float, ell: float, tol: float = 1e-6) -> bool:
    """Verify α̂(k) = 1/(1+k²ℓ²) matches numerical integration of α(x)."""
    k = fftfreq(N, d=dx) * 2.0 * np.pi
    x = np.arange(N) * dx - N * dx / 2.0  # centered

    alpha_x = nonlocal_kernel_direct(x, ell)
    alpha_x_norm = alpha_x / (alpha_x.sum() * dx)  # ensure normalised

    alpha_hat_numeric = np.abs(fft(alpha_x_norm * dx))
    alpha_hat_analytic = nonlocal_kernel_fourier(k, ell)

    # Compare at low-k modes where periodic approximation is good
    n_modes = min(N // 8, 16)
    rel_err = np.abs(alpha_hat_numeric[:n_modes] - alpha_hat_analytic[:n_modes]) / (
        np.abs(alpha_hat_analytic[:n_modes]) + 1e-30)
    return bool(rel_err.max() < tol)
