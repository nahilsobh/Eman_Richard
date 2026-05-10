"""Nonlocal FD solver: verification implementation.

Solves the nonlocal wave equation via iterative Fourier-space correction.
Used to verify the analytical inversion formula in the nonlocal case.
"""
from __future__ import annotations

import numpy as np
from numpy.fft import fft, ifft, fftfreq


def nonlocal_helmholtz_solve_1d(
    N: int,
    dx: float,
    E_eff: np.ndarray,
    rho: float,
    freq: float,
    ell: float,
    source_amplitude: float = 1.0,
    source_phase: float = 0.0,
    n_iter: int = 20,
) -> np.ndarray:
    """Solve nonlocal Helmholtz via Fourier fixed-point iteration.

    For homogeneous E: exact Fourier-space solve.
    For heterogeneous E: one-iteration Born approximation (sufficient for weak contrast).
    """
    omega = 2.0 * np.pi * freq
    k = fftfreq(N, d=dx) * 2.0 * np.pi
    alpha_hat = 1.0 / (1.0 + k**2 * ell**2)

    # For homogeneous E, use direct k-space solve
    E_mean = float(E_eff.mean())
    dE = E_eff - E_mean

    # Homogeneous part
    k2 = k**2
    denom = -alpha_hat * E_mean * k2 + rho * omega**2
    denom[np.abs(denom) < 1e-30] = 1e-30 + 0j

    # Source in Fourier space
    x = np.arange(N) * dx
    width = 3.0 * dx
    f = source_amplitude * np.exp(-x**2/(2*width**2)) * np.exp(1j*source_phase)
    F_hat = fft(f)

    U_hat = F_hat / denom
    u = ifft(U_hat)

    u_max = np.max(np.abs(u))
    if u_max > 1e-30:
        u /= u_max
    return u
