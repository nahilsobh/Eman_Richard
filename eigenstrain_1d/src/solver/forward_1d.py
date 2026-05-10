"""1D time-harmonic Helmholtz FD solver on a periodic domain.

    -ρω²·u = d/dx[E_eff(x)·du/dx] + f(x)

Second-order finite differences with half-point arithmetic averaging:
    E_{j+1/2} = (E_j + E_{j+1}) / 2

Periodic boundary conditions via circulant stiffness matrix.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


DAMPING = 0.05   # loss tangent ξ; E* = E(1+iξ) regularises near-resonance


def helmholtz_solve_1d(
    N: int,
    dx: float,
    E_eff: np.ndarray,
    rho: float,
    freq: float,
    source_phase: float = 0.0,
    source_amplitude: float = 1.0,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
    eps_star: np.ndarray | None = None,
    normalize: bool = True,
    damping: float = DAMPING,
) -> np.ndarray:
    """Solve 1D Helmholtz on periodic domain with distributed source near x=0.

    Parameters
    ----------
    N, dx            : grid points and spacing
    E_eff            : (N,) spatially varying effective modulus [Pa]
    rho              : density [kg/m³]
    freq             : drive frequency [Hz]
    source_phase     : complex phase of source [rad]
    source_amplitude : amplitude of source
    snr_db           : if given, add complex noise at this SNR
    rng              : random number generator
    eps_star         : (N,) eigenstrain field; adds source term f += -E_eff * d(eps_star)/dx
    normalize        : if True, divide u by max|u| before returning (default True)
    damping          : loss tangent ξ; E* = E(1+iξ) regularises near-resonance

    Returns
    -------
    u : (N,) complex128 — displacement field
    """
    assert E_eff.shape == (N,)
    omega = 2.0 * np.pi * freq

    # Complex modulus: E* = E(1+iξ) regularises near-resonance singularity
    E_c = E_eff * (1.0 + 1j * damping)

    # Half-point moduli (arithmetic average for smooth interfaces)
    E_half = 0.5 * (E_c + np.roll(E_c, -1))   # E*_{j+1/2}
    E_half_m = np.roll(E_half, 1)                   # E*_{j-1/2}

    # Stiffness matrix: -d/dx[E du/dx] on periodic domain
    # Diagonal: (E_{j+1/2} + E_{j-1/2}) / dx²
    # Off-diagonal ±1: -E_{j±1/2} / dx²
    diag_main = (E_half + E_half_m) / dx**2
    diag_plus = -E_half / dx**2                     # upper diagonal
    diag_minus = -E_half_m / dx**2                  # lower diagonal

    # Build circulant sparse matrix
    # Corner wraps: K[0,N-1] = diag_minus[0], K[N-1,0] = diag_plus[-1]
    d = [diag_minus, diag_main, diag_plus,
         diag_minus[[0]], diag_plus[[-1]]]
    offsets = [-1, 0, 1, N-1, -(N-1)]

    K = diags(d, offsets, shape=(N, N), format='csr', dtype=complex)

    # Mass matrix: rho*omega²
    M = rho * omega**2 * eye(N, format='csr', dtype=complex)

    # System: (K - M) u = f
    A = K - M

    # Source: smooth Gaussian near x=0 with width ~3*dx
    x = np.arange(N) * dx
    width = 3.0 * dx
    f_real = source_amplitude * np.exp(-x**2 / (2*width**2)) * np.cos(source_phase)
    f_imag = source_amplitude * np.exp(-x**2 / (2*width**2)) * np.sin(source_phase)
    f = (f_real + 1j * f_imag).astype(complex)
    # Eigenstrain source: A·u = f = -d/dx[E*·ε*]  (Eshelby form, complex E)
    if eps_star is not None:
        deps_dx = (np.roll(eps_star, -1) - np.roll(eps_star, 1)) / (2.0 * dx)
        f = f - (E_c * deps_dx).astype(complex)

    u = spsolve(A, f)

    # Normalise
    u_max = np.max(np.abs(u))
    if normalize and u_max > 1e-30:
        u = u / u_max

    if snr_db is not None:
        if rng is None:
            rng = np.random.default_rng()
        snr_lin = 10 ** (snr_db / 10.0)
        sig_power = float(np.mean(np.abs(u)**2))
        sigma_n = np.sqrt(sig_power / snr_lin / 2.0)
        u = u + sigma_n * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

    return u
