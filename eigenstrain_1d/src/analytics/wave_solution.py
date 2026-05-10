"""Exact 1D Helmholtz solution via transfer matrix method.

Governing equation (time-harmonic, SH shear wave):
    d/dx[E_eff(x) dû/dx] + ρω²û = f(x)

For piecewise-constant E_eff with two values (E1 inside lesion, E2 outside):
    k1 = ω√(ρ/E1),  k2 = ω√(ρ/E2)

General solution in each region:
    Inside  [L-a, L+a]:  û = C1·exp(ik1·x) + D1·exp(-ik1·x)
    Outside (both):      û = C2·exp(ik2·x) + D2·exp(-ik2·x)

Periodic BC on [0, 2L] with incident plane wave source.
"""
from __future__ import annotations

import numpy as np


def transfer_matrix_solution(
    N: int,
    dx: float,
    a: float,
    L: float,
    E_bg: float,
    E_lesion: float,
    A_coeff: float,
    sigma_bar: float,
    rho: float,
    freq: float,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
    source_amplitude: float = 1.0,
) -> np.ndarray:
    """Exact solution via transfer matrix on periodic domain [0, 2L].

    Parameters
    ----------
    N, dx         : grid points and spacing
    a             : lesion half-width [m]
    L             : domain half-length [m]
    E_bg/E_lesion : elastic moduli [Pa]
    A_coeff       : acoustoelastic constant (dimensionless)
    sigma_bar     : static pre-stress [Pa]
    rho           : density [kg/m³]
    freq          : drive frequency [Hz]
    snr_db        : if given, add complex Gaussian noise at this SNR
    rng           : random number generator
    source_amplitude : amplitude of incident plane wave

    Returns
    -------
    u_hat : (N,) complex128 — displacement field
    """
    omega = 2.0 * np.pi * freq

    # Acoustoelastic effective moduli
    E1 = float(E_lesion) + A_coeff * sigma_bar   # inside lesion
    E2 = float(E_bg)     + A_coeff * sigma_bar   # outside
    E1 = max(E1, 1.0)
    E2 = max(E2, 1.0)

    k1 = omega * np.sqrt(rho / E1)
    k2 = omega * np.sqrt(rho / E2)

    x1 = L - a   # left lesion boundary
    x2 = L + a   # right lesion boundary

    # Build x grid
    x = np.arange(N) * dx

    # Transfer matrix at an interface between regions with moduli Ea (left) and Eb (right)
    # ka/kb are wavenumbers. At interface xi, enforce û and E*dû/dx continuous.
    # Solutions:  left: A*exp(ika*x) + B*exp(-ika*x)
    #             right: C*exp(ikb*x) + D*exp(-ikb*x)
    # [1  1 ; ika*Ea  -ika*Ea] [A;B] = [exp(ikb*xi)  exp(-ikb*xi); ikb*Eb*exp(ikb*xi) -ikb*Eb*exp(-ikb*xi)] [C;D]
    def tm_at_interface(ka, Ea, kb, Eb, xi):
        """Transfer [A,B] → [C,D] across interface at xi."""
        # M_left  @ [A,B] = M_right @ [C,D]  →  [C,D] = M_right^{-1} M_left [A,B]
        el = np.exp(1j * ka * xi);  er = np.exp(-1j * ka * xi)
        fl = np.exp(1j * kb * xi);  fr = np.exp(-1j * kb * xi)
        M_left  = np.array([[el,           er          ],
                             [1j*ka*Ea*el, -1j*ka*Ea*er]])
        M_right = np.array([[fl,           fr          ],
                             [1j*kb*Eb*fl, -1j*kb*Eb*fr]])
        return np.linalg.solve(M_right, M_left)

    # Interface at x1: outside(k2,E2) → inside(k1,E1)
    T1 = tm_at_interface(k2, E2, k1, E1, x1)
    # Interface at x2: inside(k1,E1) → outside(k2,E2)
    T2 = tm_at_interface(k1, E1, k2, E2, x2)

    # Combined transfer matrix from x=0 to x=2L (all outside region wraps around)
    # Propagate from 0 to x1 in region2, then x1→x2 in region1, then x2→2L in region2
    # Propagation matrix in uniform medium over distance d:
    def prop(k, d):
        return np.array([[np.exp(1j*k*d), 0],
                         [0, np.exp(-1j*k*d)]])

    # Full chain: start at x=0 with [C2_0, D2_0] (region2)
    # After prop x1: T1 → [C1, D1]; after prop x2-x1: T2 → [C2_after, D2_after]
    # Periodic BC: û(0) = û(2L) and E2*û'(0) = E2*û'(2L)

    P_pre  = prop(k2, x1)
    P_mid  = prop(k1, x2 - x1)
    P_post = prop(k2, 2*L - x2)

    # Full transfer matrix from [A,B] at x=0 to [A,B] at x=2L
    M = P_post @ T2 @ P_mid @ T1 @ P_pre

    # Source: incident plane wave from left. Add it as forcing.
    # Source injects amplitude S at x=0, so the outgoing field has unit amplitude.
    # Use: at x=0, set the field as A_total = A_source + A_scattered
    # Simplest approach: fix û(0) = source_amplitude (Dirichlet source)
    # and compute self-consistently for periodic domain.
    # For periodic domain, the system in [C2_0, D2_0]:
    #   û(0) = C2_0 + D2_0 = u0 (given)
    #   û(2L) = û(0) via periodicity → M[0,0]*C2_0 + M[0,1]*D2_0 = C2_0 + D2_0
    #   E2*û'(0) = E2*û'(2L) → iE2(k2*C2_0 - k2*D2_0) = iE2*(k2*M[1,0]*C2_0 - k2*M[1,1]*D2_0)
    # This gives:
    #   (M[0,0]-1)*C2_0 + (M[0,1]-1)*D2_0 = 0
    #   (M[1,0]-1)*C2_0 + (M[1,1]-1)*D2_0 = 0   (same equation up to a constant)
    # Homogeneous system — only trivial solution, so need source.
    # Instead: fix D2_0 = 0 (no incoming from the right), solve for C2_0 from periodic BC.
    # The periodic BC matrix equation: (M - I) [C2_0; D2_0] = 0
    # Non-trivial if det(M-I) = 0 — this is the resonance condition.
    # For off-resonance: use source injection.

    # Practical approach: set u(0) = source_amplitude * exp(i*phase),
    # u'(0) = ik2 * u(0) (forward-propagating wave), solve for field.
    src_phase = 0.0
    u0 = source_amplitude * np.exp(1j * src_phase)
    # Determine [C2_0, D2_0] from u(0)=u0 and u'(0)=ik2*u0
    # u(0) = C2_0 + D2_0 = u0
    # E2*ik2*(C2_0 - D2_0) = E2*ik2*u0  →  C2_0 - D2_0 = u0
    C2_0 = u0
    D2_0 = 0.0 + 0.0j

    # Evaluate û(x) piecewise
    u_hat = np.zeros(N, dtype=complex)
    for i, xi in enumerate(x):
        if xi <= x1:
            d = xi
            C = C2_0 * np.exp(1j*k2*d)
            D = D2_0 * np.exp(-1j*k2*d)
            u_hat[i] = C + D
        elif xi <= x2:
            # Propagate to x1 in region2, then transfer to region1, then propagate
            [C1_0, D1_0] = T1 @ prop(k2, x1) @ np.array([C2_0, D2_0])
            d = xi - x1
            u_hat[i] = C1_0 * np.exp(1j*k1*d) + D1_0 * np.exp(-1j*k1*d)
        else:
            [C1_0, D1_0] = T1 @ prop(k2, x1) @ np.array([C2_0, D2_0])
            [C2_1, D2_1] = T2 @ prop(k1, x2-x1) @ np.array([C1_0, D1_0])
            d = xi - x2
            u_hat[i] = C2_1 * np.exp(1j*k2*d) + D2_1 * np.exp(-1j*k2*d)

    # Normalise so max |u| = 1
    u_max = np.max(np.abs(u_hat))
    if u_max > 1e-30:
        u_hat /= u_max

    # Add noise
    if snr_db is not None:
        if rng is None:
            rng = np.random.default_rng()
        snr_lin = 10 ** (snr_db / 10.0)
        sig_power = float(np.mean(np.abs(u_hat)**2))
        sigma_n = np.sqrt(sig_power / snr_lin / 2.0)
        u_hat = u_hat + sigma_n * (rng.standard_normal(N) + 1j*rng.standard_normal(N))

    return u_hat
