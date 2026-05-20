"""1D brain-scale Helmholtz problem with continuous complex G(x).

Governing equation (time-harmonic shear wave, 1D):
    d/dx [ G(x) du/dx ] + rho * omega^2 * u = 0,    x in [0, L]

With the power-law profile
    G(x) = G0 * (1 + alpha*x)^2 * (1 + i*xi)
the substitution s = 1 + alpha*x reduces the equation to the Euler form
    s^2 u''(s) + 2 s u'(s) + nu^2 u(s) = 0,
    nu^2 = rho * omega^2 / ( alpha^2 * G0 * (1 + i*xi) ).

The general solution is
    u(s) = A * s^p1 + B * s^p2,    p1,p2 = -1/2 +- i*mu,
    mu = sqrt(nu^2 - 1/4).

Boundary conditions for the analytical / numerical match:
    u(0) = 1            (driver displacement)
    u(L) = 0            (clamped distal boundary)

Physical scales chosen to match published in vivo brain MRE:
    L  = 0.16 m   (~human head)
    G0 = 1500 Pa, alpha chosen so G(L) ~ 3500 Pa (cortex -> deep grey)
    xi = 0.10     (loss tangent; published brain values 0.05-0.20)
    f  = 50 Hz    (pneumatic driver, e.g. Heras Rivera 2025 oNLI)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ── physical defaults ─────────────────────────────────────────────────────────

L_DEFAULT     = 0.16        # m, ~ human head front-to-back
G0_DEFAULT    = 1500.0      # Pa, white matter
G_END_DEFAULT = 3500.0      # Pa, deep grey matter
XI_DEFAULT    = 0.10        # loss tangent
RHO_DEFAULT   = 1000.0      # kg/m^3
FREQ_DEFAULT  = 50.0        # Hz


@dataclass
class BrainParams:
    L:    float = L_DEFAULT
    G0:   float = G0_DEFAULT
    Gend: float = G_END_DEFAULT
    xi:   float = XI_DEFAULT
    rho:  float = RHO_DEFAULT
    freq: float = FREQ_DEFAULT

    @property
    def alpha(self) -> float:
        # G(L) = G0 (1 + alpha L)^2  ->  alpha = (sqrt(Gend/G0) - 1) / L
        return (np.sqrt(self.Gend / self.G0) - 1.0) / self.L

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.freq


def G_profile(x: np.ndarray, p: BrainParams) -> np.ndarray:
    """Complex shear modulus G(x) = G0 (1 + alpha x)^2 (1 + i xi)."""
    base = p.G0 * (1.0 + p.alpha * x) ** 2
    return base * (1.0 + 1j * p.xi)


# ── analytical solution ──────────────────────────────────────────────────────

def analytical_solution(x: np.ndarray, p: BrainParams) -> np.ndarray:
    """Closed-form u(x) for the power-law G profile with u(0)=1, u(L)=0."""
    s = 1.0 + p.alpha * x
    s_L = 1.0 + p.alpha * p.L

    # Complex nu^2 because of the loss tangent
    nu2 = p.rho * p.omega ** 2 / (p.alpha ** 2 * p.G0 * (1.0 + 1j * p.xi))
    mu = np.sqrt(nu2 - 0.25 + 0j)   # complex; principal branch
    p1, p2 = -0.5 + 1j * mu, -0.5 - 1j * mu

    # BCs: A + B = 1; A s_L^p1 + B s_L^p2 = 0  =>  B = -A s_L^(p1-p2)
    delta = s_L ** (p1 - p2)
    A = 1.0 / (1.0 - delta)
    B = 1.0 - A

    return A * s ** p1 + B * s ** p2


# ── numerical solver (Dirichlet-Dirichlet FD with variable G) ────────────────

def numerical_solution(N: int, p: BrainParams) -> tuple[np.ndarray, np.ndarray]:
    """Second-order FD discretisation of d/dx[G(x) du/dx] + rho omega^2 u = 0,
    with u(0)=1 and u(L)=0 enforced on cell centres at x_j = j * dx,
    j = 0, 1, ..., N-1, dx = L/(N-1).

    Returns (x, u) with shapes (N,), (N,) complex.
    """
    x  = np.linspace(0.0, p.L, N)
    dx = x[1] - x[0]
    G  = G_profile(x, p)

    # Half-point moduli using the harmonic mean preserves flux for jumps
    # but here G is smooth so arithmetic mean is fine and consistent with
    # the analytical operator written above:
    G_half_p = 0.5 * (G + np.roll(G, -1))     # G_{j+1/2}
    G_half_m = 0.5 * (G + np.roll(G, 1))      # G_{j-1/2}

    diag_main  = (G_half_p + G_half_m) / dx ** 2 - p.rho * p.omega ** 2
    diag_upper = -G_half_p / dx ** 2          # links u_j -> u_{j+1}
    diag_lower = -G_half_m / dx ** 2          # links u_j -> u_{j-1}

    # Enforce Dirichlet BCs by overwriting first and last rows
    # u_0 = 1: row 0 = [1, 0, 0, ..., 0],  RHS_0 = 1
    # u_{N-1} = 0: row N-1 = [0, ..., 0, 1],  RHS_{N-1} = 0
    Adiag = diag_main.astype(complex)
    Aup   = diag_upper[:-1].astype(complex)   # upper diag has length N-1
    Alo   = diag_lower[1:].astype(complex)    # lower diag has length N-1

    A = diags(
        [Alo, Adiag, Aup],
        offsets=[-1, 0, 1],
        format="lil",
        dtype=complex,
    )

    # Dirichlet rows
    A[0, :] = 0;     A[0, 0] = 1.0
    A[-1, :] = 0;    A[-1, -1] = 1.0
    A = A.tocsr()

    rhs = np.zeros(N, dtype=complex)
    rhs[0]  = 1.0
    rhs[-1] = 0.0

    u = spsolve(A, rhs)
    return x, u
