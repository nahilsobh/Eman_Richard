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


# ── symmetric (mirrored Euler) variant ───────────────────────────────────────

@dataclass
class SymmetricBrainParams:
    """Symmetric power-law profile G(x) = Gc (1 + alpha |x - L/2|)^2 (1 + i xi).

    G is smallest at the centre (Gc) and largest at the two boundaries
    (Gb = Gc * (1 + alpha L/2)^2).
    """
    L:    float = L_DEFAULT
    Gc:   float = 1500.0     # Pa, centre (minimum)
    Gb:   float = 3500.0     # Pa, boundaries (maximum)
    xi:   float = XI_DEFAULT
    rho:  float = RHO_DEFAULT
    freq: float = FREQ_DEFAULT

    @property
    def alpha(self) -> float:
        # Gb = Gc (1 + alpha L/2)^2  ->  alpha = 2 (sqrt(Gb/Gc) - 1) / L
        return 2.0 * (np.sqrt(self.Gb / self.Gc) - 1.0) / self.L

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.freq


def G_profile_symmetric(x: np.ndarray, p: SymmetricBrainParams) -> np.ndarray:
    """Symmetric complex shear modulus, V-shape with minimum at x = L/2.

    G(x) = Gc * (1 + alpha |x - L/2|)^2 * (1 + i xi).  C^0 at x=L/2 but
    G'(x) is discontinuous (kink).  See ``SmoothSymmetricBrainParams``
    for the C^infty (quadratic) variant.
    """
    zeta = x - 0.5 * p.L
    base = p.Gc * (1.0 + p.alpha * np.abs(zeta)) ** 2
    return base * (1.0 + 1j * p.xi)


# ── smooth symmetric (quadratic) variant ──────────────────────────────────────

@dataclass
class SmoothSymmetricBrainParams:
    """Smooth symmetric profile  G(x) = Gc + beta (x - L/2)^2  with G' and G''
    continuous everywhere (no absolute values).  Min at the centre (Gc),
    max at the boundaries (Gb = Gc + beta (L/2)^2).
    """
    L:    float = L_DEFAULT
    Gc:   float = 1500.0
    Gb:   float = 3500.0
    xi:   float = XI_DEFAULT
    rho:  float = RHO_DEFAULT
    freq: float = FREQ_DEFAULT

    @property
    def beta(self) -> float:
        # Gb = Gc + beta (L/2)^2  ->  beta = 4 (Gb - Gc) / L^2
        return 4.0 * (self.Gb - self.Gc) / self.L ** 2

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.freq


def G_profile_smooth_symmetric(x: np.ndarray,
                                p: SmoothSymmetricBrainParams) -> np.ndarray:
    """Smooth symmetric complex modulus G(x) = (Gc + beta zeta^2)(1 + i xi)."""
    zeta = x - 0.5 * p.L
    return (p.Gc + p.beta * zeta ** 2) * (1.0 + 1j * p.xi)


def _power_series_solution(zeta: np.ndarray, a: complex, b: complex,
                            rhsq: complex, n_terms: int = 80
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Even and odd power-series solutions of
        (a + b zeta^2) u'' + 2 b zeta u' + rhsq u = 0.

    The recurrence is
        c_{k+2} = -( b k(k+1) + rhsq ) / ( a (k+2)(k+1) ) * c_k.
    The 'even' solution sets c_0 = 1, c_1 = 0; the 'odd' solution sets
    c_0 = 0, c_1 = 1.  Returns (u_even(zeta), u_odd(zeta)).
    """
    zeta = np.asarray(zeta, dtype=complex)
    # Even series: only even powers
    c_even = np.zeros(n_terms, dtype=complex)
    c_even[0] = 1.0 + 0.0j
    for k in range(0, n_terms - 2):
        if (k % 2) == 0:
            c_even[k + 2] = -(b * k * (k + 1) + rhsq) / (a * (k + 2) * (k + 1)) * c_even[k]
    # Odd series: only odd powers
    c_odd = np.zeros(n_terms, dtype=complex)
    c_odd[1] = 1.0 + 0.0j
    for k in range(1, n_terms - 2):
        if (k % 2) == 1:
            c_odd[k + 2] = -(b * k * (k + 1) + rhsq) / (a * (k + 2) * (k + 1)) * c_odd[k]

    # Evaluate by Horner-style accumulation over zeta
    u_even = np.zeros_like(zeta, dtype=complex)
    u_odd  = np.zeros_like(zeta, dtype=complex)
    zk = np.ones_like(zeta, dtype=complex)
    for k in range(n_terms):
        u_even = u_even + c_even[k] * zk
        u_odd  = u_odd  + c_odd[k]  * zk
        zk = zk * zeta
    return u_even, u_odd


def analytical_solution_smooth_symmetric(
    x: np.ndarray, p: SmoothSymmetricBrainParams,
    u0: complex = 1.0, n_terms: int = 80,
) -> np.ndarray:
    """Closed-form (power-series) u(x) for the smooth symmetric profile with
    Dirichlet BCs u(0) = u0 and u(L) = 0.

    The wave equation
        (a + b zeta^2) u''(zeta) + 2b zeta u'(zeta) + rho omega^2 u(zeta) = 0
    with a = Gc(1+i xi), b = beta(1+i xi), has two independent series
    solutions u_even (symmetric about zeta=0) and u_odd (antisymmetric).
    The general u = A u_even + B u_odd; A and B are fixed by the two
    boundary conditions u(-L/2) = u0 and u(L/2) = 0.

    Because u_even(-z) = u_even(z) and u_odd(-z) = -u_odd(z),
        u(-L/2) = A u_even(L/2) - B u_odd(L/2) = u0
        u(+L/2) = A u_even(L/2) + B u_odd(L/2) = 0
    so A = u0 / (2 u_even(L/2)),  B = -u0 / (2 u_odd(L/2)).
    """
    a    = p.Gc   * (1.0 + 1j * p.xi)
    b    = p.beta * (1.0 + 1j * p.xi)
    rhsq = p.rho  * p.omega ** 2

    zeta_field = x - 0.5 * p.L
    z_half     = np.array([0.5 * p.L], dtype=float)
    u_e_field, u_o_field = _power_series_solution(zeta_field, a, b, rhsq, n_terms)
    u_e_h,     u_o_h     = _power_series_solution(z_half,     a, b, rhsq, n_terms)
    A =  u0 / (2.0 * u_e_h[0])
    B = -u0 / (2.0 * u_o_h[0])
    return A * u_e_field + B * u_o_field


def numerical_solution_smooth_symmetric(
    N: int, p: SmoothSymmetricBrainParams, u0: complex = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """FD solver for the smooth symmetric problem, Dirichlet u(0)=u0, u(L)=0."""
    x  = np.linspace(0.0, p.L, N)
    dx = x[1] - x[0]
    G  = G_profile_smooth_symmetric(x, p)

    G_half_p = 0.5 * (G + np.roll(G, -1))
    G_half_m = 0.5 * (G + np.roll(G, 1))
    diag_main  = (G_half_p + G_half_m) / dx ** 2 - p.rho * p.omega ** 2
    diag_upper = -G_half_p / dx ** 2
    diag_lower = -G_half_m / dx ** 2

    from scipy.sparse import diags as _diags
    from scipy.sparse.linalg import spsolve as _spsolve

    A = _diags(
        [diag_lower[1:].astype(complex), diag_main.astype(complex),
         diag_upper[:-1].astype(complex)],
        offsets=[-1, 0, 1], format="lil", dtype=complex,
    )
    A[0, :] = 0;  A[0, 0]   = 1.0
    A[-1, :] = 0; A[-1, -1] = 1.0
    A = A.tocsr()

    rhs = np.zeros(N, dtype=complex)
    rhs[0]  = u0
    rhs[-1] = 0.0
    u = _spsolve(A, rhs)
    return x, u


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

def analytical_solution_symmetric(
    x: np.ndarray, p: SymmetricBrainParams, u0: complex = 1.0
) -> np.ndarray:
    """Closed-form u(x) for the symmetric profile with u(0)=u0, u(L)=0.

    Each half admits the same Euler-form solution u = A s^p1 + B s^p2 with
    s = 1 + alpha|x - L/2|. The four constants (A_L, B_L, A_R, B_R) are
    fixed by two Dirichlet BCs plus continuity of u and of the traction
    G du/dx at the centre.
    """
    nu2 = p.rho * p.omega ** 2 / (p.alpha ** 2 * p.Gc * (1.0 + 1j * p.xi))
    mu  = np.sqrt(nu2 - 0.25 + 0j)
    p1, p2 = -0.5 + 1j * mu, -0.5 - 1j * mu

    sB  = 1.0 + p.alpha * 0.5 * p.L     # s at the two boundaries (and at x=0, x=L)
    sM  = 1.0                            # s at the centre x = L/2

    # 4x4 linear system in [A_L, B_L, A_R, B_R]:
    #   1: A_L sB^p1 + B_L sB^p2                          = u0     (u(0)=u0)
    #   2:                       A_R sB^p1 + B_R sB^p2    = 0      (u(L)=0)
    #   3: A_L sM^p1 + B_L sM^p2 - A_R sM^p1 - B_R sM^p2  = 0      (continuity of u)
    #   4: -p1 A_L - p2 B_L      - p1 A_R - p2 B_R        = 0      (continuity of G u_x;
    #                                                                left side has
    #                                                                ds/dx = -alpha)
    M = np.array([
        [sB ** p1, sB ** p2,    0,          0       ],
        [0,         0,           sB ** p1,  sB ** p2],
        [sM ** p1, sM ** p2,    -sM ** p1, -sM ** p2],
        [-p1,      -p2,          -p1,       -p2     ],
    ], dtype=complex)
    rhs = np.array([u0, 0.0, 0.0, 0.0], dtype=complex)
    A_L, B_L, A_R, B_R = np.linalg.solve(M, rhs)

    zeta = x - 0.5 * p.L
    s    = 1.0 + p.alpha * np.abs(zeta)
    is_L = zeta < 0
    u = np.where(
        is_L,
        A_L * s ** p1 + B_L * s ** p2,
        A_R * s ** p1 + B_R * s ** p2,
    )
    return u


def numerical_solution_symmetric(
    N: int, p: SymmetricBrainParams, u0: complex = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """FD solver for the symmetric problem, Dirichlet BCs u(0)=u0, u(L)=0."""
    x  = np.linspace(0.0, p.L, N)
    dx = x[1] - x[0]
    G  = G_profile_symmetric(x, p)

    G_half_p = 0.5 * (G + np.roll(G, -1))
    G_half_m = 0.5 * (G + np.roll(G, 1))
    diag_main  = (G_half_p + G_half_m) / dx ** 2 - p.rho * p.omega ** 2
    diag_upper = -G_half_p / dx ** 2
    diag_lower = -G_half_m / dx ** 2

    from scipy.sparse import diags as _diags
    from scipy.sparse.linalg import spsolve as _spsolve

    A = _diags(
        [diag_lower[1:].astype(complex), diag_main.astype(complex),
         diag_upper[:-1].astype(complex)],
        offsets=[-1, 0, 1], format="lil", dtype=complex,
    )
    A[0, :] = 0;  A[0, 0]   = 1.0
    A[-1, :] = 0; A[-1, -1] = 1.0
    A = A.tocsr()

    rhs = np.zeros(N, dtype=complex)
    rhs[0]  = u0
    rhs[-1] = 0.0
    u = _spsolve(A, rhs)
    return x, u


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
