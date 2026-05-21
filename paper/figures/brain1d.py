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


# ── embedded-BC synthetic generator ──────────────────────────────────────────
#
# Real MRE acquisitions have unknown boundary conditions at the edges of the
# imaged region: the wave field continues into tissue beyond the FOV.  To
# match this in training we (a) randomise the BC type per sample, and
# (b) optionally solve on a domain larger than the inversion ROI and crop
# the central window.  The PDE-residual / SSL fine-tuning stage uses only
# interior pixels, so it is BC-agnostic by construction.


@dataclass
class BCSpec:
    """Boundary-condition specification at one end of the domain.

    kind: 'dirichlet' (u = value), 'neumann' (du/dx = value), or
          'absorbing' (Sommerfeld: outgoing wave with local wavenumber).
    value: complex amplitude (dirichlet) or gradient (neumann).
           Ignored for 'absorbing'.
    """
    kind: str    = "dirichlet"
    value: complex = 1.0 + 0.0j


def _apply_bc_row(A_lil, rhs, idx: int, side: str, bc: "BCSpec",
                  k_local: float, dx: float) -> None:
    """In-place: overwrite the row of A at `idx` to encode BC `bc`."""
    if bc.kind == "dirichlet":
        A_lil[idx, :] = 0
        A_lil[idx, idx] = 1.0 + 0.0j
        rhs[idx] = bc.value

    elif bc.kind == "neumann":
        # First-order one-sided FD for du/dx = bc.value
        A_lil[idx, :] = 0
        if side == "left":
            A_lil[idx, idx]     = -1.0 / dx
            A_lil[idx, idx + 1] =  1.0 / dx
        else:  # right
            A_lil[idx, idx]     =  1.0 / dx
            A_lil[idx, idx - 1] = -1.0 / dx
        rhs[idx] = bc.value

    elif bc.kind == "absorbing":
        # Sommerfeld outgoing-wave condition with local wavenumber k:
        #   left  end:  du/dx = -i k u   (wave leaves to the left)
        #   right end:  du/dx = +i k u   (wave leaves to the right)
        # First-order FD then folds the BC into the boundary row.
        A_lil[idx, :] = 0
        if side == "left":
            A_lil[idx, idx]     = (1.0 / dx) + 1j * k_local
            A_lil[idx, idx + 1] = -1.0 / dx
        else:
            A_lil[idx, idx]     = (1.0 / dx) - 1j * k_local
            A_lil[idx, idx - 1] = -1.0 / dx
        rhs[idx] = 0.0 + 0.0j

    else:
        raise ValueError(f"unknown BC kind: {bc.kind!r}")


def solve_helmholtz_1d_with_bcs(
    x: np.ndarray, G: np.ndarray, rho: float, freq: float,
    bc_left: "BCSpec", bc_right: "BCSpec",
) -> np.ndarray:
    """Generic 1D Helmholtz FD solver with arbitrary BCs at both ends.

    Same arithmetic-mean half-point averaging as the existing solvers.  The
    full system is built first (assuming generic boundaries), then the first
    and last rows are overwritten by the BC routine.
    """
    from scipy.sparse import diags as _diags
    from scipy.sparse.linalg import spsolve as _spsolve
    N    = len(x)
    dx   = float(x[1] - x[0])
    omega = 2.0 * np.pi * freq

    G_half_p = 0.5 * (G + np.roll(G, -1))
    G_half_m = 0.5 * (G + np.roll(G, 1))
    diag_main  = (G_half_p + G_half_m) / dx ** 2 - rho * omega ** 2
    diag_upper = -G_half_p / dx ** 2
    diag_lower = -G_half_m / dx ** 2

    A = _diags(
        [diag_lower[1:].astype(complex), diag_main.astype(complex),
         diag_upper[:-1].astype(complex)],
        offsets=[-1, 0, 1], format="lil", dtype=complex,
    )
    rhs = np.zeros(N, dtype=complex)

    # Local wavenumber at each boundary (used for absorbing BCs)
    k_left  = omega * np.sqrt(rho / float(np.abs(G[0])))
    k_right = omega * np.sqrt(rho / float(np.abs(G[-1])))
    _apply_bc_row(A, rhs, 0,     "left",  bc_left,  k_left,  dx)
    _apply_bc_row(A, rhs, N - 1, "right", bc_right, k_right, dx)

    return _spsolve(A.tocsr(), rhs)


def _G_extended_constant(x_ext: np.ndarray, roi_profile_fn, L_roi: float,
                          *args, **kwargs) -> np.ndarray:
    """Extend a G profile constant outside the inversion ROI [0, L_roi].

    `roi_profile_fn` is one of G_profile, G_profile_symmetric,
    G_profile_smooth_symmetric.  Its signature is (x, params).
    """
    x_clipped = np.clip(x_ext, 0.0, L_roi)
    return roi_profile_fn(x_clipped, *args, **kwargs)


def _extended_grid(N_roi: int, L_roi: float, ext_factor: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_ext, roi_mask, x_roi) for a grid extended by `ext_factor`
    in *both* directions (so the extended length is (1 + 2 ext_factor) L_roi).

    The grids are aligned so that the ROI corresponds to an exact subset of
    integer indices.
    """
    # spacing matches the ROI grid exactly
    dx = L_roi / (N_roi - 1)
    n_pad   = int(round(ext_factor * (N_roi - 1)))
    N_ext   = N_roi + 2 * n_pad
    x_ext   = np.linspace(-n_pad * dx, L_roi + n_pad * dx, N_ext)
    roi_idx = slice(n_pad, n_pad + N_roi)
    return x_ext, roi_idx, x_ext[roi_idx]


BC_KINDS = ("dirichlet", "neumann", "absorbing")


def _sample_bc(rng: np.random.Generator) -> "BCSpec":
    """Random BC: type uniformly from BC_KINDS, value as a unit complex."""
    kind = str(rng.choice(BC_KINDS))
    if kind == "dirichlet":
        value = complex(rng.standard_normal(), rng.standard_normal())
        # normalise so |value| ~ 1
        value = value / max(abs(value), 1e-12)
    elif kind == "neumann":
        # Gradient scale chosen so that the implied wave amplitude is ~ 1 / dx
        value = complex(rng.standard_normal(), rng.standard_normal())
    else:
        value = 0.0 + 0.0j
    return BCSpec(kind=kind, value=value)


PROFILE_FAMILIES = ("asym", "sym_euler", "sym_smooth")


def _sample_profile(rng: np.random.Generator, freq: float, rho: float,
                     L_roi: float):
    """Sample a random profile from one of the three families."""
    kind = str(rng.choice(PROFILE_FAMILIES))
    Gc = float(rng.uniform(800.0,  2500.0))
    Gb = float(rng.uniform(max(Gc + 300.0, 1500.0), 4500.0))
    xi = float(rng.uniform(0.05, 0.20))
    if kind == "asym":
        p = BrainParams(L=L_roi, G0=Gc, Gend=Gb, xi=xi, freq=freq, rho=rho)
        pf = G_profile
    elif kind == "sym_euler":
        p = SymmetricBrainParams(L=L_roi, Gc=Gc, Gb=Gb, xi=xi, freq=freq, rho=rho)
        pf = G_profile_symmetric
    else:
        p = SmoothSymmetricBrainParams(L=L_roi, Gc=Gc, Gb=Gb, xi=xi,
                                        freq=freq, rho=rho)
        pf = G_profile_smooth_symmetric
    return kind, p, pf, dict(Gc=Gc, Gb=Gb, xi=xi)


def make_embedded_sample(
    N_roi: int = 96, L_roi: float = L_DEFAULT, freq: float = FREQ_DEFAULT,
    rho: float = RHO_DEFAULT, ext_factor: float = 1.0, snr_db: float = 25.0,
    rng: np.random.Generator | None = None,
):
    """Generate one BC-randomised, domain-embedded training sample.

    Pipeline:
      1. Choose a profile family (asym / sym_euler / sym_smooth) and sample
         its parameters from the published-brain-MRE ranges.
      2. Build an extended grid that overhangs the ROI by `ext_factor` * L_roi
         on each side.  Extend G constant outside the ROI.
      3. Sample two random BCs (one per end).  At least one is not Dirichlet
         in expectation, exposing the network to BC types it would meet in
         vivo.
      4. Solve the FD Helmholtz on the extended domain.
      5. Crop the central window (the ROI) and treat that as the
         "observation": the network sees a wave field whose edge values are
         whatever the surrounding tissue + outer BC delivered, not a clean
         driver+clamped pair.
      6. Add complex-Gaussian noise at the requested SNR and rescale so
         ||u||_inf = 1.

    Returns
    -------
    u_roi : (N_roi,) complex64  -- noisy normalised wave in the ROI
    G_roi : (N_roi,) complex64  -- ground-truth modulus in the ROI
    meta  : dict                  -- sampled hyperparameters
    """
    if rng is None:
        rng = np.random.default_rng()

    family, p_obj, profile_fn, gparams = _sample_profile(rng, freq, rho, L_roi)
    x_ext, roi_idx, x_roi = _extended_grid(N_roi, L_roi, ext_factor)
    G_ext = _G_extended_constant(x_ext, profile_fn, L_roi, p_obj)

    bc_left  = _sample_bc(rng)
    bc_right = _sample_bc(rng)

    u_ext = solve_helmholtz_1d_with_bcs(x_ext, G_ext, rho, freq,
                                         bc_left, bc_right)
    u_roi = u_ext[roi_idx]
    G_roi = G_ext[roi_idx]

    # Complex Gaussian noise + normalise to unit max
    sig_power = float(np.mean(np.abs(u_roi) ** 2))
    if sig_power > 0.0:
        sigma_n = float(np.sqrt(sig_power / (10.0 ** (snr_db / 10.0)) / 2.0))
        u_roi = u_roi + sigma_n * (rng.standard_normal(N_roi)
                                    + 1j * rng.standard_normal(N_roi))
    u_roi = u_roi / max(np.abs(u_roi).max(), 1e-12)

    meta = {
        "family":    family,
        "bc_left":   bc_left.kind,
        "bc_right":  bc_right.kind,
        "Gc":        gparams["Gc"],
        "Gb":        gparams["Gb"],
        "xi":        gparams["xi"],
        "ext_factor": ext_factor,
        "snr_db":    snr_db,
        "freq":      freq,
    }
    return u_roi.astype(np.complex64), G_roi.astype(np.complex64), meta
