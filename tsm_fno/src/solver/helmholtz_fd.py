"""2D time-harmonic Helmholtz solver with multi-source Dirichlet excitation.

Reproduces Phase 0's `mre_pipeline/src/fem_solver.py` and adds a
`random_source_phase` argument so each sample can use an independent
phase offset on the prescribed boundary, which the prompt requires for
the TSM training distribution.

    ∇·(G* ∇u) + ρω² u = 0,   G* = G(1 + iξ)

with second-order finite differences and half-point harmonic averaging
at material interfaces. All boundaries are Dirichlet; non-source
boundary nodes are u = 0.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def helmholtz_solve(
    G: np.ndarray,
    freq: float = 60.0,
    rho: float = 1000.0,
    dx: float = 0.002,
    damping: float = 0.05,
    sources: list[tuple[int, int, complex]] | None = None,
) -> np.ndarray:
    """Solve the 2D scalar Helmholtz equation for the complex shear field u.

    Parameters
    ----------
    G : (N, N) ndarray
        Spatially varying shear modulus [Pa].
    freq : float
        Drive frequency [Hz].
    rho : float
        Mass density [kg/m^3].
    dx : float
        Voxel pitch [m].
    damping : float
        Loss tangent ξ in G* = G(1 + iξ).
    sources : list of (i, j, complex_amplitude), optional
        Dirichlet point sources (or patch nodes). When None the legacy
        Phase 0 left-column source u = 1 is used.
    """
    N = G.shape[0]
    assert G.shape == (N, N)
    omega = 2.0 * np.pi * freq
    Gc = G * (1.0 + 1j * damping)

    def idx(i, j):
        return i * N + j

    boundary: set[int] = set()
    bc_values: dict[int, complex] = {}
    for i in range(N):
        boundary.add(idx(i, 0))
        boundary.add(idx(i, N - 1))
    for j in range(N):
        boundary.add(idx(0, j))
        boundary.add(idx(N - 1, j))

    if sources is None:
        for i in range(N):
            bc_values[idx(i, 0)] = 1.0 + 0.0j
    else:
        for (i, j, amp) in sources:
            k = idx(i, j)
            bc_values[k] = complex(amp)
            boundary.add(k)

    n_dof = N * N
    A = lil_matrix((n_dof, n_dof), dtype=complex)
    b = np.zeros(n_dof, dtype=complex)

    def G_half(ga, gb):
        return 2.0 * ga * gb / (ga + gb)

    for i in range(N):
        for j in range(N):
            k = idx(i, j)
            if k in boundary:
                A[k, k] = 1.0
                b[k] = bc_values.get(k, 0.0 + 0.0j)
                continue
            g_c = Gc[i, j]
            g_e = G_half(g_c, Gc[i, j + 1]) if j + 1 < N else g_c
            g_w = G_half(g_c, Gc[i, j - 1]) if j - 1 >= 0 else g_c
            g_n = G_half(g_c, Gc[i - 1, j]) if i - 1 >= 0 else g_c
            g_s = G_half(g_c, Gc[i + 1, j]) if i + 1 < N else g_c
            diag = -(g_e + g_w + g_n + g_s) / dx ** 2 + rho * omega ** 2
            A[k, k] = diag
            A[k, idx(i, j + 1)] = g_e / dx ** 2
            A[k, idx(i, j - 1)] = g_w / dx ** 2
            A[k, idx(i - 1, j)] = g_n / dx ** 2
            A[k, idx(i + 1, j)] = g_s / dx ** 2

    u_flat = spsolve(A.tocsr(), b)
    return u_flat.reshape(N, N)


def helmholtz_eshelby_solve(
    G: np.ndarray,
    eps_star: np.ndarray,
    center: tuple[float, float],
    freq: float = 80.0,
    rho: float = 1000.0,
    dx: float = 0.003,
    damping: float = 0.05,
    sources: list[tuple[int, int, complex]] | None = None,
) -> np.ndarray:
    """Solve ∇·[G*(x)(∇u − ε̄*(x))] + ρω²u = 0 (Eshelby inclusion form).

    The eigenstrain ε̄*(x) is radially oriented from ``center``:
        ε*_x(i,j) = eps_star[i,j] * (j − cx) / r(i,j)
        ε*_y(i,j) = eps_star[i,j] * (i − cy) / r(i,j)

    The stiffness matrix A is identical to ``helmholtz_solve``; the
    eigenstrain contributes a divergence source term to the RHS:
        f*(i,j) = [G*(i,j+½)ε*_x(j+½) − G*(i,j-½)ε*_x(j-½)] / dx
                + [G*(i+½,j)ε*_y(i+½) − G*(i-½,j)ε*_y(i-½)] / dx

    Parameters
    ----------
    G : (N, N)  Intrinsic shear modulus (no acoustoelastic folding).
    eps_star : (N, N)  Scalar eigenstrain magnitude ε*(x) = A·Δσ/G_bg.
    center : (cy, cx) in grid-pixel coordinates (row, col).
    """
    N = G.shape[0]
    assert G.shape == (N, N) and eps_star.shape == (N, N)
    omega = 2.0 * np.pi * freq
    Gc = G * (1.0 + 1j * damping)

    cy, cx = center

    # Build radial unit-vector components for eigenstrain orientation
    rows, cols = np.mgrid[0:N, 0:N].astype(float)
    dy = rows - cy
    dx_r = cols - cx
    r = np.sqrt(dy ** 2 + dx_r ** 2)
    r[r < 1e-12] = 1e-12          # avoid division by zero at lesion center
    ex = eps_star * dx_r / r       # x-component (column direction)
    ey = eps_star * dy / r         # y-component (row direction)

    def idx(i, j):
        return i * N + j

    boundary: set[int] = set()
    bc_values: dict[int, complex] = {}
    for i in range(N):
        boundary.add(idx(i, 0))
        boundary.add(idx(i, N - 1))
    for j in range(N):
        boundary.add(idx(0, j))
        boundary.add(idx(N - 1, j))

    if sources is None:
        for i in range(N):
            bc_values[idx(i, 0)] = 1.0 + 0.0j
    else:
        for (i, j, amp) in sources:
            k = idx(i, j)
            bc_values[k] = complex(amp)
            boundary.add(k)

    n_dof = N * N
    A_mat = lil_matrix((n_dof, n_dof), dtype=complex)
    b = np.zeros(n_dof, dtype=complex)

    def G_half(ga, gb):
        return 2.0 * ga * gb / (ga + gb)

    for i in range(N):
        for j in range(N):
            k = idx(i, j)
            if k in boundary:
                A_mat[k, k] = 1.0
                b[k] = bc_values.get(k, 0.0 + 0.0j)
                continue

            g_c = Gc[i, j]
            g_e = G_half(g_c, Gc[i, j + 1]) if j + 1 < N else g_c
            g_w = G_half(g_c, Gc[i, j - 1]) if j - 1 >= 0 else g_c
            g_n = G_half(g_c, Gc[i - 1, j]) if i - 1 >= 0 else g_c
            g_s = G_half(g_c, Gc[i + 1, j]) if i + 1 < N else g_c

            diag = -(g_e + g_w + g_n + g_s) / dx ** 2 + rho * omega ** 2
            A_mat[k, k] = diag
            A_mat[k, idx(i, j + 1)] = g_e / dx ** 2
            A_mat[k, idx(i, j - 1)] = g_w / dx ** 2
            A_mat[k, idx(i - 1, j)] = g_n / dx ** 2
            A_mat[k, idx(i + 1, j)] = g_s / dx ** 2

            # Eigenstrain RHS: f*(i,j) = ∇·[G* ε̄*] at (i,j)
            # x-flux: G*(i,j+½)·ex(i,j+½) − G*(i,j-½)·ex(i,j-½)
            ex_e = 0.5 * (ex[i, j] + ex[i, j + 1]) if j + 1 < N else ex[i, j]
            ex_w = 0.5 * (ex[i, j] + ex[i, j - 1]) if j - 1 >= 0 else ex[i, j]
            ey_s = 0.5 * (ey[i, j] + ey[i + 1, j]) if i + 1 < N else ey[i, j]
            ey_n = 0.5 * (ey[i, j] + ey[i - 1, j]) if i - 1 >= 0 else ey[i, j]

            f_star = (g_e * ex_e - g_w * ex_w) / dx + (g_s * ey_s - g_n * ey_n) / dx
            b[k] += f_star   # moves to RHS: A u = b + f*

    u_flat = spsolve(A_mat.tocsr(), b)
    return u_flat.reshape(N, N)


def solve_two_frequencies(
    G: np.ndarray,
    freq1: float = 60.0,
    freq2: float = 120.0,
    rho: float = 1000.0,
    dx: float = 0.002,
    damping: float = 0.05,
    sources: list[tuple[int, int, complex]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: solve at two frequencies and return both fields."""
    u1 = helmholtz_solve(G, freq=freq1, rho=rho, dx=dx, damping=damping, sources=sources)
    u2 = helmholtz_solve(G, freq=freq2, rho=rho, dx=dx, damping=damping, sources=sources)
    return u1, u2


def random_sources(
    N: int,
    rng: np.random.Generator,
    n_min: int = 1,
    n_max: int = 10,
    patch_min: int = 4,
    patch_max: int = 12,
    random_phase: bool = True,
) -> list[tuple[int, int, complex]]:
    """1–10 random source patches on the boundary, each with random complex amplitude."""
    n_src = int(rng.integers(n_min, n_max + 1))
    sources: list[tuple[int, int, complex]] = []
    used: set[tuple[int, int]] = set()
    for _ in range(n_src):
        edge = int(rng.integers(0, 4))
        patch_len = int(rng.integers(patch_min, patch_max + 1))
        start = int(rng.integers(0, N - patch_len))
        phase = float(rng.uniform(0, 2 * np.pi)) if random_phase else 0.0
        amp = complex(np.exp(1j * phase))
        for k in range(patch_len):
            if edge == 0:
                i, j = 0, start + k
            elif edge == 1:
                i, j = N - 1, start + k
            elif edge == 2:
                i, j = start + k, 0
            else:
                i, j = start + k, N - 1
            if (i, j) in used:
                continue
            used.add((i, j))
            sources.append((i, j, amp))
    return sources
