"""3D time-harmonic Helmholtz solver — the 3D analogue of ``helmholtz_fd``.

Solves the scalar shear-wave equation
    ∇·(G* ∇u) + ρω² u = 0,   G* = G(1 + iξ)
on a regular Cartesian grid with a 7-point finite-difference stencil and
half-point harmonic averaging of G at material interfaces.

Boundary conditions
-------------------
By default all six faces are Dirichlet u = 0 (a fully clamped cube).
Pass ``top_free=True`` to switch the top face (i = 0 slab, excluding
edges) to a traction-free Neumann BC ∂u/∂z = 0 — the physically correct
BC for a container open at the top. Edges of the top face remain
Dirichlet u = 0 to keep the corner behavior well-defined.

Sources are Dirichlet overrides at listed (i, j, k) nodes and take
precedence over the face BC (including the free-top face).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def helmholtz_solve_3d(
    G: np.ndarray,
    freq: float = 60.0,
    rho: float = 1000.0,
    dx: float = 0.003,
    damping: float = 0.05,
    sources: list[tuple[int, int, int, complex]] | None = None,
    top_free: bool = False,
    viscosity: float | None = None,
) -> np.ndarray:
    """Solve the 3D scalar Helmholtz equation for the complex shear field u.

    Parameters
    ----------
    G : (N, N, N) ndarray
        Spatially varying shear modulus [Pa].
    freq, rho, dx, damping : same as ``helmholtz_solve`` (2D).
        ``damping`` gives a hysteretic (frequency-independent Q) loss via
        ``G*(x) = G(x)·(1 + i·damping)``.
    sources : list of (i, j, k, complex_amplitude), optional
        Dirichlet overrides. When None, no sources are placed — user must
        supply them explicitly for a 3D run (no meaningful default here).
    top_free : bool
        If True, the top face (i = 0, non-edge) uses ∂u/∂z = 0
        (ghost-node mirror) instead of Dirichlet u = 0.
    viscosity : float, optional
        Kelvin-Voigt dynamic viscosity η [Pa·s]. When set, adds a
        frequency-dependent damping term:
            G*(x, ω) = G(x)·(1 + i·damping) + i·ω·η
        This makes the effective loss grow linearly with ω, matching real
        soft tissue at low-MHz-and-below. Typical gelatin η ≈ 0.5–2 Pa·s
        (→ ξ_KV ≈ 0.075 at 60 Hz for G_bg=2500 Pa).
    """
    N = G.shape[0]
    assert G.shape == (N, N, N), f"expected cube, got {G.shape}"
    omega = 2.0 * np.pi * freq
    Gc = G * (1.0 + 1j * damping)
    if viscosity is not None:
        Gc = Gc + 1j * omega * float(viscosity)

    def idx(i, j, k):
        return (i * N + j) * N + k

    # ── Boundary set ────────────────────────────────────────────────────────
    boundary: set[int] = set()
    bc_values: dict[int, complex] = {}

    # Side faces (j = 0, j = N-1) — always Dirichlet.
    for i in range(N):
        for k in range(N):
            boundary.add(idx(i, 0, k))
            boundary.add(idx(i, N - 1, k))
    # Side faces (k = 0, k = N-1) — always Dirichlet.
    for i in range(N):
        for j in range(N):
            boundary.add(idx(i, j, 0))
            boundary.add(idx(i, j, N - 1))
    # Bottom face (i = N-1) — always Dirichlet.
    for j in range(N):
        for k in range(N):
            boundary.add(idx(N - 1, j, k))
    # Top face (i = 0) — Dirichlet unless top_free.
    if not top_free:
        for j in range(N):
            for k in range(N):
                boundary.add(idx(0, j, k))

    if sources is not None:
        for (i, j, k, amp) in sources:
            m = idx(i, j, k)
            bc_values[m] = complex(amp)
            boundary.add(m)

    n_dof = N ** 3
    A = lil_matrix((n_dof, n_dof), dtype=complex)
    b = np.zeros(n_dof, dtype=complex)

    def G_half(ga, gb):
        return 2.0 * ga * gb / (ga + gb)

    inv_dx2 = 1.0 / dx ** 2

    for i in range(N):
        for j in range(N):
            for k in range(N):
                m = idx(i, j, k)
                if m in boundary:
                    A[m, m] = 1.0
                    b[m] = bc_values.get(m, 0.0 + 0.0j)
                    continue
                g_c = Gc[i, j, k]
                g_e = G_half(g_c, Gc[i, j + 1, k]) if j + 1 < N else g_c
                g_w = G_half(g_c, Gc[i, j - 1, k]) if j - 1 >= 0 else g_c
                g_n = G_half(g_c, Gc[i - 1, j, k]) if i - 1 >= 0 else g_c
                g_s = G_half(g_c, Gc[i + 1, j, k]) if i + 1 < N else g_c
                g_u = G_half(g_c, Gc[i, j, k + 1]) if k + 1 < N else g_c
                g_d = G_half(g_c, Gc[i, j, k - 1]) if k - 1 >= 0 else g_c
                diag = -(g_e + g_w + g_n + g_s + g_u + g_d) * inv_dx2 + rho * omega ** 2
                A[m, m] = diag
                A[m, idx(i, j + 1, k)] = g_e * inv_dx2
                A[m, idx(i, j - 1, k)] = g_w * inv_dx2
                A[m, idx(i, j, k + 1)] = g_u * inv_dx2
                A[m, idx(i, j, k - 1)] = g_d * inv_dx2
                if i - 1 >= 0:
                    A[m, idx(i - 1, j, k)] = g_n * inv_dx2
                    A[m, idx(i + 1, j, k)] = g_s * inv_dx2
                else:
                    # Neumann top: ghost mirror folds north coeff onto south.
                    A[m, idx(i + 1, j, k)] = (g_n + g_s) * inv_dx2

    u_flat = spsolve(A.tocsr(), b)
    return u_flat.reshape(N, N, N)


def directional_filter_3d(
    u: np.ndarray,
    khat: np.ndarray,
    angular_width: float = 0.35,
    kmin_frac: float = 0.02,
    kmax_frac: float = 0.45,
) -> np.ndarray:
    """Isolate the ±k̂-propagating component of a complex 3D wave field.

    Implements the k-space wedge filter that underlies Yin's directional
    filtering step. In Fourier space, keeps K-vectors whose direction is
    close to ±k̂ (symmetric because ±K are the same physical wave) and
    within a band-pass |K| range.

    The angular weight is a smooth Gaussian in ``sin²(θ)``:
        W_ang(K) = exp( -(1 - (K̂·k̂)²) / angular_width² )
    where K̂ = K/|K|. The band-pass keeps normalised |K|/K_Nyquist in
    [kmin_frac, kmax_frac] — excludes DC (kmin) and grid-aliasing (kmax).

    Parameters
    ----------
    u : (N, N, N) complex ndarray
    khat : (3,) unit vector — filter direction
    angular_width : Gaussian σ in sin(θ) space. Smaller = narrower wedge.
        0.35 rad ≈ 20° full-width half-max, typical for MRE DF sets.
    kmin_frac, kmax_frac : band-pass corners as fraction of Nyquist.

    Returns
    -------
    u_khat : (N, N, N) complex field, the k̂-propagating component.
    """
    khat = np.asarray(khat, dtype=np.float64)
    khat = khat / (np.linalg.norm(khat) + 1e-30)
    N = u.shape[0]
    U = np.fft.fftn(u)
    # k-space grid: fftshift-natural ordering, values in [-π/dx, π/dx].
    kk = np.fft.fftfreq(N)          # in cycles/voxel, range [-0.5, 0.5)
    Kx, Ky, Kz = np.meshgrid(kk, kk, kk, indexing="ij")
    K_norm = np.sqrt(Kx ** 2 + Ky ** 2 + Kz ** 2)
    # Unit-vector K̂; zero norm → keep zero (DC gets killed by band-pass anyway).
    safe = np.maximum(K_norm, 1e-30)
    dot  = (Kx * khat[0] + Ky * khat[1] + Kz * khat[2]) / safe
    # Angular weight — Gaussian in sin²(θ). |dot|² = cos², so 1 - |dot|² = sin².
    sin2 = np.clip(1.0 - dot ** 2, 0.0, 1.0)
    W_ang = np.exp(-sin2 / (angular_width ** 2 + 1e-30))
    # Band-pass |K|.
    band = ((K_norm >= kmin_frac) & (K_norm <= kmax_frac)).astype(float)
    mask = W_ang * band
    return np.fft.ifftn(U * mask)


def multi_face_broadband_sources(
    N: int,
    radius_frac: float = 0.5,
    faces: tuple[str, ...] = ("iN", "jN", "j0", "kN", "k0"),
) -> list[tuple[int, int, int, complex]]:
    """Coherent-phase source disks on multiple faces at once.

    Launches a naturally multi-directional wave field for the filter-based
    TSM pipeline (contrast to the per-direction bottom-plate driver used in
    the N-solves pipeline). Default keeps the bottom face (iN) as the
    primary driver and adds four side faces; ``i0`` is intentionally
    omitted to leave the "top" as the free/reflecting surface if the caller
    uses ``top_free=True``.

    Face codes: 'iN' = i=N-1 (bottom), 'i0' = i=0 (top),
                'jN' = j=N-1,   'j0' = j=0,
                'kN' = k=N-1,   'k0' = k=0.
    """
    cy = (N - 1) / 2.0
    r_max = (N / 2.0) * radius_frac
    face_map = {
        "i0": ("i", 0), "iN": ("i", N - 1),
        "j0": ("j", 0), "jN": ("j", N - 1),
        "k0": ("k", 0), "kN": ("k", N - 1),
    }
    src = []
    for f in faces:
        axis, idx = face_map[f]
        for a in range(N):
            for b in range(N):
                if (a - cy) ** 2 + (b - cy) ** 2 > r_max ** 2:
                    continue
                if axis == "i":
                    src.append((idx, a, b, 1.0 + 0.0j))
                elif axis == "j":
                    src.append((a, idx, b, 1.0 + 0.0j))
                else:  # k
                    src.append((a, b, idx, 1.0 + 0.0j))
    return src


def lfe_inversion_3d(
    u: np.ndarray,
    freq: float,
    rho: float = 1000.0,
    dx: float = 0.003,
    smoothing_sigma_vx: float | None = 1.0,
    median_filter_size: int | None = None,
) -> np.ndarray:
    """Local Frequency Estimation inversion (first-derivative form).

    For a locally-plane-wave field u(x) ≈ A·exp(i·k·x + φ),
        ∇u = i·k·u   →   |∇u|² = |k|²·|u|²   →   |k|² = |∇u|²/|u|²
    hence
        G(x) ≈ ρω² / |k|² = ρω²·|u|² / |∇u|²
    Uses ONLY first derivatives, so it's much less noise-sensitive than
    the second-derivative direct inversion (∇²u). Widely used in MRE
    as an alternative to DI (Manduca et al.).

    Parameters
    ----------
    smoothing_sigma_vx : Gaussian σ in voxels applied to numerator
        (|u|²) and denominator (|∇u|²) separately before dividing. This
        stabilises the ratio in low-amplitude regions. Set None to skip.
        Default 1.0 (mild).
    median_filter_size : optional post-inversion cubic median.

    Boundary voxels (first & last plane on each axis) are set to NaN.
    """
    N = u.shape[0]
    omega = 2.0 * np.pi * freq
    # Central-difference first derivatives.
    inv_2dx = 0.5 / dx
    gx = np.zeros_like(u); gy = np.zeros_like(u); gz = np.zeros_like(u)
    gx[1:-1, :, :] = (u[2:, :, :] - u[:-2, :, :]) * inv_2dx
    gy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) * inv_2dx
    gz[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) * inv_2dx
    grad_sq = np.abs(gx) ** 2 + np.abs(gy) ** 2 + np.abs(gz) ** 2
    u_sq    = np.abs(u) ** 2

    if smoothing_sigma_vx and smoothing_sigma_vx > 0:
        from scipy.ndimage import gaussian_filter
        grad_sq = gaussian_filter(grad_sq, sigma=float(smoothing_sigma_vx))
        u_sq    = gaussian_filter(u_sq,    sigma=float(smoothing_sigma_vx))

    with np.errstate(divide="ignore", invalid="ignore"):
        k_sq = grad_sq / u_sq
        G    = rho * omega ** 2 / k_sq

    # Mask boundaries and non-finite voxels.
    G[0, :, :] = G[-1, :, :] = np.nan
    G[:, 0, :] = G[:, -1, :] = np.nan
    G[:, :, 0] = G[:, :, -1] = np.nan
    G[~np.isfinite(G)] = np.nan

    if median_filter_size and median_filter_size > 1:
        from scipy.ndimage import median_filter
        valid = np.isfinite(G)
        interior_median = float(np.nanmedian(G))
        G_fill = np.where(valid, G, interior_median)
        G = median_filter(G_fill, size=int(median_filter_size), mode="mirror")
        G[~valid] = np.nan

    return G


def bottom_plate_driver_sources_3d(
    N: int,
    radius_frac: float = 0.5,
    amp: complex = 1.0 + 0.0j,
) -> list[tuple[int, int, int, complex]]:
    """Coherent-phase disk source on the bottom face (i = N-1).

    Mimics a real MRE mechanical driver — a circular rigid piston plate
    under the phantom, all its contact nodes at one complex amplitude.

    Parameters
    ----------
    N : grid size (assumes N × N × N).
    radius_frac : disk radius as fraction of N/2. 0.5 → radius = N/4.
    amp : complex amplitude applied uniformly.
    """
    if not (0.0 < radius_frac <= 1.0):
        raise ValueError(f"radius_frac must be in (0, 1], got {radius_frac}")
    cy = (N - 1) / 2.0
    cz = (N - 1) / 2.0
    r_max = (N / 2.0) * radius_frac
    src: list[tuple[int, int, int, complex]] = []
    for j in range(N):
        for k in range(N):
            if (j - cy) ** 2 + (k - cz) ** 2 <= r_max ** 2:
                src.append((N - 1, j, k, complex(amp)))
    return src


def direct_inversion_3d(
    u: np.ndarray,
    freq: float,
    rho: float = 1000.0,
    dx: float = 0.003,
    median_filter_size: int | None = None,
) -> np.ndarray:
    """Voxel-wise direct inversion for the shear modulus (real part).

    Uses the locally-homogeneous approximation
        G(x) ≈ -ρω² u(x) / ∇²u(x)
    with a 7-point centered Laplacian. Boundary voxels are set to NaN.

    Parameters
    ----------
    median_filter_size : int, optional
        If given (e.g. 3), applies a size × size × size spatial median
        filter to the recovered G map to suppress inversion artifacts at
        stiffness gradients. Matches Yin's ``3 × 3 × 3 cubic spatial
        median filter to improve regional homogeneity of stiffness
        estimates'' (paper Methods). NaN voxels are protected via a
        fill-mask-filter-remask pattern so the filter doesn't leak NaN
        into interior voxels. Default None = no filtering.
    """
    N = u.shape[0]
    omega = 2.0 * np.pi * freq
    lap = np.zeros_like(u)
    inv_dx2 = 1.0 / dx ** 2
    lap[1:-1, 1:-1, 1:-1] = (
        u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]
        + u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1]
        + u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2]
        - 6.0 * u[1:-1, 1:-1, 1:-1]
    ) * inv_dx2
    # Voxels with vanishing Laplacian → NaN (would explode).
    with np.errstate(divide="ignore", invalid="ignore"):
        G_c = -rho * omega ** 2 * u / lap
    G_real = G_c.real
    # Zero out boundary voxels (undefined Laplacian).
    G_real[0, :, :] = G_real[-1, :, :] = np.nan
    G_real[:, 0, :] = G_real[:, -1, :] = np.nan
    G_real[:, :, 0] = G_real[:, :, -1] = np.nan
    # Mask non-physical values.
    G_real[~np.isfinite(G_real)] = np.nan

    if median_filter_size and median_filter_size > 1:
        from scipy.ndimage import median_filter
        valid = np.isfinite(G_real)
        # Fill NaNs with the interior median so the filter has no unusual
        # boundary values to smear inward; then re-apply the mask.
        interior_median = float(np.nanmedian(G_real))
        G_fill = np.where(valid, G_real, interior_median)
        G_real = median_filter(G_fill, size=int(median_filter_size), mode="mirror")
        G_real[~valid] = np.nan

    return G_real
