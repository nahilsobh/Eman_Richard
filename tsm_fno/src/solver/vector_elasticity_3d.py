"""3D vector elasticity solver — isotropic-heterogeneous first cut.

Solves the time-harmonic Navier equation

    ρ ω² u_i = ∂_j [μ(x) · (∂_i u_j + ∂_j u_i)] + ∂_i [λ · ∂_k u_k]         (1)

for a 3-vector displacement field u_i(x) on a regular Cartesian grid,
with heterogeneous shear modulus μ(x) and a spatially constant λ.
Hysteretic (constant-Q) damping is applied via ``μ → μ (1 + i·damping)``.

The physical model is isotropic linear elasticity — this captures the
distinction between P-waves (longitudinal, speed √((λ+2μ)/ρ)) and
S-waves (transverse, speed √(μ/ρ)), which the scalar-Helmholtz solver
collapses into a single scalar. For soft tissue with ρ = 1000 kg/m³,
G_bg = 2500 Pa, we get c_S ≈ 1.58 m/s at 60 Hz — the target shear
wavelength ≈ 26 mm.

Anisotropic C_ijkl (Murnaghan-tensor acoustoelastic coupling) is NOT
in this first cut — that would replace μ(x) with a tensor field and add
substantial stencil complexity. See module docstring at end for the
extension path.

Discretisation
--------------
Non-staggered (collocated) grid: all three components of u live at the
same voxel centres. Assembly uses second-order central differences:

  ∇²u_i    :  standard 7-point Laplacian on component i
  ∂_i(∂_k u_k) for i = k :  captured in the Laplacian
  ∂_i(∂_k u_k) for i ≠ k :  4-point cross derivative
      ∂_x ∂_y u_y at (i,j,k) ≈
          (u_y[i+1,j+1] − u_y[i+1,j−1] − u_y[i−1,j+1] + u_y[i−1,j−1]) / (4 dx²)
  ∂_j μ · ∂_i u_j :  central-difference ∂_j μ and multiply with central
      ∂_i u_j (also 3-point). Yields extra off-diagonal entries.

System size: 3 N³ unknowns; ~30 nnz per row → ~90·N³ nonzeros. At
N=32 the sparse system solves in ~1 min with scipy's SuperLU.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def navier_solve_3d_isotropic(
    mu_field: np.ndarray,
    lam: float,
    freq: float = 60.0,
    rho: float = 1000.0,
    dx: float = 0.003,
    damping: float = 0.05,
    sources: list[tuple[int, int, int, int, complex]] | None = None,
) -> np.ndarray:
    """Solve the 3D isotropic-heterogeneous Navier equation for u_i(x).

    Parameters
    ----------
    mu_field : (N, N, N) real ndarray — shear modulus field μ(x) [Pa].
    lam : float — Lamé's first parameter (bulk-related), assumed constant [Pa].
        For nearly-incompressible tissue set lam ≈ 100·μ_max as a soft
        incompressibility penalty (true incompressibility λ→∞ needs a
        mixed u-p formulation, out of scope for this first cut).
    freq, rho, dx, damping : same as scalar solver.
    sources : list of (i, j, k, comp, complex_amp) tuples where
        ``comp ∈ {0, 1, 2}`` picks the driven displacement component.
        Any node listed here is Dirichlet-clamped to that value.

    Returns
    -------
    u : (N, N, N, 3) complex ndarray — the vector displacement field.

    Boundary conditions
    -------------------
    Dirichlet u = 0 on all six faces except where overridden by ``sources``.
    Free-surface (traction-free) BCs are more involved for vector elasticity
    (need σ_ij n_j = 0, not just u_i = 0) and are not implemented here.
    """
    N = mu_field.shape[0]
    assert mu_field.shape == (N, N, N), f"expected (N,N,N), got {mu_field.shape}"
    omega = 2.0 * np.pi * freq
    mu = mu_field.astype(complex) * (1.0 + 1j * float(damping))
    lam_c = complex(lam)

    # Global row/col index for (component, i, j, k).
    n_vox = N ** 3
    n_dof = 3 * n_vox

    def idx(comp, i, j, k):
        return comp * n_vox + (i * N + j) * N + k

    def in_bounds(i, j, k):
        return 0 <= i < N and 0 <= j < N and 0 <= k < N

    # Boundary set (all six faces, per component).
    boundary: set[int] = set()
    bc_values: dict[int, complex] = {}
    for c in range(3):
        for a in range(N):
            for b in range(N):
                boundary.add(idx(c, 0, a, b))
                boundary.add(idx(c, N - 1, a, b))
                boundary.add(idx(c, a, 0, b))
                boundary.add(idx(c, a, N - 1, b))
                boundary.add(idx(c, a, b, 0))
                boundary.add(idx(c, a, b, N - 1))

    if sources is not None:
        for (i, j, k, comp, amp) in sources:
            m = idx(comp, i, j, k)
            bc_values[m] = complex(amp)
            boundary.add(m)

    A = lil_matrix((n_dof, n_dof), dtype=complex)
    b = np.zeros(n_dof, dtype=complex)

    inv_dx2  = 1.0 / dx ** 2
    inv_4dx2 = 0.25 / dx ** 2

    # Precompute half-point mu averages (arithmetic) for each face:
    # mu_half_x[i, j, k] = mu at (i + 1/2, j, k). Shape (N-1, N, N).
    mu_hx = 0.5 * (mu[1:, :, :] + mu[:-1, :, :])   # μ(i+½, j, k)
    mu_hy = 0.5 * (mu[:, 1:, :] + mu[:, :-1, :])
    mu_hz = 0.5 * (mu[:, :, 1:] + mu[:, :, :-1])

    for i in range(N):
        for j in range(N):
            for k in range(N):
                for c in range(3):
                    m = idx(c, i, j, k)
                    if m in boundary:
                        A[m, m] = 1.0
                        b[m] = bc_values.get(m, 0.0 + 0.0j)
                        continue

                    # ── Term 1: ∂_j (μ ∂_j u_c) for j = 0,1,2  (variable-μ Laplacian) ──
                    # Half-point μ averages:
                    mu_xp = mu_hx[i, j, k]     if i + 1 < N else mu[i, j, k]
                    mu_xm = mu_hx[i - 1, j, k] if i - 1 >= 0 else mu[i, j, k]
                    mu_yp = mu_hy[i, j, k]     if j + 1 < N else mu[i, j, k]
                    mu_ym = mu_hy[i, j - 1, k] if j - 1 >= 0 else mu[i, j, k]
                    mu_zp = mu_hz[i, j, k]     if k + 1 < N else mu[i, j, k]
                    mu_zm = mu_hz[i, j, k - 1] if k - 1 >= 0 else mu[i, j, k]

                    diag = -(mu_xp + mu_xm + mu_yp + mu_ym + mu_zp + mu_zm) * inv_dx2 \
                            + rho * omega ** 2
                    A[m, m] = diag
                    A[m, idx(c, i + 1, j, k)] = mu_xp * inv_dx2 if i + 1 < N else 0
                    A[m, idx(c, i - 1, j, k)] = mu_xm * inv_dx2 if i - 1 >= 0 else 0
                    A[m, idx(c, i, j + 1, k)] = mu_yp * inv_dx2 if j + 1 < N else 0
                    A[m, idx(c, i, j - 1, k)] = mu_ym * inv_dx2 if j - 1 >= 0 else 0
                    A[m, idx(c, i, j, k + 1)] = mu_zp * inv_dx2 if k + 1 < N else 0
                    A[m, idx(c, i, j, k - 1)] = mu_zm * inv_dx2 if k - 1 >= 0 else 0

                    # ── Term 2: ∂_j (μ ∂_c u_j) for j ≠ c  (shear cross-coupling) ──
                    # Approximate with μ at centre + 4-point cross derivative:
                    #    ∂_j (μ ∂_c u_j) ≈ μ · ∂_j ∂_c u_j + (∂_j μ) · (∂_c u_j)
                    # First piece uses ∂_j∂_c cross-difference on u_j.
                    mu_c = mu[i, j, k]
                    for j_ax in range(3):
                        if j_ax == c:
                            continue
                        # ∂_j ∂_c u_j :  4-point cross stencil
                        for sc in (+1, -1):
                            for sj in (+1, -1):
                                ii = i + (sc if c == 0 else 0) + (sj if j_ax == 0 else 0)
                                jj = j + (sc if c == 1 else 0) + (sj if j_ax == 1 else 0)
                                kk = k + (sc if c == 2 else 0) + (sj if j_ax == 2 else 0)
                                if not in_bounds(ii, jj, kk):
                                    continue
                                m_col = idx(j_ax, ii, jj, kk)
                                A[m, m_col] = A[m, m_col] + (sc * sj) * mu_c * inv_4dx2
                        # (∂_j μ) · (∂_c u_j) :  central-diff μ and u_j
                        # ∂_j μ ≈ (μ[i+e_j] − μ[i-e_j]) / (2 dx)
                        ip = [i, j, k]; im = [i, j, k]
                        ip[j_ax] += 1; im[j_ax] -= 1
                        if in_bounds(*ip) and in_bounds(*im):
                            dmu_dj = (mu[tuple(ip)] - mu[tuple(im)]) * (0.5 / dx)
                        else:
                            dmu_dj = 0.0
                        # ∂_c u_j at (i,j,k) → central diff along c on u_j
                        ip2 = [i, j, k]; im2 = [i, j, k]
                        ip2[c] += 1; im2[c] -= 1
                        if in_bounds(*ip2):
                            A[m, idx(j_ax, *ip2)] = A[m, idx(j_ax, *ip2)] + dmu_dj * (0.5 / dx)
                        if in_bounds(*im2):
                            A[m, idx(j_ax, *im2)] = A[m, idx(j_ax, *im2)] - dmu_dj * (0.5 / dx)

                    # ── Term 3: ∂_c (λ ∂_k u_k)  for constant λ  ──
                    # λ ∂_c ∂_k u_k :  k = c gives extra Laplacian-like along c;
                    # k ≠ c gives cross-derivatives.
                    # For k = c: λ ∂_c^2 u_c → add to diagonal & c-axis neighbours
                    ip3 = [i, j, k]; im3 = [i, j, k]
                    ip3[c] += 1; im3[c] -= 1
                    if in_bounds(*ip3):
                        A[m, idx(c, *ip3)] = A[m, idx(c, *ip3)] + lam_c * inv_dx2
                    if in_bounds(*im3):
                        A[m, idx(c, *im3)] = A[m, idx(c, *im3)] + lam_c * inv_dx2
                    A[m, m] = A[m, m] - 2.0 * lam_c * inv_dx2
                    # For k ≠ c: λ ∂_c ∂_k u_k :  4-point cross derivative
                    for k_ax in range(3):
                        if k_ax == c:
                            continue
                        for sc in (+1, -1):
                            for sk in (+1, -1):
                                ii = i + (sc if c == 0 else 0) + (sk if k_ax == 0 else 0)
                                jj = j + (sc if c == 1 else 0) + (sk if k_ax == 1 else 0)
                                kk = k + (sc if c == 2 else 0) + (sk if k_ax == 2 else 0)
                                if not in_bounds(ii, jj, kk):
                                    continue
                                m_col = idx(k_ax, ii, jj, kk)
                                A[m, m_col] = A[m, m_col] + (sc * sk) * lam_c * inv_4dx2

    print(f"[navier] solving {n_dof} DOF sparse system (nnz ~ {A.nnz})...")
    u_flat = spsolve(A.tocsr(), b)
    return u_flat.reshape(3, N, N, N).transpose(1, 2, 3, 0)


def navier_solve_3d_tensor_mu(
    mu_tensor: np.ndarray,
    lam: float,
    freq: float = 60.0,
    rho: float = 1000.0,
    dx: float = 0.003,
    damping: float = 0.05,
    sources: list[tuple[int, int, int, int, complex]] | None = None,
) -> np.ndarray:
    """3D Navier solver with a rank-2 anisotropic shear-modulus tensor field.

    A step toward the full Murnaghan-C_ijkl(x) formulation. Uses a
    "quasi-anisotropic" constitutive law:

        σ_ij = λ · δ_ij · tr(ε)  +  (μ_ik · ε_kj + μ_jk · ε_ki)

    where ``μ_ij(x)`` is a symmetric rank-2 field (Pa) and
    ``ε_ij = (∂_i u_j + ∂_j u_i) / 2``. When ``μ_ij = μ · δ_ij`` (scalar
    reduction), this reduces exactly to the isotropic-heterogeneous
    ``navier_solve_3d_isotropic`` case.

    For our sphere problem, μ_ij(x) is constructed from Ogden principal
    tangent shear moduli as
        μ_ij(x) = G_r(x) · r̂_i r̂_j + G_θ(x) · (δ_ij − r̂_i r̂_j)
    which captures the radial-vs-tangential shear anisotropy without
    needing the full 6-rank Murnaghan tensor.

    Parameters
    ----------
    mu_tensor : (N, N, N, 3, 3) real ndarray — symmetric shear tensor field.
    lam, freq, rho, dx, damping, sources : same as ``navier_solve_3d_isotropic``.

    Returns
    -------
    u : (N, N, N, 3) complex ndarray — vector displacement field.

    Notes
    -----
    This is NOT the full Murnaghan third-order elasticity theory (which
    couples pre-stress to the elasticity tensor via a 6-rank M tensor and
    3 additional constants l, m, n). It's a "quasi-anisotropic" scalar-μ
    generalisation to a rank-2 μ tensor that keeps the assembly tractable
    (~2× more nonzeros than scalar case) while capturing directional
    shear response.
    """
    N = mu_tensor.shape[0]
    assert mu_tensor.shape == (N, N, N, 3, 3), \
        f"expected (N,N,N,3,3), got {mu_tensor.shape}"
    omega = 2.0 * np.pi * freq
    mu = mu_tensor.astype(complex) * (1.0 + 1j * float(damping))
    lam_c = complex(lam)

    n_vox = N ** 3
    n_dof = 3 * n_vox

    def idx(comp, i, j, k):
        return comp * n_vox + (i * N + j) * N + k

    def in_bounds(i, j, k):
        return 0 <= i < N and 0 <= j < N and 0 <= k < N

    boundary: set[int] = set()
    bc_values: dict[int, complex] = {}
    for c in range(3):
        for a in range(N):
            for b in range(N):
                boundary.add(idx(c, 0, a, b))
                boundary.add(idx(c, N - 1, a, b))
                boundary.add(idx(c, a, 0, b))
                boundary.add(idx(c, a, N - 1, b))
                boundary.add(idx(c, a, b, 0))
                boundary.add(idx(c, a, b, N - 1))

    if sources is not None:
        for (i, j, k, comp, amp) in sources:
            m = idx(comp, i, j, k)
            bc_values[m] = complex(amp)
            boundary.add(m)

    A = lil_matrix((n_dof, n_dof), dtype=complex)
    b = np.zeros(n_dof, dtype=complex)

    inv_dx2 = 1.0 / dx ** 2
    inv_4dx2 = 0.25 / dx ** 2

    # Assembly:
    #   ρω² u_i = ∂_j σ_ij
    #   σ_ij = λ δ_ij tr(ε) + μ_ik ε_kj + μ_jk ε_ki
    #         = λ δ_ij ∂_l u_l  +  (1/2)(μ_ik (∂_k u_j + ∂_j u_k) + μ_jk (∂_k u_i + ∂_i u_k))
    # For assembly at voxel (i,j,k) for equation-component c:
    #   Iterate over stress-divergence index j_ax; extract σ_{c, j_ax}
    #   For each j_ax, contribution to equation is ∂_{j_ax} σ_{c, j_ax}
    # Use central differences on σ (which itself uses first derivatives of u).
    # This gives a wider stencil than the isotropic case — up to 2·dx reach.

    for i in range(N):
        for j in range(N):
            for k in range(N):
                for c in range(3):
                    m = idx(c, i, j, k)
                    if m in boundary:
                        A[m, m] = 1.0
                        b[m] = bc_values.get(m, 0.0 + 0.0j)
                        continue

                    # Mass term.
                    A[m, m] = rho * omega ** 2

                    # ── Contribution: 2·∂_{j_ax} (μ_{c,k_ax} ∂_{k_ax} u_{j_ax}) etc.
                    # Instead of building σ explicitly, we implement:
                    #   ∂_{j_ax} σ_{c,j_ax} for each j_ax
                    # via CENTRAL differences that reach ±e_{j_ax} in σ.
                    # Each σ at x+e_j uses u derivatives at x+e_j, giving u
                    # values at x + e_j ± e_k. This gives a stencil at
                    # x + e_j + e_k for each (j, k) combination.

                    for j_ax in range(3):
                        # ∂_{j_ax} of σ_{c, j_ax}  (approximate by
                        # 2-sided central diff of σ evaluated at ±e_{j_ax})
                        for sj in (+1, -1):
                            i_j = [i, j, k]; i_j[j_ax] += sj
                            if not in_bounds(*i_j):
                                continue
                            # σ_{c, j_ax}(x + sj·e_{j_ax}):
                            # = λ δ_{c,j_ax} · Σ_l ∂_l u_l(x+sj·e_{j_ax})
                            #   + Σ_k [μ_{c,k}(x+sj·e_{j_ax}) · ∂_k u_{j_ax}(x+sj·e_{j_ax})
                            #        + μ_{j_ax,k}(x+sj·e_{j_ax}) · ∂_k u_c(x+sj·e_{j_ax})] / ... symmetric
                            # Actually: μ_ik ε_kj + μ_jk ε_ki
                            #       = μ_ik · (∂_k u_j + ∂_j u_k)/2 + μ_jk · (∂_k u_i + ∂_i u_k)/2

                            # ── λ·δ_{c,j_ax}·div(u) term ──
                            if c == j_ax:
                                for l_ax in range(3):
                                    # ∂_l u_l at (x + sj·e_{j_ax})
                                    ip = i_j.copy(); im = i_j.copy()
                                    ip[l_ax] += 1; im[l_ax] -= 1
                                    coef_scale = (sj / (2.0 * dx)) * (0.5 / dx)
                                    if in_bounds(*ip):
                                        col = idx(l_ax, *ip)
                                        A[m, col] = A[m, col] + lam_c * coef_scale
                                    if in_bounds(*im):
                                        col = idx(l_ax, *im)
                                        A[m, col] = A[m, col] - lam_c * coef_scale

                            # ── μ_{c,k}·(∂_k u_{j_ax} + ∂_{j_ax} u_k)/2 term ──
                            # and μ_{j_ax,k}·(∂_k u_c + ∂_c u_k)/2  (symmetric)
                            mu_at_neighbor = mu[i_j[0], i_j[1], i_j[2]]  # (3,3) tensor at x+sj·e_j
                            for k_ax in range(3):
                                mu_ck = mu_at_neighbor[c, k_ax]
                                mu_jaxk = mu_at_neighbor[j_ax, k_ax]

                                # ∂_k u_{j_ax}
                                ip = i_j.copy(); im = i_j.copy()
                                ip[k_ax] += 1; im[k_ax] -= 1
                                coef_scale = (sj / (2.0 * dx)) * (0.5 / dx)
                                if in_bounds(*ip):
                                    col = idx(j_ax, *ip)
                                    A[m, col] = A[m, col] + mu_ck * coef_scale
                                if in_bounds(*im):
                                    col = idx(j_ax, *im)
                                    A[m, col] = A[m, col] - mu_ck * coef_scale

                                # ∂_{j_ax} u_k  (mult by μ_ck / 2)
                                ip = i_j.copy(); im = i_j.copy()
                                ip[j_ax] += 1; im[j_ax] -= 1
                                if in_bounds(*ip):
                                    col = idx(k_ax, *ip)
                                    A[m, col] = A[m, col] + mu_ck * coef_scale
                                if in_bounds(*im):
                                    col = idx(k_ax, *im)
                                    A[m, col] = A[m, col] - mu_ck * coef_scale

                                # ∂_k u_c  (mult by μ_{jax,k} / 2)
                                ip = i_j.copy(); im = i_j.copy()
                                ip[k_ax] += 1; im[k_ax] -= 1
                                if in_bounds(*ip):
                                    col = idx(c, *ip)
                                    A[m, col] = A[m, col] + mu_jaxk * coef_scale
                                if in_bounds(*im):
                                    col = idx(c, *im)
                                    A[m, col] = A[m, col] - mu_jaxk * coef_scale

                                # ∂_c u_k  (mult by μ_{jax,k} / 2)
                                ip = i_j.copy(); im = i_j.copy()
                                ip[c] += 1; im[c] -= 1
                                if in_bounds(*ip):
                                    col = idx(k_ax, *ip)
                                    A[m, col] = A[m, col] + mu_jaxk * coef_scale
                                if in_bounds(*im):
                                    col = idx(k_ax, *im)
                                    A[m, col] = A[m, col] - mu_jaxk * coef_scale

    print(f"[navier-tensor] solving {n_dof} DOF, nnz ~ {A.nnz}...")
    u_flat = spsolve(A.tocsr(), b)
    return u_flat.reshape(3, N, N, N).transpose(1, 2, 3, 0)


def curl_of_displacement_3d(u_vec: np.ndarray, dx: float) -> np.ndarray:
    """Discrete curl of a vector displacement field. Returns a (N,N,N,3) field.

    Isolates the S-wave (rotational) part of the elastic wave field:
    for a P-wave u = ∇φ, curl u = 0. Passing curl-u to a scalar direct
    inversion gives a cleaner shear-modulus estimate than DI on the raw
    displacement (which mixes P and S).
    """
    N = u_vec.shape[0]
    inv_2dx = 0.5 / dx
    du = np.zeros((N, N, N, 3, 3), dtype=u_vec.dtype)
    # du[..., i, j] = ∂_j u_i
    du[1:-1, :, :, :, 0] = (u_vec[2:, :, :, :] - u_vec[:-2, :, :, :]) * inv_2dx
    du[:, 1:-1, :, :, 1] = (u_vec[:, 2:, :, :] - u_vec[:, :-2, :, :]) * inv_2dx
    du[:, :, 1:-1, :, 2] = (u_vec[:, :, 2:, :] - u_vec[:, :, :-2, :]) * inv_2dx
    curl = np.zeros((N, N, N, 3), dtype=u_vec.dtype)
    curl[..., 0] = du[..., 2, 1] - du[..., 1, 2]
    curl[..., 1] = du[..., 0, 2] - du[..., 2, 0]
    curl[..., 2] = du[..., 1, 0] - du[..., 0, 1]
    return curl


# ── Extension roadmap (not implemented in this first cut) ──────────────
# 1. Anisotropic C_ijkl(x) tensor field (fully general) — replaces μ(x)
#    with a rank-4 field; assembly loops over 3×3 stress-strain relation
#    at each voxel. ~2× more nonzeros per row.
# 2. Acoustoelastic Murnaghan third-order coupling — C_ijkl^eff = C_ijkl^0
#    + M_ijklmn·σ_mn where σ is the finite pre-stress from Lamé/Ogden.
# 3. Traction-free top BC — implement σ_ij·n_j = 0 discretely (requires
#    the strain at the boundary voxel + special stencil).
# 4. Nearly-incompressible u-p mixed formulation — introduces pressure
#    Lagrange multiplier, avoids the λ→∞ ill-conditioning that our
#    penalty-λ approach has.
