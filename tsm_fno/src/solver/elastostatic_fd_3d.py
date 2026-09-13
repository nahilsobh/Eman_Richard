"""3D static elastostatic solver on a rectangular grid — for option-3 baseline.

Solves the static Navier equation for linear isotropic homogeneous
nearly-incompressible material:

    (λ + μ) ∂_i (∂_k u_k) + μ ∂_j ∂_j u_i = 0                              (*)

on a rectangular grid (Nx, Ny, Nz) with per-face boundary conditions
and optional internal Dirichlet nodes.

Purpose (option 3): compute the balloon-inflation displacement field
inside Yin's actual 15×15×18 cm container:
  - 5 rigid walls (fixed u = 0)
  - 1 free top surface (∂u/∂z = 0 mirror approximation)
  - internal Dirichlet: prescribe radial displacement u_r on all voxels
    on the balloon surface

The displacement field is then used to compute the deformation-gradient
tensor F = I + ∇u, right Cauchy-Green tensor C = F^T F, and principal
stretches (λ_1, λ_2, λ_3). Ogden principal shear moduli are computed
ex-post from these stretches:

    G_p(x) = Σ_q μ_q · λ_p(x)^(α_q − 2)                                   (**)

Scope / limitations of this first cut
-------------------------------------
1. Small-strain LINEAR elasticity. For balloon inflations of ~40%
   strain near the surface, det(F) = 1 (finite-strain incompressibility)
   is NOT the same as ∇·u = 0 (small-strain incompressibility). The
   stretch field extracted from the linear solve deviates from truth
   by O(strain²) at large deformation. Correct at small volumes;
   approximate at large volumes.
2. Free-top BC is a mirror approximation (∂u/∂z = 0), not the exact
   traction-free condition (σ·n = 0 componentwise).
3. Homogeneous material only (μ constant). Heterogeneity per-voxel
   would require the general-purpose harmonic solver.

For a scientifically-rigorous finite-strain baseline, use FEniCS or
similar with Ogden hyperelastic material and nonlinear Newton — out of
scope here (weeks of work). This first cut is the honest baseline for
comparing container-confined vs infinite-matrix stretch fields.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def elastostatic_solve_3d(
    Nx: int,
    Ny: int,
    Nz: int,
    dx: float,
    mu: float,
    lam: float,
    fixed_faces: tuple[str, ...] = ("-x", "+x", "-y", "+y", "-z"),
    mirror_faces: tuple[str, ...] = ("+z",),
    dirichlet: list[tuple[int, int, int, int, float]] | None = None,
) -> np.ndarray:
    """Solve static Navier equation on a rectangular grid.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid dimensions. Voxel spacing is uniform ``dx``.
    mu, lam : float
        Isotropic Lamé constants (Pa). For near-incompressibility use
        ``lam ≈ 1e3 · mu``.
    fixed_faces : tuple of str
        Face labels where u = 0. Labels: ``"-x", "+x", "-y", "+y", "-z", "+z"``.
    mirror_faces : tuple of str
        Face labels where ∂u/∂n = 0 (mirror approximation for free surface).
    dirichlet : list of tuples
        Internal Dirichlet nodes as ``(i, j, k, comp, value)`` — comp ∈ {0,1,2}.

    Returns
    -------
    u : (Nx, Ny, Nz, 3) real ndarray
        The displacement field.
    """
    fixed = set(fixed_faces)
    mirror = set(mirror_faces)
    assert fixed.isdisjoint(mirror), "Face cannot be both fixed and mirror."

    n_vox = Nx * Ny * Nz
    n_dof = 3 * n_vox

    def idx(comp, i, j, k):
        return comp * n_vox + (i * Ny + j) * Nz + k

    inv_dx2  = 1.0 / dx ** 2
    inv_4dx2 = 0.25 / dx ** 2

    # ── Identify boundary DOFs ──────────────────────────────────────
    fixed_dofs: set[int] = set()
    mirror_dofs: set[int] = set()

    face_slices = {
        "-x": (0,          slice(None), slice(None)),
        "+x": (Nx - 1,     slice(None), slice(None)),
        "-y": (slice(None), 0,          slice(None)),
        "+y": (slice(None), Ny - 1,     slice(None)),
        "-z": (slice(None), slice(None), 0),
        "+z": (slice(None), slice(None), Nz - 1),
    }
    face_normal = {"-x": 0, "+x": 0, "-y": 1, "+y": 1, "-z": 2, "+z": 2}
    face_side   = {"-x": -1, "+x": +1, "-y": -1, "+y": +1, "-z": -1, "+z": +1}

    def _dof_list(face):
        ii, jj, kk = face_slices[face]
        I, J, K = np.mgrid[0:Nx, 0:Ny, 0:Nz]
        mask = np.zeros((Nx, Ny, Nz), dtype=bool)
        mask[ii, jj, kk] = True
        i_arr, j_arr, k_arr = np.where(mask)
        return [(i, j, k) for i, j, k in zip(i_arr, j_arr, k_arr)]

    for face in fixed:
        for (i, j, k) in _dof_list(face):
            for c in range(3):
                fixed_dofs.add(idx(c, i, j, k))
    for face in mirror:
        for (i, j, k) in _dof_list(face):
            for c in range(3):
                mirror_dofs.add(idx(c, i, j, k))

    dirichlet_values: dict[int, float] = {}
    if dirichlet is not None:
        for (i, j, k, c, val) in dirichlet:
            m = idx(c, i, j, k)
            dirichlet_values[m] = float(val)
            fixed_dofs.add(m)
            mirror_dofs.discard(m)

    # ── Assemble ────────────────────────────────────────────────────
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b_vec = np.zeros(n_dof, dtype=float)

    def add(row, col, val):
        rows.append(row); cols.append(col); vals.append(val)

    def in_bounds(i, j, k):
        return 0 <= i < Nx and 0 <= j < Ny and 0 <= k < Nz

    lam_p_mu = lam + mu

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for c in range(3):
                    m = idx(c, i, j, k)

                    if m in fixed_dofs:
                        add(m, m, 1.0)
                        b_vec[m] = dirichlet_values.get(m, 0.0)
                        continue

                    if m in mirror_dofs:
                        # Neumann-mirror: u(ghost) = u(interior neighbor).
                        # We encode this as a stencil equation using the
                        # ghost value substituted by the interior neighbor.
                        # For simplicity here we treat mirror as: assemble
                        # the standard PDE stencil, replacing out-of-domain
                        # ghost with the reflected value.
                        pass

                    # ── (λ+μ) ∂_i div(u) term ─────────────────────
                    # ∂_c(∂_c u_c) = ∂²_c u_c → part of Laplacian on u_c
                    # ∂_c(∂_k u_k) for k ≠ c → cross derivative on u_k
                    # ── μ ∇² u_c term (Laplacian on u_c) ───────────

                    # Handle diagonal Laplacian on u_c
                    # (μ + (λ+μ) δ direction) coefficient
                    for ax in range(3):
                        # coeff = μ (from Laplacian) + (λ+μ) if ax == c
                        coeff_lap = mu
                        if ax == c:
                            coeff_lap += lam_p_mu

                        # neighbor +ax
                        ip = [i, j, k]; ip[ax] += 1
                        im = [i, j, k]; im[ax] -= 1

                        if in_bounds(*ip):
                            add(m, idx(c, *ip), coeff_lap * inv_dx2)
                        else:
                            # Ghost handling: is +ax face mirror? Fold into interior
                            face_pos = {"+x": (ax == 0), "+y": (ax == 1), "+z": (ax == 2)}
                            if "+x" in mirror and ax == 0:
                                add(m, idx(c, *im), coeff_lap * inv_dx2)
                            elif "+y" in mirror and ax == 1:
                                add(m, idx(c, *im), coeff_lap * inv_dx2)
                            elif "+z" in mirror and ax == 2:
                                add(m, idx(c, *im), coeff_lap * inv_dx2)
                            # Otherwise Dirichlet 0 → no contribution

                        if in_bounds(*im):
                            add(m, idx(c, *im), coeff_lap * inv_dx2)
                        else:
                            if "-x" in mirror and ax == 0:
                                add(m, idx(c, *ip), coeff_lap * inv_dx2)
                            elif "-y" in mirror and ax == 1:
                                add(m, idx(c, *ip), coeff_lap * inv_dx2)
                            elif "-z" in mirror and ax == 2:
                                add(m, idx(c, *ip), coeff_lap * inv_dx2)

                        # diagonal
                        add(m, m, -2.0 * coeff_lap * inv_dx2)

                    # ── (λ+μ) ∂_c ∂_k u_k for k ≠ c ────────────────
                    # Cross derivative: 4-point stencil
                    for k_ax in range(3):
                        if k_ax == c:
                            continue
                        for sc in (+1, -1):
                            for sk in (+1, -1):
                                ii = i + (sc if c    == 0 else 0) + (sk if k_ax == 0 else 0)
                                jj = j + (sc if c    == 1 else 0) + (sk if k_ax == 1 else 0)
                                kk = k + (sc if c    == 2 else 0) + (sk if k_ax == 2 else 0)
                                if not in_bounds(ii, jj, kk):
                                    continue
                                add(m, idx(k_ax, ii, jj, kk),
                                    (sc * sk) * lam_p_mu * inv_4dx2)

    A = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    print(f"[elastostatic] {n_dof} DOF, nnz = {A.nnz}, solving...")
    u_flat = spsolve(A, b_vec)
    u = u_flat.reshape(3, Nx, Ny, Nz).transpose(1, 2, 3, 0)
    return u


def deformation_stretches(u_vec: np.ndarray, dx: float) -> np.ndarray:
    """Return principal stretches (λ_1, λ_2, λ_3) at each voxel.

    Given u(x), compute F = I + ∇u and C = F^T F, then eigenvalues of C
    are (λ_p)². Returns per-voxel sorted stretches (descending).

    Parameters
    ----------
    u_vec : (Nx, Ny, Nz, 3) ndarray
    dx : float

    Returns
    -------
    stretches : (Nx, Ny, Nz, 3) ndarray — principal stretches sorted descending.
    """
    Nx, Ny, Nz = u_vec.shape[:3]
    inv_2dx = 0.5 / dx

    # ∇u[..., i, j] = ∂_j u_i  → shape (Nx, Ny, Nz, 3, 3)
    grad = np.zeros((Nx, Ny, Nz, 3, 3), dtype=u_vec.dtype)
    grad[1:-1, :, :, :, 0] = (u_vec[2:, :, :, :] - u_vec[:-2, :, :, :]) * inv_2dx
    grad[:, 1:-1, :, :, 1] = (u_vec[:, 2:, :, :] - u_vec[:, :-2, :, :]) * inv_2dx
    grad[:, :, 1:-1, :, 2] = (u_vec[:, :, 2:, :] - u_vec[:, :, :-2, :]) * inv_2dx

    # F = I + ∇u
    F = grad.copy()
    F[..., 0, 0] += 1
    F[..., 1, 1] += 1
    F[..., 2, 2] += 1

    # C = F^T F
    C = np.einsum("...ki,...kj->...ij", F, F)

    # Principal stretches: sqrt of eigenvalues of C (symmetric)
    # eigvalsh returns ascending; reverse to descending
    C_flat = C.reshape(-1, 3, 3)
    eigvals_sorted = np.linalg.eigvalsh(C_flat)      # ascending
    eigvals_desc   = eigvals_sorted[:, ::-1]
    eigvals_desc[eigvals_desc < 1e-12] = 1e-12
    stretches = np.sqrt(eigvals_desc)
    return stretches.reshape(Nx, Ny, Nz, 3)
