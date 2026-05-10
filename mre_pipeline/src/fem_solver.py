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
    """Solve 2D time-harmonic shear wave equation with Dirichlet BCs.

    The four edges are zero-displacement Dirichlet boundaries.
    Wave excitation is provided by `sources`: a list of (i, j, amplitude)
    triples placing a point Dirichlet source at interior or boundary nodes.
    If `sources` is None, the legacy left-column source is used (back-compat).

    Returns complex (N,N) displacement array.
    """
    N = G.shape[0]
    assert G.shape == (N, N)

    omega = 2.0 * np.pi * freq
    Gc = G * (1.0 + 1j * damping)

    def idx(i, j):
        return i * N + j

    # Build BC node sets
    boundary = set()
    bc_values: dict[int, complex] = {}
    for i in range(N):
        boundary.add(idx(i, 0))
        boundary.add(idx(i, N - 1))
    for j in range(N):
        boundary.add(idx(0, j))
        boundary.add(idx(N - 1, j))

    if sources is None:
        # Back-compat: full left column = 1
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

            diag = -(g_e + g_w + g_n + g_s) / dx**2 + rho * omega**2
            A[k, k] = diag
            A[k, idx(i, j + 1)] = g_e / dx**2
            A[k, idx(i, j - 1)] = g_w / dx**2
            A[k, idx(i - 1, j)] = g_n / dx**2
            A[k, idx(i + 1, j)] = g_s / dx**2

    A_csr = A.tocsr()
    u_flat = spsolve(A_csr, b)
    return u_flat.reshape(N, N)


def random_sources(N: int, rng: np.random.Generator,
                    n_min: int = 1, n_max: int = 10,
                    patch_min: int = 4, patch_max: int = 12) -> list[tuple[int, int, complex]]:
    """Draw 1–10 random sources on the boundary of an N×N grid.

    Each "source" is a contiguous patch of `patch_min`–`patch_max` boundary
    pixels sharing the same complex amplitude (unit magnitude, random phase).
    Patches are larger than single pixels so the wave radiates coherently
    without single-cell discontinuity artifacts. Multiple patches give
    multi-directional wave fields, matching the spirit of ILI's 1–10 random
    force generators on the boundary.
    """
    n_src = int(rng.integers(n_min, n_max + 1))
    sources: list[tuple[int, int, complex]] = []
    used: set[tuple[int, int]] = set()

    for _ in range(n_src):
        edge = int(rng.integers(0, 4))
        patch_len = int(rng.integers(patch_min, patch_max + 1))
        start = int(rng.integers(0, N - patch_len))
        phase = float(rng.uniform(0, 2 * np.pi))
        amp = complex(np.exp(1j * phase))

        for k in range(patch_len):
            if edge == 0:    # top    (i=0)
                i, j = 0, start + k
            elif edge == 1:  # bottom (i=N-1)
                i, j = N - 1, start + k
            elif edge == 2:  # left   (j=0)
                i, j = start + k, 0
            else:            # right  (j=N-1)
                i, j = start + k, N - 1
            if (i, j) in used:
                continue
            used.add((i, j))
            sources.append((i, j, amp))
    return sources
