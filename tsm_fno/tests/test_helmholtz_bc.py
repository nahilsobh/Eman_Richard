"""Tests for helmholtz_fd boundary conditions.

Verifies:
  1. Default (clamped) BC is unchanged — regression guard.
  2. top_free=True enforces Neumann ∂u/∂z = 0 on the top row (non-corner).
  3. Skipping top sources works with top_free.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.solver.helmholtz_fd import (
    bottom_driver_sources,
    helmholtz_solve,
    random_sources,
)


def _uniform_G(N: int = 32, G_val: float = 2500.0) -> np.ndarray:
    return np.full((N, N), G_val, dtype=float)


def test_default_bc_clamps_top():
    G = _uniform_G(N=24)
    rng = np.random.default_rng(0)
    src = random_sources(24, rng, n_min=2, n_max=2)
    u = helmholtz_solve(G, freq=60.0, dx=0.003, sources=src, top_free=False)
    # With Dirichlet top, u[0, j] must be 0 at non-source cols.
    src_cols = {c for (i, c, _) in src if i == 0}
    interior_top = [u[0, j] for j in range(1, 23) if j not in src_cols]
    top_max = max(abs(v) for v in interior_top)
    assert top_max < 1e-12, f"top row should be pinned to 0, got max|u|={top_max}"


def test_top_free_soft_reflection_vs_clamped():
    # A clamped (Dirichlet) top pins the wave field to zero at row 0 —
    # a hard reflection. A free (Neumann) top lets the wave amplitude
    # be finite at row 0 — an antinode. So RMS amplitude at the top row
    # should be dramatically larger under top_free.
    G = _uniform_G(N=64)
    src = [(32, 0, 1.0 + 0.0j)]  # side source, mid-height

    u_clamp = helmholtz_solve(G, freq=60.0, dx=0.003, sources=src, top_free=False)
    u_free  = helmholtz_solve(G, freq=60.0, dx=0.003, sources=src, top_free=True)

    # Exclude corners.
    top_clamp = np.sqrt(np.mean(np.abs(u_clamp[0, 2:62]) ** 2))
    top_free  = np.sqrt(np.mean(np.abs(u_free[0,  2:62]) ** 2))
    # Sanity: the two solutions must differ meaningfully at the top face.
    assert top_free > 10 * top_clamp, \
        f"top-free row should be much larger than clamped: {top_free:.3g} vs {top_clamp:.3g}"
    # Sanity: far from the top face (row 32 = mid-height) the two agree
    # roughly, because BC influence hasn't propagated as strongly.
    mid_clamp = np.abs(u_clamp[32, 2:62])
    mid_free  = np.abs(u_free[32,  2:62])
    rel_mid   = np.linalg.norm(mid_clamp - mid_free) / (np.linalg.norm(mid_clamp) + 1e-30)
    assert rel_mid < 0.5, \
        f"mid-height fields diverge too much: rel diff {rel_mid:.3g}"


def test_top_free_flux_residual_at_top_row():
    # Direct algebraic check: the ghost-mirror stencil at row 0 should
    # give the Helmholtz equation residual = 0 when we substitute the
    # solved field. Uses the same coefficients as the assembly.
    G_val = 2500.0
    dx    = 0.003
    freq  = 60.0
    rho   = 1000.0
    xi    = 0.05
    G = np.full((32, 32), G_val)
    src = [(16, 0, 1.0 + 0.0j)]
    u = helmholtz_solve(G, freq=freq, dx=dx, damping=xi, sources=src, top_free=True)

    omega = 2 * np.pi * freq
    Gc = G_val * (1 + 1j * xi)
    # For uniform G, all half-point conductances = Gc.
    # Row 0 (non-corner), ghost mirror: u[-1] = u[1] ⇒
    #   residual = Gc/dx² * (u[1] + u[0,j-1] - 3 u[0] + u[0,j+1]) + ρω² u[0] - 2 Gc/dx² u[0] + 2 Gc/dx² u[1]
    # Cleaner form using the assembled equation:
    #   -(g_e+g_w+g_n+g_s)/dx² * u[0,j] + g_e u[0,j+1]/dx² + g_w u[0,j-1]/dx² + (g_n+g_s) u[1,j]/dx² + ρω² u[0,j] = 0
    j_range = np.arange(2, 30)  # interior columns
    resid = (
        -(4 * Gc) / dx ** 2 * u[0, j_range]
        + Gc / dx ** 2 * u[0, j_range + 1]
        + Gc / dx ** 2 * u[0, j_range - 1]
        + (2 * Gc) / dx ** 2 * u[1, j_range]
        + rho * omega ** 2 * u[0, j_range]
    )
    scale = np.max(np.abs(u)) * (4 * abs(Gc) / dx ** 2 + rho * omega ** 2)
    rel   = np.max(np.abs(resid)) / (scale + 1e-30)
    assert rel < 1e-10, f"row-0 stencil residual too large: {rel:.3g}"


def test_random_sources_skip_top_excludes_top_edge():
    # skip_top must exclude the top *edge* (edge index 0); corners of the
    # top row can still appear via the side edges (edges 2, 3) — those are
    # boundary nodes regardless.
    rng = np.random.default_rng(42)
    for _ in range(50):
        N = 40
        src = random_sources(N, rng, n_min=5, n_max=10, skip_top=True)
        for (i, j, _) in src:
            if i == 0 and j not in (0, N - 1):
                pytest.fail(f"non-corner top-row source (i=0, j={j}) with skip_top")


def test_top_free_matches_dirichlet_when_source_on_top():
    # If a source is placed on the top edge, top_free still honors it
    # as a Dirichlet override at that node. Verify the solver runs and
    # the source value is preserved.
    G = _uniform_G(N=20)
    src = [(0, 10, 1.0 + 0.0j)]  # single point source on top
    u = helmholtz_solve(G, freq=60.0, dx=0.003, sources=src, top_free=True)
    assert np.isclose(u[0, 10], 1.0 + 0.0j), \
        f"source override not honored under top_free; got u[0,10]={u[0, 10]}"


def test_bottom_driver_geometry():
    N = 80
    src = bottom_driver_sources(N, width_frac=0.5)
    # All nodes on the bottom row.
    assert all(i == N - 1 for (i, _, _) in src), \
        "bottom driver must place all sources on i = N-1"
    # Exactly 40 nodes for width_frac=0.5, N=80.
    assert len(src) == 40, f"expected 40 nodes, got {len(src)}"
    # Centred: columns span [20, 59].
    cols = sorted(j for (_, j, _) in src)
    assert cols == list(range(20, 60)), f"columns wrong: {cols[:3]}...{cols[-3:]}"
    # Coherent phase — all amps identical.
    amps = {a for (_, _, a) in src}
    assert amps == {1.0 + 0.0j}, f"driver phase not coherent: {amps}"


def test_bottom_driver_produces_upward_wave():
    # With a bottom driver and a free top, the wave should propagate upward
    # from row N-1 toward row 0 — expect meaningful field amplitude across
    # the whole interior, not just near the driver.
    G = _uniform_G(N=64)
    src = bottom_driver_sources(64, width_frac=0.5)
    u = helmholtz_solve(G, freq=60.0, dx=0.003, sources=src, top_free=True)
    # Amplitude near the driver (row 60).
    near = np.max(np.abs(u[60, 2:62]))
    # Amplitude in the middle (row 32).
    mid  = np.max(np.abs(u[32, 2:62]))
    # Amplitude near the free top (row 3).
    far  = np.max(np.abs(u[3,  2:62]))
    assert near > 0.5, f"near-driver amplitude too small: {near:.3g}"
    # Wave must reach the middle at meaningful amplitude.
    assert mid > 0.05 * near, \
        f"wave not propagating: mid/near = {mid/near:.3g}"
    # And should still show at the free top (soft reflection = antinode).
    assert far > 1e-3, f"no field at free top: {far:.3g}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
