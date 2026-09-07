"""Tests for the isotropic-heterogeneous vector Navier solver.

Small grid tests (N=12) to keep the direct sparse solve fast (~10 s each).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.solver.vector_elasticity_3d import (
    curl_of_displacement_3d,
    navier_solve_3d_isotropic,
)


N_SMALL = 12
DX = 0.005
FREQ = 60.0
RHO = 1000.0
MU0 = 2500.0


def _uniform_mu(N=N_SMALL, val=MU0):
    return np.full((N, N, N), val, dtype=float)


def test_solver_runs_and_returns_correct_shape():
    N = N_SMALL
    mu = _uniform_mu(N)
    # Single side-face source: y-polarised in the middle of the +i face.
    src = [(N - 1, N // 2, N // 2, 1, 1.0 + 0.0j)]
    u = navier_solve_3d_isotropic(mu, lam=0.0, freq=FREQ, rho=RHO, dx=DX,
                                    damping=0.0, sources=src)
    assert u.shape == (N, N, N, 3), f"expected (N,N,N,3), got {u.shape}"
    assert np.issubdtype(u.dtype, np.complexfloating)


def test_source_dirichlet_honoured():
    N = N_SMALL
    mu = _uniform_mu(N)
    amp = 0.7 + 0.3j
    src = [(N - 1, N // 2, N // 2, 1, amp)]
    u = navier_solve_3d_isotropic(mu, lam=0.0, freq=FREQ, rho=RHO, dx=DX,
                                    damping=0.05, sources=src)
    assert np.isclose(u[N - 1, N // 2, N // 2, 1], amp), \
        f"Dirichlet source not honoured: got {u[N-1, N//2, N//2, 1]}"


def test_uniform_mu_wave_amplitude_finite():
    # Solver must produce a bounded finite field on a uniform-μ homogeneous
    # medium — no NaN/Inf.
    N = N_SMALL
    mu = _uniform_mu(N)
    src = [(N - 1, N // 2, N // 2, 1, 1.0 + 0.0j)]
    u = navier_solve_3d_isotropic(mu, lam=0.0, freq=FREQ, rho=RHO, dx=DX,
                                    damping=0.05, sources=src)
    assert np.all(np.isfinite(u)), "vector u contains non-finite entries"
    # Interior amplitude should be nonzero.
    inner = u[2:-2, 2:-2, 2:-2]
    assert np.max(np.abs(inner)) > 1e-4, \
        f"interior amplitude vanishingly small: {np.max(np.abs(inner)):.3g}"


def test_both_stiffnesses_give_finite_nonzero_field():
    # Different μ values give different standing-wave patterns in a
    # bounded Dirichlet-walled cube (which particular pattern is
    # brighter at a specific voxel depends on cavity resonances). Just
    # check that both solves produce a physical (finite, nonzero) field.
    N = N_SMALL
    src = [(N - 1, N // 2, N // 2, 1, 1.0 + 0.0j)]
    for mu_val in (500.0, 2500.0, 20000.0):
        u = navier_solve_3d_isotropic(_uniform_mu(N, mu_val), lam=0.0,
                                        freq=FREQ, rho=RHO, dx=DX,
                                        damping=0.05, sources=src)
        assert np.all(np.isfinite(u)), f"non-finite u at μ={mu_val}"
        amp = np.max(np.abs(u[2:-2, 2:-2, 2:-2, :]))
        assert amp > 1e-4, f"amplitude vanishing at μ={mu_val}: {amp:.3g}"


def test_curl_of_uniform_vec_is_zero():
    N = N_SMALL
    u = np.ones((N, N, N, 3), dtype=complex) * (2.0 + 1.0j)
    curl = curl_of_displacement_3d(u, dx=DX)
    assert np.allclose(curl[2:-2, 2:-2, 2:-2], 0.0)


def test_curl_extracts_rotational_component():
    # Analytic vector field with a nonzero curl: u = (−y, x, 0) → curl = (0, 0, 2)
    # meshgrid indexing="ij" returns arrays that vary along axes 0, 1, 2.
    # Naming convention here: axis 0 = x, axis 1 = y, axis 2 = z.
    N = 16
    xs = np.arange(N).astype(float)
    x, y, z = np.meshgrid(xs, xs, xs, indexing="ij")
    u = np.zeros((N, N, N, 3), dtype=complex)
    u[..., 0] = -y   # u_x = -y
    u[..., 1] =  x   # u_y =  x
    curl = curl_of_displacement_3d(u, dx=1.0)
    # ∂_x u_y - ∂_y u_x = 1 - (-1) = 2, in the z-component.
    assert np.allclose(curl[2:-2, 2:-2, 2:-2, 0], 0.0, atol=1e-8)
    assert np.allclose(curl[2:-2, 2:-2, 2:-2, 1], 0.0, atol=1e-8)
    assert np.allclose(curl[2:-2, 2:-2, 2:-2, 2], 2.0, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
