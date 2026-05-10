import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.solver.hodge import hodge_decompose_2d, directional_filter_tsm


def _make_test_field(N=64, dx=1.0, seed=0):
    """A complex 2D vector field with both div and curl content."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N] * dx
    # Pure shear (div-free): u = ∇⊥ψ for ψ = sin(2π x/L)·sin(2π y/L)
    L = N * dx
    psi = np.sin(2 * np.pi * xx / L) * np.sin(2 * np.pi * yy / L)
    # ∇⊥ψ = (∂ψ/∂y, -∂ψ/∂x)
    ux_df = (2 * np.pi / L) * np.sin(2 * np.pi * xx / L) * np.cos(2 * np.pi * yy / L)
    uy_df = -(2 * np.pi / L) * np.cos(2 * np.pi * xx / L) * np.sin(2 * np.pi * yy / L)
    # Pure compressional (curl-free): u = ∇φ for φ = sin(2π x/L)·cos(2π y/L)
    ux_cf = (2 * np.pi / L) * np.cos(2 * np.pi * xx / L) * np.cos(2 * np.pi * yy / L)
    uy_cf = -(2 * np.pi / L) * np.sin(2 * np.pi * xx / L) * np.sin(2 * np.pi * yy / L)
    # Combine, with random complex amplitudes per channel
    ax_df = rng.standard_normal() + 1j * rng.standard_normal()
    ax_cf = rng.standard_normal() + 1j * rng.standard_normal()
    ux = ax_df * ux_df + ax_cf * ux_cf
    uy = ax_df * uy_df + ax_cf * uy_cf
    return ux.astype(complex), uy.astype(complex), dx


def test_decomposition_complete():
    ux, uy, dx = _make_test_field()
    out = hodge_decompose_2d(ux, uy, dx=dx)
    ux_df, uy_df = out["div_free"]
    ux_cf, uy_cf = out["curl_free"]
    err_x = np.max(np.abs(ux - (ux_df + ux_cf)))
    err_y = np.max(np.abs(uy - (uy_df + uy_cf)))
    assert err_x < 1e-9, f"x reconstruction err {err_x}"
    assert err_y < 1e-9, f"y reconstruction err {err_y}"


def _spectral_div(ux, uy, dx):
    Ny, Nx = ux.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dx)[:, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)[None, :]
    Ux = np.fft.fft2(ux)
    Uy = np.fft.fft2(uy)
    return np.fft.ifft2(1j * (kx * Ux + ky * Uy))


def _spectral_curl(ux, uy, dx):
    Ny, Nx = ux.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dx)[:, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)[None, :]
    Ux = np.fft.fft2(ux)
    Uy = np.fft.fft2(uy)
    return np.fft.ifft2(1j * (kx * Uy - ky * Ux))


def test_div_free_is_divergence_free():
    ux, uy, dx = _make_test_field()
    out = hodge_decompose_2d(ux, uy, dx=dx)
    ux_df, uy_df = out["div_free"]
    div = _spectral_div(ux_df, uy_df, dx)
    assert np.max(np.abs(div)) < 1e-8


def test_curl_free_is_curl_free():
    ux, uy, dx = _make_test_field()
    out = hodge_decompose_2d(ux, uy, dx=dx)
    ux_cf, uy_cf = out["curl_free"]
    curl = _spectral_curl(ux_cf, uy_cf, dx)
    assert np.max(np.abs(curl)) < 1e-8


def test_directional_filter_shape():
    ux, uy, dx = _make_test_field()
    out = hodge_decompose_2d(ux, uy, dx=dx)
    ux_df, uy_df = out["div_free"]
    N = ux_df.shape[0]
    normals = np.zeros((N, N, 2), dtype=np.float64)
    normals[..., 0] = 1.0
    res = directional_filter_tsm(ux_df, uy_df, normals, dx=dx, n_angles=8)
    assert res["u_tsm"].shape == ux_df.shape
    assert res["mip_index"].shape == ux_df.shape
