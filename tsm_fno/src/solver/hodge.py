"""2D Helmholtz-Hodge decomposition and TSM directional filter.

Decomposes a complex 2D vector displacement field u = (u_x, u_y) into

    u = ∇φ  (curl-free, compressional) + ∇⊥ψ (divergence-free, shear)

via the Fourier projection

    φ̂(k) = (kx · Û_x + ky · Û_y) / k²
    ψ̂(k) = (kx · Û_y − ky · Û_x) / k²

The directional filter selects shear-wave components propagating
parallel to the lesion-host interface (i.e. perpendicular to the local
outward normal n̂), which is the operation that exposes the TSM
stiffening ring.
"""
from __future__ import annotations

import numpy as np


def hodge_decompose_2d(
    ux: np.ndarray,
    uy: np.ndarray,
    dx: float = 1.0,
) -> dict:
    """Helmholtz-Hodge decomposition of a complex 2D vector field.

    Returns a dict with keys:
      'div_free'  : (ux_df, uy_df) — shear (curl-bearing) component
      'curl_free' : (ux_cf, uy_cf) — compressional (div-bearing) component
      'phi'       : scalar potential of the curl-free part
      'psi'       : stream function of the div-free part
    """
    assert ux.shape == uy.shape and ux.ndim == 2
    Ny, Nx = ux.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dx)[:, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)[None, :]
    k2 = kx ** 2 + ky ** 2
    k2_safe = np.where(k2 == 0.0, 1.0, k2)

    Ux = np.fft.fft2(ux)
    Uy = np.fft.fft2(uy)

    phi_hat = (kx * Ux + ky * Uy) / k2_safe
    psi_hat = (kx * Uy - ky * Ux) / k2_safe
    phi_hat[k2 == 0.0] = 0.0
    psi_hat[k2 == 0.0] = 0.0

    Ux_cf = kx * phi_hat
    Uy_cf = ky * phi_hat
    Ux_df = Ux - Ux_cf
    Uy_df = Uy - Uy_cf

    ux_df = np.fft.ifft2(Ux_df)
    uy_df = np.fft.ifft2(Uy_df)
    ux_cf = np.fft.ifft2(Ux_cf)
    uy_cf = np.fft.ifft2(Uy_cf)
    phi   = np.fft.ifft2(phi_hat)
    psi   = np.fft.ifft2(psi_hat)

    return {
        "div_free":  (ux_df, uy_df),
        "curl_free": (ux_cf, uy_cf),
        "phi":       phi,
        "psi":       psi,
    }


def _angle_window(theta_grid: np.ndarray, theta_centre: float,
                   half_width: float) -> np.ndarray:
    """Smooth angular bandpass: 1 on the centre, 0 outside half_width."""
    diff = np.angle(np.exp(1j * (theta_grid - theta_centre)))
    return np.clip(1.0 - np.abs(diff) / half_width, 0.0, 1.0)


def directional_filter_tsm(
    ux_df: np.ndarray,
    uy_df: np.ndarray,
    boundary_normals: np.ndarray,
    dx: float = 1.0,
    n_angles: int = 36,
    delta_threshold: float = 0.30,
) -> dict:
    """TSM-style angular MIP over directionally filtered shear components.

    Parameters
    ----------
    ux_df, uy_df : (N, N) complex
        Divergence-free shear components from `hodge_decompose_2d`.
    boundary_normals : (N, N, 2)
        Local outward normals to the lesion surface (zero inside lesion).
    n_angles : int
        Number of propagation directions sampled in [0, π].
    delta_threshold : float
        Maximum |k̂ · n̂| accepted into the tangential band — directions
        within ±delta_threshold radians of perpendicular to n̂ are kept.

    Returns
    -------
    dict with:
      'u_tsm'      : (N, N) complex — sum of tangentially propagating shear
                      components (the TSM signal).
      'mip_index'  : (N, N) int     — direction index that contributes the
                      maximum magnitude to each pixel.
      'angles'     : (n_angles,)    — propagation angles in radians.
    """
    Ny, Nx = ux_df.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dx)[:, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)[None, :]
    k_mag = np.sqrt(kx ** 2 + ky ** 2)
    k_mag_safe = np.where(k_mag == 0.0, 1.0, k_mag)
    theta = np.arctan2(ky, kx)

    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    half_width = float(np.pi / n_angles)

    Ux_df = np.fft.fft2(ux_df)
    Uy_df = np.fft.fft2(uy_df)

    nx = boundary_normals[..., 0]
    ny = boundary_normals[..., 1]

    u_mag_max = np.zeros((Ny, Nx), dtype=np.float64)
    mip_index = np.full((Ny, Nx), -1, dtype=np.int32)
    u_tsm = np.zeros((Ny, Nx), dtype=complex)

    for d, theta_c in enumerate(angles):
        win = _angle_window(theta, theta_c, half_width)
        win_neg = _angle_window(theta, theta_c - np.pi, half_width)  # symmetric
        win = np.maximum(win, win_neg)
        comp = np.fft.ifft2(win * (np.cos(theta_c) * Ux_df +
                                     np.sin(theta_c) * Uy_df))

        # Tangential gate based on local surface normal
        khat = np.array([np.cos(theta_c), np.sin(theta_c)])
        # |k̂ · n̂| close to zero ⇒ tangential ⇒ keep
        gate = np.clip(1.0 - np.abs(khat[0] * nx + khat[1] * ny) /
                              max(delta_threshold, 1e-9),
                        0.0, 1.0)
        comp_gated = comp * gate

        mag = np.abs(comp_gated)
        replace = mag > u_mag_max
        u_mag_max = np.where(replace, mag, u_mag_max)
        mip_index = np.where(replace, d, mip_index)
        u_tsm = np.where(replace, comp_gated, u_tsm)

    return {
        "u_tsm":     u_tsm,
        "mip_index": mip_index,
        "angles":    angles,
    }
