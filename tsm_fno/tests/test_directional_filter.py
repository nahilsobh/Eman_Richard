"""Tests for the k-space directional filter and broadband sources."""
from __future__ import annotations

import numpy as np
import pytest

from src.solver.helmholtz_fd_3d import (
    directional_filter_3d,
    multi_face_broadband_sources,
)


def _plane_wave(N: int, khat: np.ndarray, wavelength_vx: float = 8.0) -> np.ndarray:
    """A complex plane wave A·exp(i·k·x) sampled on the N³ grid."""
    khat = np.asarray(khat, dtype=np.float64)
    khat = khat / np.linalg.norm(khat)
    ii, jj, kk = np.mgrid[0:N, 0:N, 0:N].astype(np.float64)
    kmag = 2.0 * np.pi / wavelength_vx
    phase = kmag * (khat[0] * ii + khat[1] * jj + khat[2] * kk)
    return np.exp(1j * phase)


def test_matching_direction_passes_plane_wave():
    # Filter aligned with the wave direction should keep most of the energy.
    N = 32
    khat = np.array([1., 0., 0.])
    u = _plane_wave(N, khat, wavelength_vx=8.0)
    u_filt = directional_filter_3d(u, khat=khat, angular_width=0.35)
    # Core of the field (avoid edges) — filtered amplitude should be close.
    core = slice(4, N - 4)
    energy_in  = np.sum(np.abs(u[core, core, core]) ** 2)
    energy_out = np.sum(np.abs(u_filt[core, core, core]) ** 2)
    keep_frac = energy_out / energy_in
    # Band-pass drops some energy (DC, high-freq); wedge keeps most of the wave.
    assert keep_frac > 0.5, f"matching-direction filter should preserve field, got {keep_frac:.2%}"


def test_orthogonal_direction_suppresses_plane_wave():
    N = 32
    khat_wave   = np.array([1., 0., 0.])
    khat_filter = np.array([0., 1., 0.])   # orthogonal
    u = _plane_wave(N, khat_wave, wavelength_vx=8.0)
    u_filt = directional_filter_3d(u, khat=khat_filter, angular_width=0.35)
    core = slice(4, N - 4)
    energy_in  = np.sum(np.abs(u[core, core, core]) ** 2)
    energy_out = np.sum(np.abs(u_filt[core, core, core]) ** 2)
    suppress = energy_out / energy_in
    # Orthogonal filter should kill nearly all of it.
    assert suppress < 0.02, f"orthogonal filter should suppress, kept {suppress:.2%}"


def test_symmetric_under_negation_of_khat():
    # ±k̂ are the same physical wave — filter must be sign-invariant.
    N = 24
    khat = np.array([0.6, 0.8, 0.0])
    u = _plane_wave(N, khat, wavelength_vx=6.0)
    u_pos = directional_filter_3d(u, khat=khat)
    u_neg = directional_filter_3d(u, khat=-khat)
    assert np.allclose(u_pos, u_neg, atol=1e-10)


def test_bandpass_excludes_dc():
    # A constant field is pure DC; must be filtered out.
    N = 24
    u = np.ones((N, N, N), dtype=complex) * (3.0 + 2.0j)
    u_filt = directional_filter_3d(u, khat=np.array([1., 0., 0.]))
    assert np.max(np.abs(u_filt)) < 1e-10


def test_narrow_wedge_kills_more_than_wide():
    N = 24
    # Wave slightly off the filter axis.
    khat_wave   = np.array([1., 0.5, 0.])
    khat_wave  /= np.linalg.norm(khat_wave)
    khat_filter = np.array([1., 0., 0.])
    u = _plane_wave(N, khat_wave, wavelength_vx=6.0)
    core = slice(4, N - 4)
    e_wide   = np.sum(np.abs(directional_filter_3d(u, khat_filter, angular_width=0.6)[core, core, core]) ** 2)
    e_narrow = np.sum(np.abs(directional_filter_3d(u, khat_filter, angular_width=0.2)[core, core, core]) ** 2)
    assert e_narrow < e_wide, f"narrower wedge must reject off-axis energy more: wide={e_wide:.3g}, narrow={e_narrow:.3g}"


def test_multi_face_source_places_on_requested_faces():
    N = 20
    src = multi_face_broadband_sources(N, radius_frac=0.5,
                                        faces=("iN", "j0", "kN"))
    faces_seen = set()
    for (i, j, k, _) in src:
        if i == N - 1: faces_seen.add("iN")
        if i == 0:     faces_seen.add("i0")
        if j == 0:     faces_seen.add("j0")
        if j == N - 1: faces_seen.add("jN")
        if k == N - 1: faces_seen.add("kN")
        if k == 0:     faces_seen.add("k0")
    assert "iN" in faces_seen and "j0" in faces_seen and "kN" in faces_seen
    assert "i0" not in faces_seen and "jN" not in faces_seen and "k0" not in faces_seen


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
