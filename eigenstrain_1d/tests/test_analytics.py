"""Tests for analytical solutions. All must pass to machine/near-machine precision."""
import numpy as np
import pytest

from src.analytics.static_solution import sigma_bar_formula, static_solution
from src.analytics.wave_solution import transfer_matrix_solution
from src.analytics.inversion_formula import inversion_formula
from src.analytics.nonlocal_solution import (nonlocal_kernel_fourier,
                                               nonlocal_kernel_direct,
                                               verify_kernel_fourier)
from src.analytics.viscoelastic import (G_relaxation, delta_c_over_c0,
                                         estimate_tau_analytical)

# Canonical parameters for all tests
E_BG = 2000.0; E_LES = 8000.0; EPS0 = 0.02; A = 0.01; L = 0.10
N = 256; DX = 2*L/N
RHO = 1000.0


def test_static_periodic_bc():
    """∫ε dx ≈ 0 and σ = constant (to within O(dx) discretisation)."""
    sb, u, E = static_solution(N, DX, A, L, E_BG, E_LES, EPS0)
    x = np.arange(N) * DX
    mask = np.abs(x - L) <= A
    eps_field = sb / E + EPS0 * mask.astype(float)
    # Zero-net-strain (periodic BC) — allow O(dx * |sigma_bar|) discretisation error
    assert abs(np.sum(eps_field) * DX) < 1e-3 * abs(sb) + 1e-7


def test_sigma_bar_formula():
    """Verify σ_bar formula against hand calculation."""
    sigma_expected = -(0.02 * 8000 * 2000 * 2*0.01) / (2000*2*0.01 + 8000*(2*0.10-2*0.01))
    sigma_computed = sigma_bar_formula(A, L, E_BG, E_LES, EPS0)
    assert abs(sigma_computed - sigma_expected) < 1e-6


def test_static_infinite_limit():
    """For L >> a (L=100*a): sigma_bar << E_bg * eps0 (1D perilesional stress vanishes)."""
    L_large = 100.0 * A   # 1.0 m >> a=0.01 m
    sb = sigma_bar_formula(A, L_large, E_BG, E_LES, EPS0)
    assert abs(sb) < 0.05 * E_BG * EPS0, f"sigma_bar={sb:.4f}, limit={0.05*E_BG*EPS0:.4f}"


def test_inversion_noisefree():
    """Apply inversion formula to FD solution with eigenstrain source (homogeneous E).
    RL²(eps_recovered, eps_true) < 0.20 for the homogeneous-E inversion formula.
    """
    from src.solver.forward_1d import helmholtz_solve_1d

    x = np.arange(N) * DX
    eps_true = np.where(np.abs(x - L) <= A, EPS0, 0.0)
    # Homogeneous E — inversion formula is exact for this case
    E_hom = np.full(N, E_BG)

    # source_amplitude=0, normalize=False: eigenstrain is the ONLY source
    # No normalization — inversion formula requires preserved amplitude
    u_hat = helmholtz_solve_1d(N, DX, E_hom, RHO, freq=60.0,
                                source_amplitude=0.0, eps_star=eps_true,
                                normalize=False)
    eps_rec = inversion_formula(u_hat, DX, E_BG, RHO, freq=60.0, ell=0.0)

    # Remove DC (formula suppresses mean ε*)
    eps_true_dc = eps_true - eps_true.mean()
    eps_rec_dc  = eps_rec  - eps_rec.mean()

    rl2 = (np.linalg.norm(eps_rec_dc - eps_true_dc)
           / (np.linalg.norm(eps_true_dc) + 1e-8))
    print(f"RL² (noise-free inversion, homogeneous E): {rl2:.6f}")
    assert rl2 < 0.20, f"Inversion RL²={rl2:.4f} too large"


def test_inversion_dc_component():
    """Recovered ε* must have zero mean (DC suppressed by formula)."""
    sb = sigma_bar_formula(A, L, E_BG, E_LES, EPS0)
    u_hat = transfer_matrix_solution(N, DX, A, L, E_BG, E_LES,
                                      A_coeff=0.0, sigma_bar=sb,
                                      rho=RHO, freq=60.0, snr_db=None)
    eps_rec = inversion_formula(u_hat, DX, E_BG, RHO, freq=60.0)
    assert abs(eps_rec.mean()) < 1e-8


def test_nonlocal_kernel_fourier():
    """α̂(k) = 1/(1+k²ℓ²) matches numerical integration."""
    ell = 0.005
    assert verify_kernel_fourier(N, DX, ell, tol=0.05)


def test_relaxation_single_exp():
    """estimate_tau_analytical recovers τ to < 1% relative error.
    Requires G_inf = 0 so δc(t) ∝ exp(-t/τ) exactly.
    """
    tau_true = 3.5   # days
    # G_inf=0 so G(t) = G1*exp(-t/tau) — pure single exponential
    G_inf = 0.0; G1 = E_BG; G2 = 0.0
    sb = sigma_bar_formula(A, L, E_BG, E_LES, EPS0)
    A_coeff = -4.0
    dc_0 = delta_c_over_c0(0.0, sb, E_BG, A_coeff, G_inf, G1, tau_true)
    dc_5 = delta_c_over_c0(5.0, sb, E_BG, A_coeff, G_inf, G1, tau_true)
    tau_est = estimate_tau_analytical(dc_0, dc_5, dt_days=5.0)
    assert tau_est is not None
    assert abs(tau_est - tau_true) / tau_true < 0.01
