"""Direct inversion (DI) baseline for 2D MRE.

Algebraic inversion of the Helmholtz equation under a local-homogeneity
assumption:
        ∇·(G ∇u) + ρω² u = 0
        G Δu + ρω² u ≈ 0
        ⇒ G = −ρω² u / Δu

We compute the Laplacian with a 3×3 finite-difference stencil on a
smoothed displacement field (Gaussian smoothing as a 2D analogue of
Romano's quartic kernel). The estimate is the magnitude of the complex
shear modulus and is clipped to the physical range used for training.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, laplace


RHO_DEFAULT = 1000.0
FREQ_DEFAULT = 60.0
SMOOTH_SIGMA = 2.5          # voxels of Gaussian smoothing (analogous to ILI's quartic kernel)
G_MIN_PA = 100.0
G_MAX_PA = 60000.0
BOUNDARY_PAD = 4            # mask N pixels near the boundary (DI is unreliable there)


def direct_inversion(
    u: np.ndarray,
    freq: float = FREQ_DEFAULT,
    rho: float = RHO_DEFAULT,
    dx: float = 0.002,
    smooth_sigma: float = SMOOTH_SIGMA,
) -> np.ndarray:
    """Algebraic DI on a complex 2D displacement field.

    Returns a real-valued (N,N) shear-modulus estimate in Pa.
    """
    omega = 2.0 * np.pi * freq

    # Smooth real and imaginary parts independently
    u_smooth = (
        gaussian_filter(u.real, smooth_sigma)
        + 1j * gaussian_filter(u.imag, smooth_sigma)
    )

    # 3×3 discrete Laplacian (use scipy.ndimage.laplace for boundary handling)
    lap = (
        laplace(u_smooth.real)
        + 1j * laplace(u_smooth.imag)
    ) / (dx ** 2)

    # Avoid division by zero / weakly-driven regions
    lap_mag = np.abs(lap)
    eps = 0.05 * lap_mag.max()
    safe_denom = np.where(lap_mag < eps, eps + 0j, lap)
    G_complex = -rho * omega ** 2 * u_smooth / safe_denom

    # Take magnitude of complex shear modulus
    G_mag = np.abs(G_complex)

    # Mask boundary band — DI cannot estimate stiffness near Dirichlet edges
    N = G_mag.shape[0]
    out = np.full_like(G_mag, np.median(G_mag))
    out[BOUNDARY_PAD:N - BOUNDARY_PAD, BOUNDARY_PAD:N - BOUNDARY_PAD] = \
        G_mag[BOUNDARY_PAD:N - BOUNDARY_PAD, BOUNDARY_PAD:N - BOUNDARY_PAD]

    return np.clip(out, G_MIN_PA, G_MAX_PA).astype(np.float32)


def direct_inversion_from_X(
    X: np.ndarray,
    freq: float = FREQ_DEFAULT,
    rho: float = RHO_DEFAULT,
    dx: float = 0.002,
    smooth_sigma: float = SMOOTH_SIGMA,
) -> np.ndarray:
    """Convenience wrapper accepting the (2,N,N) network-input format.

    X[0] = Re(u),  X[1] = Im(u). We assume the field has been globally
    rescaled so |u| ≤ 1; the rescaling cancels in the algebraic inversion.
    """
    assert X.shape[0] == 2, f"Expected 2 channels, got {X.shape}"
    u = X[0] + 1j * X[1]
    return direct_inversion(u, freq=freq, rho=rho, dx=dx, smooth_sigma=smooth_sigma)
