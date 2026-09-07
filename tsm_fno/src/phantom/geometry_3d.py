"""3D spherical-balloon phantom for the 3D Helmholtz demo.

Analogue of ``geometry.py`` + ``acoustoelastic.py``, restricted to the
spherical case needed for the paper demo. Provides the intrinsic and
acoustoelastic-effective stiffness fields, plus the perilesional shell.

The Lamé pre-stress around a pressurised spherical inclusion in an
infinite matrix is
    σ_rr(r) = -p · (a/r)^3
    σ_θθ(r) = σ_φφ(r) = +½ p · (a/r)^3
    Δσ(r)   = σ_θθ − σ_rr = (3/2) p · (a/r)^3    (outside, r ≥ a)
    Δσ      = p                                    (inside, uniform)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


G_MIN_PA = 200.0
G_MAX_PA = 80000.0


@dataclass
class SphericalBalloon:
    center: tuple[float, float, float]   # (i, j, k) in voxels
    radius_vx: float
    pressure: float                       # Pa

    def _coords(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ii, jj, kk = np.mgrid[0:N, 0:N, 0:N].astype(np.float64)
        return (ii - self.center[0],
                jj - self.center[1],
                kk - self.center[2])

    def mask(self, N: int) -> np.ndarray:
        di, dj, dk = self._coords(N)
        return (di ** 2 + dj ** 2 + dk ** 2) <= self.radius_vx ** 2

    def distance_field(self, N: int, dx: float) -> np.ndarray:
        outside = ~self.mask(N)
        d = distance_transform_edt(outside).astype(np.float64)
        return d * dx


def lame_field_sphere(balloon: SphericalBalloon, N: int) -> np.ndarray:
    """Lamé deviatoric pre-stress Δσ(x) for a pressurised sphere."""
    if balloon.pressure == 0.0:
        return np.zeros((N, N, N), dtype=np.float64)

    di, dj, dk = balloon._coords(N)
    r = np.sqrt(di ** 2 + dj ** 2 + dk ** 2)
    a = balloon.radius_vx
    inside = balloon.mask(N)
    safe_r = np.maximum(r, a)
    out_decay = 1.5 * balloon.pressure * (a / safe_r) ** 3
    field = np.where(inside, balloon.pressure, out_decay)
    return field.astype(np.float64)


def stress_tensor_sphere(balloon: SphericalBalloon, N: int) -> np.ndarray:
    """Cauchy pre-stress tensor σ_ij(x) for a pressurised sphere.

    In spherical coordinates centred on the balloon, the classical Lamé
    solution outside a pressurised spherical inclusion gives
        σ_rr(r)  = -p · (a/r)^3        (radial compression)
        σ_θθ(r)  = σ_φφ(r) = +½ p · (a/r)^3   (tangential tension)
    with all off-diagonal components zero in the spherical frame.

    Converting to Cartesian via r̂ = x/|x|:
        σ_ij(x) = σ_rr(r) · r̂_i r̂_j + σ_θθ(r) · (δ_ij − r̂_i r̂_j)
    Inside the balloon, σ_ij = p · δ_ij (uniform isotropic).

    Returns
    -------
    (N, N, N, 3, 3) ndarray of stress tensors [Pa].
    """
    if balloon.pressure == 0.0:
        return np.zeros((N, N, N, 3, 3), dtype=np.float64)

    di, dj, dk = balloon._coords(N)
    r = np.sqrt(di ** 2 + dj ** 2 + dk ** 2)
    a = balloon.radius_vx
    safe_r = np.maximum(r, 1e-12)
    rhat = np.stack([di / safe_r, dj / safe_r, dk / safe_r], axis=-1)  # (N,N,N,3)

    outside_r = np.maximum(r, a)
    sigma_rr = -balloon.pressure * (a / outside_r) ** 3
    sigma_tt =  0.5 * balloon.pressure * (a / outside_r) ** 3

    # σ_ij = σ_rr r̂_i r̂_j + σ_θθ (δ_ij − r̂_i r̂_j)
    rr = rhat[..., :, None] * rhat[..., None, :]           # (N,N,N,3,3)
    delta = np.eye(3)                                       # (3,3)
    sigma = (sigma_rr[..., None, None] * rr
             + sigma_tt[..., None, None] * (delta - rr))

    # Inside the balloon: uniform isotropic pressure σ_ij = p·δ_ij.
    inside = balloon.mask(N)
    sigma[inside] = balloon.pressure * delta
    return sigma.astype(np.float64)


def make_anisotropic_G_tensor(N: int, balloon: SphericalBalloon,
                               G_bg: float, G_lesion: float,
                               A_coeff: float,
                               stiffening_exponent: float = 1.0,
                               constitutive: str = "powerlaw") -> np.ndarray:
    """Rank-2 anisotropic stiffness tensor field G_ij(x) [Pa].

    Linear form (m=1, default): `G_ij = G_base·δ_ij + A·σ_ij`. For plane
    waves this gives effective scalar `k̂·G·k̂ = G_base + A·(k̂·σ·k̂)`,
    matching `effective_G_for_direction`.

    Nonlinear form (m > 1): apply the constitutive law in the *principal
    axis frame* of the stress tensor, then rotate back. For the sphere
    the principal directions are (r̂, θ̂, φ̂) with principal stresses
    (σ_rr, σ_θθ, σ_φφ = σ_θθ) — a spherically symmetric decomposition.
    The principal stiffnesses are

        powerlaw:  G_r/θ = G_base · (1 + A·σ_r/θ/G_base)^m
        ogden:     G_r/θ = G_base · ½·[(1+A·σ_r/θ/G_base)^m
                                       + (1+A·σ_r/θ/G_base)^-m]

    Then the Cartesian tensor is
        G_ij(x) = G_r · r̂_i r̂_j + G_θ · (δ_ij − r̂_i r̂_j).
    At m=1 this reduces exactly to the linear form above. For m>1 this
    is the correct extension — element-wise nonlinearity of σ_ij would
    mix off-diagonal components incoherently.
    """
    G_base = np.full((N, N, N), float(G_bg), dtype=np.float64)
    G_base[balloon.mask(N)] = float(G_lesion)

    if stiffening_exponent == 1.0 and constitutive == "powerlaw":
        # Fast path: element-wise linear form.
        sigma = stress_tensor_sphere(balloon, N)
        G_tensor = float(A_coeff) * sigma
        for c in range(3):
            G_tensor[..., c, c] += G_base
        return G_tensor

    # Nonlinear path: work in principal-axis frame.
    m = float(stiffening_exponent)
    # Radial unit vector at each voxel.
    di, dj, dk = balloon._coords(N)
    r = np.sqrt(di ** 2 + dj ** 2 + dk ** 2)
    safe_r = np.maximum(r, 1e-12)
    rhat = np.stack([di / safe_r, dj / safe_r, dk / safe_r], axis=-1)

    # Principal stresses (spherically-symmetric Lamé for a pressurised sphere).
    p = balloon.pressure
    a = balloon.radius_vx
    outside_r = np.maximum(r, a)
    sigma_rr =        -p * (a / outside_r) ** 3
    sigma_tt =   0.5 * p * (a / outside_r) ** 3

    # Apply constitutive law element-wise on principal stresses.
    lam_r  = 1.0 + float(A_coeff) * sigma_rr / G_base
    lam_tt = 1.0 + float(A_coeff) * sigma_tt / G_base
    if constitutive == "powerlaw":
        G_r  = G_base * np.power(lam_r,  m)
        G_tt = G_base * np.power(lam_tt, m)
    elif constitutive == "ogden":
        G_r  = G_base * 0.5 * (np.power(lam_r,  m) + np.power(lam_r,  -m))
        G_tt = G_base * 0.5 * (np.power(lam_tt, m) + np.power(lam_tt, -m))
    else:
        raise ValueError(f"unknown constitutive: {constitutive!r}")

    # Rebuild Cartesian tensor G_ij = G_r·r̂r̂ + G_tt·(δ − r̂r̂).
    rr = rhat[..., :, None] * rhat[..., None, :]                # (N,N,N,3,3)
    delta = np.eye(3)
    G_tensor = (G_r[..., None, None] * rr
                + G_tt[..., None, None] * (delta - rr))

    # Inside the balloon: isotropic G_lesion (no acoustoelastic — uniform p).
    inside = balloon.mask(N)
    G_tensor[inside] = float(G_lesion) * delta
    return G_tensor


def effective_G_for_direction(
    sigma: np.ndarray,
    khat: np.ndarray,
    G_base: np.ndarray,
    A_coeff: float,
    stiffening_exponent: float = 1.0,
) -> np.ndarray:
    """Direction-dependent acoustoelastic G_eff for wave propagating along k̂.

    Shear-wave apparent modulus for waves propagating along unit direction
    ``khat`` and polarised perpendicular to it depends on the *normal*
    component of the pre-stress in that direction:

        G_eff(k̂, x) = G_base(x) · (1 + A_coeff · (k̂·σ·k̂)(x) / G_base(x))^m

    where m is the hyperelastic exponent (default 1 = linear). Since Δσ
    can be negative in the radial direction, G_eff can drop *below*
    G_base — the physical softening a shear wave sees when propagating
    along a compression axis (Yin's radial-propagation case).

    Parameters
    ----------
    sigma : (N, N, N, 3, 3) stress tensor field
    khat  : (3,) unit vector (propagation direction)
    G_base : (N, N, N) baseline stiffness
    A_coeff : acoustoelastic constant
    stiffening_exponent : hyperelastic exponent
    """
    khat = np.asarray(khat, dtype=np.float64)
    khat = khat / (np.linalg.norm(khat) + 1e-30)
    # k·σ·k contraction (scalar field).
    k_sigma_k = np.einsum("i,xyzij,j->xyz", khat, sigma, khat)
    ratio = 1.0 + float(A_coeff) * k_sigma_k / G_base
    # Guard against negative ratios (extreme compression) before the power.
    ratio = np.maximum(ratio, 1e-6)
    return G_base * np.power(ratio, float(stiffening_exponent))


def make_effective_G_3d(N: int, balloon: SphericalBalloon,
                         G_bg: float, G_lesion: float,
                         A_coeff: float,
                         stiffening_exponent: float = 1.0,
                         G_max_pa: float = G_MAX_PA,
                         constitutive: str = "powerlaw") -> np.ndarray:
    """Acoustoelastic-effective shear modulus for a pressurised spherical balloon.

    Parameters
    ----------
    N, balloon, G_bg, G_lesion, A_coeff : geometry and coupling as before.
    stiffening_exponent : float
        Nonlinearity exponent. For ``constitutive='powerlaw'`` this is the
        power in `(1 + A·Δσ/G_base)^m`. For ``constitutive='ogden'`` this
        is the Ogden α.
    G_max_pa : float
        Hard clip on the returned stiffness — numerical safety only.
    constitutive : {'powerlaw', 'ogden'}
        - ``'powerlaw'`` (default, backward compatible): asymmetric
          `G_eff = G_base · (1 + A·Δσ/G_base)^m`.  Linear at m=1
          (Phantom 1 analogue), super-linear for m>1 (Phantom 2).
        - ``'ogden'``: single-term Ogden hyperelastic form
          `G_eff = G_base · ½·(λ^α + λ^(-α))` with `λ = 1 + A·Δσ/G_base`.
          Symmetric in compression/tension, quadratic small-strain rise
          (softer near baseline than the power law), diverges as λ^α at
          large stretch. Common tissue α ≈ 3–7. At α = 2 reduces to a
          Mooney-Rivlin-like form.

    Both forms are memoryless — inflation and deflation at the same
    pre-stress give identical G_eff. Real gel hysteresis lives in the
    quasi-static viscoelastic response (see viscoelastic.py).
    """
    G_base = np.full((N, N, N), float(G_bg), dtype=np.float64)
    G_base[balloon.mask(N)] = float(G_lesion)
    dsig = lame_field_sphere(balloon, N)
    lam  = 1.0 + float(A_coeff) * dsig / G_base
    m    = float(stiffening_exponent)
    if constitutive == "powerlaw":
        G_eff = G_base * np.power(lam, m)
    elif constitutive == "ogden":
        G_eff = G_base * 0.5 * (np.power(lam, m) + np.power(lam, -m))
    else:
        raise ValueError(f"unknown constitutive: {constitutive!r}; "
                         "expected 'powerlaw' or 'ogden'")
    return np.clip(G_eff, G_MIN_PA, float(G_max_pa))


def perilesional_shell_3d(lesion_mask: np.ndarray, shell_mm: float,
                          dx: float, inner_offset_mm: float = 0.0) -> np.ndarray:
    """Perilesional shell — voxels within `shell_mm` of the lesion boundary.

    Parameters
    ----------
    lesion_mask, shell_mm, dx : as before.
    inner_offset_mm : float
        Distance (in mm) from the lesion boundary to exclude before the
        shell begins. Matches Yin's protocol: ``the inner boundary
        positioned three pixels away from the balloon edge to minimize
        edge effects''. Default 0 (backward compatible). Yin's 3 px at
        typical 3 mm voxels ≈ 9 mm.
    """
    outside = ~lesion_mask
    dist_vox = distance_transform_edt(outside).astype(np.float64)
    dist_mm = dist_vox * dx * 1000.0
    return outside & (dist_mm > inner_offset_mm) & (dist_mm <= inner_offset_mm + shell_mm)
