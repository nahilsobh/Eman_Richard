#!/usr/bin/env python3
"""Full vector Navier + rotated Cartesian tensor stiffness on the phantoms.

Upgrades `paper_wave_sim_sobh_initial.py` from the scalar Helmholtz
formulation to the vector Navier equation:

    ρ ω² u_i = ∂_j [ μ_ij(x) · ∂_j u_i + µ_ij ε_… ]                (schematic)

with **rank-2 tensor stiffness** µ_ij(x) built from the Sobh–Ehman
principal-frame acoustoelastic moduli and rotated into Cartesian:

    µ_ij(x) = µ_rr(r) · r̂_i r̂_j + µ_θθ(r) · (δ_ij − r̂_i r̂_j)

For incompressible spherical cavity expansion (paper eqs. 1–2):
    λ_r = λ_θ⁻²        so     µ_rr = µ · λ_r² = µ · λ_θ⁻⁴  (P1 matrix)

Radial-vs-tangential anisotropy at 250 mL (λ_θ = 1.71):
    µ_θθ / µ_rr = λ_θ⁶ ≈ 25×
so the tensor version is materially different from the scalar
µ_app,θθ we used in the scalar run.

Solver limits (matching `navier_solve_3d_tensor_mu`):
    - all 6 walls Dirichlet u = 0 (no top-free option in this solver
      — the tensor comparison is done with fully-clamped container to
      isolate the tensor-vs-scalar effect from the BC difference)
    - Bottom-plate driver: prescribed vector displacement at bottom
      disk nodes, comp = 0 (the vertical/i-axis), amp = 1 µm.

Grid: 32³ hex at dx = 6 mm → 19.2 cm cube (slightly larger than Yin's
15 × 15 × 18 cm; the extra 1 cm of gel on each side is background
material and does not affect the ring recovery).

Outputs:
    u_P1, u_P2 vector fields (N, N, N, 3) complex
    |curl u| — the shear content used for direct-inversion analogue
    Ring-mean stiffness comparison against scalar run and ground truth
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solver.vector_elasticity_3d import (
    curl_of_displacement_3d,
    navier_solve_3d_tensor_mu,
)


# ── Grid ───────────────────────────────────────────────────────────
N   = 32
DX  = 0.006             # 6 mm
CUBE_L_M = N * DX       # 19.2 cm

# Physics
FREQ_HZ   = 60.0
RHO       = 1000.0
DAMPING   = 0.05
LAM_C     = 1.0e5       # near-incompressible bulk penalty (Pa)
DRIVER_AMP = 1.0e-6

# Balloon at 250 mL (Sobh-Ehman)
A_REF_M = 0.02285
V_INJECTION_ML = 250
A_INFL_M = ((3 * V_INJECTION_ML * 1e-6) / (4 * math.pi)) ** (1/3)

# Mean-of-band material constants
MU_MEAN  = 2750.0
K1_MEAN  = 3250.0
K2_MEAN  = 1.25
G_WATER  = 1.0          # Pa (no shear support inside balloon)


def _W1_P1(lam_theta):
    """W1 = ∂W/∂I1 for P1 neo-Hookean matrix."""
    return np.full_like(lam_theta, MU_MEAN / 2.0)


def _W1_P2(lam_theta):
    """W1 = ∂W/∂I1 for P2 matrix + fiber."""
    eps = lam_theta - 1.0
    lam6_m1 = lam_theta ** 6 - 1.0
    safe = np.where(np.abs(lam6_m1) < 1e-12, 1e-12, lam6_m1)
    # W1 fiber contribution: from chain rule through ∂λθ/∂I1 = 1/(4(λθ-λθ⁻⁵))
    fiber_W1 = (K1_MEAN * eps * np.exp(K2_MEAN * eps ** 2) * lam_theta ** 5
                  / (4.0 * safe))
    # limit at λθ → 1 is k1/24 (verify: k1/12 for µ = 2 W1 λθ² formula at λ=1)
    fiber_W1 = np.where(np.abs(lam_theta - 1.0) < 1e-8,
                          K1_MEAN / 24.0, fiber_W1)
    return MU_MEAN / 2.0 + fiber_W1


def build_tensor_field(W1_fn, name):
    """Build Cartesian mu_ij(x) using the spherical-frame rotation.

    Convention: (i, j, k) with i vertical. Balloon centred at N/2.
    """
    print(f"  building tensor field for {name}...")
    ii, jj, kk = np.indices((N, N, N))
    c = (N - 1) / 2.0
    dz = (ii - c) * DX   # i axis
    dy = (jj - c) * DX
    dxv = (kk - c) * DX
    r = np.sqrt(dxv ** 2 + dy ** 2 + dz ** 2)
    in_balloon = r < A_INFL_M
    in_gel = ~in_balloon
    r_safe = np.where(r < 1e-12, 1e-12, r)

    # incompressible Lamé stretch
    R_ref = np.cbrt(r_safe ** 3 - A_INFL_M ** 3 + A_REF_M ** 3)
    lam_theta = np.where(in_gel, r_safe / R_ref, 1.0)

    W1 = W1_fn(lam_theta)

    # Principal-frame acoustoelastic moduli (paper eq 4)
    mu_theta_theta = 2.0 * W1 * lam_theta ** 2                # tangential
    mu_rr          = 2.0 * W1 * lam_theta ** (-4)             # radial (λ_r² = λ_θ⁻⁴)

    # Water inside balloon: soft isotropic
    mu_theta_theta = np.where(in_balloon, G_WATER, mu_theta_theta)
    mu_rr          = np.where(in_balloon, G_WATER, mu_rr)

    # Unit radial vector (avoid /0 at centre by any_r=0)
    rhat_i = dz  / r_safe
    rhat_j = dy  / r_safe
    rhat_k = dxv / r_safe
    rhat = np.stack([rhat_i, rhat_j, rhat_k], axis=-1)         # (N,N,N,3)

    # µ_ij = µ_rr r̂_i r̂_j + µ_θθ (δ_ij - r̂_i r̂_j)
    mu_tensor = np.zeros((N, N, N, 3, 3), dtype=float)
    RR = rhat[..., :, None] * rhat[..., None, :]              # (N,N,N,3,3) outer product
    dij = np.eye(3)[None, None, None, :, :]
    mu_tensor = (mu_rr[..., None, None] * RR
                    + mu_theta_theta[..., None, None] * (dij - RR))

    return mu_tensor, lam_theta, in_balloon, mu_rr, mu_theta_theta


def bottom_disk_driver(radius_frac=0.5, amp=DRIVER_AMP, comp=0):
    """Bottom face (i = N-1) coherent disk driver, comp is the driven component."""
    cy = (N - 1) / 2.0
    cz = (N - 1) / 2.0
    r_max = (N / 2.0) * radius_frac
    src = []
    for j in range(N):
        for k in range(N):
            if (j - cy) ** 2 + (k - cz) ** 2 <= r_max ** 2:
                src.append((N - 1, j, k, comp, complex(amp)))
    return src


def ring_stats(field, in_balloon, r_ref=A_INFL_M, shell_mm=12.0):
    ii, jj, kk = np.indices(field.shape)
    c = (N - 1) / 2.0
    r = np.sqrt(((ii - c) * DX) ** 2 + ((jj - c) * DX) ** 2 + ((kk - c) * DX) ** 2)
    ring = (~in_balloon) & (r >= r_ref) & (r <= r_ref + shell_mm * 1e-3)
    vals = field[ring]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), 0
    return float(vals.mean()), int(vals.size)


def main():
    out_dir = ROOT / "results" / "paper_wave_sim_sobh_vector"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Vector Navier grid: {N}³ hex at dx = {DX*1000:.1f} mm "
          f"→ cube {CUBE_L_M*100:.1f}³ cm")
    print(f"Balloon: a_infl = {A_INFL_M*100:.3f} cm (250 mL)")
    print(f"BCs: all 6 walls Dirichlet u = 0 (no top-free in this solver)")
    print(f"Driver: bottom disk, comp = 0 (vertical axis i), amp = "
          f"{DRIVER_AMP*1e6:.1f} µm")
    print()

    print("[P1] Building Cartesian tensor stiffness...")
    mu_P1, lam_theta, balloon, mu_rr_P1, mu_tt_P1 = \
        build_tensor_field(_W1_P1, "P1")
    print(f"  P1  µ_θθ range: {mu_tt_P1.min()/1000:.2f} – {mu_tt_P1.max()/1000:.2f} kPa")
    print(f"  P1  µ_rr range: {mu_rr_P1.min()/1000:.4f} – {mu_rr_P1.max()/1000:.4f} kPa")

    print("[P2] Building Cartesian tensor stiffness...")
    mu_P2, _, _, mu_rr_P2, mu_tt_P2 = build_tensor_field(_W1_P2, "P2")
    print(f"  P2  µ_θθ range: {mu_tt_P2.min()/1000:.2f} – {mu_tt_P2.max()/1000:.2f} kPa")
    print(f"  P2  µ_rr range: {mu_rr_P2.min()/1000:.4f} – {mu_rr_P2.max()/1000:.4f} kPa")
    print()

    sources = bottom_disk_driver()
    print(f"Driver: {len(sources)} bottom-face nodes at amp = {DRIVER_AMP:.1e} m")
    print()

    print("=" * 65)
    print("[P1] Solving vector Navier with tensor µ (this may take a bit)...")
    print("=" * 65, flush=True)
    t0 = time.time()
    u_P1 = navier_solve_3d_tensor_mu(mu_P1, lam=LAM_C, freq=FREQ_HZ, rho=RHO,
                                          dx=DX, damping=DAMPING, sources=sources)
    print(f"  → {time.time()-t0:.1f} s")
    print(f"  |u_P1| range: {np.abs(u_P1).min():.2e} – {np.abs(u_P1).max():.2e} m")

    print("=" * 65)
    print("[P2] Solving vector Navier with tensor µ...")
    print("=" * 65, flush=True)
    t0 = time.time()
    u_P2 = navier_solve_3d_tensor_mu(mu_P2, lam=LAM_C, freq=FREQ_HZ, rho=RHO,
                                          dx=DX, damping=DAMPING, sources=sources)
    print(f"  → {time.time()-t0:.1f} s")
    print(f"  |u_P2| range: {np.abs(u_P2).min():.2e} – {np.abs(u_P2).max():.2e} m")

    # curl for the S-wave (shear) content
    print("\nComputing curl(u) for the shear content...")
    curl_P1 = curl_of_displacement_3d(u_P1, DX)
    curl_P2 = curl_of_displacement_3d(u_P2, DX)
    curl_mag_P1 = np.linalg.norm(np.abs(curl_P1), axis=-1)
    curl_mag_P2 = np.linalg.norm(np.abs(curl_P2), axis=-1)

    # Ring stats on ground-truth µ_θθ, |u|, |curl u|
    ring_gt_P1, n_ring = ring_stats(mu_tt_P1, balloon)
    ring_gt_P2, _      = ring_stats(mu_tt_P2, balloon)
    ring_ur_P1, _      = ring_stats(np.abs(u_P1).sum(-1), balloon)
    ring_ur_P2, _      = ring_stats(np.abs(u_P2).sum(-1), balloon)
    ring_cu_P1, _      = ring_stats(curl_mag_P1, balloon)
    ring_cu_P2, _      = ring_stats(curl_mag_P2, balloon)

    lines = [
        "Vector Navier + rotated Cartesian tensor µ_ij(x) — 250 mL water balloon",
        "=" * 76,
        f"Grid: {N}³ at dx = {DX*1000:.1f} mm → {CUBE_L_M*100:.1f}³ cm cube",
        f"BCs:  all 6 walls Dirichlet u = 0 (no top-free in this solver)",
        f"      → NOT the same BC as scalar Helmholtz run (top-free there);",
        f"        the comparison isolates the tensor-vs-scalar effect only.",
        f"Driver: bottom disk, comp = 0 (i-axis / vertical), amp = {DRIVER_AMP*1e6:.1f} µm",
        f"Balloon at 250 mL (a = {A_INFL_M*100:.3f} cm), water G = {G_WATER} Pa",
        f"Perilesional shell: r ∈ [a, a+12 mm], {n_ring} voxels",
        "",
        "Cartesian tensor stiffness (rotated from spherical frame):",
        "  µ_ij(x) = µ_rr(r) · r̂_i r̂_j + µ_θθ(r) · (δ_ij − r̂_i r̂_j)",
        "",
        f"Anisotropy at the cavity edge (λ_θ = 1.71):",
        f"  P1: µ_θθ = {mu_tt_P1.max()/1000:.2f} kPa,  "
        f"µ_rr = {mu_rr_P1[balloon].mean()/1000 if balloon.any() else float('nan'):.4f} kPa (ex-balloon min "
        f"{mu_rr_P1[~balloon].min()/1000:.4f}), ratio θθ/rr up to "
        f"{mu_tt_P1[~balloon].max()/max(mu_rr_P1[~balloon].min(),1e-6):.1f}×",
        f"  P2: µ_θθ = {mu_tt_P2.max()/1000:.2f} kPa (fiber-boosted)",
        "",
        "Ring-averaged fields:",
        f"  P1:  GT µ_θθ = {ring_gt_P1/1000:.3f} kPa    "
        f"⟨Σ|u|⟩ = {ring_ur_P1*1e6:.3f} µm    "
        f"⟨|curl u|⟩ = {ring_cu_P1:.2e} rad/m",
        f"  P2:  GT µ_θθ = {ring_gt_P2/1000:.3f} kPa    "
        f"⟨Σ|u|⟩ = {ring_ur_P2*1e6:.3f} µm    "
        f"⟨|curl u|⟩ = {ring_cu_P2:.2e} rad/m",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))

    # ── Save ──
    np.savez_compressed(
        out_dir / "wave_sim_vector.npz",
        mu_P1=mu_P1, mu_P2=mu_P2, balloon=balloon, lam_theta=lam_theta,
        mu_rr_P1=mu_rr_P1, mu_tt_P1=mu_tt_P1,
        mu_rr_P2=mu_rr_P2, mu_tt_P2=mu_tt_P2,
        u_P1=u_P1, u_P2=u_P2,
        curl_P1=curl_P1, curl_P2=curl_P2,
        curl_mag_P1=curl_mag_P1, curl_mag_P2=curl_mag_P2,
        dx_m=DX, freq_hz=FREQ_HZ, driver_amp_m=DRIVER_AMP,
        N=N, cube_L_m=CUBE_L_M,
    )
    print(f"\nSaved {out_dir / 'wave_sim_vector.npz'}")

    # ── Figure ──
    mid = N // 2

    fig, axes = plt.subplots(2, 5, figsize=(22, 9.5), constrained_layout=True)
    for row, (name, mu_tt, mu_rr, u, curl_mag) in enumerate([
        ("P1", mu_tt_P1, mu_rr_P1, u_P1, curl_mag_P1),
        ("P2", mu_tt_P2, mu_rr_P2, u_P2, curl_mag_P2),
    ]):
        vmax = mu_tt.max() / 1000
        # µ_θθ
        im = axes[row, 0].imshow(np.where(balloon[:, :, mid], np.nan,
                                             mu_tt[:, :, mid]) / 1000,
                                    origin="upper", vmin=0, vmax=vmax, cmap="viridis")
        axes[row, 0].set_title(f"{name} µ_θθ (tangential) [kPa]")
        plt.colorbar(im, ax=axes[row, 0], shrink=0.85)

        # µ_rr
        im = axes[row, 1].imshow(np.where(balloon[:, :, mid], np.nan,
                                             mu_rr[:, :, mid]) / 1000,
                                    origin="upper", vmin=0, vmax=vmax/5,
                                    cmap="viridis")
        axes[row, 1].set_title(f"{name} µ_rr (radial) [kPa]")
        plt.colorbar(im, ax=axes[row, 1], shrink=0.85)

        # |u|
        u_mag = np.linalg.norm(np.abs(u), axis=-1)
        im = axes[row, 2].imshow(u_mag[:, :, mid] * 1e6, origin="upper", cmap="magma")
        axes[row, 2].set_title(f"{name} |u_vector| [µm]")
        plt.colorbar(im, ax=axes[row, 2], shrink=0.85)

        # u_i (vertical component real part)
        im = axes[row, 3].imshow(np.real(u[:, :, mid, 0]) * 1e6, origin="upper", cmap="RdBu_r")
        axes[row, 3].set_title(f"{name} Re(u_z) [µm]")
        plt.colorbar(im, ax=axes[row, 3], shrink=0.85)

        # |curl u|
        im = axes[row, 4].imshow(curl_mag[:, :, mid], origin="upper", cmap="magma")
        axes[row, 4].set_title(f"{name} |curl u| (S-wave content)")
        plt.colorbar(im, ax=axes[row, 4], shrink=0.85)

        for ax in axes[row]:
            ax.set_xlabel("j (y)"); ax.set_ylabel("i (z, top=0)")

    plt.suptitle(
        f"Vector Navier + rotated Cartesian µ_ij(x) at 250 mL — bottom driver, "
        f"all walls fixed\n"
        f"{N}³ at dx = {DX*1000:.1f} mm ({CUBE_L_M*100:.1f}³ cm cube)",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_dir / "wave_sim_vector.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'wave_sim_vector.png'}")


if __name__ == "__main__":
    main()
