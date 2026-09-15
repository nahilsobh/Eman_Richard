#!/usr/bin/env python3
"""3D time-harmonic wave propagation in the phantoms at 250 mL.

Consumes the initial stiffness field G_P1(x), G_P2(x) built from the
Sobh-Ehman paper (`paper_initial_stiffness_from_sobh.py`), maps it into
the scalar-Helmholtz forward-model grid, and runs a 60 Hz MRE simulation
with:

  Container BCs
    - side walls (j = 0, j = N−1, k = 0, k = N−1) : u = 0  (rigid)
    - bottom (i = N−1) : Dirichlet from the driver source below
    - top (i = 0)      : ∂u/∂z = 0  (traction-free, `top_free=True`)

  Driver
    Coherent bottom-plate piston: disk source on the bottom face
    covering the central 50 % of the plate area, amplitude 1 µm at 60 Hz.

  Balloon at 250 mL (water-filled → no shear support)
    G_water = 1 Pa inside the balloon (softer than gel by ~3 orders of
    magnitude → the scalar Helmholtz equation reduces to ρω²u ≈ 0 in
    the balloon, so u decays there; the interior contributes no shear
    wave propagation, matching real water behaviour at MRE frequencies).

Grid
----
Solver uses a cubic mesh, so we pad Yin's 15 × 15 × 18 cm container into
an 18 × 18 × 18 cm cube at dx = 3 mm → N = 60. The extra 1.5 cm on each
lateral side is filled with the background gel (µ = 2.75 kPa unstrained).

Output
------
- Complex displacement fields u_P1(x), u_P2(x) → `.npz`
- Direct inversion G_est(x) per phantom  → `.npz`
- Midplane snapshots (|u| and Re(u)) and radial G comparison
- Ring-mean G comparison: ground truth (from initial stiffness) vs DI
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solver.helmholtz_fd_3d import (
    bottom_plate_driver_sources_3d,
    direct_inversion_3d,
    helmholtz_solve_3d,
)


# ── Solver grid (must be cubic) ─────────────────────────────────────
N = 60
DX_M = 0.003            # 3 mm ≈ Yin acquisition voxel
CUBE_L_M = N * DX_M     # 0.18 m = 18 cm

# Physical parameters
FREQ_HZ = 60.0
RHO = 1000.0            # gel density kg/m³
DAMPING = 0.05          # hysteretic Q ≈ 1/(2·damping) = 10

# Driver
DRIVER_AMP = 1.0e-6     # 1 µm displacement at 60 Hz  (MRE typical)
DRIVER_RADIUS_FRAC = 0.5   # covers central 50% of bottom plate

# Balloon at 250 mL (from Sobh-Ehman paper, §2)
A_REF_M = 0.02285
V_INJECTION_ML = 250
A_INFL_M = ((3 * V_INJECTION_ML * 1e-6) / (4 * math.pi)) ** (1/3)

# Mean-of-band material constants (Table 2 of the paper)
MU_MEAN = 2750.0
K1_MEAN = 3250.0
K2_MEAN = 1.25

# Water inside balloon — no shear support
G_WATER = 1.0           # Pa (essentially zero shear; ρω²u ≈ 0 inside → u → 0)


def _mu_P1(lam):
    return MU_MEAN * lam ** 2


def _mu_P2(lam):
    eps = lam - 1.0
    matrix = MU_MEAN * lam ** 2
    lam6_m1 = lam ** 6 - 1.0
    safe = np.where(np.abs(lam6_m1) < 1e-12, 1e-12, lam6_m1)
    fiber = K1_MEAN * lam ** 7 * eps * np.exp(K2_MEAN * eps ** 2) / (2.0 * safe)
    fiber = np.where(np.abs(lam - 1.0) < 1e-8, K1_MEAN / 12.0, fiber)
    return matrix + fiber


def build_stiffness_solver_grid():
    """Build G_P1, G_P2 on the (i,j,k) solver grid with i=vertical (top=0).

    Balloon centred at (i,j,k) = (N/2, N/2, N/2).
    Water inside r < a_infl_m; gel (with prestrain-induced stiffening) outside.
    """
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    cx = cy = cz = (N - 1) / 2.0
    # physical coords (all in metres, all relative to balloon centre)
    dz = (i_idx - cz) * DX_M    # i is vertical
    dy = (j_idx - cy) * DX_M
    dx_ = (k_idx - cx) * DX_M
    r = np.sqrt(dx_ ** 2 + dy ** 2 + dz ** 2)

    in_balloon = r < A_INFL_M
    in_gel = ~in_balloon

    # Prestrain via incompressible Lamé (paper eq 2)
    r_safe = np.where(r < 1e-12, 1e-12, r)
    R_ref = np.cbrt(r_safe ** 3 - A_INFL_M ** 3 + A_REF_M ** 3)
    lam_theta = np.where(in_gel, r_safe / R_ref, 1.0)

    G_P1 = np.where(in_gel, _mu_P1(lam_theta), G_WATER).astype(float)
    G_P2 = np.where(in_gel, _mu_P2(lam_theta), G_WATER).astype(float)

    balloon_mask = in_balloon
    return G_P1, G_P2, lam_theta, balloon_mask


def ring_stats(G, balloon_mask, dx=DX_M, shell_inner_mm=0.0, shell_outer_mm=12.0):
    """Mean & median of G within a spherical shell around balloon surface."""
    i_idx, j_idx, k_idx = np.indices(G.shape)
    cx = cy = cz = (N - 1) / 2.0
    r = np.sqrt(((i_idx - cz) * dx) ** 2 +
                  ((j_idx - cy) * dx) ** 2 +
                  ((k_idx - cx) * dx) ** 2)
    ring = (~balloon_mask) & (r >= A_INFL_M + shell_inner_mm * 1e-3) \
                            & (r <= A_INFL_M + shell_outer_mm * 1e-3)
    vals = G[ring]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), 0
    return float(vals.mean()), float(np.median(vals)), int(vals.size)


def main():
    out_dir = ROOT / "results" / "paper_wave_sim_sobh_initial"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Solver grid: {N}³ hex at dx = {DX_M*1000:.1f} mm "
          f"→ cube {CUBE_L_M*100:.1f}³ cm")
    print(f"Balloon: a_infl = {A_INFL_M*100:.3f} cm ({V_INJECTION_ML} mL)")
    print(f"Freq: {FREQ_HZ} Hz, ρ = {RHO} kg/m³, damping = {DAMPING}")
    print(f"Water inside balloon: G = {G_WATER} Pa (no shear support)")
    print(f"Driver: bottom-plate disk (radius_frac = {DRIVER_RADIUS_FRAC}), "
          f"amp = {DRIVER_AMP*1e6:.1f} µm")
    print()

    G_P1, G_P2, lam_theta, balloon_mask = build_stiffness_solver_grid()

    # Ground-truth ring stiffness (paper's shell: r ∈ [a, a+12mm])
    G_P1_ring_gt, _, n_ring = ring_stats(G_P1, balloon_mask)
    G_P2_ring_gt, _, _      = ring_stats(G_P2, balloon_mask)
    print(f"GT ring-mean stiffness (r ∈ [a, a+12 mm], {n_ring} voxels):")
    print(f"  P1: {G_P1_ring_gt/1000:.3f} kPa  (paper band 4.28–5.14)")
    print(f"  P2: {G_P2_ring_gt/1000:.3f} kPa  (paper band 4.70–6.94)")
    print()

    # Driver
    sources = bottom_plate_driver_sources_3d(N, radius_frac=DRIVER_RADIUS_FRAC,
                                                amp=DRIVER_AMP)
    print(f"Driver: {len(sources)} bottom-face nodes at amp = {DRIVER_AMP:.1e} m")

    # ── Forward solves ────────────────────────────────────────────────
    print("\nSolving Helmholtz (P1)...", flush=True)
    u_P1 = helmholtz_solve_3d(G_P1, freq=FREQ_HZ, rho=RHO, dx=DX_M,
                                 damping=DAMPING, sources=sources, top_free=True)
    print(f"  |u_P1| range: {np.abs(u_P1).min():.2e} – {np.abs(u_P1).max():.2e} m")

    print("Solving Helmholtz (P2)...", flush=True)
    u_P2 = helmholtz_solve_3d(G_P2, freq=FREQ_HZ, rho=RHO, dx=DX_M,
                                 damping=DAMPING, sources=sources, top_free=True)
    print(f"  |u_P2| range: {np.abs(u_P2).min():.2e} – {np.abs(u_P2).max():.2e} m")

    # ── Direct inversion (recover G from u) ──────────────────────────
    print("\nDirect inversion (P1)...", flush=True)
    G_P1_est = direct_inversion_3d(u_P1, freq=FREQ_HZ, rho=RHO, dx=DX_M,
                                       median_filter_size=3)
    print("Direct inversion (P2)...", flush=True)
    G_P2_est = direct_inversion_3d(u_P2, freq=FREQ_HZ, rho=RHO, dx=DX_M,
                                       median_filter_size=3)

    G_P1_ring_di, _, _ = ring_stats(G_P1_est, balloon_mask)
    G_P2_ring_di, _, _ = ring_stats(G_P2_est, balloon_mask)
    print(f"\nDI-recovered ring-mean stiffness:")
    print(f"  P1: {G_P1_ring_di/1000:.3f} kPa  (GT: {G_P1_ring_gt/1000:.3f})")
    print(f"  P2: {G_P2_ring_di/1000:.3f} kPa  (GT: {G_P2_ring_gt/1000:.3f})")

    # ── Save snapshots ────────────────────────────────────────────────
    np.savez_compressed(
        out_dir / "wave_sim.npz",
        G_P1=G_P1, G_P2=G_P2,
        u_P1=u_P1, u_P2=u_P2,
        G_P1_est=G_P1_est, G_P2_est=G_P2_est,
        lam_theta=lam_theta, balloon_mask=balloon_mask,
        dx_m=DX_M, freq_hz=FREQ_HZ, driver_amp_m=DRIVER_AMP,
        mu_mean=MU_MEAN, k1_mean=K1_MEAN, k2_mean=K2_MEAN,
        A_ref_m=A_REF_M, a_infl_m=A_INFL_M, V_ml=V_INJECTION_ML,
    )
    print(f"\nSaved {out_dir / 'wave_sim.npz'}")

    # ── Figures ───────────────────────────────────────────────────────
    mid = N // 2
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), constrained_layout=True)

    # Top row: P1
    im0 = axes[0, 0].imshow(G_P1[:, :, mid] / 1000, origin="upper",
                              vmin=0, vmax=10, cmap="viridis")
    axes[0, 0].set_title("P1 ground-truth µ_app,θθ [kPa]")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.85)

    im1 = axes[0, 1].imshow(np.abs(u_P1[:, :, mid]) * 1e6, origin="upper",
                              cmap="magma")
    axes[0, 1].set_title("P1 |u| [µm]")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.85)

    im2 = axes[0, 2].imshow(np.real(u_P1[:, :, mid]) * 1e6, origin="upper",
                              cmap="RdBu_r")
    axes[0, 2].set_title("P1 Re(u) [µm]")
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.85)

    G_P1_est_masked = np.where(np.isfinite(G_P1_est), G_P1_est, 0)
    im3 = axes[0, 3].imshow(G_P1_est_masked[:, :, mid] / 1000, origin="upper",
                              vmin=0, vmax=10, cmap="viridis")
    axes[0, 3].set_title(f"P1 DI-recovered G [kPa]\n"
                         f"ring GT {G_P1_ring_gt/1000:.2f} vs DI {G_P1_ring_di/1000:.2f}")
    plt.colorbar(im3, ax=axes[0, 3], shrink=0.85)

    # Bottom row: P2
    im0 = axes[1, 0].imshow(G_P2[:, :, mid] / 1000, origin="upper",
                              vmin=0, vmax=10, cmap="viridis")
    axes[1, 0].set_title("P2 ground-truth µ_app,θθ [kPa]")
    plt.colorbar(im0, ax=axes[1, 0], shrink=0.85)

    im1 = axes[1, 1].imshow(np.abs(u_P2[:, :, mid]) * 1e6, origin="upper",
                              cmap="magma")
    axes[1, 1].set_title("P2 |u| [µm]")
    plt.colorbar(im1, ax=axes[1, 1], shrink=0.85)

    im2 = axes[1, 2].imshow(np.real(u_P2[:, :, mid]) * 1e6, origin="upper",
                              cmap="RdBu_r")
    axes[1, 2].set_title("P2 Re(u) [µm]")
    plt.colorbar(im2, ax=axes[1, 2], shrink=0.85)

    G_P2_est_masked = np.where(np.isfinite(G_P2_est), G_P2_est, 0)
    im3 = axes[1, 3].imshow(G_P2_est_masked[:, :, mid] / 1000, origin="upper",
                              vmin=0, vmax=10, cmap="viridis")
    axes[1, 3].set_title(f"P2 DI-recovered G [kPa]\n"
                         f"ring GT {G_P2_ring_gt/1000:.2f} vs DI {G_P2_ring_di/1000:.2f}")
    plt.colorbar(im3, ax=axes[1, 3], shrink=0.85)

    for ax in axes.flat:
        ax.set_xlabel("j (y)")
        ax.set_ylabel("i (z, top = 0)")

    plt.suptitle(
        f"Bottom-driven MRE forward sim, 60 Hz, 18 cm cube grid\n"
        f"container fixed sides + free top, water balloon at 250 mL, "
        f"Sobh-Ehman mean-of-band material",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_dir / "wave_sim_midplane.png", dpi=140,
                 bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'wave_sim_midplane.png'}")

    # ── Summary text ──────────────────────────────────────────────────
    lines = [
        f"Bottom-driven MRE forward sim at 250 mL — Sobh-Ehman initial G",
        "=" * 72,
        f"Solver grid: {N}³ hex at dx = {DX_M*1000:.1f} mm → cube {CUBE_L_M*100:.1f}³ cm",
        f"Container: 15×15×18 cm (padded to 18³ cube — extra 1.5 cm/side of gel background)",
        f"BCs: side + bottom walls Dirichlet, top ∂u/∂z=0 (traction-free)",
        f"Driver: bottom-plate disk, radius_frac={DRIVER_RADIUS_FRAC}, amp={DRIVER_AMP*1e6:.1f} µm",
        f"Balloon at 250 mL (a = {A_INFL_M*100:.3f} cm) filled with water:",
        f"  G_water = {G_WATER} Pa (no shear support → wave decays inside)",
        f"Gel: Sobh-Ehman mean-of-band  µ = {MU_MEAN:.0f} Pa, "
        f"k₁ = {K1_MEAN:.0f} Pa, k₂ = {K2_MEAN}",
        "",
        "Perilesional shell (r ∈ [a, a+12 mm]) stiffness:",
        f"                Ground truth        DI-recovered",
        f"  P1:            {G_P1_ring_gt/1000:7.3f} kPa       {G_P1_ring_di/1000:7.3f} kPa",
        f"  P2:            {G_P2_ring_gt/1000:7.3f} kPa       {G_P2_ring_di/1000:7.3f} kPa",
        "",
        f"Peak |u|: P1 = {np.abs(u_P1).max()*1e6:.2f} µm,  P2 = {np.abs(u_P2).max()*1e6:.2f} µm",
        f"Solver output: 60 Hz shear wave; expected λ_shear ≈ 28 mm in unstrained gel",
        f"               (voxels per wavelength ≈ 9 at dx = 3 mm)",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
