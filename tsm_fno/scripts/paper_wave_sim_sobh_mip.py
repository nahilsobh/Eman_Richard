#!/usr/bin/env python3
"""TSM MIP inversion on the Sobh-Ehman initial-stiffness phantoms.

Extends `paper_wave_sim_sobh_initial.py` by applying Yin's TSM MIP
protocol on top of the same 250 mL water-balloon phantoms:

  1. Load the (i, j, k) ground-truth stiffness fields G_P1(x), G_P2(x)
     built from the Sobh-Ehman mean-of-band material.
  2. Run ONE broadband multi-face solve per phantom — 5 faces driven
     simultaneously (bottom + 4 sides), top left free — to produce a
     wave field with rich direction content.
  3. Apply the k-space directional filter for each of 20 Fibonacci-
     sphere unit vectors (Yin's dodecahedral direction set).
  4. Direct-invert each filtered field with a 3×3×3 median filter.
  5. Combine across directions:
        µ_TSM  = voxelwise MAX (Yin's TSM MIP)
        µ_conv = voxelwise amplitude-weighted MEAN (conventional MRE)

Result comparison per phantom:
    ground-truth ring | µ_conv ring | µ_TSM ring
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
    directional_filter_3d,
    direct_inversion_3d,
    helmholtz_solve_3d,
    multi_face_broadband_sources,
)


# ── Solver grid (matches paper_wave_sim_sobh_initial) ───────────────
N = 60
DX_M = 0.003
CUBE_L_M = N * DX_M

FREQ_HZ  = 60.0
RHO      = 1000.0
DAMPING  = 0.05
DRIVER_AMP = 1.0e-6
DRIVER_RADIUS_FRAC = 0.5

A_REF_M = 0.02285
V_INJECTION_ML = 250
A_INFL_M = ((3 * V_INJECTION_ML * 1e-6) / (4 * math.pi)) ** (1/3)

MU_MEAN = 2750.0
K1_MEAN = 3250.0
K2_MEAN = 1.25
G_WATER = 1.0

N_DIRECTIONS = 20      # Fibonacci sphere → dodecahedral-like coverage
WEDGE_WIDTH  = 0.35    # ~20° FWHM angular filter (matches Yin)
MEDIAN_FILTER = 3      # 3×3×3 spatial median (Yin's Methods step)


def _mu_P1(lam):
    return MU_MEAN * lam ** 2


def _mu_P2(lam):
    eps = lam - 1.0
    lam6_m1 = lam ** 6 - 1.0
    safe = np.where(np.abs(lam6_m1) < 1e-12, 1e-12, lam6_m1)
    fiber = K1_MEAN * lam ** 7 * eps * np.exp(K2_MEAN * eps ** 2) / (2.0 * safe)
    fiber = np.where(np.abs(lam - 1.0) < 1e-8, K1_MEAN / 12.0, fiber)
    return MU_MEAN * lam ** 2 + fiber


def build_stiffness():
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    c = (N - 1) / 2.0
    dz = (i_idx - c) * DX_M
    dy = (j_idx - c) * DX_M
    dx_ = (k_idx - c) * DX_M
    r = np.sqrt(dx_ ** 2 + dy ** 2 + dz ** 2)
    in_balloon = r < A_INFL_M
    in_gel = ~in_balloon
    r_safe = np.where(r < 1e-12, 1e-12, r)
    R_ref = np.cbrt(r_safe ** 3 - A_INFL_M ** 3 + A_REF_M ** 3)
    lam = np.where(in_gel, r_safe / R_ref, 1.0)
    G_P1 = np.where(in_gel, _mu_P1(lam), G_WATER).astype(float)
    G_P2 = np.where(in_gel, _mu_P2(lam), G_WATER).astype(float)
    return G_P1, G_P2, in_balloon


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n approximately equidistributed unit vectors."""
    i = np.arange(n)
    phi = (1 + 5 ** 0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - 2 * (i + 0.5) / n
    rxy = np.sqrt(np.maximum(0.0, 1 - z * z))
    return np.stack([rxy * np.cos(theta), rxy * np.sin(theta), z], axis=1)


def ring_stats(G, balloon_mask,
                shell_inner_mm=0.0, shell_outer_mm=12.0):
    """Ring-mean of G in perilesional shell around balloon (r ∈ [a+in, a+out])."""
    i_idx, j_idx, k_idx = np.indices(G.shape)
    c = (N - 1) / 2.0
    r = np.sqrt(((i_idx - c) * DX_M) ** 2 +
                  ((j_idx - c) * DX_M) ** 2 +
                  ((k_idx - c) * DX_M) ** 2)
    ring = (~balloon_mask) & (r >= A_INFL_M + shell_inner_mm * 1e-3) \
                            & (r <= A_INFL_M + shell_outer_mm * 1e-3)
    vals = G[ring]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def main():
    out_dir = ROOT / "results" / "paper_wave_sim_sobh_mip"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid: {N}³ hex at dx = {DX_M*1000:.1f} mm → cube {CUBE_L_M*100:.1f}³ cm")
    print(f"Freq: {FREQ_HZ} Hz, ρ = {RHO} kg/m³, damping = {DAMPING}")
    print(f"Directions: {N_DIRECTIONS} (Fibonacci sphere), wedge σ = {WEDGE_WIDTH} rad")
    print(f"Median filter: {MEDIAN_FILTER}³")
    print()

    G_P1, G_P2, balloon = build_stiffness()
    G_P1_ring_gt = ring_stats(G_P1, balloon)
    G_P2_ring_gt = ring_stats(G_P2, balloon)
    print(f"GT ring stiffness (r ∈ [a, a+12 mm]):")
    print(f"  P1: {G_P1_ring_gt/1000:.3f} kPa")
    print(f"  P2: {G_P2_ring_gt/1000:.3f} kPa")
    print()

    # Multi-face broadband source (bottom + 4 sides, top left free)
    sources = multi_face_broadband_sources(
        N, radius_frac=DRIVER_RADIUS_FRAC,
        faces=("iN", "jN", "j0", "kN", "k0"),
    )
    # Rescale to physical driver amp
    sources = [(i, j, k, DRIVER_AMP + 0.0j) for (i, j, k, _) in sources]
    print(f"Driver: multi-face broadband, {len(sources)} source nodes total")

    directions = fibonacci_sphere(N_DIRECTIONS)
    print(f"Filtering into {N_DIRECTIONS} directions...")

    def _tsm_recover(G_field, phantom_name):
        print(f"\n[{phantom_name}] Solving broadband Helmholtz...", flush=True)
        u = helmholtz_solve_3d(G_field, freq=FREQ_HZ, rho=RHO, dx=DX_M,
                                  damping=DAMPING, sources=sources,
                                  top_free=True)
        print(f"    |u| range: {np.abs(u).min():.2e} – {np.abs(u).max():.2e} m")

        # Per-direction filter + DI
        G_est_stack = np.zeros((N_DIRECTIONS, N, N, N))
        amp_stack   = np.zeros((N_DIRECTIONS, N, N, N))
        for d, khat in enumerate(directions):
            u_k = directional_filter_3d(u, khat=khat,
                                            angular_width=WEDGE_WIDTH)
            G_k = direct_inversion_3d(u_k, freq=FREQ_HZ, rho=RHO, dx=DX_M,
                                          median_filter_size=MEDIAN_FILTER)
            # NaN → 0 for combination (they'll be re-masked at the end)
            G_est_stack[d] = np.where(np.isfinite(G_k), G_k, 0.0)
            amp_stack[d]   = np.abs(u_k)
            if (d + 1) % 5 == 0:
                print(f"    direction {d+1}/{N_DIRECTIONS} done", flush=True)

        # µ_TSM = voxelwise max (only positive stiffness — negatives are noise)
        stack_pos = np.where(G_est_stack > 0, G_est_stack, 0.0)
        G_tsm = stack_pos.max(axis=0)

        # µ_conv = amplitude-weighted mean
        wsum = amp_stack.sum(axis=0)
        wsum_safe = np.where(wsum > 1e-30, wsum, 1e-30)
        G_conv = (stack_pos * amp_stack).sum(axis=0) / wsum_safe

        # Mask balloon interior
        G_tsm  = np.where(balloon, np.nan, G_tsm)
        G_conv = np.where(balloon, np.nan, G_conv)

        return u, G_tsm, G_conv

    u_P1, G_P1_tsm, G_P1_conv = _tsm_recover(G_P1, "P1")
    u_P2, G_P2_tsm, G_P2_conv = _tsm_recover(G_P2, "P2")

    # Ring-mean summary
    r_conv_p1 = ring_stats(G_P1_conv, balloon)
    r_tsm_p1  = ring_stats(G_P1_tsm,  balloon)
    r_conv_p2 = ring_stats(G_P2_conv, balloon)
    r_tsm_p2  = ring_stats(G_P2_tsm,  balloon)

    lines = [
        "TSM MIP recovery on Sobh-Ehman phantoms at 250 mL",
        "=" * 72,
        f"Grid: {N}³ hex at dx = {DX_M*1000:.1f} mm → {CUBE_L_M*100:.1f}³ cm cube",
        f"BCs: side + bottom walls u=0, top ∂u/∂z=0 (free)",
        f"Driver: multi-face broadband (bottom + 4 sides), amp = {DRIVER_AMP*1e6:.1f} µm",
        f"Balloon at 250 mL (a={A_INFL_M*100:.3f} cm), water inside (G = {G_WATER} Pa)",
        f"Directions: {N_DIRECTIONS} (Fibonacci), wedge σ = {WEDGE_WIDTH} rad",
        f"Median filter: {MEDIAN_FILTER}×{MEDIAN_FILTER}×{MEDIAN_FILTER}",
        "",
        "Perilesional ring (r ∈ [a, a+12 mm]) mean stiffness:",
        f"{'':16s}{'GT':>8s}   {'µ_conv':>8s}   {'µ_TSM':>8s}   "
        f"{'TSM/conv':>9s}   {'TSM/GT':>8s}",
        f"  P1 (neo-Hookean)     {G_P1_ring_gt/1000:6.3f}     "
        f"{r_conv_p1/1000:6.3f}     {r_tsm_p1/1000:6.3f}     "
        f"{r_tsm_p1/max(r_conv_p1, 1):>7.2f}      {r_tsm_p1/max(G_P1_ring_gt, 1):>6.2f}",
        f"  P2 (matrix+fiber)    {G_P2_ring_gt/1000:6.3f}     "
        f"{r_conv_p2/1000:6.3f}     {r_tsm_p2/1000:6.3f}     "
        f"{r_tsm_p2/max(r_conv_p2, 1):>7.2f}      {r_tsm_p2/max(G_P2_ring_gt, 1):>6.2f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))

    np.savez_compressed(
        out_dir / "wave_sim_mip.npz",
        G_P1=G_P1, G_P2=G_P2, balloon=balloon,
        u_P1=u_P1, u_P2=u_P2,
        G_P1_conv=G_P1_conv, G_P1_tsm=G_P1_tsm,
        G_P2_conv=G_P2_conv, G_P2_tsm=G_P2_tsm,
        directions=directions,
        dx_m=DX_M, freq_hz=FREQ_HZ, driver_amp_m=DRIVER_AMP,
        n_directions=N_DIRECTIONS, wedge_width=WEDGE_WIDTH,
        median_filter=MEDIAN_FILTER,
    )

    # Figure: 2 phantoms × (GT, |u|, µ_conv, µ_TSM)
    mid = N // 2
    def _slice(arr): return arr[:, :, mid]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), constrained_layout=True)
    for row, (name, G_gt, u, G_conv, G_tsm) in enumerate([
        ("P1", G_P1, u_P1, G_P1_conv, G_P1_tsm),
        ("P2", G_P2, u_P2, G_P2_conv, G_P2_tsm),
    ]):
        gt_s   = np.where(balloon[:, :, mid], np.nan, _slice(G_gt) / 1000)
        u_s    = np.abs(_slice(u)) * 1e6
        conv_s = _slice(G_conv) / 1000
        tsm_s  = _slice(G_tsm) / 1000

        vmax_g = max(np.nanmax(gt_s), np.nanmax(tsm_s), np.nanmax(conv_s))

        im = axes[row, 0].imshow(gt_s, origin="upper", vmin=0, vmax=vmax_g, cmap="viridis")
        axes[row, 0].set_title(f"{name} GT µ_app,θθ [kPa]"); plt.colorbar(im, ax=axes[row, 0], shrink=0.85)

        im = axes[row, 1].imshow(u_s, origin="upper", cmap="magma")
        axes[row, 1].set_title(f"{name} |u| [µm]"); plt.colorbar(im, ax=axes[row, 1], shrink=0.85)

        im = axes[row, 2].imshow(conv_s, origin="upper", vmin=0, vmax=vmax_g, cmap="viridis")
        axes[row, 2].set_title(f"{name} µ_conv [kPa]  "
                                f"(ring {ring_stats(G_conv, balloon)/1000:.2f})")
        plt.colorbar(im, ax=axes[row, 2], shrink=0.85)

        im = axes[row, 3].imshow(tsm_s, origin="upper", vmin=0, vmax=vmax_g, cmap="viridis")
        axes[row, 3].set_title(f"{name} µ_TSM (MIP) [kPa]  "
                                f"(ring {ring_stats(G_tsm, balloon)/1000:.2f})")
        plt.colorbar(im, ax=axes[row, 3], shrink=0.85)

        for ax in axes[row]:
            ax.set_xlabel("j (y)"); ax.set_ylabel("i (z, top=0)")

    plt.suptitle(
        f"Yin TSM MIP inversion on Sobh-Ehman initial G — 250 mL, "
        f"multi-face broadband, {N_DIRECTIONS}-direction filter, top free",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_dir / "wave_sim_mip.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_dir / 'wave_sim_mip.png'}")


if __name__ == "__main__":
    main()
