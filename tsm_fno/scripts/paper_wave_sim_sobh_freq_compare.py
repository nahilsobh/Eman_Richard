#!/usr/bin/env python3
"""60 Hz vs 80 Hz comparison for the Sobh-Ehman phantom MIP sim.

Yin et al. (2026) drove the phantoms at 80 Hz; our earlier runs used
60 Hz.  This script runs the same multi-face broadband + 20-direction
Fibonacci filter + DI + MIP pipeline at both frequencies with the
same phantoms, same driver amplitude, same BCs, and produces:

  - Ring-mean stiffness table for each phantom at each frequency
  - Two-panel side-by-side heatmaps of µ_TSM at 60 vs 80 Hz
  - Radial profiles overlaid (GT + µ_conv + µ_TSM per frequency)

Only µ_θθ (scalar acoustoelastic tangential modulus) is used, and only
the multi-face broadband source drives the field (matches paper_wave_sim_sobh_mip).
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

from src.solver.helmholtz_fd_3d import (
    direct_inversion_3d,
    directional_filter_3d,
    helmholtz_solve_3d,
    multi_face_broadband_sources,
)


# ── Grid / geometry ───────────────────────────────────────────────
N = 60
DX_M = 0.003
RHO = 1000.0
DAMPING = 0.05

A_REF_M = 0.02285
V_INJECTION_ML = 250
A_INFL_M = ((3 * V_INJECTION_ML * 1e-6) / (4 * math.pi)) ** (1/3)

MU_MEAN = 2750.0
K1_MEAN = 3250.0
K2_MEAN = 1.25
G_WATER = 1.0

DRIVER_AMP = 1.0e-6
DRIVER_R_FRAC = 0.5
N_DIRECTIONS = 20
WEDGE_WIDTH = 0.35
MEDIAN_FILTER = 3


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
    dxv = (k_idx - c) * DX_M
    r = np.sqrt(dxv ** 2 + dy ** 2 + dz ** 2)
    in_balloon = r < A_INFL_M
    r_safe = np.where(r < 1e-12, 1e-12, r)
    R_ref = np.cbrt(r_safe ** 3 - A_INFL_M ** 3 + A_REF_M ** 3)
    lam = np.where(~in_balloon, r_safe / R_ref, 1.0)
    G_P1 = np.where(~in_balloon, _mu_P1(lam), G_WATER).astype(float)
    G_P2 = np.where(~in_balloon, _mu_P2(lam), G_WATER).astype(float)
    return G_P1, G_P2, in_balloon


def fibonacci_sphere(n):
    i = np.arange(n)
    phi = (1 + 5 ** 0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - 2 * (i + 0.5) / n
    rxy = np.sqrt(np.maximum(0.0, 1 - z * z))
    return np.stack([rxy * np.cos(theta), rxy * np.sin(theta), z], axis=1)


def ring_stats(G, balloon, shell_inner_mm=0.0, shell_outer_mm=12.0):
    i_idx, j_idx, k_idx = np.indices(G.shape)
    c = (N - 1) / 2.0
    r = np.sqrt(((i_idx - c) * DX_M) ** 2 +
                  ((j_idx - c) * DX_M) ** 2 +
                  ((k_idx - c) * DX_M) ** 2)
    ring = (~balloon) & (r >= A_INFL_M + shell_inner_mm * 1e-3) \
                        & (r <= A_INFL_M + shell_outer_mm * 1e-3)
    vals = G[ring]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def run_mip(G, balloon, freq, label):
    print(f"\n[{label}] running MIP pipeline at {freq} Hz...", flush=True)
    directions = fibonacci_sphere(N_DIRECTIONS)
    sources = multi_face_broadband_sources(N, radius_frac=DRIVER_R_FRAC,
                                                faces=("iN", "jN", "j0", "kN", "k0"))
    sources = [(i, j, k, DRIVER_AMP + 0.0j) for (i, j, k, _) in sources]

    t0 = time.time()
    u = helmholtz_solve_3d(G, freq=freq, rho=RHO, dx=DX_M, damping=DAMPING,
                              sources=sources, top_free=True)
    print(f"  Helmholtz: {time.time() - t0:.1f} s  |u| range "
          f"{np.abs(u).min():.2e} – {np.abs(u).max():.2e} m")

    G_est_stack = np.zeros((N_DIRECTIONS, N, N, N))
    amp_stack   = np.zeros((N_DIRECTIONS, N, N, N))
    for d, khat in enumerate(directions):
        u_k = directional_filter_3d(u, khat=khat, angular_width=WEDGE_WIDTH)
        G_k = direct_inversion_3d(u_k, freq=freq, rho=RHO, dx=DX_M,
                                      median_filter_size=MEDIAN_FILTER)
        G_est_stack[d] = np.where(np.isfinite(G_k), G_k, 0.0)
        amp_stack[d]   = np.abs(u_k)
        if (d + 1) % 5 == 0:
            print(f"    direction {d+1}/{N_DIRECTIONS}", flush=True)

    pos = np.where(G_est_stack > 0, G_est_stack, 0.0)
    G_tsm = pos.max(axis=0)
    wsum = amp_stack.sum(axis=0)
    wsafe = np.where(wsum > 1e-30, wsum, 1e-30)
    G_conv = (pos * amp_stack).sum(axis=0) / wsafe

    G_tsm  = np.where(balloon, np.nan, G_tsm)
    G_conv = np.where(balloon, np.nan, G_conv)
    return u, G_conv, G_tsm


def main():
    out_dir = ROOT / "results" / "paper_wave_sim_sobh_freq_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid: {N}³ hex at dx = {DX_M * 1000:.1f} mm → cube "
          f"{N * DX_M * 100:.1f}³ cm")
    print(f"Balloon: 250 mL (a = {A_INFL_M*100:.3f} cm)")
    print(f"Freq comparison: 60 Hz  vs  80 Hz")
    print()

    G_P1, G_P2, balloon = build_stiffness()
    ring_gt_P1 = ring_stats(G_P1, balloon)
    ring_gt_P2 = ring_stats(G_P2, balloon)
    print(f"GT ring: P1 = {ring_gt_P1/1000:.3f} kPa, "
          f"P2 = {ring_gt_P2/1000:.3f} kPa")

    results = {}
    for freq in (60.0, 80.0):
        cs = math.sqrt(MU_MEAN / RHO)
        lam_s = cs / freq
        print(f"\nAt {freq} Hz — unstretched-gel shear: "
              f"c_s = {cs:.2f} m/s, λ = {lam_s*1000:.1f} mm, "
              f"λ/dx = {lam_s/DX_M:.1f}")
        u_P1, conv_P1, tsm_P1 = run_mip(G_P1, balloon, freq, f"{freq}Hz P1")
        u_P2, conv_P2, tsm_P2 = run_mip(G_P2, balloon, freq, f"{freq}Hz P2")
        results[freq] = {
            "u_P1": u_P1, "conv_P1": conv_P1, "tsm_P1": tsm_P1,
            "u_P2": u_P2, "conv_P2": conv_P2, "tsm_P2": tsm_P2,
        }

    # ── Numeric summary ─────────────────────────────────────────────
    lines = [
        "60 Hz vs 80 Hz — Sobh-Ehman phantoms at 250 mL",
        "=" * 78,
        f"Grid: {N}³ at dx = {DX_M*1000:.1f} mm (18 cm cube)",
        f"Balloon 250 mL, gel: µ = {MU_MEAN:.0f} Pa, k1 = {K1_MEAN:.0f}, k2 = {K2_MEAN}",
        f"12 mm perilesional shell ring means:",
        "",
        f"{'':10s}{'GT':>10s}{'µ_conv (60)':>14s}{'µ_TSM (60)':>14s}"
        f"{'µ_conv (80)':>14s}{'µ_TSM (80)':>14s}",
    ]
    for phantom, gt in (("P1", ring_gt_P1), ("P2", ring_gt_P2)):
        row_vals = [gt / 1000]
        for freq in (60.0, 80.0):
            row_vals.append(ring_stats(results[freq][f"conv_{phantom}"], balloon) / 1000)
            row_vals.append(ring_stats(results[freq][f"tsm_{phantom}"], balloon) / 1000)
        lines.append(f"{phantom:10s}"
                     f"{row_vals[0]:>10.3f}"
                     f"{row_vals[1]:>14.3f}{row_vals[2]:>14.3f}"
                     f"{row_vals[3]:>14.3f}{row_vals[4]:>14.3f}")

    # TSM/conv ratios per phantom per frequency (Yin's TSM signature)
    lines.append("")
    lines.append("TSM / conv ratio (Yin's discriminating signature; Yin P1 measured 1.57):")
    for phantom in ("P1", "P2"):
        for freq in (60.0, 80.0):
            c_val = ring_stats(results[freq][f"conv_{phantom}"], balloon)
            t_val = ring_stats(results[freq][f"tsm_{phantom}"], balloon)
            ratio = t_val / max(c_val, 1)
            lines.append(f"  {phantom} @ {freq} Hz:   TSM/conv = {ratio:.3f}")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))

    np.savez_compressed(
        out_dir / "freq_compare.npz",
        G_P1=G_P1, G_P2=G_P2, balloon=balloon,
        conv_P1_60=results[60.0]["conv_P1"], tsm_P1_60=results[60.0]["tsm_P1"],
        conv_P2_60=results[60.0]["conv_P2"], tsm_P2_60=results[60.0]["tsm_P2"],
        conv_P1_80=results[80.0]["conv_P1"], tsm_P1_80=results[80.0]["tsm_P1"],
        conv_P2_80=results[80.0]["conv_P2"], tsm_P2_80=results[80.0]["tsm_P2"],
        u_P1_60=results[60.0]["u_P1"], u_P2_60=results[60.0]["u_P2"],
        u_P1_80=results[80.0]["u_P1"], u_P2_80=results[80.0]["u_P2"],
        dx_m=DX_M, driver_amp_m=DRIVER_AMP,
    )
    print(f"\nSaved {out_dir / 'freq_compare.npz'}")

    # ── Figure: shell heatmaps + radial profiles ────────────────────
    i_idx, j_idx, k_idx = np.indices(G_P1.shape)
    c_grid = (N - 1) / 2.0
    r = np.sqrt(((i_idx - c_grid) * DX_M) ** 2 +
                  ((j_idx - c_grid) * DX_M) ** 2 +
                  ((k_idx - c_grid) * DX_M) ** 2)
    shell = (~balloon) & (r >= A_INFL_M) & (r <= A_INFL_M + 0.012)
    mid = N // 2
    extent = [(-c_grid) * DX_M * 100, (N - 1 - c_grid) * DX_M * 100,
              (-c_grid) * DX_M * 100, (N - 1 - c_grid) * DX_M * 100]

    def _mask(field):
        return np.where(shell, field, np.nan)

    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.9])

    vmax = 0
    for freq in (60.0, 80.0):
        for phantom in ("P1", "P2"):
            for kind in ("conv", "tsm"):
                vmax = max(vmax, np.nanmax(_mask(results[freq][f"{kind}_{phantom}"])))
    vmax = max(vmax, np.nanmax(_mask(G_P1)), np.nanmax(_mask(G_P2))) / 1000
    vmin = 0

    # Row 0: GT (once for reference)
    for col, (name, G) in enumerate([("P1", G_P1), ("P2", G_P2)]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(_mask(G)[:, :, mid] / 1000, origin="upper",
                        extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
        ang = np.linspace(0, 2*np.pi, 400)
        ax.plot(A_INFL_M*100*np.cos(ang), A_INFL_M*100*np.sin(ang), "w-", lw=1.5)
        ax.plot((A_INFL_M+0.012)*100*np.cos(ang),
                (A_INFL_M+0.012)*100*np.sin(ang), "w--", lw=1.0)
        ax.set_title(f"{name} GT in 12 mm shell   "
                     f"⟨G⟩ = {ring_stats(G, balloon)/1000:.2f} kPa",
                     fontsize=11)
        ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.85)

    # Row 1: µ_TSM at 60 Hz
    for col, phantom in enumerate(("P1", "P2")):
        ax = fig.add_subplot(gs[1, col])
        arr = _mask(results[60.0][f"tsm_{phantom}"])
        im = ax.imshow(arr[:, :, mid] / 1000, origin="upper", extent=extent,
                        vmin=vmin, vmax=vmax, cmap="viridis")
        ang = np.linspace(0, 2*np.pi, 400)
        ax.plot(A_INFL_M*100*np.cos(ang), A_INFL_M*100*np.sin(ang), "w-", lw=1.5)
        ax.plot((A_INFL_M+0.012)*100*np.cos(ang),
                (A_INFL_M+0.012)*100*np.sin(ang), "w--", lw=1.0)
        m_val = np.nanmean(arr) / 1000
        ax.set_title(f"{phantom} $\\mu_{{TSM}}$ MIP  at 60 Hz   "
                     f"⟨G⟩ = {m_val:.2f} kPa", fontsize=11)
        ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.85)

    # Row 2: µ_TSM at 80 Hz
    for col, phantom in enumerate(("P1", "P2")):
        ax = fig.add_subplot(gs[2, col])
        arr = _mask(results[80.0][f"tsm_{phantom}"])
        im = ax.imshow(arr[:, :, mid] / 1000, origin="upper", extent=extent,
                        vmin=vmin, vmax=vmax, cmap="viridis")
        ang = np.linspace(0, 2*np.pi, 400)
        ax.plot(A_INFL_M*100*np.cos(ang), A_INFL_M*100*np.sin(ang), "w-", lw=1.5)
        ax.plot((A_INFL_M+0.012)*100*np.cos(ang),
                (A_INFL_M+0.012)*100*np.sin(ang), "w--", lw=1.0)
        m_val = np.nanmean(arr) / 1000
        ax.set_title(f"{phantom} $\\mu_{{TSM}}$ MIP  at 80 Hz (Yin)   "
                     f"⟨G⟩ = {m_val:.2f} kPa", fontsize=11)
        ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.85)

    # Row 3: radial profile with all curves
    ax = fig.add_subplot(gs[3, :])
    bins = np.linspace(A_INFL_M, A_INFL_M + 0.012, 24)
    r_flat = r.ravel()
    shell_flat = shell.ravel()
    def _radial(field):
        v = field.ravel()
        rm = r_flat[shell_flat]; vm = v[shell_flat]
        idx = np.digitize(rm, bins) - 1
        profile = np.zeros(len(bins) - 1)
        counts = np.zeros(len(bins) - 1, dtype=int)
        for i2, ri in enumerate(idx):
            if 0 <= ri < len(profile) and np.isfinite(vm[i2]):
                profile[ri] += vm[i2]
                counts[ri] += 1
        with np.errstate(invalid="ignore"):
            return profile / np.where(counts > 0, counts, 1)
    r_cm = 0.5 * (bins[:-1] + bins[1:]) * 100

    ax.plot(r_cm, _radial(G_P1)/1000, "-", color="darkred", lw=2.4, label="P1 GT")
    ax.plot(r_cm, _radial(results[60.0]["tsm_P1"])/1000, "--",
             color="tab:red", lw=2.0, label="P1 $\\mu_{TSM}$ 60 Hz")
    ax.plot(r_cm, _radial(results[80.0]["tsm_P1"])/1000, ":",
             color="tab:red", lw=2.0, label="P1 $\\mu_{TSM}$ 80 Hz")
    ax.plot(r_cm, _radial(G_P2)/1000, "-", color="darkblue", lw=2.4, label="P2 GT")
    ax.plot(r_cm, _radial(results[60.0]["tsm_P2"])/1000, "--",
             color="tab:blue", lw=2.0, label="P2 $\\mu_{TSM}$ 60 Hz")
    ax.plot(r_cm, _radial(results[80.0]["tsm_P2"])/1000, ":",
             color="tab:blue", lw=2.0, label="P2 $\\mu_{TSM}$ 80 Hz")
    ax.axvline(A_INFL_M*100, color="black", ls=":", alpha=0.6, label="cavity edge")
    ax.axvline((A_INFL_M+0.012)*100, color="black", ls=":", alpha=0.35,
                 label="shell outer")
    ax.set_xlabel("radial distance from balloon centre [cm]", fontsize=11)
    ax.set_ylabel("stiffness [kPa]", fontsize=11)
    ax.set_title("Radial profiles inside the 12 mm shell — GT vs $\\mu_{TSM}$ at 60 Hz vs 80 Hz",
                  fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc="upper right")

    plt.suptitle("Frequency comparison — Yin used 80 Hz for the phantoms",
                  fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "freq_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_dir / 'freq_compare.png'}")


if __name__ == "__main__":
    main()
