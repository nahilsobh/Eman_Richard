#!/usr/bin/env python3
"""Simple per-phantom Ogden-vs-Yin comparison plots.

One panel per phantom, three curves each:
  1. Yin's measured TSM (from Fig 6)   — star markers, black
  2. Ogden analytical G_θ (tangential principal, what tangential-wave
     measurement would see if it picked the max direction)
  3. Ogden analytical tr(G)/3 (isotropic mean, what direction-averaged
     inversion recovers)

The Ogden curves bracket where the measurement should sit for
principal-axis-aligned wave physics. Yin's TSM = 20-direction MIP falls
somewhere in that range plus a constant MIP-upward-bias.

Composition-based Ogden parameters (NOT tuned to Yin):
  P1: 10% bovine gelatin, 5-day RT cure
      μ = (1800, 700) Pa,   α = (2.5, 3.0),   G₀ = 2500 Pa
  P2: 8% gelatin + 7% cellulose fiber
      μ = (2500, 1500) Pa,  α = (2.5, 5.0),   G₀ = 4000 Pa
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

from src.phantom.geometry_3d import (
    ogden_G_tensor_field,
    ogden_lame_stretch_field,
    perilesional_shell_3d,
    SphericalBalloon,
)


N        = 32
DX       = 0.003
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM        = 5.0
SHELL_OFFSET_MM = 3.0

BALLOON_VOLUMES_ML = [50, 100, 150, 200, 250]
def _r_vx(vol):
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX
BALLOON_RADII_VX = [_r_vx(v) for v in BALLOON_VOLUMES_ML]
A0_VX = BALLOON_RADII_VX[0]

SCHEDULE = list(zip(BALLOON_RADII_VX, BALLOON_VOLUMES_ML, ["infl"] * 5))
for i in range(3, -1, -1):
    SCHEDULE.append((BALLOON_RADII_VX[i], BALLOON_VOLUMES_ML[i], "defl"))

PHANTOMS = [
    dict(name="Phantom 1 — 10% bovine gelatin",
         mu=[1800.0,  700.0], alpha=[2.5, 3.0],
         yin_tsm=[3.5, 3.8, 3.9, 4.2, 4.4, 4.2, 3.7, 3.6, 3.6]),
    dict(name="Phantom 2 — 8% gelatin + 7% cellulose fiber",
         mu=[2500.0, 1500.0], alpha=[2.5, 5.0],
         yin_tsm=[3.5, 3.9, 4.2, 4.9, 5.15, 4.6, 3.9, 3.6, 3.6]),
]


def analytical_ring(a_vx, mu, alpha):
    balloon = SphericalBalloon(center=CENTER, radius_vx=a_vx, pressure=0.0)
    G_tensor = ogden_G_tensor_field(N=N, dx=DX, center=CENTER,
                                     a0_vx=A0_VX, a_vx=a_vx,
                                     mu_list=mu, alpha_list=alpha,
                                     G_lesion=1500.0,
                                     balloon_mask=balloon.mask(N))
    G_iso = np.einsum("xyzii->xyz", G_tensor) / 3.0

    _lam_r, lam_theta, _rhat = ogden_lame_stretch_field(N, DX, CENTER, A0_VX, a_vx)
    G_theta = np.zeros_like(lam_theta)
    for m, al in zip(mu, alpha):
        G_theta = G_theta + float(m) * np.power(lam_theta, al - 2.0)

    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=SHELL_MM, dx=DX,
                                   inner_offset_mm=SHELL_OFFSET_MM)

    def _ring(f):
        v = f[shell]; v = v[np.isfinite(v)]
        if v.size == 0: return float("nan")
        lo, hi = np.percentile(v, [10, 90])
        tr = v[(v >= lo) & (v <= hi)]
        return float(tr.mean()) if tr.size else float("nan")

    return _ring(G_theta) / 1000, _ring(G_iso) / 1000


def main():
    out_dir = ROOT / "results" / "paper_ogden_vs_yin_per_phantom"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in PHANTOMS:
        g_th_curve, g_iso_curve = [], []
        for a_vx, vol, branch in SCHEDULE:
            g_th, g_iso = analytical_ring(a_vx, p["mu"], p["alpha"])
            g_th_curve.append(g_th)
            g_iso_curve.append(g_iso)
        p["g_theta"] = g_th_curve
        p["g_iso"] = g_iso_curve

    n = len(SCHEDULE)
    x = list(range(n))
    x_labels = [s[1] for s in SCHEDULE]

    # ── One panel per phantom, stacked vertically ────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

    for ax, p in zip(axes, PHANTOMS):
        # Background: inflation blue, deflation orange.
        ax.axvspan(-0.5, 4.5,   alpha=0.06, color="tab:blue")
        ax.axvspan(4.5, n - 0.5, alpha=0.06, color="tab:orange")

        # Yin measured (heavy black stars).
        ax.plot(x, p["yin_tsm"], "*-", color="black", ms=14, lw=1.8,
                label="Yin μ_TSM (measured, MIP-based)")

        # Ogden analytical: two curves bracketing the prediction range.
        ax.plot(x, p["g_theta"], "o-", color="tab:red", ms=8, lw=2.4,
                label=f"Ogden G_θ tangential (max-direction estimate)")
        ax.plot(x, p["g_iso"], "s--", color="tab:blue", ms=8, lw=2.0, alpha=0.85,
                label=f"Ogden tr(G)/3 mean (direction-averaged estimate)")

        # Annotate parameters on the plot.
        params_str = (f"Ogden N=2: μ = {p['mu']} Pa,  α = {p['alpha']}"
                      f"    (composition-based, not tuned to Yin)")
        ax.text(0.02, 0.97, params_str, transform=ax.transAxes,
                 fontsize=9, va="top", fontfamily="monospace",
                 bbox=dict(facecolor="white", alpha=0.85, pad=3, edgecolor="lightgray"))

        ax.set_title(p["name"], fontsize=12, fontweight="bold")
        ax.set_ylabel("Ring stiffness [kPa]", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    axes[0].set_ylim(1.8, 5.5)
    axes[1].set_ylim(3.0, 7.0)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{v}" for v in x_labels], fontsize=10)
    axes[-1].set_xlabel("Balloon water volume (mL)     "
                        "[inflation ← 50→250 | 200→50 → deflation]",
                        fontsize=11)

    plt.suptitle(
        "Ogden analytical prediction (composition-based) vs Yin Fig 6 — per phantom",
        fontsize=13, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    out_fig = out_dir / "ogden_vs_yin_per_phantom.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")

    # Summary text
    lines = [
        "Simple per-phantom Ogden vs Yin comparison",
        "=" * 78,
        "",
    ]
    for p in PHANTOMS:
        lines.append(f"{p['name']}")
        lines.append(f"  Ogden N=2: μ = {p['mu']} Pa,  α = {p['alpha']}    G₀ = {sum(p['mu'])} Pa")
        lines.append(f"  Composition: from published gel rheometry (not tuned to Yin)")
        lines.append("")
        lines.append(f"  Volume(mL): {'  '.join(f'{v:>5d}' for v in BALLOON_VOLUMES_ML)}")
        lines.append(f"  Yin TSM  :  {'  '.join(f'{v:5.2f}' for v in p['yin_tsm'][:5])}")
        lines.append(f"  Ogden G_θ:  {'  '.join(f'{v:5.2f}' for v in p['g_theta'][:5])}")
        lines.append(f"  Ogden tr/3: {'  '.join(f'{v:5.2f}' for v in p['g_iso'][:5])}")
        lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
