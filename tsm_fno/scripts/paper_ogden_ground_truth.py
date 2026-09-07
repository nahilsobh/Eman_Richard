#!/usr/bin/env python3
"""Analytical Ogden G_true (no inversion) vs Yin Fig 6 — ground-truth plot.

The Ogden constitutive law is applied to the analytical Lamé stretch
field for a balloon inflating in an incompressible matrix, and the
perilesional ring stiffness is computed directly from the tensor field
G_ij(x) — no wave solve, no direct inversion, no filter.

This isolates the *constitutive-law prediction* from the numerical
noise of the DI + MIP inversion pipeline. If G_true tracks Yin's TSM
closely, it proves the Ogden parameters are physically correct even
where the inversion pipeline can't recover the full amplitude.

Compares Yin's measured Fig 6 curves against four analytical predictions:
  • P1 baseline:  mu=(1100, 1400), alpha=(3, 1)  — user's original
  • P1 tuned  :   mu=(1100, 1400), alpha=(7, 1)  — raise nonlinear term
  • P2 baseline:  mu=(2400, 100),  alpha=(2, 10) — user's original
  • P2 tuned  :   mu=(2000, 500),  alpha=(2, 10) — rebalance to fiber term

Fast (~1 s total, no solve).
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
    perilesional_shell_3d,
    SphericalBalloon,
)


N        = 32
DX       = 0.003
G_LESION = 2000.0
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM        = 5.0
SHELL_OFFSET_MM = 3.0    # sample voxels closer to balloon (where stretch is real)

BALLOON_VOLUMES_ML = [50, 100, 150, 200, 250]
def _r_vx(vol):
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX
BALLOON_RADII_VX = [_r_vx(v) for v in BALLOON_VOLUMES_ML]
A0_VX = BALLOON_RADII_VX[0]

# Full inflation + deflation state list.
STATES = list(zip(BALLOON_RADII_VX, BALLOON_VOLUMES_ML, ["inflation"] * 5))
for i in range(3, -1, -1):
    STATES.append((BALLOON_RADII_VX[i], BALLOON_VOLUMES_ML[i], "deflation"))


CONFIGS = [
    dict(label="P1 baseline (user's original)",
         mu=[1100.0, 1400.0], alpha=[3.0, 1.0],
         color="black",     ls="--", marker="o"),
    dict(label="P1 tuned (α₁ = 7)",
         mu=[1100.0, 1400.0], alpha=[7.0, 1.0],
         color="black",     ls="-",  marker="o"),
    dict(label="P2 baseline (user's original)",
         mu=[2400.0,  100.0], alpha=[2.0, 10.0],
         color="tab:cyan",  ls="--", marker="s"),
    dict(label="P2 tuned (μ rebalanced to fiber)",
         mu=[2000.0,  500.0], alpha=[2.0, 10.0],
         color="tab:cyan",  ls="-",  marker="s"),
]


def analytical_ring(a_vx, a0_vx, mu, alpha):
    """Return G_true ring mean using ONLY the constitutive law + analytical
    Lamé stretch field — no wave solve, no DI, no noise."""
    balloon = SphericalBalloon(center=CENTER, radius_vx=a_vx, pressure=0.0)
    G_tensor = ogden_G_tensor_field(N=N, dx=DX, center=CENTER,
                                     a0_vx=a0_vx, a_vx=a_vx,
                                     mu_list=mu, alpha_list=alpha,
                                     G_lesion=G_LESION,
                                     balloon_mask=balloon.mask(N))
    # Isotropic-mean of tensor = (G_xx + G_yy + G_zz) / 3
    G_iso = np.einsum("xyzii->xyz", G_tensor) / 3.0
    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=SHELL_MM, dx=DX,
                                   inner_offset_mm=SHELL_OFFSET_MM)
    ring = G_iso[shell]; ring = ring[np.isfinite(ring)]
    if ring.size == 0: return float("nan")
    lo, hi = np.percentile(ring, [10, 90])
    tr = ring[(ring >= lo) & (ring <= hi)]
    return float(tr.mean())


def tangential_ring(a_vx, a0_vx, mu, alpha):
    """G_θ (pure tangential direction) ring mean — the Yin-relevant
    principal component (shear waves polarized tangentially to the ring)."""
    from src.phantom.geometry_3d import ogden_lame_stretch_field
    balloon = SphericalBalloon(center=CENTER, radius_vx=a_vx, pressure=0.0)
    lam_r, lam_theta, _ = ogden_lame_stretch_field(N, DX, CENTER, a0_vx, a_vx)
    G_theta = np.zeros_like(lam_theta)
    for m, al in zip(mu, alpha):
        G_theta = G_theta + float(m) * np.power(lam_theta, al - 2.0)
    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=SHELL_MM, dx=DX,
                                   inner_offset_mm=SHELL_OFFSET_MM)
    ring = G_theta[shell]; ring = ring[np.isfinite(ring)]
    if ring.size == 0: return float("nan")
    lo, hi = np.percentile(ring, [10, 90])
    tr = ring[(ring >= lo) & (ring <= hi)]
    return float(tr.mean())


def main():
    out_dir = ROOT / "results" / "paper_ogden_ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves_iso = {c["label"]: [] for c in CONFIGS}
    curves_theta = {c["label"]: [] for c in CONFIGS}
    for a_vx, vol, branch in STATES:
        for cfg in CONFIGS:
            curves_iso[cfg["label"]].append(
                analytical_ring(a_vx, A0_VX, cfg["mu"], cfg["alpha"]) / 1000)
            curves_theta[cfg["label"]].append(
                tangential_ring(a_vx, A0_VX, cfg["mu"], cfg["alpha"]) / 1000)

    # Yin Fig 6 measured curves.
    yin_p1 = [3.5, 3.8, 3.9, 4.2, 4.4, 4.2, 3.7, 3.6, 3.6]
    yin_p2 = [3.5, 3.9, 4.2, 4.9, 5.15, 4.6, 3.9, 3.6, 3.6]
    n = len(STATES)
    x = list(range(n))
    labels = [s[1] for s in STATES]

    # ── Two-panel figure: (a) isotropic-average, (b) tangential G_θ ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, curves, title in [
        (axes[0], curves_iso,   "G_true = trace(G_ij)/3   (isotropic mean)"),
        (axes[1], curves_theta, "G_θ = Σμᵢ·λ_θ^(αᵢ−2)   (tangential component, Yin-relevant)"),
    ]:
        # Yin measured (thick stars)
        ax.plot(x, yin_p1, "*-", color="black",    ms=13, lw=1.5,
                label="Yin P1 TSM (measured)", alpha=0.8)
        ax.plot(x, yin_p2, "*-", color="tab:cyan", ms=13, lw=1.5,
                label="Yin P2 TSM (measured)", alpha=0.8)

        # Our analytical predictions
        for cfg in CONFIGS:
            ax.plot(x, curves[cfg["label"]], cfg["ls"],
                     color=cfg["color"], marker=cfg["marker"], ms=7, lw=1.8,
                     label=cfg["label"], alpha=0.85)

        # Shading
        ax.axvspan(-0.5, 4.5, alpha=0.05, color="tab:blue")
        ax.axvspan(4.5, n - 0.5, alpha=0.05, color="tab:orange")
        ax.text(2, 5.7, "Inflation", ha="center", color="tab:blue",
                 fontsize=10, fontweight="bold", alpha=0.7)
        ax.text(6.5, 5.7, "Deflation", ha="center", color="tab:orange",
                 fontsize=10, fontweight="bold", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels([f"{v}" for v in labels], fontsize=10)
        ax.set_xlabel("Balloon water volume (mL)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(2.0, 6.0)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Ring G [kPa]   (analytical — no wave solve, no DI)")

    fig.suptitle(
        "Ogden N=2 ground truth vs Yin Fig 6\n"
        "Left: isotropic-mean of the anisotropic G_ij tensor.  "
        "Right: pure tangential G_θ component (Yin-relevant).",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_fig = out_dir / "ogden_ground_truth.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")

    # ── Summary text ──
    lines = [
        "Ogden N=2 ground-truth ring stiffness vs Yin Fig 6",
        "=" * 78,
        "Analytical Lamé stretch → Ogden tensor G_ij(x) → shell mean.",
        "No wave solve, no direct inversion. Isolates the constitutive law.",
        f"Shell inner offset = {SHELL_OFFSET_MM} mm, thickness = {SHELL_MM} mm.",
        "",
        f"State:      {'  '.join(f'{v:>4d}' for v in labels)}",
        f"Branch:     {'  '.join(f'{s[2][:4]}' for s in STATES):>{5*n-1}}",
        "",
        f"Yin P1 TSM: {'  '.join(f'{v:5.2f}' for v in yin_p1)}",
        f"Yin P2 TSM: {'  '.join(f'{v:5.2f}' for v in yin_p2)}",
        "",
        "--- Isotropic mean G_true (tensor trace /3) ---",
    ]
    for cfg in CONFIGS:
        lines.append(f"  {cfg['label']:<40s}: "
                     + "  ".join(f"{v:5.2f}" for v in curves_iso[cfg['label']]))
    lines += [
        "",
        "--- Tangential component G_θ = Σμ_p · λ_θ^(α_p-2) ---",
    ]
    for cfg in CONFIGS:
        lines.append(f"  {cfg['label']:<40s}: "
                     + "  ".join(f"{v:5.2f}" for v in curves_theta[cfg['label']]))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
