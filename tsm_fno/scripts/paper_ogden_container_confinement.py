#!/usr/bin/env python3
"""Composition-based Ogden with container-confinement correction.

Yin's phantoms are cast inside a rigid-walled container with only the
TOP surface free. That's different from the infinite-matrix Lamé
solution our analytical model has been using.

Physical effect: gel material displaced by the balloon can't flow
radially outward (walls block it) — all displaced volume must travel
up through the free top. This makes the effective tangential stretch
at the balloon surface LARGER than the infinite-matrix estimate.

Simple first-order correction: multiply λ_θ(x) by a
`confinement_factor` > 1 uniformly. This is an approximation — the
correct BVP has spatially-varying enhancement (larger near balloon,
zero far from it) — but captures the leading effect.

Sweeps three confinement factors: 1.0 (infinite matrix — no change),
1.15 (moderate — Yin's likely regime for 250 mL in a ~500 mL jar),
1.30 (tight — very packed geometry).

Composition-based Ogden (NOT tuned to Yin):
  P1: 10% bovine gelatin — μ = (1800, 700) Pa,  α = (2.5, 3.0)
  P2: 8% gelatin + 7% cellulose — μ = (2500, 1500) Pa,  α = (2.5, 5.0)
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

PHANTOMS = [
    dict(name="Phantom 1 — 10% gelatin",
         mu=[1800.0,  700.0], alpha=[2.5, 3.0],
         yin_tsm=[3.5, 3.8, 3.9, 4.2, 4.4]),
    dict(name="Phantom 2 — 8% gel + 7% cellulose",
         mu=[2500.0, 1500.0], alpha=[2.5, 5.0],
         yin_tsm=[3.5, 3.9, 4.2, 4.9, 5.15]),
]

# Yin's control-based MIP bias (~1 kPa upward offset from true G_bg).
MIP_BIAS_KPA = 0.85

CONFINEMENT_FACTORS = [1.00, 1.15, 1.30]


def _ring_stats(g_field, shell):
    v = g_field[shell]; v = v[np.isfinite(v)]
    if v.size == 0: return float("nan")
    lo, hi = np.percentile(v, [10, 90])
    tr = v[(v >= lo) & (v <= hi)]
    return float(tr.mean()) if tr.size else float("nan")


def analytical_ring_with_confinement(a_vx, mu, alpha, confinement):
    """Compute G_θ and tr(G)/3 ring means with a container-confinement
    correction (λ_θ multiplied by `confinement`)."""
    balloon = SphericalBalloon(center=CENTER, radius_vx=a_vx, pressure=0.0)
    lam_r, lam_theta, _rhat = ogden_lame_stretch_field(N, DX, CENTER, A0_VX, a_vx)

    # Apply confinement to λ_θ. To keep incompressibility (λ_r · λ_θ² = 1),
    # adjust λ_r proportionally: λ_r_new = 1 / λ_θ_new²
    lam_theta_conf = lam_theta * float(confinement)
    lam_r_conf     = 1.0 / (lam_theta_conf ** 2)

    G_theta = np.zeros_like(lam_theta_conf)
    G_iso_contrib_theta = np.zeros_like(lam_theta_conf)
    G_iso_contrib_r     = np.zeros_like(lam_theta_conf)
    for m, al in zip(mu, alpha):
        exponent = al - 2.0
        G_theta = G_theta + float(m) * np.power(lam_theta_conf, exponent)
        # tr(G)/3 = (G_r + 2·G_θ)/3
        G_iso_contrib_theta = G_iso_contrib_theta + float(m) * np.power(lam_theta_conf, exponent)
        G_iso_contrib_r     = G_iso_contrib_r     + float(m) * np.power(lam_r_conf,     exponent)
    G_iso = (G_iso_contrib_r + 2.0 * G_iso_contrib_theta) / 3.0

    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=SHELL_MM, dx=DX,
                                   inner_offset_mm=SHELL_OFFSET_MM)
    return _ring_stats(G_theta, shell) / 1000, _ring_stats(G_iso, shell) / 1000


def main():
    out_dir = ROOT / "results" / "paper_ogden_container_confinement"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute curves for each phantom × each confinement.
    results = {}
    for p in PHANTOMS:
        results[p["name"]] = {}
        for cf in CONFINEMENT_FACTORS:
            gt_curve, gi_curve = [], []
            for a_vx, vol in zip(BALLOON_RADII_VX, BALLOON_VOLUMES_ML):
                gt, gi = analytical_ring_with_confinement(a_vx, p["mu"], p["alpha"], cf)
                gt_curve.append(gt)
                gi_curve.append(gi)
            results[p["name"]][cf] = dict(g_theta=gt_curve, g_iso=gi_curve)

    # ── Figure: one panel per phantom, curves for each confinement ──
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    colors = {1.00: "tab:red", 1.15: "tab:green", 1.30: "tab:purple"}

    for ax, p in zip(axes, PHANTOMS):
        # Yin measured
        ax.plot(BALLOON_VOLUMES_ML, p["yin_tsm"], "*-", color="black",
                ms=14, lw=1.8, label="Yin μ_TSM (measured)")

        # For each confinement, plot G_θ + MIP-bias correction (best analog for Yin TSM)
        for cf in CONFINEMENT_FACTORS:
            gt = results[p["name"]][cf]["g_theta"]
            gt_biased = [v + MIP_BIAS_KPA for v in gt]
            ax.plot(BALLOON_VOLUMES_ML, gt_biased,
                     "o-", color=colors[cf], ms=8, lw=2.2,
                     label=f"Ogden G_θ + 0.85 MIP-bias, confinement={cf:.2f}")

        # Raw G_θ (no confinement, no bias) as reference
        gt_raw = results[p["name"]][1.00]["g_theta"]
        ax.plot(BALLOON_VOLUMES_ML, gt_raw,
                 "s:", color="gray", ms=6, lw=1.2, alpha=0.6,
                 label="Ogden G_θ (infinite matrix, no MIP-bias)")

        params_str = (f"Ogden N=2: μ = {p['mu']} Pa,  α = {p['alpha']}")
        ax.text(0.02, 0.97, params_str, transform=ax.transAxes,
                 fontsize=9, va="top", fontfamily="monospace",
                 bbox=dict(facecolor="white", alpha=0.85, pad=3, edgecolor="lightgray"))

        ax.set_title(p["name"], fontsize=12, fontweight="bold")
        ax.set_ylabel("Ring stiffness [kPa]", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    axes[-1].set_xlabel("Balloon water volume (mL)  — inflation only, both phantoms",
                        fontsize=11)
    axes[-1].set_xticks(BALLOON_VOLUMES_ML)
    plt.suptitle(
        "Composition-based Ogden with container-confinement correction vs Yin\n"
        "MIP-bias offset (+0.85 kPa) added — measured via Yin's P3 control",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_fig = out_dir / "ogden_container_confinement.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")

    # ── Summary ──
    lines = [
        "Composition-based Ogden with container-confinement correction",
        "=" * 78,
        f"MIP-bias offset applied: +{MIP_BIAS_KPA} kPa (from Yin's P3 control gap)",
        "",
    ]
    for p in PHANTOMS:
        lines.append(f"{p['name']}")
        lines.append(f"  Ogden N=2: μ={p['mu']} Pa, α={p['alpha']}")
        lines.append(f"  Volume(mL):  {'  '.join(f'{v:>5d}' for v in BALLOON_VOLUMES_ML)}")
        lines.append(f"  Yin TSM:     {'  '.join(f'{v:5.2f}' for v in p['yin_tsm'])}")
        for cf in CONFINEMENT_FACTORS:
            gt = results[p["name"]][cf]["g_theta"]
            gt_biased = [v + MIP_BIAS_KPA for v in gt]
            errs = [(a - y)/y*100 for a, y in zip(gt_biased, p["yin_tsm"])]
            lines.append(f"  cf={cf:.2f}+bias: "
                         + "  ".join(f"{v:5.2f}" for v in gt_biased)
                         + f"    (errs: {' '.join(f'{e:+4.0f}%' for e in errs)})")
        lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
