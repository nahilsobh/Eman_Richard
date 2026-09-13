#!/usr/bin/env python3
"""Ogden-only baseline: pure physics prediction, no corrections.

Reports what the Ogden constitutive law predicts for both phantoms
using ONLY composition-based parameters. No container confinement,
no MIP bias, no fitting to Yin.

Purpose: establish an honest baseline before layering any corrections.
Any subsequent deviations between this and Yin's MRE readouts represent
some combination of:
    (a) our composition-based Ogden parameters differing from Yin's
        actual gel constants (composite rheology uncertainty),
    (b) Yin's MRE measurement operator (MIP-based inversion) systematic
        biases relative to the true material G,
    (c) container-geometry effects not present in our infinite-matrix
        Lamé assumption,
without independent mechanical characterization of Yin's gel we cannot
attribute the gap to any one of these.

Composition-based Ogden N=2 parameters (from published gel rheometry):
    P1 (10% bovine gelatin, 5-day RT cure):
        μ = (1800, 700) Pa,  α = (2.5, 3.0),  G₀ = 2500 Pa
    P2 (8% gelatin + 7% cellulose fiber, 5-day RT cure):
        μ = (2500, 1500) Pa, α = (2.5, 5.0),  G₀ = 4000 Pa
        (initial composite estimate — significant literature uncertainty)

Reports two Ogden analog quantities per phantom:
    G_θ    = Σ μ_p · λ_θ^(α_p−2)   ← tangential principal (max-direction)
    tr(G)/3 = isotropic tensor mean (direction-averaged)

Yin's Fig 6 curves are overlaid as a REFERENCE, not a ground-truth target.
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

PHANTOMS = [
    dict(name="Phantom 1 — 10% bovine gelatin",
         mu=[1800.0,  700.0], alpha=[2.5, 3.0],
         yin_ref=[3.5, 3.8, 3.9, 4.2, 4.4]),
    dict(name="Phantom 2 — 8% gelatin + 7% cellulose fiber",
         mu=[2500.0, 1500.0], alpha=[2.5, 5.0],
         yin_ref=[3.5, 3.9, 4.2, 4.9, 5.15]),
]


def analytical_ring(a_vx, mu, alpha):
    """G_θ (tangential) and tr(G)/3 (isotropic mean) shell means, no corrections."""
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
    out_dir = ROOT / "results" / "paper_ogden_only_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in PHANTOMS:
        p["g_theta"] = []
        p["g_iso"] = []
        for a_vx in BALLOON_RADII_VX:
            gt, gi = analytical_ring(a_vx, p["mu"], p["alpha"])
            p["g_theta"].append(gt)
            p["g_iso"].append(gi)

    # ── Figure: two panels, one per phantom ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, p in zip(axes, PHANTOMS):
        # Yin as REFERENCE (not target)
        ax.plot(BALLOON_VOLUMES_ML, p["yin_ref"], "*-", color="black",
                ms=14, lw=1.5, alpha=0.55,
                label="Yin μ_TSM (MIP-based measurement, reference only)")
        # Ogden predictions
        ax.plot(BALLOON_VOLUMES_ML, p["g_theta"], "o-", color="tab:red",
                ms=9, lw=2.4,
                label="Ogden G_θ (tangential principal)")
        ax.plot(BALLOON_VOLUMES_ML, p["g_iso"], "s--", color="tab:blue",
                ms=8, lw=2.0, alpha=0.85,
                label="Ogden tr(G)/3 (isotropic mean)")

        params_str = (f"Ogden N=2: μ = {p['mu']} Pa,  α = {p['alpha']}\n"
                      f"G₀ = {sum(p['mu']):.0f} Pa  (from composition, not Yin)")
        ax.text(0.02, 0.97, params_str, transform=ax.transAxes,
                 fontsize=9, va="top", fontfamily="monospace",
                 bbox=dict(facecolor="white", alpha=0.85, pad=4,
                           edgecolor="lightgray"))

        ax.set_title(p["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Balloon water volume (mL)", fontsize=11)
        ax.set_ylabel("Ring stiffness [kPa]", fontsize=11)
        ax.set_xticks(BALLOON_VOLUMES_ML)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    plt.suptitle(
        "Ogden-only baseline — pure physics prediction, NO corrections\n"
        "(no container confinement, no MIP-bias, no Yin fitting)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_fig = out_dir / "ogden_only_baseline.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")

    # ── Summary ──────────────────────────────────────────────────────
    lines = [
        "Ogden-only baseline — pure physics prediction, NO corrections",
        "=" * 78,
        "",
        "IMPORTANT: Yin's Fig 6 values are her MIP-based MRE readouts, NOT",
        "independent ground truth for the phantom material G. The comparison",
        "below is between two INDEPENDENT measurements of the same unknown",
        "underlying quantity — neither is the 'right' answer without independent",
        "mechanical characterization of Yin's actual gel.",
        "",
        "PHANTOM PARAMETERS (from composition, not tuned to Yin):",
    ]
    for p in PHANTOMS:
        lines.append(f"  {p['name']}:")
        lines.append(f"    μ = {p['mu']} Pa,  α = {p['alpha']},  G₀ = {sum(p['mu'])} Pa")
    lines.append("")
    lines.append("OGDEN PREDICTIONS (analytical, no wave solve, no inversion):")
    lines.append("")
    for p in PHANTOMS:
        lines.append(f"{p['name']}:")
        lines.append(f"  Volume(mL):    {'  '.join(f'{v:>5d}' for v in BALLOON_VOLUMES_ML)}")
        lines.append(f"  Yin (ref):     {'  '.join(f'{v:5.2f}' for v in p['yin_ref'])}")
        lines.append(f"  Ogden G_θ:     {'  '.join(f'{v:5.2f}' for v in p['g_theta'])}")
        lines.append(f"  Ogden tr(G)/3: {'  '.join(f'{v:5.2f}' for v in p['g_iso'])}")
        lines.append("")

    lines += [
        "The gap between Ogden and Yin reflects some combination of:",
        "  (a) composition-based Ogden parameters differing from Yin's actual gel,",
        "  (b) Yin's MRE inversion introducing systematic biases (MIP artifact),",
        "  (c) container-geometry effects (Yin's finite bounded matrix vs",
        "      our infinite-matrix Lamé assumption).",
        "",
        "Without independent mechanical testing of Yin's specific gel, we cannot",
        "attribute the gap to any single cause. Subsequent scripts explore each",
        "of these three contributions (see paper_ogden_container_confinement.py",
        "and paper_ogden_composition_final.py).",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
