#!/usr/bin/env python3
"""Plot FEniCS FEM results (from paper_ogden_fenics_fem.py) vs Yin.

Reads `results.json` from a `paper_ogden_fenics_fem_dx{N}mm/` folder,
produces a 2-panel figure (one per phantom) with:
  - FEniCS FEM ring G_θ (physics ground truth for the Ogden phantom model)
  - Linear FDM ring G_θ (from paper_ogden_option3_container_fem.py, for context)
  - Yin's MIP-MRE measurement
Annotates the per-state deviation and reports RMS(MIP−FEM).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PHANTOMS = [
    dict(short="P1", name="Phantom 1 — 10% bovine gelatin",
         mu=[1800.0, 700.0], alpha=[2.5, 3.0],
         yin_tsm=[3.5, 3.8, 3.9, 4.2, 4.4]),
    dict(short="P2", name="Phantom 2 — 8% gel + 7% cellulose",
         mu=[2500.0, 1500.0], alpha=[2.5, 5.0],
         yin_tsm=[3.5, 3.9, 4.2, 4.9, 5.15]),
]

# Reference linear-FDM option-3 numbers (from paper_ogden_option3_container_fem)
LINEAR_FDM = {
    "P1": [2.50, 2.71, 2.87, 3.00, 3.11],
    "P2": [4.00, 4.87, 5.65, 6.37, 7.06],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx-mm", type=int, required=True)
    args = ap.parse_args()

    result_dir = ROOT / "results" / f"paper_ogden_fenics_fem_dx{args.dx_mm}mm"
    js_path = result_dir / "results.json"
    if not js_path.exists():
        raise FileNotFoundError(f"{js_path} not found. Run FEM first.")
    data = json.loads(js_path.read_text())

    vols = [r["vol_ml"] for r in data["results"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, p in zip(axes, PHANTOMS):
        key = f"G_{p['short']}"
        fem  = [r[key] for r in data["results"]]
        lin  = LINEAR_FDM[p["short"]][:len(vols)]
        yin  = p["yin_tsm"][:len(vols)]

        ax.plot(vols, fem, "o-", color="tab:red", ms=10, lw=2.6,
                label=f"FEM ground truth ({args.dx_mm} mm hex, near-incomp neo-Hookean + Ogden)")
        ax.plot(vols, lin, "s:", color="tab:blue", ms=6, lw=1.4, alpha=0.6,
                label="Linear FDM option-3 (small-strain, ex-post Ogden)")
        ax.plot(vols, yin, "*--", color="black", ms=14, lw=1.5,
                label="Yin MIP-MRE (measured)")

        # Annotate FEM - Yin deviation
        for v, f, y in zip(vols, fem, yin):
            d = y - f
            ax.annotate(f"{d:+.2f}", xy=(v, y),
                         xytext=(0, 10 if d > 0 else -18),
                         textcoords="offset points",
                         ha="center", fontsize=8)

        info = (f"Ogden N=2: μ = {p['mu']} Pa\n"
                f"           α = {p['alpha']}\n"
                f"G₀ = {sum(p['mu']):.0f} Pa (composition)")
        ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8,
                 va="top", fontfamily="monospace",
                 bbox=dict(facecolor="white", alpha=0.85, pad=4,
                           edgecolor="lightgray"))

        errs = np.array(yin) - np.array(fem)
        rms = float(np.sqrt(np.mean(errs ** 2)))
        rms_pct = float(np.sqrt(np.mean((errs / np.array(fem)) ** 2))) * 100
        ax.set_title(
            f"{p['name']}\nMIP-MRE − FEM: RMS = {rms:.2f} kPa ({rms_pct:.1f}%)",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Balloon water volume (mL)", fontsize=11)
        ax.set_ylabel("Ring stiffness [kPa]", fontsize=11)
        ax.set_xticks(vols)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle(
        f"FEniCS finite-strain FEM (dx = {args.dx_mm} mm hex, "
        f"{data['results'][0]['n_ring']}+ ring cells, "
        f"{data['wall_time_min']:.1f} min wall-clock)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_fig = result_dir / "ogden_fenics_vs_yin.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
