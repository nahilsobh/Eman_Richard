#!/usr/bin/env python3
"""Build a single 'problem setup' figure: BCs, material, simulated displacement."""
import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DX = 0.002       # m
N = 64
EXTENT = [0, N * DX * 100, 0, N * DX * 100]   # cm
SAMPLE_IDX = 48270


def main():
    f = h5py.File("data/mre_synthetic_50000.h5", "r")
    Y = f["Y"][SAMPLE_IDX]               # G [Pa], (64, 64)
    X = f["X"][SAMPLE_IDX]               # (4, 64, 64)
    u_re_60, u_im_60 = X[0], X[1]
    u_re_120         = X[2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle("Problem setup — 2D time-harmonic shear wave (Helmholtz) simulation",
                 fontsize=15, y=0.995)

    # ── (a) Domain + BCs schematic ────────────────────────
    ax = axes[0, 0]
    ax.imshow(np.log10(Y), cmap="viridis", origin="lower", extent=EXTENT)
    ax.set_title("(a) Domain & boundary conditions", fontsize=12)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    # Left edge: harmonic source u = 1+0j
    ax.annotate("", xy=(0.6, 6.4), xytext=(-1.2, 6.4),
                arrowprops=dict(arrowstyle="->", color="cyan", lw=2.5))
    ax.text(-1.6, 6.4, "u = 1 + 0j\n(harmonic\n source)",
            color="cyan", fontsize=10, ha="right", va="center", fontweight="bold")

    # Other three edges: Dirichlet 0
    for x_pos, y_pos, lbl in [
        (6.4, 13.5, "u = 0"),  # top
        (6.4,  -0.7, "u = 0"),  # bottom
        (13.5, 6.4, "u = 0"),  # right
    ]:
        ax.text(x_pos, y_pos, lbl, color="black", fontsize=11,
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    # Domain box
    ax.plot([0, EXTENT[1], EXTENT[1], 0, 0],
            [0, 0, EXTENT[3], EXTENT[3], 0],
            color="white", lw=2)
    ax.set_xlim(-3.5, 15)
    ax.set_ylim(-1.5, 14)
    ax.set_aspect("equal")

    # ── (b) Material property: shear modulus G(x,y) ──────
    ax = axes[0, 1]
    im = ax.imshow(np.log10(Y), cmap="viridis", origin="lower", extent=EXTENT)
    ax.set_title(f"(b) Material:  G(x,y)  [Pa]  (log scale)", fontsize=12)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    log_ticks = np.array([3.0, 3.3, 3.6, 4.0, 4.3])
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"{10**t:.0f}" for t in log_ticks])
    cbar.set_label("Shear modulus G [Pa]", fontsize=10)

    bg = float(np.median(Y))
    inc_max = float(Y.max())
    contrast = inc_max / bg
    ax.text(0.5, -0.15,
            f"background ≈ {bg:.0f} Pa     inclusion peak = {inc_max:.0f} Pa     "
            f"contrast = {contrast:.1f}×     ξ = 0.05  (damping)",
            transform=ax.transAxes, ha="center", fontsize=10, color="0.2")

    # ── (c) Re(u)  60 Hz  ─────────────────────────────────
    ax = axes[1, 0]
    vmax = max(abs(u_re_60.min()), abs(u_re_60.max()))
    im = ax.imshow(u_re_60, cmap="RdBu_r", origin="lower",
                   extent=EXTENT, vmin=-vmax, vmax=vmax)
    ax.set_title("(c) Simulated displacement:  Re(u)  at 60 Hz", fontsize=12)
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── (d) Im(u)  60 Hz  ─────────────────────────────────
    ax = axes[1, 1]
    vmax = max(abs(u_im_60.min()), abs(u_im_60.max()))
    im = ax.imshow(u_im_60, cmap="RdBu_r", origin="lower",
                   extent=EXTENT, vmin=-vmax, vmax=vmax)
    ax.set_title("(d) Simulated displacement:  Im(u)  at 60 Hz", fontsize=12)
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    # Caption
    fig.text(0.5, 0.005,
             "Equation:  ∇·(G* ∇u) + ρω²u = 0,    G* = G(1 + iξ),   "
             "ρ = 1000 kg/m³,  dx = 2 mm,  domain 12.8 cm × 12.8 cm.    "
             "Wave propagates left → right; scattering at the inclusion is visible "
             "in both Re(u) and Im(u).",
             ha="center", fontsize=10, color="0.3", style="italic")

    out = Path("data/problem_setup.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
