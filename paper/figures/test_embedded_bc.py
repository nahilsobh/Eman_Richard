#!/usr/bin/env python3
"""Visual smoke-test for the embedded-BC synthetic generator.

Generates one sample per (family, BC-pair) combination and plots G(x)
alongside Re u(x).  The point is to confirm:
  - Wave fields look continuous through the ROI window (no obvious edge
    artefacts from constant extension).
  - Different BC pairs produce visibly different wave fields for the same
    underlying G.

Output: paper/figures/test_embedded_bc.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brain1d import (
    BC_KINDS, PROFILE_FAMILIES, BCSpec,
    _extended_grid, _G_extended_constant, _sample_profile,
    solve_helmholtz_1d_with_bcs, make_embedded_sample,
    BrainParams, SymmetricBrainParams, SmoothSymmetricBrainParams,
    G_profile, G_profile_symmetric, G_profile_smooth_symmetric,
    L_DEFAULT, FREQ_DEFAULT, RHO_DEFAULT,
)


def show_one(ax_G, ax_u, family: str, bc_l_kind: str, bc_r_kind: str,
             N_roi: int = 96, ext_factor: float = 1.0,
             rng=None):
    if rng is None:
        rng = np.random.default_rng(7)
    # Fix the profile so the only thing changing between panels is BC
    Gc, Gb, xi = 1500.0, 3000.0, 0.10
    if family == "asym":
        p = BrainParams(L=L_DEFAULT, G0=Gc, Gend=Gb, xi=xi)
        pf = G_profile
    elif family == "sym_euler":
        p = SymmetricBrainParams(L=L_DEFAULT, Gc=Gc, Gb=Gb, xi=xi)
        pf = G_profile_symmetric
    else:
        p = SmoothSymmetricBrainParams(L=L_DEFAULT, Gc=Gc, Gb=Gb, xi=xi)
        pf = G_profile_smooth_symmetric

    x_ext, roi_idx, x_roi = _extended_grid(N_roi, L_DEFAULT, ext_factor)
    G_ext = _G_extended_constant(x_ext, pf, L_DEFAULT, p)

    bcs = {
        "dirichlet": BCSpec("dirichlet", 1.0),
        "neumann":   BCSpec("neumann",   1.0),  # nonzero so the system isn't degenerate
        "absorbing": BCSpec("absorbing"),
    }
    u_ext = solve_helmholtz_1d_with_bcs(x_ext, G_ext, RHO_DEFAULT, FREQ_DEFAULT,
                                          bcs[bc_l_kind], bcs[bc_r_kind])
    u_roi = u_ext[roi_idx]
    G_roi = G_ext[roi_idx]
    u_roi = u_roi / np.abs(u_roi).max()

    # Plot G on ROI alone; plot u on extended domain with ROI highlighted
    ax_G.plot(x_roi * 100, np.abs(G_roi) / 1000, color="black", lw=1.5)
    ax_G.set_ylim(0, 4.5)
    ax_G.set_title(f"{family}\nBC=({bc_l_kind[:3]},{bc_r_kind[:3]})", fontsize=8)
    ax_G.set_xticks([0, 8, 16]); ax_G.tick_params(labelsize=7)
    ax_G.grid(alpha=0.3)

    u_norm_ext = u_ext / np.abs(u_ext).max()
    ax_u.plot(x_ext * 100, u_norm_ext.real, color="steelblue", lw=1.0)
    ax_u.axvspan(0, L_DEFAULT * 100, color="seagreen", alpha=0.10,
                  label="ROI")
    ax_u.axhline(0, color="black", lw=0.4)
    ax_u.set_ylim(-1.1, 1.1)
    ax_u.set_xticks([-16, 0, 16, 32]); ax_u.tick_params(labelsize=7)
    ax_u.grid(alpha=0.3)


def main():
    out = Path(__file__).parent / "test_embedded_bc.png"

    families = ["asym", "sym_euler", "sym_smooth"]
    bc_combos = [("dirichlet", "dirichlet"),
                  ("dirichlet", "absorbing"),
                  ("absorbing", "absorbing"),
                  ("neumann",   "absorbing")]

    fig, axes = plt.subplots(len(families) * 2, len(bc_combos),
                              figsize=(11, 9), sharex=False)

    for fi, family in enumerate(families):
        for bi, (bl, br) in enumerate(bc_combos):
            show_one(axes[2 * fi,     bi], axes[2 * fi + 1, bi],
                     family, bl, br)
            if bi == 0:
                axes[2 * fi,     bi].set_ylabel("|G| [kPa]", fontsize=8)
                axes[2 * fi + 1, bi].set_ylabel(r"Re $u$ (norm)", fontsize=8)
            if fi == len(families) - 1:
                axes[2 * fi + 1, bi].set_xlabel("x [cm]", fontsize=8)
                axes[2 * fi,     bi].set_xlabel("x [cm]", fontsize=8)

    fig.suptitle("Embedded-BC generator: 3 families × 4 BC pairs.\n"
                  "Top of each cell: |G(x)| on ROI [0, 16] cm.\n"
                  "Bottom: Re u(x) on extended domain [-16, 32] cm; "
                  "shaded band = ROI window seen by the network.",
                  fontsize=10, y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
