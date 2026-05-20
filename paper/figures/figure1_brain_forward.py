#!/usr/bin/env python3
"""Figure 1 — 1D brain-scale forward problem: analytical vs numerical.

Four panels:
  (a) Continuous complex shear modulus G(x) = G0(1+alpha x)^2 (1+i xi)
      with values matched to published in vivo brain MRE.
  (b) Re(u(x)) and Im(u(x)) from the closed-form solution.
  (c) Re(u(x)) from the FD numerical solver overlaid on the analytical.
  (d) Pointwise relative error |u_num - u_an| / max|u_an| in log scale.

Output: paper/figures/fig1_brain_forward.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brain1d import BrainParams, G_profile, analytical_solution, numerical_solution


def main():
    out = Path(__file__).parent / "fig1_brain_forward.png"

    p = BrainParams()  # defaults match human head
    print(f"Brain 1D problem:")
    print(f"  L      = {p.L*100:.1f} cm  (~human head)")
    print(f"  G(0)   = {p.G0:.0f} Pa,  G(L) = {p.Gend:.0f} Pa")
    print(f"  alpha  = {p.alpha:.3f} /m")
    print(f"  xi     = {p.xi}")
    print(f"  f      = {p.freq:.0f} Hz")
    print(f"  rho    = {p.rho:.0f} kg/m^3")
    print(f"  omega  = {p.omega:.3f} rad/s")

    # Analytical on a dense grid
    x_dense = np.linspace(0.0, p.L, 2001)
    u_an_dense = analytical_solution(x_dense, p)

    # Numerical at moderate resolution
    Ns = [64, 128, 256, 512, 1024]
    print("\nConvergence:")
    errs = []
    for N in Ns:
        x_n, u_n = numerical_solution(N, p)
        u_an_at_n = analytical_solution(x_n, p)
        err = np.linalg.norm(u_n - u_an_at_n) / np.linalg.norm(u_an_at_n)
        errs.append(err)
        print(f"  N={N:5d}  rel err = {err:.3e}")

    # The displayed numerical solution uses N=512 (a sensible default)
    N_show = 512
    x_show, u_show = numerical_solution(N_show, p)
    u_an_show = analytical_solution(x_show, p)
    pointwise = np.abs(u_show - u_an_show) / np.abs(u_an_show).max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.0))

    # (a) G(x): real and imaginary parts
    G_dense = G_profile(x_dense, p)
    ax = axes[0]
    ax.plot(x_dense * 100, G_dense.real / 1000, color="steelblue", lw=2.0,
            label=r"Re$\,G$")
    ax.plot(x_dense * 100, G_dense.imag / 1000, color="tomato", lw=2.0,
            label=r"Im$\,G$ ($\xi=0.10$)")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("shear modulus [kPa]")
    ax.set_title(r"(a) Continuous $G(x)=G_0(1+\alpha x)^2(1+i\xi)$")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Analytical u(x): real and imaginary
    ax = axes[1]
    ax.plot(x_dense * 100, u_an_dense.real, color="steelblue", lw=1.6,
            label=r"Re$\,u_{\rm an}$")
    ax.plot(x_dense * 100, u_an_dense.imag, color="tomato", lw=1.6,
            label=r"Im$\,u_{\rm an}$")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel(r"displacement $u(x)$")
    ax.set_title(r"(b) Closed-form solution $u(s)=A s^{p_1}+B s^{p_2}$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (c) Numerical vs analytical (real parts overlaid)
    ax = axes[2]
    ax.plot(x_dense * 100, u_an_dense.real, color="black", lw=1.4,
            label="analytical")
    ax.plot(x_show * 100, u_show.real, "o", color="seagreen", ms=3,
            markevery=20, label=f"FD (N={N_show})", alpha=0.85)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel(r"Re $u(x)$")
    ax.set_title(f"(c) FD solver overlaid on analytical")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (d) Convergence + pointwise error
    ax = axes[3]
    # Main: convergence rate
    ax.loglog(Ns, errs, "o-", color="seagreen", lw=1.8, ms=8,
              label=r"$\|u_{\rm FD} - u_{\rm an}\| / \|u_{\rm an}\|$")
    # Reference 2nd-order line
    ref = errs[0] * (Ns[0] / np.array(Ns)) ** 2
    ax.loglog(Ns, ref, "k--", lw=1.0,
              label=r"$\mathcal{O}(N^{-2})$ reference")
    ax.set_xlabel("FD grid resolution N")
    ax.set_ylabel("relative L2 error")
    ax.set_title("(d) Spatial convergence  ($f=50$ Hz)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nWrote {out}")
    return errs[-1]


if __name__ == "__main__":
    main()
