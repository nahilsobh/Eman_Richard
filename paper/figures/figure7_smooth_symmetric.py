#!/usr/bin/env python3
"""Figure 7 - Smooth symmetric profile: no absolute-value kink.

A third 1D analytical test case, distinct from Fig 3's mirrored Euler
profile. Here

    G(x) = (Gc + beta (x - L/2)^2)(1 + i xi)

is C^infty in x: G'(x) = 2 beta (x - L/2) is continuous everywhere
(zero at the centre), and G''(x) = 2 beta is constant. No |.|.

Closed-form analytical solution exists as a convergent power series
about the symmetry point zeta = x - L/2 = 0. The recurrence is

    c_{k+2} = -( b k(k+1) + rho omega^2 ) / ( a (k+2)(k+1) ) * c_k

with a = Gc(1+i xi), b = beta(1+i xi); the convergence radius is
sqrt(|a/b|) which must exceed L/2 for the series to cover the domain.
At brain-scale parameters (L=16 cm, Gc=2 kPa, Gb=2.4 kPa) the radius
is 17.9 cm > 8 cm.  We use 200 terms for safety.

Output:
  paper/figures/fig7_smooth_symmetric.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brain1d import (
    SmoothSymmetricBrainParams, G_profile_smooth_symmetric,
    analytical_solution_smooth_symmetric, numerical_solution_smooth_symmetric,
)


def main():
    out = Path(__file__).parent / "fig7_smooth_symmetric.png"

    p = SmoothSymmetricBrainParams(L=0.16, Gc=2000.0, Gb=2400.0,
                                    xi=0.10, freq=50.0)
    print(f"Smooth symmetric profile:")
    print(f"  L      = {p.L*100:.1f} cm")
    print(f"  Gc     = {p.Gc:.0f} Pa  (centre, min)")
    print(f"  Gb     = {p.Gb:.0f} Pa  (boundaries, max)")
    print(f"  beta   = {p.beta:.1f} Pa/m^2")
    print(f"  xi     = {p.xi}")
    print(f"  f      = {p.freq:.0f} Hz")
    a_over_b = abs(p.Gc * (1 + 1j * p.xi)) / abs(p.beta * (1 + 1j * p.xi))
    print(f"  power-series convergence radius = {np.sqrt(a_over_b)*100:.2f} cm "
          f"(need > {p.L/2*100:.2f} cm)")

    # ── Forward convergence
    print("\nForward convergence (analytical = 200-term power series):")
    Ns = [64, 128, 256, 512, 1024, 2048]
    errs = []
    for N in Ns:
        x_n, u_n = numerical_solution_smooth_symmetric(N, p)
        u_an = analytical_solution_smooth_symmetric(x_n, p, n_terms=200)
        err = np.linalg.norm(u_n - u_an) / np.linalg.norm(u_an)
        errs.append(err)
        print(f"  N={N:5d}  rel L2 err = {err:.3e}")

    # Dense fields for plotting
    x_dense   = np.linspace(0, p.L, 2001)
    G_dense   = G_profile_smooth_symmetric(x_dense, p)
    u_an_dense = analytical_solution_smooth_symmetric(x_dense, p, n_terms=200)
    N_show = 512
    x_show, u_show = numerical_solution_smooth_symmetric(N_show, p)

    # ── Derivatives of G(x) for the smoothness panel
    G_re = G_dense.real
    dG  = np.gradient(G_re, x_dense)
    ddG = np.gradient(dG, x_dense)

    fig = plt.figure(figsize=(15, 8.5))
    gs  = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30)

    # (a) G(x): real and imaginary
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x_dense * 100, G_dense.real / 1000, color="steelblue", lw=2.0,
            label=r"Re$\,G$")
    ax.plot(x_dense * 100, G_dense.imag / 1000, color="tomato", lw=2.0,
            label=r"Im$\,G$  ($\xi=0.10$)")
    ax.axvline(p.L * 100 / 2, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("shear modulus [kPa]")
    ax.set_title(r"(a) Smooth $G(x) = G_c + \beta(x - L/2)^2$")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Derivatives - show G' and G'' both continuous (zero at center for G')
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x_dense * 100, dG, color="seagreen", lw=1.8,
            label=r"$\mathrm{d}G/\mathrm{d}x$  [Pa/m]")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel(r"$\mathrm{d}G/\mathrm{d}x$  [Pa/m]")
    ax.set_title(r"(b) First derivative $\mathrm{d}G/\mathrm{d}x$ is continuous, "
                  r"zero at centre")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(p.L * 100 / 2, color="gray", lw=0.6, ls=":")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # (c) Re u(x): analytical and FD
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(x_dense * 100, u_an_dense.real, color="black",   lw=1.5,
            label="analytical (power series)")
    ax.plot(x_show * 100,  u_show.real,     "o", color="seagreen", ms=3,
            markevery=20, alpha=0.85, label=f"FD ($N=512$)")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel(r"Re $u(x)$")
    ax.set_title("(c) Forward solution at $f=50$ Hz, real part")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # (d) Convergence
    ax = fig.add_subplot(gs[1, 0])
    ax.loglog(Ns, errs, "o-", color="seagreen", lw=1.8, ms=8,
              label=r"$\|u_{\rm FD} - u_{\rm AN}\| / \|u_{\rm AN}\|$")
    ref = errs[0] * (Ns[0] / np.array(Ns)) ** 2
    ax.loglog(Ns, ref, "k--", lw=1.0, label=r"$\mathcal{O}(N^{-2})$")
    ax.set_xlabel("FD grid resolution N")
    ax.set_ylabel(r"relative L$^2$ error")
    ax.set_title("(d) Second-order convergence")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # (e) Im u(x): analytical and FD
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x_dense * 100, u_an_dense.imag, color="black", lw=1.5,
            label="analytical")
    ax.plot(x_show * 100,  u_show.imag, "o", color="tomato", ms=3,
            markevery=20, alpha=0.85, label=f"FD ($N=512$)")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel(r"Im $u(x)$")
    ax.set_title("(e) Imaginary part (damping signature)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # (f) Pointwise residual at N=512
    ax = fig.add_subplot(gs[1, 2])
    u_an_at_show = analytical_solution_smooth_symmetric(x_show, p, n_terms=200)
    pointwise = np.abs(u_show - u_an_at_show)
    norm = np.abs(u_an_dense).max()
    ax.semilogy(x_show * 100, pointwise / max(norm, 1e-12),
                 color="gray", lw=1.0)
    ax.axvline(p.L * 100 / 2, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel(r"$|u_{\rm FD} - u_{\rm AN}| / \max|u_{\rm AN}|$")
    ax.set_title(f"(f) Pointwise residual at $N=512$  (max {pointwise.max()/norm:.1e})")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(r"Smooth symmetric brain-scale profile $G(x)=G_c+\beta(x-L/2)^2$: "
                  "C$^\infty$, no absolute-value kink",
                  fontsize=11, y=1.00)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
