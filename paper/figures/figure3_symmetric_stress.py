#!/usr/bin/env python3
"""Figure 3 - Symmetric (mirrored-Euler) stress test.

The 1D FNO was trained on the asymmetric power-law family
    G_asym(x) = G0 (1 + alpha x)^2 (1 + i xi),
with G strictly increasing across the domain. We now evaluate it
zero-shot on the *symmetric* mirrored-Euler family
    G_sym(x)  = Gc (1 + alpha |x - L/2|)^2 (1 + i xi),
which has G minimum at the centre and equal maxima at both boundaries.
The only structural change is the shape of G(x); BCs, normalisation,
SNR, and grid size are kept identical to training.

Panels:
  (a) Two representative symmetric G(x) profiles, real and imaginary parts.
  (b) Analytical u(x) and FD overlay on one of those samples; convergence
      table inset.
  (c) FNO zero-shot prediction overlay on four held-out symmetric samples.
  (d) Per-sample histogram of validation RL2 on the symmetric distribution
      versus the asymmetric training distribution -- shows the
      distribution shift impact.

Output: paper/figures/fig3_symmetric.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from brain1d import (
    SymmetricBrainParams, G_profile_symmetric,
    analytical_solution_symmetric, numerical_solution_symmetric,
)
from figure2_brain_inverse import FNO1d, make_sample, build_tensors


# Match the per-endpoint prior used in training (asymmetric case)
GC_RANGE_PA = (800.0, 2500.0)      # centre value -- maps to G0 in training
GB_RANGE_PA = (1500.0, 4500.0)     # boundary value -- maps to G(L) in training
XI_RANGE    = (0.05, 0.20)
SNR_DB      = 25.0


def make_symmetric_sample(N: int, rng: np.random.Generator
                           ) -> tuple[np.ndarray, np.ndarray, SymmetricBrainParams]:
    Gc = float(rng.uniform(*GC_RANGE_PA))
    Gb = float(rng.uniform(*GB_RANGE_PA))
    if Gb <= Gc:
        Gb = Gc + rng.uniform(300.0, 2500.0)
    xi = float(rng.uniform(*XI_RANGE))
    p  = SymmetricBrainParams(L=0.16, Gc=Gc, Gb=Gb, xi=xi, freq=50.0)

    x, u = numerical_solution_symmetric(N, p)
    G    = G_profile_symmetric(x, p)

    sig_power = float(np.mean(np.abs(u) ** 2))
    sigma_n   = float(np.sqrt(sig_power / (10.0 ** (SNR_DB / 10.0)) / 2.0))
    u_noisy   = u + sigma_n * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    u_norm    = u_noisy / max(np.abs(u_noisy).max(), 1e-12)

    return u_norm.astype(np.complex64), G.astype(np.complex64), p


def build_symmetric_tensors(N: int, n: int, seed: int, G_scale: float = 5000.0
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 2, N), dtype=np.float32)
    Y = np.zeros((n, 2, N), dtype=np.float32)
    for i in range(n):
        u, G, _ = make_symmetric_sample(N, rng)
        X[i, 0] = u.real
        X[i, 1] = u.imag
        Y[i, 0] = G.real / G_scale
        Y[i, 1] = G.imag / G_scale
    return torch.from_numpy(X), torch.from_numpy(Y)


def load_trained_fno(checkpoint: Path, device: torch.device,
                     width: int = 24, modes: int = 12, blocks: int = 3) -> FNO1d:
    """Load the 1D FNO trained in figure2_brain_inverse.py."""
    model = FNO1d(in_ch=2, out_ch=2, width=width, modes=modes, n_blocks=blocks)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


def per_sample_rl2(X: torch.Tensor, Y: torch.Tensor, model: FNO1d,
                   device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 64):
            xb = X[k:k + 64].to(device)
            yb = Y[k:k + 64].to(device)
            yp = model(xb)
            num = torch.linalg.vector_norm(yp - yb, dim=(-2, -1))
            den = torch.linalg.vector_norm(yb,      dim=(-2, -1)).clamp_min(1e-6)
            out.append((num / den).cpu().numpy())
    return np.concatenate(out)


def main():
    out_dir = Path(__file__).parent
    fig_path = out_dir / "fig3_symmetric.png"
    N = 96   # matches the model trained in figure2_brain_inverse.py
    G_scale = 5000.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # ── Convergence on a representative symmetric sample ────────────────────
    p_demo = SymmetricBrainParams(L=0.16, Gc=1500.0, Gb=3500.0, xi=0.10, freq=50.0)
    print(f"Demo sample: Gc={p_demo.Gc:.0f}, Gb={p_demo.Gb:.0f}, alpha={p_demo.alpha:.3f} /m")
    print("Symmetric forward convergence:")
    errs, Ns = [], [64, 128, 256, 512, 1024]
    for N_test in Ns:
        x_n, u_n = numerical_solution_symmetric(N_test, p_demo)
        u_an = analytical_solution_symmetric(x_n, p_demo)
        err = np.linalg.norm(u_n - u_an) / np.linalg.norm(u_an)
        errs.append(err)
        print(f"  N={N_test:5d}  rel L2 err = {err:.3e}")

    x_dense  = np.linspace(0, p_demo.L, 2001)
    G_dense  = G_profile_symmetric(x_dense, p_demo)
    u_an_dense = analytical_solution_symmetric(x_dense, p_demo)
    x_fd, u_fd = numerical_solution_symmetric(512, p_demo)

    # ── FNO zero-shot evaluation on the symmetric distribution ──────────────
    print("\nLoading trained FNO checkpoint ...")
    model = load_trained_fno(out_dir / "fig2_fno1d.pt", device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FNO loaded: {n_params:,} params")

    print("Building symmetric validation set (300 samples) ...")
    X_sym, Y_sym = build_symmetric_tensors(N, 300, seed=2026, G_scale=G_scale)
    print("Building asymmetric reference set (300 samples) ...")
    X_asym, Y_asym = build_tensors(N, 300, seed=2027, snr_db=SNR_DB,
                                    G_scale=G_scale)

    rl2_sym  = per_sample_rl2(X_sym,  Y_sym,  model, device)
    rl2_asym = per_sample_rl2(X_asym, Y_asym, model, device)
    print(f"Mean RL2 on asymmetric (in-distribution) : {rl2_asym.mean():.4f}")
    print(f"Mean RL2 on symmetric  (out-of-dist)     : {rl2_sym.mean():.4f}")

    # Four sample predictions on symmetric
    rng = np.random.default_rng(11)
    idx = rng.choice(len(X_sym), 4, replace=False)
    with torch.no_grad():
        Yp = model(X_sym[idx].to(device)).cpu().numpy() * G_scale
    Yt = Y_sym[idx].numpy() * G_scale

    # ── Plot ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 9.0))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.32)

    # (a) Two illustrative G(x) profiles
    ax_a = fig.add_subplot(gs[0, 0])
    rng_ax = np.random.default_rng(101)
    for ci, color in zip(range(2), ["steelblue", "tomato"]):
        pi = SymmetricBrainParams(
            L=0.16,
            Gc=float(rng_ax.uniform(*GC_RANGE_PA)),
            Gb=float(rng_ax.uniform(*GB_RANGE_PA)),
            xi=float(rng_ax.uniform(*XI_RANGE)),
            freq=50.0,
        )
        if pi.Gb < pi.Gc:
            pi.Gb = pi.Gc + 500
        Gi = G_profile_symmetric(x_dense, pi)
        ax_a.plot(x_dense * 100, Gi.real / 1000, color=color, lw=1.8,
                  label=fr"Re G: Gc={pi.Gc:.0f}, Gb={pi.Gb:.0f} Pa")
    ax_a.axvline(p_demo.L * 100 / 2, color="black", lw=0.6, ls=":")
    ax_a.set_xlabel("x [cm]")
    ax_a.set_ylabel("Re G(x) [kPa]")
    ax_a.set_title(r"(a) Symmetric profiles $G_c(1+\alpha|x-L/2|)^2$")
    ax_a.legend(fontsize=8, loc="upper center")
    ax_a.grid(alpha=0.3)

    # (b) Analytical vs FD on the demo sample
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(x_dense * 100, u_an_dense.real, color="black", lw=1.4,
              label="analytical")
    ax_b.plot(x_fd * 100, u_fd.real, "o", color="seagreen", ms=3,
              markevery=20, alpha=0.85, label="FD (N=512)")
    ax_b.set_xlabel("x [cm]")
    ax_b.set_ylabel(r"Re $u(x)$")
    ax_b.set_title("(b) Forward: analytical and FD agree")
    ax_b.legend(fontsize=9, loc="lower right")
    ax_b.grid(alpha=0.3)

    # (c) Convergence
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.loglog(Ns, errs, "o-", color="seagreen", lw=1.8, ms=8,
                label="symmetric FD vs analytical")
    ref = errs[0] * (Ns[0] / np.array(Ns)) ** 2
    ax_c.loglog(Ns, ref, "k--", lw=1.0, label=r"$\mathcal{O}(N^{-2})$")
    ax_c.set_xlabel("FD grid resolution N")
    ax_c.set_ylabel(r"$\| u_{\rm FD} - u_{\rm an}\| / \| u_{\rm an}\|$")
    ax_c.set_title("(c) Second-order convergence")
    ax_c.legend(fontsize=9, loc="upper right")
    ax_c.grid(alpha=0.3, which="both")

    # (d) RL2 distribution: in-distribution vs OOD
    ax_d = fig.add_subplot(gs[0, 3])
    bins = np.linspace(0, max(rl2_sym.max(), rl2_asym.max()) * 1.05, 36)
    ax_d.hist(rl2_asym, bins=bins, alpha=0.55, color="steelblue",
              label=fr"asymmetric (train dist)  mean {rl2_asym.mean():.3f}",
              edgecolor="black", lw=0.4)
    ax_d.hist(rl2_sym,  bins=bins, alpha=0.55, color="tomato",
              label=fr"symmetric (zero-shot)     mean {rl2_sym.mean():.3f}",
              edgecolor="black", lw=0.4)
    ax_d.set_xlabel(r"per-sample RL$^2$ on G")
    ax_d.set_ylabel("count (out of 300)")
    ax_d.set_title("(d) FNO zero-shot vs in-distribution")
    ax_d.legend(fontsize=9)
    ax_d.grid(axis="y", alpha=0.3)

    # (e-h) FNO predictions on 4 symmetric samples (bottom row: G real)
    x_grid = np.linspace(0, p_demo.L, N) * 100
    for k in range(4):
        ax = fig.add_subplot(gs[1, k])
        # Plot G_true real
        ax.plot(x_grid, Yt[k, 0] / 1000, color="black", lw=1.8,
                label=r"true Re$\,G$")
        ax.plot(x_grid, Yp[k, 0] / 1000, color="steelblue", lw=1.4, ls="--",
                label=r"FNO Re$\,G$")
        # Plot G_true imag
        ax.plot(x_grid, Yt[k, 1] / 1000, color="black", lw=1.4, alpha=0.5)
        ax.plot(x_grid, Yp[k, 1] / 1000, color="tomato", lw=1.4, ls="--",
                label=r"FNO Im$\,G$")
        ax.axvline(p_demo.L * 100 / 2, color="gray", lw=0.5, ls=":")
        # per-sample RL2 score
        rl2_i = rl2_sym[idx[k]]
        ax.set_title(f"sample #{idx[k]}  RL$^2$={rl2_i:.3f}")
        ax.set_xlabel("x [cm]")
        if k == 0:
            ax.set_ylabel("G(x) [kPa]")
            ax.legend(fontsize=8, loc="upper center")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "FNO trained on asymmetric ramps (Fig. 2) evaluated zero-shot on "
        "symmetric V-shaped G(x). The forward problem (a-c) is analytically "
        "exact; the inverse (d, e-h) tests the trained operator's ability "
        "to handle structurally different G under unchanged BCs and SNR.",
        fontsize=10, y=1.02,
    )
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"\nWrote {fig_path}")


if __name__ == "__main__":
    main()
