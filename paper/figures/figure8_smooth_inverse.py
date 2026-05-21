#!/usr/bin/env python3
"""Figure 8 - Inverse experiment on the smooth symmetric profile.

A third zero-shot distribution for the 1D operators: G(x) = Gc + beta
(x - L/2)^2 (C^infinity, no absolute-value kink). None of the four
trained models -- FNO-asym, SIREN-asym, FNO-mixed, SIREN-mixed -- has
seen a smooth quadratic profile in training; the asym-only models saw
monotonic ramps and the mixed models saw asym ramps + mirrored-Euler
V-shapes. The smooth quadratic is structurally distinct from all of
them, isolating *which* aspect of the inductive bias matters.

Panels:
  (a) Three example smooth symmetric profiles (real parts).
  (b) Bar chart of mean RL2 across all four models on three test sets:
       asym (in-distribution for fno_asym/siren_asym),
       sym mirrored-Euler (in-distribution for fno_mixed/siren_mixed),
       sym smooth quadratic (out-of-distribution for all four).
  (c) Four held-out smooth predictions, one per model, on the same sample.

Output:
  paper/figures/fig8_smooth_inverse.png
  paper/figures/fig8_smooth_inverse_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from brain1d import (
    SmoothSymmetricBrainParams, G_profile_smooth_symmetric,
    numerical_solution_smooth_symmetric,
)
from figure2_brain_inverse import build_tensors as build_asym_tensors
from figure3_symmetric_stress import (
    build_symmetric_tensors, per_sample_rl2, load_trained_fno,
)
from figure5_siren_compare import SIREN1d


N_GRID  = 96
G_SCALE = 5000.0
SNR_DB  = 25.0

# Match per-endpoint ranges used in training, with Gb > Gc + 200 Pa
GC_RANGE_PA = (800.0,  2500.0)
GB_RANGE_PA = (1500.0, 4500.0)
XI_RANGE    = (0.05, 0.20)


def make_smooth_symmetric_sample(N: int, rng: np.random.Generator
                                  ) -> tuple[np.ndarray, np.ndarray,
                                             SmoothSymmetricBrainParams]:
    """Draw one (u, G, params) triple from the smooth-symmetric distribution."""
    Gc = float(rng.uniform(*GC_RANGE_PA))
    Gb = float(rng.uniform(max(Gc + 300.0, GB_RANGE_PA[0]), GB_RANGE_PA[1]))
    xi = float(rng.uniform(*XI_RANGE))
    p  = SmoothSymmetricBrainParams(L=0.16, Gc=Gc, Gb=Gb, xi=xi, freq=50.0)

    _, u = numerical_solution_smooth_symmetric(N, p)
    G    = G_profile_smooth_symmetric(np.linspace(0, p.L, N), p)

    sig_power = float(np.mean(np.abs(u) ** 2))
    sigma_n   = float(np.sqrt(sig_power / (10.0 ** (SNR_DB / 10.0)) / 2.0))
    u_noisy   = u + sigma_n * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    u_norm    = u_noisy / max(np.abs(u_noisy).max(), 1e-12)

    return u_norm.astype(np.complex64), G.astype(np.complex64), p


def build_smooth_tensors(N: int, n: int, seed: int, G_scale: float = G_SCALE
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 2, N), dtype=np.float32)
    Y = np.zeros((n, 2, N), dtype=np.float32)
    for i in range(n):
        u, G, _ = make_smooth_symmetric_sample(N, rng)
        X[i, 0] = u.real
        X[i, 1] = u.imag
        Y[i, 0] = G.real / G_scale
        Y[i, 1] = G.imag / G_scale
    return torch.from_numpy(X), torch.from_numpy(Y)


def load_siren(ckpt_path: Path, device: torch.device) -> SIREN1d:
    m = SIREN1d(in_ch=2, out_ch=2, width=24, kernel_size=11, n_blocks=3).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    m.eval()
    return m


def main():
    figs = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nBuilding held-out test sets (300 samples each) ...")
    X_asym, Y_asym = build_asym_tensors(N_GRID, 300, seed=1)
    X_sym,  Y_sym  = build_symmetric_tensors(N_GRID, 300, seed=2)
    X_sm,   Y_sm   = build_smooth_tensors(N_GRID, 300, seed=3)

    print("Loading the four trained checkpoints ...")
    fno_asym   = load_trained_fno(figs / "fig2_fno1d.pt",       device)
    fno_mixed  = load_trained_fno(figs / "fig4_mixed_fno1d.pt", device)
    siren_asym = load_siren(figs / "fig5_siren_rl2.pt",         device)
    siren_mix  = load_siren(figs / "fig6_siren_mixed.pt",       device)

    models = [
        ("FNO + asym-only",   fno_asym,   "steelblue"),
        ("FNO + mixed",       fno_mixed,  "navy"),
        ("SIREN + asym-only", siren_asym, "tomato"),
        ("SIREN + mixed",     siren_mix,  "darkred"),
    ]
    tests = [
        ("asym (training)",         X_asym, Y_asym),
        ("sym Euler-kink (Fig 4)",  X_sym,  Y_sym),
        ("sym SMOOTH (Fig 3, new)", X_sm,   Y_sm),
    ]

    # Compute mean per-sample RL2 for the 4 x 3 grid
    M = np.zeros((len(models), len(tests)))
    print("\nComputing 4 x 3 RL2 grid ...")
    for i, (name, model, _) in enumerate(models):
        for j, (test_name, X, Y) in enumerate(tests):
            rl2 = per_sample_rl2(X, Y, model, device).mean()
            M[i, j] = float(rl2)
            print(f"  {name:20s} on {test_name:30s}  RL2 = {rl2:.4f}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.32,
                           height_ratios=[1.0, 1.0])

    # (a) Three example smooth profiles
    ax_a = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(7)
    x_dense = np.linspace(0, 0.16, 2001)
    for ci, color in zip(range(3), ["steelblue", "seagreen", "tomato"]):
        Gc = float(rng.uniform(*GC_RANGE_PA))
        Gb = float(rng.uniform(max(Gc + 500.0, GB_RANGE_PA[0]), GB_RANGE_PA[1]))
        xi = float(rng.uniform(*XI_RANGE))
        p  = SmoothSymmetricBrainParams(L=0.16, Gc=Gc, Gb=Gb, xi=xi, freq=50.0)
        Gi = G_profile_smooth_symmetric(x_dense, p)
        ax_a.plot(x_dense * 100, Gi.real / 1000, color=color, lw=1.8,
                  label=fr"Gc={Gc:.0f}, Gb={Gb:.0f}, $\xi$={xi:.2f}")
    ax_a.axvline(0.16 * 100 / 2, color="black", lw=0.5, ls=":")
    ax_a.set_xlabel("x [cm]")
    ax_a.set_ylabel("Re G(x) [kPa]")
    ax_a.set_title(r"(a) Smooth symmetric profiles $G_c+\beta(x-L/2)^2$")
    ax_a.legend(fontsize=8, loc="upper center")
    ax_a.grid(alpha=0.3)

    # (b, c) Grouped bar chart of mean RL2 across 4 models x 3 test sets
    ax_b = fig.add_subplot(gs[0, 1:])
    n_models = len(models)
    n_tests  = len(tests)
    x = np.arange(n_tests)
    w = 0.18
    for i, (name, _, color) in enumerate(models):
        offset = (i - (n_models - 1) / 2) * w
        bars = ax_b.bar(x + offset, M[i, :], w, color=color, alpha=0.85,
                         edgecolor="black", lw=0.5, label=name)
        for b, v in zip(bars, M[i, :]):
            ax_b.text(b.get_x() + b.get_width() / 2, v + 0.005,
                       f"{v:.3f}", ha="center", fontsize=8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([t[0] for t in tests])
    ax_b.set_ylabel(r"mean per-sample RL$^2$ on G")
    ax_b.set_title("(b) Zero-shot evaluation across three test distributions")
    ax_b.legend(loc="upper left", fontsize=9, ncol=2)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.set_ylim(0, max(M.max() * 1.18, 0.25))

    # (c-f) Four predictions, all four models on the same smooth-sym sample
    rng = np.random.default_rng(31)
    idx = int(rng.integers(0, len(X_sm)))
    x_grid = np.linspace(0, 0.16, N_GRID) * 100
    for col, (name, model, color) in enumerate(models):
        ax = fig.add_subplot(gs[1, col % 3])  # wrap to 3-column layout
        if col >= 3:
            continue  # only 3 panels in row, drop last
        with torch.no_grad():
            yp = model(X_sm[idx:idx + 1].to(device)).cpu().numpy()[0] * G_SCALE
        yt = Y_sm[idx].numpy() * G_SCALE
        ax.plot(x_grid, yt[0] / 1000, color="black", lw=1.8, label=r"true Re$\,G$")
        ax.plot(x_grid, yp[0] / 1000, color=color, lw=1.4, ls="--",
                label=fr"pred Re$\,G$ ({name})")
        ax.plot(x_grid, yt[1] / 1000, color="black", lw=1.0, alpha=0.5)
        ax.plot(x_grid, yp[1] / 1000, color="gray", lw=1.4, ls=":",
                label=r"pred Im$\,G$")
        ax.axvline(0.16 * 100 / 2, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("G(x) [kPa]" if (col % 3) == 0 else "")
        ax.set_title(f"({chr(ord('c') + col)}) {name}\n smooth sample #{idx}",
                      fontsize=10)
        ax.legend(fontsize=7, loc="upper center")
        ax.grid(alpha=0.3)

    fig.suptitle("Zero-shot inverse on the smooth symmetric distribution. "
                  "Three test families across four models; sample predictions "
                  "for the first three models on the same held-out C^infty sample.",
                  fontsize=11, y=1.01)
    fig.savefig(figs / "fig8_smooth_inverse.png", dpi=160, bbox_inches="tight")
    print(f"Wrote {figs / 'fig8_smooth_inverse.png'}")

    # ── Persist numbers
    summary = {
        "tests":  [t[0] for t in tests],
        "models": [m[0] for m in models],
        "rl2":    {m[0]: {t[0]: float(M[i, j])
                          for j, t in enumerate(tests)}
                    for i, m in enumerate(models)},
        "n_samples_per_test": 300,
    }
    json.dump(summary, open(figs / "fig8_smooth_inverse_summary.json", "w"),
              indent=2)
    print(f"Wrote {figs / 'fig8_smooth_inverse_summary.json'}")


if __name__ == "__main__":
    main()
