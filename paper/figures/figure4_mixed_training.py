#!/usr/bin/env python3
"""Figure 4 - Mixed-distribution training closes the symmetric gap.

We retrain the same 1D FNO architecture (23k params) on a mixed
distribution where each sample is drawn with probability 1/2 from the
asymmetric power-law family and 1/2 from the symmetric mirrored-Euler
family. All other settings (grid size, SNR, optimiser, epochs) are
identical to the asymmetric-only run from figure2_brain_inverse.py.

The hypothesis: the architecture is expressive enough to fit both
families if shown both during training, so the symmetric-distribution
RL2 should drop from 0.21 (zero-shot) to a value comparable to the
asymmetric-distribution baseline (~0.046).

Panels:
  (a) Training/validation loss for the mixed model (separately tracking
      asymmetric-val and symmetric-val).
  (b) 2x2 grouped bar chart of mean RL2:
        asym-only model vs mixed model, on asym test set vs sym test set.
  (c) Four sample predictions from the mixed model on symmetric samples,
      to be compared visually with Fig 3(e-h).

Output:
  paper/figures/fig4a_mixed_loss.png
  paper/figures/fig4b_mixed_bars.png
  paper/figures/fig4c_mixed_predict.png
  paper/figures/fig4_mixed_fno1d.pt
  paper/figures/fig4_mixed_history.json
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from figure2_brain_inverse import (
    FNO1d, make_sample as make_asym_sample, build_tensors as build_asym_tensors,
)
from figure3_symmetric_stress import (
    make_symmetric_sample, build_symmetric_tensors, per_sample_rl2,
    load_trained_fno,
)


def build_mixed_tensors(N: int, n: int, seed: int, snr_db: float = 25.0,
                        G_scale: float = 5000.0, p_sym: float = 0.5
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Half asymmetric, half symmetric (per sample, by Bernoulli with p_sym)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 2, N), dtype=np.float32)
    Y = np.zeros((n, 2, N), dtype=np.float32)
    for i in range(n):
        if rng.random() < p_sym:
            u, G, _ = make_symmetric_sample(N, rng)
        else:
            u, G, _ = make_asym_sample(N, rng, snr_db=snr_db)
        X[i, 0] = u.real
        X[i, 1] = u.imag
        Y[i, 0] = G.real / G_scale
        Y[i, 1] = G.imag / G_scale
    return torch.from_numpy(X), torch.from_numpy(Y)


def train_mixed(N: int = 96, n_train: int = 1500, n_val_asym: int = 300,
                n_val_sym: int = 300, epochs: int = 25, batch_size: int = 64,
                lr: float = 1e-3, width: int = 24, modes: int = 12,
                n_blocks: int = 3, threads: int = 16, seed: int = 0):
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  threads: {torch.get_num_threads()}")

    print(f"Generating {n_train} mixed training samples (N={N}) ...")
    t0 = time.time()
    X_tr, Y_tr = build_mixed_tensors(N, n_train, seed, p_sym=0.5)
    print(f"  train gen: {time.time()-t0:.1f}s")

    print(f"Generating {n_val_asym} asymmetric val + {n_val_sym} symmetric val ...")
    X_va_asym, Y_va_asym = build_asym_tensors(N, n_val_asym, seed + 1)
    X_va_sym,  Y_va_sym  = build_symmetric_tensors(N, n_val_sym,  seed + 2)

    train_dl = DataLoader(TensorDataset(X_tr, Y_tr),
                           batch_size=batch_size, shuffle=True)

    model = FNO1d(in_ch=2, out_ch=2, width=width, modes=modes,
                  n_blocks=n_blocks).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"FNO1d: width={width}, modes={modes}, blocks={n_blocks}, "
          f"{n_params:,} params")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)

    history = {"train": [], "val_asym": [], "val_sym": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            yp = model(xb)
            num  = torch.linalg.vector_norm(yp - yb, dim=(-2, -1))
            den  = torch.linalg.vector_norm(yb,      dim=(-2, -1)).clamp_min(1e-6)
            loss = (num / den).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        sched.step()
        train_loss = tot / n

        rl2_asym = float(per_sample_rl2(X_va_asym, Y_va_asym, model, device).mean())
        rl2_sym  = float(per_sample_rl2(X_va_sym,  Y_va_sym,  model, device).mean())
        history["train"].append(train_loss)
        history["val_asym"].append(rl2_asym)
        history["val_sym"].append(rl2_sym)
        print(f"  epoch {ep:3d}  train={train_loss:.4f}  "
              f"val_asym={rl2_asym:.4f}  val_sym={rl2_sym:.4f}",
              flush=True)

    return {
        "model": model, "history": history, "device": device,
        "X_va_asym": X_va_asym, "Y_va_asym": Y_va_asym,
        "X_va_sym":  X_va_sym,  "Y_va_sym":  Y_va_sym,
        "G_scale": 5000.0, "N": N,
    }


def plot_loss(history: dict, out: Path):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    ep = np.arange(1, len(history["train"]) + 1)
    ax.plot(ep, history["train"],     color="black",    lw=1.6, label="train (mixed)")
    ax.plot(ep, history["val_asym"],  color="steelblue", lw=1.6, label="val: asymmetric")
    ax.plot(ep, history["val_sym"],   color="tomato",    lw=1.6, label="val: symmetric")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"relative L$^2$ on G")
    ax.set_title("1D FNO trained on 50/50 asymmetric + symmetric mixture")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_bars(asym_only: dict, mixed: dict, out: Path):
    """Grouped bar chart: two models x two distributions, mean RL2."""
    labels = ["asym test set", "sym test set"]
    asym_only_vals = [asym_only["asym"], asym_only["sym"]]
    mixed_vals     = [mixed["asym"],     mixed["sym"]]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.4))
    b1 = ax.bar(x - w / 2, asym_only_vals, w, color="steelblue", alpha=0.85,
                edgecolor="black", lw=0.5,
                label="asymmetric-only model (Fig 2)")
    b2 = ax.bar(x + w / 2, mixed_vals, w, color="seagreen", alpha=0.85,
                edgecolor="black", lw=0.5,
                label="mixed model (this figure)")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"mean per-sample RL$^2$ on G")
    ax.set_title("Mixed training closes the symmetric gap")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(asym_only_vals + mixed_vals) * 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_predict(result: dict, out: Path, n_show: int = 4):
    model  = result["model"]
    device = result["device"]
    Gs     = result["G_scale"]
    N      = result["N"]
    X_sym  = result["X_va_sym"]
    Y_sym  = result["Y_va_sym"]
    L_m    = 0.16

    rng = np.random.default_rng(13)
    idx = rng.choice(len(X_sym), n_show, replace=False)
    with torch.no_grad():
        Yp = model(X_sym[idx].to(device)).cpu().numpy() * Gs
    Yt = Y_sym[idx].numpy() * Gs

    x_grid = np.linspace(0, L_m, N) * 100
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4.0), sharey=True)
    for k, ax in enumerate(axes):
        ax.plot(x_grid, Yt[k, 0] / 1000, color="black",    lw=1.8, label=r"true Re$\,G$")
        ax.plot(x_grid, Yp[k, 0] / 1000, color="seagreen", lw=1.4, ls="--",
                label=r"mixed-FNO Re$\,G$")
        ax.plot(x_grid, Yt[k, 1] / 1000, color="black",    lw=1.2, alpha=0.5)
        ax.plot(x_grid, Yp[k, 1] / 1000, color="tomato",   lw=1.4, ls="--",
                label=r"mixed-FNO Im$\,G$")
        ax.axvline(L_m * 100 / 2, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("x [cm]")
        ax.set_title(f"sym sample #{idx[k]}")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("G(x) [kPa]")
            ax.legend(fontsize=8, loc="upper center")
    fig.suptitle("Mixed-trained FNO on symmetric samples: kink now resolved",
                  fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",   type=int, default=25)
    ap.add_argument("--n_train",  type=int, default=1500)
    ap.add_argument("--threads",  type=int, default=16)
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    figs = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the mixed model
    result = train_mixed(epochs=args.epochs, n_train=args.n_train,
                         threads=args.threads)

    # Save artifacts
    torch.save(result["model"].state_dict(), figs / "fig4_mixed_fno1d.pt")
    json.dump(result["history"], open(figs / "fig4_mixed_history.json", "w"),
              indent=2)

    # Evaluate the asymmetric-only model on the same val sets for the bar chart
    asym_only = load_trained_fno(figs / "fig2_fno1d.pt", device)
    asym_only_asym = float(per_sample_rl2(result["X_va_asym"],
                                          result["Y_va_asym"],
                                          asym_only, device).mean())
    asym_only_sym  = float(per_sample_rl2(result["X_va_sym"],
                                          result["Y_va_sym"],
                                          asym_only, device).mean())

    mixed_asym = result["history"]["val_asym"][-1]
    mixed_sym  = result["history"]["val_sym"][-1]

    print("\n========== Summary ==========")
    print(f"Asymmetric-only model:  asym RL2 = {asym_only_asym:.4f}, "
          f"sym RL2 = {asym_only_sym:.4f}")
    print(f"Mixed model:            asym RL2 = {mixed_asym:.4f}, "
          f"sym RL2 = {mixed_sym:.4f}")

    plot_loss(result["history"], figs / "fig4a_mixed_loss.png")
    plot_bars(
        {"asym": asym_only_asym, "sym": asym_only_sym},
        {"asym": mixed_asym,     "sym": mixed_sym},
        figs / "fig4b_mixed_bars.png",
    )
    plot_predict(result, figs / "fig4c_mixed_predict.png")

    summary = {
        "asym_only_on_asym": asym_only_asym,
        "asym_only_on_sym":  asym_only_sym,
        "mixed_on_asym":     mixed_asym,
        "mixed_on_sym":      mixed_sym,
    }
    json.dump(summary, open(figs / "fig4_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
