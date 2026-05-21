#!/usr/bin/env python3
"""Figure 6 - SIREN trained on the mixed asym + sym distribution.

Combines the two strongest findings from the 1D narrative:
  * SIREN beats FNO on both asym and sym test sets (Fig 5)
  * Mixed training closes the symmetric OOD gap (Fig 4) for the FNO,
    but with a 2x degradation on asym

This experiment asks: does SIREN trained on the same 50/50 mixed
distribution as the FNO+mixed model achieve specialist-level performance
on *both* distributions simultaneously?

Output:
  paper/figures/fig6_siren_mixed_summary.json
  paper/figures/fig6_siren_mixed_bars.png         - 4-bar comparison
  paper/figures/fig6_siren_mixed_predict.png      - sample predictions
  paper/figures/fig6_siren_mixed_loss.png         - loss curves
  paper/figures/fig6_siren_mixed.pt               - checkpoint
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
from torch.utils.data import DataLoader, TensorDataset

from figure2_brain_inverse import build_tensors as build_asym_tensors
from figure3_symmetric_stress import (
    build_symmetric_tensors, per_sample_rl2, load_trained_fno,
)
from figure4_mixed_training import build_mixed_tensors
from figure5_siren_compare import SIREN1d


N_GRID  = 96
G_SCALE = 5000.0


def train_siren_mixed(N: int = N_GRID, n_train: int = 1500, n_val: int = 300,
                      epochs: int = 25, batch_size: int = 64, lr: float = 1e-3,
                      width: int = 24, kernel: int = 11, n_blocks: int = 3,
                      threads: int = 16, seed: int = 0,
                      ) -> dict:
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  threads: {torch.get_num_threads()}")

    print(f"Generating {n_train} mixed train + {n_val} asym val + {n_val} sym val ...")
    t0 = time.time()
    X_tr, Y_tr  = build_mixed_tensors(N, n_train, seed=seed, p_sym=0.5)
    X_va_a, Y_va_a = build_asym_tensors(N, n_val, seed=seed + 1)
    X_va_s, Y_va_s = build_symmetric_tensors(N, n_val, seed=seed + 2)
    print(f"  data gen: {time.time() - t0:.1f}s")

    model = SIREN1d(in_ch=2, out_ch=2, width=width, kernel_size=kernel,
                    n_blocks=n_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SIREN1d: width={width}, kernel={kernel}, blocks={n_blocks}, "
          f"{n_params:,} params")

    train_dl = DataLoader(TensorDataset(X_tr, Y_tr),
                           batch_size=batch_size, shuffle=True)
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
        rl2_a = float(per_sample_rl2(X_va_a, Y_va_a, model, device).mean())
        rl2_s = float(per_sample_rl2(X_va_s, Y_va_s, model, device).mean())
        history["train"].append(train_loss)
        history["val_asym"].append(rl2_a)
        history["val_sym"].append(rl2_s)
        print(f"  epoch {ep:3d}  train={train_loss:.4f}  "
              f"val_asym={rl2_a:.4f}  val_sym={rl2_s:.4f}", flush=True)

    return {"model": model, "history": history, "n_params": n_params,
            "device": device,
            "X_va_a": X_va_a, "Y_va_a": Y_va_a,
            "X_va_s": X_va_s, "Y_va_s": Y_va_s}


def plot_bars(siren_mixed: dict, fno_mixed: dict, siren_asym_only: dict,
              fno_asym_only: dict, out: Path):
    """Four-model comparison: FNO-asym/mixed vs SIREN-asym/mixed on asym/sym test."""
    models = [
        ("FNO + asym-only",  fno_asym_only,    "steelblue"),
        ("FNO + mixed",      fno_mixed,        "navy"),
        ("SIREN + asym-only",siren_asym_only,  "tomato"),
        ("SIREN + mixed",    siren_mixed,      "darkred"),
    ]
    n = len(models)
    x = np.arange(2)
    w = 0.18
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.6))
    for i, (label, vals, color) in enumerate(models):
        bars = ax.bar(x + (i - (n - 1) / 2) * w,
                       [vals["asym"], vals["sym"]],
                       w, color=color, alpha=0.85,
                       edgecolor="black", lw=0.5, label=label)
        for b, v in zip(bars, (vals["asym"], vals["sym"])):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004,
                    f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["asym test set", "sym test set"])
    ax.set_ylabel(r"mean per-sample RL$^2$ on G")
    ax.set_title("1D inversion: architecture x training distribution")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.28)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_loss(history: dict, out: Path):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    ep = np.arange(1, len(history["train"]) + 1)
    ax.plot(ep, history["train"],     color="black",  lw=1.6, label="train (mixed)")
    ax.plot(ep, history["val_asym"],  color="steelblue", lw=1.6, label="val: asymmetric")
    ax.plot(ep, history["val_sym"],   color="tomato",    lw=1.6, label="val: symmetric")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"relative L$^2$ on G")
    ax.set_yscale("log")
    ax.set_title("SIREN trained on 50/50 asym+sym mixture")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_predict(result: dict, out: Path):
    model  = result["model"]
    device = result["device"]
    L_m    = 0.16
    N      = N_GRID
    x_grid = np.linspace(0, L_m, N) * 100

    rng = np.random.default_rng(17)
    idx_a = int(rng.integers(0, len(result["X_va_a"])))
    idx_s = int(rng.integers(0, len(result["X_va_s"])))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0), sharey=True)
    for ax, X_va, Y_va, idx, label in [
        (axes[0], result["X_va_a"], result["Y_va_a"], idx_a, "asym in-dist"),
        (axes[1], result["X_va_s"], result["Y_va_s"], idx_s, "sym OOD"),
    ]:
        model.eval()
        with torch.no_grad():
            yp = model(X_va[idx:idx + 1].to(device)).cpu().numpy()[0] * G_SCALE
        yt = Y_va[idx].numpy() * G_SCALE
        ax.plot(x_grid, yt[0] / 1000, color="black", lw=1.8, label=r"true Re$\,G$")
        ax.plot(x_grid, yp[0] / 1000, color="steelblue", lw=1.4, ls="--", label="SIREN-mixed Re G")
        ax.plot(x_grid, yt[1] / 1000, color="black", lw=1.0, alpha=0.5)
        ax.plot(x_grid, yp[1] / 1000, color="tomato",    lw=1.4, ls="--", label="SIREN-mixed Im G")
        ax.axvline(L_m * 100 / 2, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{label}  sample #{idx}")
        ax.set_xlabel("x [cm]")
        if ax is axes[0]:
            ax.set_ylabel("G(x) [kPa]")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)

    fig.suptitle("SIREN+mixed predictions on held-out asym and sym samples",
                  fontsize=11, y=1.02)
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

    # Train SIREN on the mixed distribution
    res = train_siren_mixed(epochs=args.epochs, n_train=args.n_train,
                             threads=args.threads)

    siren_mixed_final = {
        "asym": res["history"]["val_asym"][-1],
        "sym":  res["history"]["val_sym"][-1],
    }

    # Evaluate the existing checkpoints on the same val sets for an apples-to-apples bar chart
    fno_asym_only_ckpt   = load_trained_fno(figs / "fig2_fno1d.pt", device)
    fno_mixed_ckpt       = load_trained_fno(figs / "fig4_mixed_fno1d.pt", device)
    siren_asym_only_ckpt = SIREN1d(in_ch=2, out_ch=2, width=24, kernel_size=11,
                                    n_blocks=3).to(device)
    siren_asym_only_ckpt.load_state_dict(
        torch.load(figs / "fig5_siren_rl2.pt", map_location=device,
                   weights_only=False)
    )
    siren_asym_only_ckpt.eval()

    fno_asym_only_vals = {
        "asym": float(per_sample_rl2(res["X_va_a"], res["Y_va_a"],
                                     fno_asym_only_ckpt, device).mean()),
        "sym":  float(per_sample_rl2(res["X_va_s"], res["Y_va_s"],
                                     fno_asym_only_ckpt, device).mean()),
    }
    fno_mixed_vals = {
        "asym": float(per_sample_rl2(res["X_va_a"], res["Y_va_a"],
                                     fno_mixed_ckpt, device).mean()),
        "sym":  float(per_sample_rl2(res["X_va_s"], res["Y_va_s"],
                                     fno_mixed_ckpt, device).mean()),
    }
    siren_asym_only_vals = {
        "asym": float(per_sample_rl2(res["X_va_a"], res["Y_va_a"],
                                     siren_asym_only_ckpt, device).mean()),
        "sym":  float(per_sample_rl2(res["X_va_s"], res["Y_va_s"],
                                     siren_asym_only_ckpt, device).mean()),
    }

    print("\n========== Final summary ==========")
    print(f"  FNO   + asym-only :  asym {fno_asym_only_vals['asym']:.4f}  "
          f"sym {fno_asym_only_vals['sym']:.4f}")
    print(f"  FNO   + mixed     :  asym {fno_mixed_vals['asym']:.4f}  "
          f"sym {fno_mixed_vals['sym']:.4f}")
    print(f"  SIREN + asym-only :  asym {siren_asym_only_vals['asym']:.4f}  "
          f"sym {siren_asym_only_vals['sym']:.4f}")
    print(f"  SIREN + mixed     :  asym {siren_mixed_final['asym']:.4f}  "
          f"sym {siren_mixed_final['sym']:.4f}")

    summary = {
        "fno_asym_only":   fno_asym_only_vals,
        "fno_mixed":       fno_mixed_vals,
        "siren_asym_only": siren_asym_only_vals,
        "siren_mixed":     siren_mixed_final,
        "n_params_siren":  res["n_params"],
    }
    json.dump(summary, open(figs / "fig6_siren_mixed_summary.json", "w"), indent=2)
    json.dump(res["history"], open(figs / "fig6_siren_mixed_history.json", "w"),
              indent=2)
    torch.save(res["model"].state_dict(), figs / "fig6_siren_mixed.pt")

    plot_loss(res["history"], figs / "fig6_siren_mixed_loss.png")
    plot_bars(siren_mixed_final, fno_mixed_vals, siren_asym_only_vals,
              fno_asym_only_vals, figs / "fig6_siren_mixed_bars.png")
    plot_predict(res, figs / "fig6_siren_mixed_predict.png")


if __name__ == "__main__":
    main()
