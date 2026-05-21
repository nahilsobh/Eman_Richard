#!/usr/bin/env python3
"""Figure 9 - Train SIREN on the embedded-BC generator.

Tests whether removing the BC assumption + adding multi-family diversity
in synthetic training produces a model that beats the existing
Dirichlet-only, single-family or two-family checkpoints across a battery
of test distributions.

Training set:
  1500 samples from make_embedded_sample (random profile in
  {asym, sym_euler, sym_smooth} x random BC in {dirichlet, neumann,
  absorbing} x extended-domain solution, cropped to ROI).
  Matched to Fig 2/5/6 training-set size so the only differences are
  BC randomisation and family diversity.

Evaluation: 4 held-out test distributions x 3 trained checkpoints
  (Fig 5 SIREN  : Dirichlet asym-only, 1500 samples)
  (Fig 6 SIREN  : Dirichlet asym+sym_euler mixed, 1500 samples)
  (Fig 9 SIREN  : embedded multi-family + random BC, 1500 samples) -- this run

Outputs:
  fig9_embedded.pt
  fig9_embedded_loss.png
  fig9_embedded_bars.png
  fig9_embedded_history.json
  fig9_embedded_summary.json
"""
from __future__ import annotations

import json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from brain1d import make_embedded_sample
from figure2_brain_inverse import (
    FNO1d, build_tensors as build_asym_tensors,
)
from figure3_symmetric_stress import (
    build_symmetric_tensors, per_sample_rl2, load_trained_fno,
)
from figure5_siren_compare import SIREN1d
from figure8_smooth_inverse import build_smooth_tensors


N_GRID  = 96
G_SCALE = 5000.0


def build_embedded_tensors(n: int, seed: int, ext_factor: float = 1.0,
                            G_scale: float = G_SCALE
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 2, N_GRID), dtype=np.float32)
    Y = np.zeros((n, 2, N_GRID), dtype=np.float32)
    for i in range(n):
        u, G, _ = make_embedded_sample(N_roi=N_GRID, ext_factor=ext_factor,
                                         rng=rng)
        X[i, 0] = u.real
        X[i, 1] = u.imag
        Y[i, 0] = G.real / G_scale
        Y[i, 1] = G.imag / G_scale
    return torch.from_numpy(X), torch.from_numpy(Y)


def train_siren_embedded(N_train: int = 1500, epochs: int = 25,
                          batch: int = 64, lr: float = 1e-3,
                          threads: int = 16, seed: int = 0):
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  threads: {torch.get_num_threads()}")

    print(f"Generating {N_train} embedded-BC training samples ...")
    t0 = time.time()
    X_tr, Y_tr = build_embedded_tensors(N_train, seed)
    print(f"  data gen: {time.time() - t0:.1f}s")

    print("Generating held-out validation sets (300 each) ...")
    X_va_emb, Y_va_emb = build_embedded_tensors(300, seed + 1)
    X_va_a,   Y_va_a   = build_asym_tensors(N_GRID, 300, seed + 2)
    X_va_se,  Y_va_se  = build_symmetric_tensors(N_GRID, 300, seed + 3)
    X_va_ss,  Y_va_ss  = build_smooth_tensors(N_GRID, 300, seed + 4)

    model = SIREN1d(in_ch=2, out_ch=2, width=24, kernel_size=11, n_blocks=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SIREN1d: {n_params:,} params")

    train_dl = DataLoader(TensorDataset(X_tr, Y_tr),
                           batch_size=batch, shuffle=True)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)

    history = {"train": [], "val_emb": [], "val_asym": [],
               "val_sym_euler": [], "val_sym_smooth": []}
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

        rl2_emb = float(per_sample_rl2(X_va_emb, Y_va_emb, model, device).mean())
        rl2_a   = float(per_sample_rl2(X_va_a,   Y_va_a,   model, device).mean())
        rl2_se  = float(per_sample_rl2(X_va_se,  Y_va_se,  model, device).mean())
        rl2_ss  = float(per_sample_rl2(X_va_ss,  Y_va_ss,  model, device).mean())
        history["train"].append(train_loss)
        history["val_emb"].append(rl2_emb)
        history["val_asym"].append(rl2_a)
        history["val_sym_euler"].append(rl2_se)
        history["val_sym_smooth"].append(rl2_ss)
        print(f"  ep {ep:3d}  train={train_loss:.4f}  "
              f"emb={rl2_emb:.4f}  asym={rl2_a:.4f}  "
              f"euler={rl2_se:.4f}  smooth={rl2_ss:.4f}", flush=True)

    return {
        "model": model, "history": history, "device": device,
        "X_va_emb": X_va_emb, "Y_va_emb": Y_va_emb,
        "X_va_a":   X_va_a,   "Y_va_a":   Y_va_a,
        "X_va_se":  X_va_se,  "Y_va_se":  Y_va_se,
        "X_va_ss":  X_va_ss,  "Y_va_ss":  Y_va_ss,
        "n_params": n_params,
    }


def plot_loss(history: dict, out: Path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.4))
    ep = np.arange(1, len(history["train"]) + 1)
    ax.plot(ep, history["train"],          color="black",     lw=1.6, label="train")
    ax.plot(ep, history["val_emb"],        color="seagreen",  lw=1.4, label="val: embedded (in-dist)")
    ax.plot(ep, history["val_asym"],       color="steelblue", lw=1.4, label="val: asym (Fig 2 dist)")
    ax.plot(ep, history["val_sym_euler"],  color="tomato",    lw=1.4, label="val: sym Euler (Fig 4 dist)")
    ax.plot(ep, history["val_sym_smooth"], color="purple",    lw=1.4, label="val: sym smooth (Fig 4 OOD)")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"relative L$^2$ on G")
    ax.set_yscale("log")
    ax.set_title("SIREN trained on embedded-BC multi-family generator")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_bars(summary: dict, out: Path):
    """Three trained models x five test distributions."""
    tests = list(summary["tests"])
    models = list(summary["rl2"].keys())
    n_t = len(tests); n_m = len(models)
    x = np.arange(n_t)
    w = 0.25
    colors = {"SIREN + asym-only (Fig 2)":  "steelblue",
              "SIREN + mixed (Fig 6)":      "tomato",
              "SIREN + embedded (new)":      "seagreen"}

    fig, ax = plt.subplots(1, 1, figsize=(13, 5.2))
    for i, m in enumerate(models):
        vals = [summary["rl2"][m][t] for t in tests]
        offset = (i - (n_m - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, color=colors.get(m, "gray"),
                      alpha=0.85, edgecolor="black", lw=0.5, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(tests, fontsize=9)
    ax.set_ylabel(r"mean per-sample RL$^2$ on G")
    ax.set_title("Three trained checkpoints × five held-out test distributions")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(summary["rl2"][m].values()) for m in models) * 1.18)
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

    # Train
    res = train_siren_embedded(N_train=args.n_train, epochs=args.epochs,
                                threads=args.threads)
    torch.save(res["model"].state_dict(), figs / "fig9_embedded.pt")
    json.dump(res["history"], open(figs / "fig9_embedded_history.json", "w"),
              indent=2)

    # Build a fifth test set so the comparison covers both "old" and "new"
    # validation distributions: original mixed (Fig 6 in-dist).  Lazy
    # version: build it inline from the existing helpers.
    from figure4_mixed_training import build_mixed_tensors
    X_mix, Y_mix = build_mixed_tensors(N_GRID, 300, seed=99, p_sym=0.5)

    # Load existing checkpoints
    device = res["device"]
    siren_asym  = SIREN1d(in_ch=2, out_ch=2, width=24, kernel_size=11,
                           n_blocks=3).to(device).eval()
    siren_asym.load_state_dict(torch.load(figs / "fig5_siren_rl2.pt",
                                            map_location=device,
                                            weights_only=False))
    siren_mixed = SIREN1d(in_ch=2, out_ch=2, width=24, kernel_size=11,
                           n_blocks=3).to(device).eval()
    siren_mixed.load_state_dict(torch.load(figs / "fig6_siren_mixed.pt",
                                            map_location=device,
                                            weights_only=False))
    siren_embed = res["model"].eval()

    tests = [
        ("asym (Fig 2 in-dist)",        res["X_va_a"],   res["Y_va_a"]),
        ("mixed asym+euler (Fig 6 in)", X_mix,          Y_mix),
        ("sym Euler (Fig 4 dist)",       res["X_va_se"],  res["Y_va_se"]),
        ("sym smooth (Fig 4 dist)",      res["X_va_ss"],  res["Y_va_ss"]),
        ("embedded (Fig 9 in-dist)",     res["X_va_emb"], res["Y_va_emb"]),
    ]
    models = [
        ("SIREN + asym-only (Fig 2)", siren_asym),
        ("SIREN + mixed (Fig 6)",     siren_mixed),
        ("SIREN + embedded (new)",    siren_embed),
    ]

    print("\n========== Three models × five test distributions ==========")
    print(f"{'Model':30s} | " + " | ".join(f"{t[0][:20]:20s}" for t in tests))
    print("-" * 145)
    grid = {m_name: {} for m_name, _ in models}
    for m_name, m in models:
        row = []
        for t_name, X, Y in tests:
            r = float(per_sample_rl2(X, Y, m, device).mean())
            grid[m_name][t_name] = r
            row.append(f"{r:>20.4f}")
        print(f"{m_name:30s} | " + " | ".join(row))

    summary = {
        "tests":  [t[0] for t in tests],
        "models": [m[0] for m, _ in models],
        "rl2":    grid,
        "n_params": res["n_params"],
        "n_train":  args.n_train,
    }
    json.dump(summary, open(figs / "fig9_embedded_summary.json", "w"), indent=2)

    plot_loss(res["history"], figs / "fig9_embedded_loss.png")
    plot_bars(summary,        figs / "fig9_embedded_bars.png")


if __name__ == "__main__":
    main()
