#!/usr/bin/env python3
"""Figure 5 - 2x4 ablation: FNO vs SIREN, four loss functions.

Trains the same data on every (architecture, loss) pair, evaluates each
trained model on the asymmetric (in-distribution) and symmetric (OOD)
held-out sets, and produces three sub-figures plus a JSON summary.

Architectures:
  FNO1d   - spectral conv blocks (figure2_brain_inverse.py)
  SIREN1d - 1D conv + sin activations with SIREN init (defined below)

Loss functions (paper/figures/losses_1d.py):
  rl2       - per-sample relative L^2 on G
  mse       - MSE on G
  pde       - Helmholtz residual only (no ground-truth)
  composite - RL^2 + 0.05 * Helmholtz residual

Output:
  paper/figures/fig5a_arch_loss_curves.png
  paper/figures/fig5b_arch_loss_heatmap.png
  paper/figures/fig5c_arch_loss_predict.png
  paper/figures/fig5_arch_loss_summary.json
  paper/figures/fig5_{arch}_{loss}.pt (eight checkpoints)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from figure2_brain_inverse import (
    FNO1d, build_tensors as build_asym_tensors,
)
from figure3_symmetric_stress import (
    build_symmetric_tensors, per_sample_rl2, load_trained_fno,
)
from losses_1d import LOSS_FNS


# Physical constants for the 1D brain-scale problem (must match brain1d.py)
N_GRID    = 96
L_M       = 0.16
DX        = L_M / (N_GRID - 1)
FREQ_HZ   = 50.0
RHO       = 1000.0
G_SCALE   = 5000.0


# ── SIREN 1D operator (convolutional, parameter-matched to FNO) ──────────────

class SineLayer(nn.Module):
    """One 1D conv + sin(omega_0 * .) layer with SIREN initialisation."""
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 1,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0  = omega_0
        self.is_first = is_first
        pad = kernel_size // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad)
        with torch.no_grad():
            fan_in = c_in * kernel_size
            if is_first:
                self.conv.weight.uniform_(-1.0 / fan_in, 1.0 / fan_in)
            else:
                bound = np.sqrt(6.0 / fan_in) / omega_0
                self.conv.weight.uniform_(-bound, bound)
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.conv(x))


class SIREN1d(nn.Module):
    """SIREN-style 1D operator for the inversion task (~20k params)."""
    def __init__(self, in_ch: int = 2, out_ch: int = 2,
                 width: int = 24, kernel_size: int = 11, n_blocks: int = 3,
                 first_omega_0: float = 30.0, hidden_omega_0: float = 30.0):
        super().__init__()
        self.first = SineLayer(in_ch + 1, width, kernel_size=1,
                                is_first=True, omega_0=first_omega_0)
        self.hidden = nn.ModuleList([
            SineLayer(width, width, kernel_size=kernel_size,
                       is_first=False, omega_0=hidden_omega_0)
            for _ in range(n_blocks)
        ])
        self.last = nn.Conv1d(width, out_ch, 1)
        with torch.no_grad():
            bound = np.sqrt(6.0 / width) / hidden_omega_0
            self.last.weight.uniform_(-bound, bound)
            self.last.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        grid = torch.linspace(0, 1, N, device=x.device).view(1, 1, N).expand(B, 1, N)
        x = torch.cat([x, grid], dim=1)
        x = self.first(x)
        for h in self.hidden:
            x = h(x)
        return self.last(x)


# ── Training driver ──────────────────────────────────────────────────────────

def train_one(arch_name: str, loss_name: str,
              X_tr: torch.Tensor, Y_tr: torch.Tensor,
              X_va_asym: torch.Tensor, Y_va_asym: torch.Tensor,
              X_va_sym: torch.Tensor, Y_va_sym: torch.Tensor,
              epochs: int, batch_size: int, lr: float,
              device: torch.device) -> dict:
    """Train one (architecture, loss) combination and return its history."""
    if arch_name == "fno":
        model = FNO1d(in_ch=2, out_ch=2, width=24, modes=12, n_blocks=3)
    elif arch_name == "siren":
        model = SIREN1d(in_ch=2, out_ch=2, width=24, kernel_size=11, n_blocks=3)
    else:
        raise ValueError(arch_name)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loss_fn = LOSS_FNS[loss_name]
    train_dl = DataLoader(TensorDataset(X_tr, Y_tr),
                           batch_size=batch_size, shuffle=True)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)

    tag = f"{arch_name.upper():5s}+{loss_name:9s}"
    print(f"[{tag}] {n_params:,} params, training {epochs} epochs ...")

    history = {"train": [], "val_asym": [], "val_sym": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            yp = model(xb)
            loss = loss_fn(yp, yb, xb, DX, FREQ_HZ, RHO, G_SCALE)
            if not torch.isfinite(loss):
                raise RuntimeError(f"[{tag}] non-finite loss at epoch {ep}")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        sched.step()
        train_loss = tot / n
        rl2_a = float(per_sample_rl2(X_va_asym, Y_va_asym, model, device).mean())
        rl2_s = float(per_sample_rl2(X_va_sym,  Y_va_sym,  model, device).mean())
        history["train"].append(train_loss)
        history["val_asym"].append(rl2_a)
        history["val_sym"].append(rl2_s)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  [{tag}] ep {ep:3d}  train={train_loss:.4f}  "
                  f"val_asym={rl2_a:.4f}  val_sym={rl2_s:.4f}", flush=True)

    return {"model": model, "history": history, "n_params": n_params,
            "arch": arch_name, "loss": loss_name}


# ── Plotting ─────────────────────────────────────────────────────────────────

ARCHES = ["fno", "siren"]
LOSSES = ["rl2", "mse", "pde", "composite"]
ARCH_LABEL = {"fno": "FNO", "siren": "SIREN"}
LOSS_LABEL = {"rl2": "RL²", "mse": "MSE", "pde": "PDE", "composite": "RL²+λ·PDE"}
ARCH_COLOR = {"fno": "steelblue", "siren": "tomato"}
LOSS_STYLE = {"rl2": "-", "mse": "--", "pde": ":", "composite": "-."}


def plot_loss_curves(results: dict, out: Path):
    """Validation RL2 vs epoch for all 8 runs, split asym/sym."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, key, title in [
        (axes[0], "val_asym", "(a) Asymmetric (in-distribution) val RL² vs epoch"),
        (axes[1], "val_sym",  "(b) Symmetric (zero-shot OOD) val RL² vs epoch"),
    ]:
        for arch in ARCHES:
            for loss in LOSSES:
                r = results[(arch, loss)]
                ep = np.arange(1, len(r["history"][key]) + 1)
                ax.plot(ep, r["history"][key],
                        color=ARCH_COLOR[arch], ls=LOSS_STYLE[loss], lw=1.5,
                        label=f"{ARCH_LABEL[arch]} + {LOSS_LABEL[loss]}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"validation RL$^2$ on G")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_heatmap(results: dict, out: Path):
    """2x4 heatmaps of final mean RL2 on asym and sym test sets."""
    asym = np.array([[results[(a, l)]["history"]["val_asym"][-1] for l in LOSSES]
                      for a in ARCHES])
    sym  = np.array([[results[(a, l)]["history"]["val_sym"][-1]  for l in LOSSES]
                      for a in ARCHES])

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    for ax, M, title in [
        (axes[0], asym, "(c) Final val RL² on ASYM test set"),
        (axes[1], sym,  "(d) Final val RL² on SYM test set (zero-shot)"),
    ]:
        im = ax.imshow(M, aspect="auto", cmap="viridis_r",
                       vmin=0, vmax=max(asym.max(), sym.max()))
        ax.set_xticks(range(len(LOSSES)))
        ax.set_xticklabels([LOSS_LABEL[l] for l in LOSSES])
        ax.set_yticks(range(len(ARCHES)))
        ax.set_yticklabels([ARCH_LABEL[a] for a in ARCHES])
        ax.set_title(title, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                color = "white" if M[i, j] > 0.5 * max(asym.max(), sym.max()) else "black"
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color=color, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04, label="RL²")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_predictions(results: dict, X_va_asym: torch.Tensor, Y_va_asym: torch.Tensor,
                     X_va_sym: torch.Tensor, Y_va_sym: torch.Tensor,
                     device: torch.device, out: Path):
    """Show four representative configurations on the same two held-out samples."""
    rng = np.random.default_rng(42)
    idx_asym = int(rng.integers(0, len(X_va_asym)))
    idx_sym  = int(rng.integers(0, len(X_va_sym)))
    x_grid = np.linspace(0, L_M, N_GRID) * 100

    chosen = [("fno", "rl2"), ("fno", "composite"),
              ("siren", "rl2"), ("siren", "composite")]

    fig, axes = plt.subplots(2, 4, figsize=(18, 7.0), sharey=True)
    for col, (arch, loss) in enumerate(chosen):
        model = results[(arch, loss)]["model"]
        model.eval()
        # asym row
        with torch.no_grad():
            yp = model(X_va_asym[idx_asym:idx_asym + 1].to(device)).cpu().numpy()[0] * G_SCALE
        yt = Y_va_asym[idx_asym].numpy() * G_SCALE
        ax = axes[0, col]
        ax.plot(x_grid, yt[0] / 1000, color="black",     lw=1.8, label=r"true Re$\,G$")
        ax.plot(x_grid, yp[0] / 1000, color="steelblue", lw=1.4, ls="--", label=r"pred Re$\,G$")
        ax.plot(x_grid, yt[1] / 1000, color="black",     lw=1.0, alpha=0.5)
        ax.plot(x_grid, yp[1] / 1000, color="tomato",    lw=1.4, ls="--", label=r"pred Im$\,G$")
        ax.set_title(f"{ARCH_LABEL[arch]} + {LOSS_LABEL[loss]} | asym",
                      fontsize=10)
        ax.set_xlabel("x [cm]")
        if col == 0:
            ax.set_ylabel("G(x) [kPa]")
            ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

        # sym row
        with torch.no_grad():
            yp = model(X_va_sym[idx_sym:idx_sym + 1].to(device)).cpu().numpy()[0] * G_SCALE
        yt = Y_va_sym[idx_sym].numpy() * G_SCALE
        ax = axes[1, col]
        ax.plot(x_grid, yt[0] / 1000, color="black",     lw=1.8, label=r"true Re$\,G$")
        ax.plot(x_grid, yp[0] / 1000, color="steelblue", lw=1.4, ls="--")
        ax.plot(x_grid, yt[1] / 1000, color="black",     lw=1.0, alpha=0.5)
        ax.plot(x_grid, yp[1] / 1000, color="tomato",    lw=1.4, ls="--")
        ax.axvline(L_M * 100 / 2, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{ARCH_LABEL[arch]} + {LOSS_LABEL[loss]} | sym (OOD)",
                      fontsize=10)
        ax.set_xlabel("x [cm]")
        if col == 0:
            ax.set_ylabel("G(x) [kPa]")
        ax.grid(alpha=0.3)

    fig.suptitle("Per-sample predictions across four representative configurations.\n"
                  "Top row: asym in-distribution sample. Bottom row: sym zero-shot OOD sample.",
                  fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",  type=int, default=25)
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    figs   = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, threads: {torch.get_num_threads()}")

    print(f"Generating {args.n_train} asym train + 300 asym val + 300 sym val ...")
    t0 = time.time()
    X_tr,      Y_tr      = build_asym_tensors(N_GRID, args.n_train, seed=0)
    X_va_asym, Y_va_asym = build_asym_tensors(N_GRID, 300,           seed=1)
    X_va_sym,  Y_va_sym  = build_symmetric_tensors(N_GRID, 300,      seed=2)
    print(f"  data gen: {time.time() - t0:.1f}s")

    # 2x4 grid
    results: dict = {}
    overall_t0 = time.time()
    for arch in ARCHES:
        for loss in LOSSES:
            run_t0 = time.time()
            res = train_one(
                arch, loss, X_tr, Y_tr, X_va_asym, Y_va_asym, X_va_sym, Y_va_sym,
                epochs=args.epochs, batch_size=64, lr=1e-3, device=device,
            )
            results[(arch, loss)] = res
            torch.save(res["model"].state_dict(), figs / f"fig5_{arch}_{loss}.pt")
            print(f"  -> {arch}+{loss} done in {time.time() - run_t0:.1f}s")
    print(f"\nAll 8 runs done in {(time.time() - overall_t0) / 60:.1f} min")

    # Summary JSON
    summary = {}
    for (arch, loss), res in results.items():
        summary[f"{arch}_{loss}"] = {
            "arch": arch, "loss": loss, "params": res["n_params"],
            "val_asym_final": res["history"]["val_asym"][-1],
            "val_sym_final":  res["history"]["val_sym"][-1],
        }
    json.dump(summary, open(figs / "fig5_arch_loss_summary.json", "w"), indent=2)
    json.dump({f"{a}_{l}": r["history"] for (a, l), r in results.items()},
              open(figs / "fig5_arch_loss_history.json", "w"), indent=2)

    print("\n========= Final mean RL² on val sets =========")
    print(f"{'Run':22s}  {'params':>8s}  {'asym':>8s}  {'sym':>8s}")
    for (arch, loss), res in results.items():
        h = res["history"]
        print(f"{ARCH_LABEL[arch]:5s} + {LOSS_LABEL[loss]:11s}  "
              f"{res['n_params']:>8,}  {h['val_asym'][-1]:>8.4f}  {h['val_sym'][-1]:>8.4f}")

    # Figures
    plot_loss_curves(results, figs / "fig5a_arch_loss_curves.png")
    plot_heatmap   (results, figs / "fig5b_arch_loss_heatmap.png")
    plot_predictions(results, X_va_asym, Y_va_asym, X_va_sym, Y_va_sym,
                     device, figs / "fig5c_arch_loss_predict.png")


if __name__ == "__main__":
    main()
