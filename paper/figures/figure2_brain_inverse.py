#!/usr/bin/env python3
"""Figure 2 - 1D FNO inversion: recover G(x) from noisy u(x) at brain scale.

Pipeline:
  1. Generate N_train training pairs (u(x), G(x)) by sampling
     (G0, Gend, xi) ~ priors matched to in vivo brain MRE, computing u
     by the FD solver, and adding complex Gaussian noise at SNR_dB.
  2. Train a small 1D Fourier neural operator that maps Re/Im u to Re/Im G.
  3. Record training and validation loss vs epoch.
  4. On held-out samples, show G_pred vs G_true.

Output:
  paper/figures/fig2a_loss.png      — training/validation loss curves
  paper/figures/fig2b_predict.png   — four-sample G prediction panel
"""
from __future__ import annotations

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

from brain1d import BrainParams, G_profile, numerical_solution


# ── 1D Fourier neural operator ───────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (c_in * c_out)
        self.W = nn.Parameter(scale * torch.randn(c_in, c_out, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, C, N)
        B, C, N = x.shape
        Xf = torch.fft.rfft(x, dim=-1)                          # (B, C, N//2+1)
        Yf = torch.zeros(B, self.W.shape[1], Xf.shape[-1],
                          dtype=torch.cfloat, device=x.device)
        m = min(self.modes, Xf.shape[-1])
        Yf[..., :m] = torch.einsum("bci,cdi->bdi", Xf[..., :m], self.W[..., :m])
        return torch.fft.irfft(Yf, n=N, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spec = SpectralConv1d(width, width, modes)
        self.lin  = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.lin(x))


class FNO1d(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 2,
                 width: int = 32, modes: int = 16, n_blocks: int = 4):
        super().__init__()
        self.lift = nn.Conv1d(in_ch + 1, width, 1)   # +1 for x-coord channel
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(n_blocks)])
        self.proj = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        grid = torch.linspace(0, 1, N, device=x.device).view(1, 1, N).expand(B, 1, N)
        x = torch.cat([x, grid], dim=1)
        x = self.lift(x)
        for b in self.blocks:
            x = b(x)
        return self.proj(x)


# ── dataset generator ────────────────────────────────────────────────────────

def make_sample(N: int, rng: np.random.Generator, snr_db: float = 25.0
                ) -> tuple[np.ndarray, np.ndarray, BrainParams]:
    p = BrainParams(
        G0   = float(rng.uniform(800.0,  2500.0)),
        Gend = float(rng.uniform(1500.0, 4500.0)),
        xi   = float(rng.uniform(0.05,   0.20)),
    )
    if p.Gend <= p.G0:                # ensure positive alpha
        p.Gend = p.G0 + rng.uniform(500.0, 2500.0)
    x, u = numerical_solution(N, p)
    G    = G_profile(x, p)

    # Add complex Gaussian noise at the requested SNR
    sig_power = float(np.mean(np.abs(u) ** 2))
    sigma_n   = float(np.sqrt(sig_power / (10.0 ** (snr_db / 10.0)) / 2.0))
    u_noisy   = u + sigma_n * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

    # Normalise displacement to unit max-magnitude
    u_norm = u_noisy / max(np.abs(u_noisy).max(), 1e-12)

    return u_norm.astype(np.complex64), G.astype(np.complex64), p


def build_tensors(N: int, n_samples: int, seed: int, snr_db: float = 25.0,
                  G_scale: float = 5000.0
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 2, N), dtype=np.float32)
    Y = np.zeros((n_samples, 2, N), dtype=np.float32)
    for i in range(n_samples):
        u, G, _ = make_sample(N, rng, snr_db)
        X[i, 0] = u.real
        X[i, 1] = u.imag
        Y[i, 0] = G.real / G_scale       # normalise to ~ O(1)
        Y[i, 1] = G.imag / G_scale
    return torch.from_numpy(X), torch.from_numpy(Y)


# ── training driver ──────────────────────────────────────────────────────────

def train(N: int = 128, n_train: int = 3000, n_val: int = 600,
          epochs: int = 60, batch_size: int = 64, lr: float = 1e-3,
          width: int = 32, modes: int = 16, n_blocks: int = 4,
          snr_db: float = 25.0, seed: int = 0,
          ) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Generating {n_train+n_val} samples at N={N} ...")
    t0 = time.time()
    X_tr, Y_tr = build_tensors(N, n_train, seed,     snr_db=snr_db)
    X_va, Y_va = build_tensors(N, n_val,   seed + 1, snr_db=snr_db)
    print(f"  data gen: {time.time()-t0:.1f}s")

    train_dl = DataLoader(TensorDataset(X_tr, Y_tr),
                           batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_va, Y_va),
                           batch_size=batch_size, shuffle=False)

    model = FNO1d(in_ch=2, out_ch=2, width=width, modes=modes,
                  n_blocks=n_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FNO1d: width={width}, modes={modes}, blocks={n_blocks},  "
          f"{n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)

    history = {"train": [], "val": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            yp = model(xb)
            num = torch.linalg.vector_norm(yp - yb, dim=(-2, -1))
            den = torch.linalg.vector_norm(yb,      dim=(-2, -1)).clamp_min(1e-6)
            loss = (num / den).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        sched.step()
        train_loss = tot / n

        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                yp = model(xb)
                num = torch.linalg.vector_norm(yp - yb, dim=(-2, -1))
                den = torch.linalg.vector_norm(yb, dim=(-2, -1)).clamp_min(1e-6)
                tot += (num / den).mean().item() * len(xb); n += len(xb)
        val_loss = tot / n
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"  epoch {ep:3d}  train RL2 = {train_loss:.4f}   val RL2 = {val_loss:.4f}",
              flush=True)

    return {"model": model, "history": history, "device": device,
            "X_va": X_va, "Y_va": Y_va, "G_scale": 5000.0, "N": N}


# ── plots ────────────────────────────────────────────────────────────────────

def plot_loss(history: dict, out: Path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ep = np.arange(1, len(history["train"]) + 1)
    ax.plot(ep, history["train"], color="steelblue", lw=1.6, label="train")
    ax.plot(ep, history["val"],   color="tomato", lw=1.6, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"relative L$^2$ on G")
    ax.set_title("1D brain-scale G inversion: training")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_predictions(result: dict, out: Path, n_show: int = 4):
    model = result["model"]
    X_va  = result["X_va"]
    Y_va  = result["Y_va"]
    Gs    = result["G_scale"]
    N     = result["N"]
    device = result["device"]

    x = np.linspace(0, 0.16, N) * 100   # cm

    rng = np.random.default_rng(7)
    idx = rng.choice(len(X_va), n_show, replace=False)
    model.eval()
    with torch.no_grad():
        Yp = model(X_va[idx].to(device)).cpu().numpy() * Gs
    Yt = Y_va[idx].numpy() * Gs

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 6.0),
                              sharex=True)
    for i in range(n_show):
        ax_r = axes[0, i]
        ax_r.plot(x, Yt[i, 0] / 1000, color="black", lw=1.8, label=r"true Re$\,G$")
        ax_r.plot(x, Yp[i, 0] / 1000, color="steelblue", lw=1.4,
                   linestyle="--", label=r"pred Re$\,G$")
        ax_r.set_title(f"sample {idx[i]}")
        ax_r.grid(alpha=0.3)
        if i == 0:
            ax_r.set_ylabel(r"Re $G$ [kPa]")
            ax_r.legend(loc="upper left", fontsize=8)

        ax_i = axes[1, i]
        ax_i.plot(x, Yt[i, 1] / 1000, color="black", lw=1.8, label=r"true Im$\,G$")
        ax_i.plot(x, Yp[i, 1] / 1000, color="tomato", lw=1.4,
                   linestyle="--", label=r"pred Im$\,G$")
        ax_i.set_xlabel("x [cm]")
        ax_i.grid(alpha=0.3)
        if i == 0:
            ax_i.set_ylabel(r"Im $G$ [kPa]")
            ax_i.legend(loc="upper left", fontsize=8)

    fig.suptitle("1D FNO recovery of complex G(x) from noisy u(x)",
                  fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",        type=int, default=128)
    ap.add_argument("--n_train",  type=int, default=3000)
    ap.add_argument("--n_val",    type=int, default=600)
    ap.add_argument("--epochs",   type=int, default=60)
    ap.add_argument("--batch",    type=int, default=64)
    ap.add_argument("--width",    type=int, default=32)
    ap.add_argument("--modes",    type=int, default=16)
    ap.add_argument("--blocks",   type=int, default=4)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--threads",  type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    print(f"torch threads: {torch.get_num_threads()}")

    figs = Path(__file__).parent
    result = train(
        N=args.N, n_train=args.n_train, n_val=args.n_val,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        width=args.width, modes=args.modes, n_blocks=args.blocks,
    )
    plot_loss(result["history"], figs / "fig2a_loss.png")
    plot_predictions(result, figs / "fig2b_predict.png")

    import json
    torch.save(result["model"].state_dict(), figs / "fig2_fno1d.pt")
    json.dump(result["history"], open(figs / "fig2_history.json", "w"), indent=2)


if __name__ == "__main__":
    main()
