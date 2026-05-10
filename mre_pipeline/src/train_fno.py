#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fno_model import FNO2d


# ── Losses ────────────────────────────────────────────────

def rel_l2(pred, target):
    return (torch.norm(pred - target, dim=(-2, -1)) /
            torch.norm(target, dim=(-2, -1)).clamp(min=1e-6)).mean()


def ssim_metric(pred, target, C1=1e-4, C2=9e-4):
    mu_p = pred.mean(dim=(-2, -1), keepdim=True)
    mu_t = target.mean(dim=(-2, -1), keepdim=True)
    sig_p = pred.var(dim=(-2, -1), keepdim=True)
    sig_t = target.var(dim=(-2, -1), keepdim=True)
    sig_pt = ((pred - mu_p) * (target - mu_t)).mean(dim=(-2, -1), keepdim=True)
    ssim = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2))
    return ssim.mean().item()


def helmholtz_residual_loss(u_re, u_im, G_pred,
                             dx=0.002, freq=60.0, rho=1000.0, damping=0.05):
    """Relative Helmholtz residual loss on interior pixels.

    Computes ||G*(Δu) + ρω²u|| / ||ρω²u|| using a 5-point FD Laplacian.
    Ignores the ∇G·∇u interface term (zero for piecewise-constant G,
    small elsewhere relative to the other terms).
    """
    omega = 2.0 * np.pi * freq
    lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                          dtype=torch.float32, device=G_pred.device
                          ).view(1, 1, 3, 3) / dx ** 2

    def lap(u):
        return F.conv2d(u.unsqueeze(1), lap_k, padding=1).squeeze(1)

    Gr = G_pred
    Gi = G_pred * damping

    lap_re = lap(u_re)
    lap_im = lap(u_im)

    # G*(Δu): complex multiply
    res_re = Gr * lap_re - Gi * lap_im + rho * omega ** 2 * u_re
    res_im = Gr * lap_im + Gi * lap_re + rho * omega ** 2 * u_im

    # Normalise by |ρω²u|, interior only (strip 1-pixel boundary)
    norm = (rho * omega ** 2 *
            (u_re[:, 1:-1, 1:-1] ** 2 + u_im[:, 1:-1, 1:-1] ** 2).sqrt()
            .clamp(min=1e-10))
    res = (res_re[:, 1:-1, 1:-1] ** 2 + res_im[:, 1:-1, 1:-1] ** 2).sqrt()
    return (res / norm).mean()


# ── Data ──────────────────────────────────────────────────

def load_data(data_path, val_frac=0.1):
    print("Loading dataset into RAM ...", flush=True)
    with h5py.File(data_path, "r") as f:
        X = torch.tensor(f["X"][:], dtype=torch.float32)
        Y = torch.tensor(f["Y"][:], dtype=torch.float32)

    n_val = int(len(X) * val_frac)
    X_tr, X_val = X[:-n_val], X[-n_val:]
    Y_tr, Y_val = Y[:-n_val], Y[-n_val:]

    # Normalise X per channel using training stats
    X_mean = X_tr.mean(dim=(0, 2, 3), keepdim=True)
    X_std  = X_tr.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    X_tr   = (X_tr  - X_mean) / X_std
    X_val  = (X_val - X_mean) / X_std

    # Log-scale Y (stiffness spans ~2 decades)
    Y_log_min = Y_tr.log().min()
    Y_log_max = Y_tr.log().max()

    def norm_Y(Y):
        return (Y.log() - Y_log_min) / (Y_log_max - Y_log_min)

    def denorm_Y(Y_n):
        return (Y_n * (Y_log_max - Y_log_min) + Y_log_min).exp()

    Y_tr_n  = norm_Y(Y_tr)
    Y_val_n = norm_Y(Y_val)

    stats = dict(X_mean=X_mean, X_std=X_std,
                 Y_log_min=Y_log_min, Y_log_max=Y_log_max)

    tr_ds  = TensorDataset(X_tr,  Y_tr_n,  Y_tr)
    val_ds = TensorDataset(X_val, Y_val_n, Y_val)
    return tr_ds, val_ds, stats, denorm_Y


# ── Training ──────────────────────────────────────────────

CHECKPOINT_EPOCHS = {25, 50, 75, 100}


def train(args):
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tr_ds, val_ds, stats, denorm_Y = load_data(args.data_path)

    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = FNO2d(modes1=args.modes, modes2=args.modes,
                  width=args.width, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    cfg = vars(args)
    cfg["n_params"] = n_params
    cfg["device"] = str(device)
    json.dump(cfg, open(run_dir / "args.json", "w"), indent=2)

    history = {"train_loss": [], "train_data_loss": [], "train_phys_loss": [],
               "val_rl2": [], "val_ssim": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses, tr_data, tr_phys = [], [], []

        for Xb, Yb_n, _ in tr_loader:
            Xb, Yb_n = Xb.to(device), Yb_n.to(device)
            pred_n = model(Xb)

            # Train with MSE in normalised log-space (numerically stable across
            # the full ILI-style stiffness range, including near-uniform targets).
            data_loss = F.mse_loss(pred_n, Yb_n)
            loss = data_loss

            if args.physics_weight > 0:
                # Denorm prediction → Pa, then compute Helmholtz residual
                pred_pa = denorm_Y(pred_n.detach().clamp(0, 1))
                u_re = Xb[:, 0]   # Re(u) at 60 Hz (normalised units — residual is relative)
                u_im = Xb[:, 1]
                phys_loss = helmholtz_residual_loss(u_re, u_im, pred_pa)
                loss = data_loss + args.physics_weight * phys_loss
                tr_phys.append(phys_loss.item())
            else:
                tr_phys.append(0.0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())
            tr_data.append(data_loss.item())

        scheduler.step()

        model.eval()
        val_rl2s, val_ssims = [], []
        with torch.no_grad():
            for Xb, Yb_n, Yb in val_loader:
                Xb, Yb_n, Yb = Xb.to(device), Yb_n.to(device), Yb.to(device)
                pred_n = model(Xb)
                pred_pa = denorm_Y(pred_n)
                val_rl2s.append(rel_l2(pred_pa, Yb).item())
                val_ssims.append(ssim_metric(pred_pa.cpu(), Yb.cpu()))

        tr_loss  = float(np.mean(tr_losses))
        val_rl2  = float(np.mean(val_rl2s))
        val_ssim = float(np.mean(val_ssims))
        history["train_loss"].append(tr_loss)
        history["train_data_loss"].append(float(np.mean(tr_data)))
        history["train_phys_loss"].append(float(np.mean(tr_phys)))
        history["val_rl2"].append(val_rl2)
        history["val_ssim"].append(val_ssim)

        elapsed = time.time() - t0
        print(f"Ep {epoch:03d}/{args.epochs} | "
              f"train={tr_loss:.4f} | val_RL2={val_rl2:.4f} | "
              f"SSIM={val_ssim:.3f} | lr={scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s", flush=True)

        json.dump(history, open(run_dir / "history.json", "w"), indent=2)

        if val_rl2 < best_val:
            best_val = val_rl2
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_rl2": val_rl2, "val_ssim": val_ssim,
                        "stats": stats}, run_dir / "best.pt")
            print(f"  → best checkpoint  val_RL2={val_rl2:.4f}", flush=True)

        if epoch in CHECKPOINT_EPOCHS:
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_rl2": val_rl2, "stats": stats},
                       run_dir / f"epoch_{epoch:03d}.pt")

    print(f"\nDone. Best val RL² = {best_val:.4f}")
    print(f"Checkpoint: {run_dir}/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",      default="data/mre_v3_50000.h5")
    parser.add_argument("--run_dir",        default="runs/phase0_v3")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--modes",          type=int,   default=12)
    parser.add_argument("--width",          type=int,   default=32)
    parser.add_argument("--n_layers",       type=int,   default=4)
    parser.add_argument("--physics_weight", type=float, default=0.05)
    train(parser.parse_args())
