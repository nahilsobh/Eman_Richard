#!/usr/bin/env python3
"""Train 1D FNO for eigenstrain inversion."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_1d import EigenstrainDataset1D
from src.model.fno_1d import FNO1d
from src.model.losses_1d import TSMLoss1D, relative_l2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/1d_pairs_20000.h5")
    p.add_argument("--run_dir",   default="runs/fno_1d")
    p.add_argument("--epochs",    type=int,   default=200)
    p.add_argument("--modes",     type=int,   default=32)
    p.add_argument("--width",     type=int,   default=64)
    p.add_argument("--n_layers",  type=int,   default=4)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--batch_size",type=int,   default=64)
    p.add_argument("--lambda_true",     type=float, default=1.0)
    p.add_argument("--lambda_analytic", type=float, default=0.1)
    p.add_argument("--lambda_pde",      type=float, default=0.0)
    p.add_argument("--lambda_expand",   type=float, default=0.2)
    return p.parse_args()


def ssim_1d(a, b, window=11):
    """Approximate SSIM for 1D signals using sliding window."""
    C1, C2 = 0.01**2, 0.03**2
    mu_a = np.convolve(a, np.ones(window)/window, mode='same')
    mu_b = np.convolve(b, np.ones(window)/window, mode='same')
    s_a  = np.convolve(a**2, np.ones(window)/window, mode='same') - mu_a**2
    s_b  = np.convolve(b**2, np.ones(window)/window, mode='same') - mu_b**2
    s_ab = np.convolve(a*b,  np.ones(window)/window, mode='same') - mu_a*mu_b
    ssim = ((2*mu_a*mu_b + C1)*(2*s_ab + C2)) / (
           (mu_a**2 + mu_b**2 + C1)*(s_a + s_b + C2))
    return float(ssim.mean())


def evaluate(model, loader, loss_fn, device):
    model.eval()
    rl2_true_list, rl2_ana_list, ssim_list = [], [], []
    rl2_exp_list = []   # RL2 vs eps_true for expanding-only cases
    labels, scores = [], []

    with torch.no_grad():
        for batch in loader:
            X       = batch["X"].to(device)
            et      = batch["eps_true"].to(device)
            ea      = batch["eps_analytic"].to(device)
            is_exp  = batch["is_expanding"].to(device)
            E_bg    = batch["E_bg"].to(device)

            eps_pred, logit = model(X)
            rl2_true_list.append(relative_l2(eps_pred, et).item())
            rl2_ana_list.append(relative_l2(eps_pred, ea).item())

            # RL2 on expanding cases only (zero-target controls skew the metric)
            exp_mask = is_exp.bool()
            if exp_mask.any():
                rl2_exp_list.append(
                    relative_l2(eps_pred[exp_mask], et[exp_mask]).item())

            # SSIM (expanding cases only — zero target gives uninformative SSIM)
            for i in range(len(X)):
                if is_exp[i].item() > 0.5:
                    ssim_list.append(ssim_1d(eps_pred[i].cpu().numpy(),
                                             et[i].cpu().numpy()))

            labels.extend(is_exp.cpu().numpy().tolist())
            scores.extend(torch.sigmoid(logit).cpu().numpy().tolist())

    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
    return {
        "rl2_true":        np.mean(rl2_true_list),
        "rl2_expanding":   np.mean(rl2_exp_list) if rl2_exp_list else float("nan"),
        "rl2_analytic":    np.mean(rl2_ana_list),
        "ssim":            np.mean(ssim_list) if ssim_list else float("nan"),
        "auc":             auc,
    }


def main():
    args = parse_args()
    run  = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Save args
    with open(run / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Dataset
    train_ds = EigenstrainDataset1D(args.data_path, split="train")
    val_ds   = EigenstrainDataset1D(args.data_path, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = FNO1d(modes=args.modes, width=args.width, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Loss and optimiser
    loss_fn = TSMLoss1D(
        lambda_true=args.lambda_true,
        lambda_analytic=args.lambda_analytic,
        lambda_pde=args.lambda_pde,
        lambda_expand=args.lambda_expand,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5)

    best_rl2 = float("inf")
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"E{epoch:03d}", leave=False):
            X      = batch["X"].to(device)
            et     = batch["eps_true"].to(device)
            ea     = batch["eps_analytic"].to(device)
            is_exp = batch["is_expanding"].to(device)
            E_bg   = batch["E_bg"].to(device)

            eps_pred, logit = model(X)
            losses = loss_fn(eps_pred, logit, et, ea, is_exp, X, E_bg)
            loss = losses["loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()

        val_metrics = evaluate(model, val_loader, loss_fn, device)
        epoch_loss /= len(train_loader)

        row = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        log_rows.append(row)

        print(f"E{epoch:03d} | loss={epoch_loss:.4f} | "
              f"rl2_exp={val_metrics['rl2_expanding']:.4f} | "
              f"rl2_all={val_metrics['rl2_true']:.4f} | "
              f"rl2_ana={val_metrics['rl2_analytic']:.4f} | "
              f"ssim={val_metrics['ssim']:.4f} | "
              f"auc={val_metrics['auc']:.4f}")

        metric = val_metrics["rl2_expanding"]
        if metric < best_rl2:
            best_rl2 = metric
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "val_metrics": val_metrics},
                       run / "best.pt")
            print(f"  → best rl2_expanding={best_rl2:.4f}")

        # Save log
        import csv
        with open(run / "logs" / "train_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

    print(f"\nBest val RL²(expanding): {best_rl2:.4f}")


if __name__ == "__main__":
    main()
