#!/usr/bin/env python3
"""Train the dual-head FNO-TSM."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import TSMDataset
from src.model.fno_tsm import FNO_TSM
from src.model.losses import (total_loss, relative_l2, masked_relative_l2,
                                ssim_metric)


def split_dataset(ds: TSMDataset, val_frac: float = 0.10):
    """Stratified split preserving expanding/control proportions.

    Reads metadata directly from the underlying HDF5 file (no augmentation).
    """
    import h5py
    with h5py.File(ds.path, "r") as f:
        is_exp = f["meta/is_expanding"][:]
    n = len(is_exp)
    rng = np.random.default_rng(0)
    idx_exp  = np.where(is_exp)[0]
    idx_ctrl = np.where(~is_exp)[0]
    rng.shuffle(idx_exp)
    rng.shuffle(idx_ctrl)
    n_val_exp  = int(round(len(idx_exp)  * val_frac))
    n_val_ctrl = int(round(len(idx_ctrl) * val_frac))
    val_idx = np.concatenate([idx_exp[:n_val_exp], idx_ctrl[:n_val_ctrl]])
    tr_idx  = np.concatenate([idx_exp[n_val_exp:], idx_ctrl[n_val_ctrl:]])
    return sorted(tr_idx.tolist()), sorted(val_idx.tolist())


def evaluate(model, loader, device):
    model.eval()
    metrics = {"L_G": [], "L_eps": [], "L_ring": [],
               "ssim_G": [], "ssim_eps": [], "auc_score": [],
               "auc_label": []}
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            G_t = batch["G"].to(device)
            eps_t = batch["epsilon"].to(device)
            ring = batch["ring"].to(device)
            is_exp = batch["is_expanding"].to(device)

            G_p, eps_p, A_p = model(X)

            metrics["L_G"].append(relative_l2(G_p, G_t).item())
            metrics["L_eps"].append(relative_l2(eps_p, eps_t).item())
            metrics["L_ring"].append(masked_relative_l2(eps_p, eps_t, ring).item())
            metrics["ssim_G"].append(ssim_metric(G_p, G_t).item())
            metrics["ssim_eps"].append(ssim_metric(eps_p, eps_t).item())
            score = (eps_p * ring).flatten(1).amax(dim=-1)
            metrics["auc_score"].append(score.cpu().numpy())
            metrics["auc_label"].append(is_exp.cpu().numpy())

    out = {k: float(np.mean(v)) for k, v in metrics.items()
           if k not in ("auc_score", "auc_label")}
    score = np.concatenate(metrics["auc_score"])
    label = np.concatenate(metrics["auc_label"])
    try:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(label, score)) \
            if len(np.unique(label)) > 1 else float("nan")
    except ImportError:
        out["auc"] = float("nan")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   default="data/tsm_50000.h5")
    p.add_argument("--run_dir",     default="runs/tsm")
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=24)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--modes",       type=int, default=12)
    p.add_argument("--width",       type=int, default=48)
    p.add_argument("--n_layers",    type=int, default=4)
    p.add_argument("--lambda_eps",      type=float, default=1.0)
    p.add_argument("--lambda_ring",     type=float, default=0.5)
    p.add_argument("--lambda_acoustic", type=float, default=0.0)
    p.add_argument("--lambda_pde",      type=float, default=0.0)
    p.add_argument("--lambda_expand",   type=float, default=0.2)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--in_channels",     type=int, default=4,
                   help="Input channels: 4 for 80Hz (Re/Im+Lame+dist), 6 for 60+120Hz")
    p.add_argument("--resume",          action="store_true",
                   help="Resume from best.pt in run_dir if it exists")
    args = p.parse_args()

    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    full_ds = TSMDataset(args.data_path, augment=False)
    tr_idx, val_idx = split_dataset(full_ds, val_frac=0.10)
    print(f"Train: {len(tr_idx):,}   Val: {len(val_idx):,}", flush=True)

    tr_ds  = TSMDataset(args.data_path, augment=True)
    val_ds = TSMDataset(args.data_path, augment=False)

    from torch.utils.data import Subset
    tr_loader  = DataLoader(Subset(tr_ds, tr_idx),  batch_size=args.batch_size,
                             shuffle=True,  num_workers=args.num_workers,
                             pin_memory=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size,
                             shuffle=False, num_workers=max(1, args.num_workers // 2),
                             pin_memory=True)

    model = FNO_TSM(in_channels=args.in_channels, modes1=args.modes,
                     modes2=args.modes, width=args.width,
                     n_layers=args.n_layers).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_p:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    cfg = vars(args).copy(); cfg["n_params"] = n_p; cfg["device"] = str(device)
    json.dump(cfg, open(run_dir / "args.json", "w"), indent=2)

    history = {k: [] for k in
               ("train_L_total", "train_L_G", "train_L_eps", "train_L_ring",
                "val_L_G", "val_L_eps", "val_L_ring",
                "val_ssim_G", "val_ssim_eps", "val_auc")}
    best_ring = float("inf")
    start_epoch = 1

    resume_path = run_dir / "best.pt"
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        best_ring = ckpt["val"].get("L_ring", float("inf"))
        history_path = run_dir / "history.json"
        if history_path.exists():
            history = json.load(open(history_path))
        # Fast-forward scheduler to correct position
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']}, best_ring={best_ring:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        e_total, e_G, e_eps, e_ring = [], [], [], []
        for batch in tr_loader:
            X = batch["X"].to(device, non_blocking=True)
            G_t = batch["G"].to(device, non_blocking=True)
            eps_t = batch["epsilon"].to(device, non_blocking=True)
            ring = batch["ring"].to(device, non_blocking=True)
            G_bg = batch["G_bg"].to(device, non_blocking=True)
            is_exp = batch["is_expanding"].to(device, non_blocking=True)

            G_p, eps_p, A_p = model(X)
            u_re, u_im = X[:, 0], X[:, 1]
            L, parts = total_loss(
                G_p, eps_p, A_p, G_t, eps_t, ring, G_bg, is_exp, u_re, u_im,
                lambda_eps=args.lambda_eps,
                lambda_ring=args.lambda_ring,
                lambda_acoustic=args.lambda_acoustic,
                lambda_pde=args.lambda_pde,
                lambda_expand=args.lambda_expand,
            )
            optimizer.zero_grad()
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            e_total.append(parts["L_total"].item())
            e_G.append(parts["L_G"].item())
            e_eps.append(parts["L_eps"].item())
            e_ring.append(parts["L_ring"].item())
        scheduler.step()

        val = evaluate(model, val_loader, device)

        history["train_L_total"].append(float(np.mean(e_total)))
        history["train_L_G"].append(float(np.mean(e_G)))
        history["train_L_eps"].append(float(np.mean(e_eps)))
        history["train_L_ring"].append(float(np.mean(e_ring)))
        for k in ("L_G", "L_eps", "L_ring", "ssim_G", "ssim_eps"):
            history["val_" + k].append(float(val[k]))
        history["val_auc"].append(float(val["auc"]))
        json.dump(history, open(run_dir / "history.json", "w"), indent=2)

        elapsed = time.time() - t0
        print(f"Ep {epoch:03d}/{args.epochs} | "
              f"train_total={np.mean(e_total):.4f} L_G={np.mean(e_G):.4f} "
              f"L_eps={np.mean(e_eps):.4f} L_ring={np.mean(e_ring):.4f} | "
              f"val_L_G={val['L_G']:.4f} val_L_ring={val['L_ring']:.4f} "
              f"val_AUC={val['auc']:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.0f}s)",
              flush=True)

        if val["L_ring"] < best_ring:
            best_ring = val["L_ring"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val": val}, run_dir / "best.pt")
            print(f"  → best val_L_ring={val['L_ring']:.4f}", flush=True)

    print(f"\nDone. Best val_L_ring = {best_ring:.4f}")


if __name__ == "__main__":
    main()
