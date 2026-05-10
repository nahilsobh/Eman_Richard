#!/usr/bin/env python3
"""Validation figures for the 1D FNO eigenstrain inversion model.

Generates five panels saved to <run_dir>/figures/:
  1. learning_curves.png  — train loss + val RL²(exp) + AUC vs epoch
  2. pred_examples.png    — 8 example eigenstrain predictions
  3. snr_curve.png        — RL²(expanding) vs SNR dB
  4. param_scatter.png    — predicted vs true eps0, A_coeff, E_bg scatter
  5. roc_curve.png        — ROC for expanding vs control classification
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc as sklearn_auc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_1d import EigenstrainDataset1D
from src.model.fno_1d import FNO1d


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(run_dir: Path, device: torch.device) -> FNO1d:
    args_path = run_dir / "args.json"
    ckpt_path = run_dir / "best.pt"
    with open(args_path) as f:
        args = json.load(f)
    model = FNO1d(modes=args["modes"], width=args["width"],
                  n_layers=args["n_layers"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device, meta: dict | None = None):
    """Returns arrays over the full validation set.

    meta: optional dict of pre-loaded numpy arrays (snr_db, eps0, A_coeff)
          keyed by name, each of length = len(val_ds).
    """
    eps_preds, eps_trues, eps_anas = [], [], []
    logits, is_exps, E_bgs = [], [], []

    for batch in loader:
        X      = batch["X"].to(device)
        et     = batch["eps_true"].to(device)
        ea     = batch["eps_analytic"].to(device)
        is_exp = batch["is_expanding"]
        E_bg   = batch["E_bg"]

        ep, logit = model(X)
        eps_preds.append(ep.cpu())
        eps_trues.append(et.cpu())
        eps_anas.append(ea.cpu())
        logits.append(logit.cpu())
        is_exps.append(is_exp)
        E_bgs.append(E_bg)

    res = {
        "eps_pred":    torch.cat(eps_preds).numpy(),
        "eps_true":    torch.cat(eps_trues).numpy(),
        "eps_analytic":torch.cat(eps_anas).numpy(),
        "logit":       torch.cat(logits).numpy(),
        "is_exp":      torch.cat(is_exps).numpy().astype(bool),
        "E_bg":        torch.cat(E_bgs).numpy(),
        "snr_db":      None,
        "eps0":        None,
        "A_coeff":     None,
    }
    if meta:
        res.update(meta)
    return res


# ── figure 1: learning curves ─────────────────────────────────────────────

def fig_learning_curves(log_csv: Path, out: Path):
    import csv
    rows = []
    with open(log_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    epochs   = [int(r["epoch"])           for r in rows]
    tr_loss  = [float(r["train_loss"])    for r in rows]
    rl2_exp  = [float(r.get("val_rl2_expanding", r.get("val_rl2_true", "nan"))) for r in rows]
    auc      = [float(r["val_auc"])       for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].semilogy(epochs, tr_loss, "b-", lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss (log scale)")
    axes[0].set_title("Training loss")

    axes[1].plot(epochs, rl2_exp, "r-", lw=1.5)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RL² (expanding only)")
    axes[1].set_title("Val RL² — expanding cases")
    axes[1].axhline(0.1, ls="--", color="gray", lw=1, label="0.1 target")
    axes[1].legend(fontsize=8)

    axes[2].plot(epochs, auc, "g-", lw=1.5)
    axes[2].set_ylim(0.4, 1.02)
    axes[2].axhline(1.0, ls="--", color="gray", lw=1)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUC")
    axes[2].set_title("Val AUC — expanding vs control")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── figure 2: example predictions ────────────────────────────────────────

def fig_pred_examples(res: dict, out: Path, n: int = 8):
    exp_idx = np.where(res["is_exp"])[0]
    rng = np.random.default_rng(42)
    chosen = rng.choice(exp_idx, size=min(n, len(exp_idx)), replace=False)
    chosen = sorted(chosen)

    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 2 * len(chosen)),
                             sharex=True)
    if len(chosen) == 1:
        axes = [axes]

    x_ax = np.linspace(0, 1, res["eps_pred"].shape[1])
    for ax, i in zip(axes, chosen):
        ax.plot(x_ax, res["eps_true"][i],     "k-",  lw=1.5, label="True ε*")
        ax.plot(x_ax, res["eps_pred"][i],     "r--", lw=1.5, label="FNO pred")
        ax.plot(x_ax, res["eps_analytic"][i], "b:",  lw=1.0, label="Analytic")
        rl2 = float(np.linalg.norm(res["eps_pred"][i] - res["eps_true"][i]) /
                    (np.linalg.norm(res["eps_true"][i]) + 1e-8))
        ax.set_ylabel(f"ε* (RL²={rl2:.3f})", fontsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("x / L")
    fig.suptitle("Eigenstrain predictions — expanding cases", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── figure 3: RL² vs SNR ─────────────────────────────────────────────────

def fig_snr_curve(res: dict, out: Path):
    if res["snr_db"] is None:
        print("Skipping snr_curve: no snr_db in dataset")
        return

    mask = res["is_exp"]
    snr  = res["snr_db"][mask]
    pred = res["eps_pred"][mask]
    true = res["eps_true"][mask]

    rl2_per = (np.linalg.norm(pred - true, axis=1)
               / (np.linalg.norm(true, axis=1) + 1e-8))

    # bin by SNR
    edges = np.arange(0, 55, 5)
    centers, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (snr >= lo) & (snr < hi)
        if sel.sum() > 0:
            centers.append((lo + hi) / 2)
            means.append(rl2_per[sel].mean())
            stds.append(rl2_per[sel].std())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(centers, means, yerr=stds, fmt="o-", capsize=4, color="steelblue")
    ax.axhline(0.1, ls="--", color="gray", lw=1, label="RL²=0.1")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("RL² (pred vs true ε*)")
    ax.set_title("Prediction accuracy vs noise level")
    ax.legend(); ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── figure 4: parameter scatter ──────────────────────────────────────────

def fig_param_scatter(res: dict, out: Path):
    """Peak ε* from FNO pred vs true eps0, A*eps0, E_bg."""
    mask = res["is_exp"]
    peak_pred = res["eps_pred"][mask].max(axis=1)
    peak_true = res["eps_true"][mask].max(axis=1)

    panels = [("Peak ε* pred vs true", peak_true, peak_pred, "True peak ε*", "FNO peak ε*")]

    if res["eps0"] is not None:
        panels.append(("Peak ε* pred vs eps0", res["eps0"][mask], peak_pred,
                        "eps0 (true)", "FNO peak ε*"))
    if res["A_coeff"] is not None and res["eps0"] is not None:
        Aeps = res["A_coeff"][mask] * res["eps0"][mask]
        panels.append(("Peak ε* pred vs A·eps0", Aeps, peak_pred,
                        "A·eps0 (true)", "FNO peak ε*"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, xv, yv, xl, yl) in zip(axes, panels):
        ax.scatter(xv, yv, alpha=0.3, s=8, color="steelblue")
        lo, hi = min(xv.min(), yv.min()), max(xv.max(), yv.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        r = float(np.corrcoef(xv, yv)[0, 1])
        ax.set_title(f"{title}\nr={r:.3f}")
        ax.set_xlabel(xl); ax.set_ylabel(yl)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── figure 5: ROC curve ───────────────────────────────────────────────────

def fig_roc(res: dict, out: Path):
    scores = 1 / (1 + np.exp(-res["logit"]))   # sigmoid
    fpr, tpr, _ = roc_curve(res["is_exp"].astype(int), scores)
    roc_auc = sklearn_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — expanding vs control (val set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir",   default="runs/fno_1d_full")
    p.add_argument("--data_path", default="data/1d_pairs_20000.h5")
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    run  = Path(args.run_dir)
    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(run, device)
    print(f"Loaded checkpoint: {run / 'best.pt'}")

    val_ds = EigenstrainDataset1D(args.data_path, split="val")
    print(f"Val samples: {len(val_ds)}")

    # Load extra metadata directly (DataLoader special-method lookup bypasses
    # instance-attribute monkey-patches, so we load outside the dataset class)
    import h5py
    meta = {}
    with h5py.File(args.data_path, "r") as f:
        n = f["X"].shape[0]
        sl = slice(int(n * 0.9), n)
        for key in ("snr_db", "eps0", "A_coeff"):
            if f"meta/{key}" in f:
                meta[key] = f[f"meta/{key}"][sl].astype(np.float32)

    loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    print("Running inference...")
    res = run_inference(model, loader, device, meta=meta)

    # Summary stats
    mask = res["is_exp"]
    rl2_exp = float(np.mean(
        np.linalg.norm(res["eps_pred"][mask] - res["eps_true"][mask], axis=1)
        / (np.linalg.norm(res["eps_true"][mask], axis=1) + 1e-8)))
    scores = 1 / (1 + np.exp(-res["logit"]))
    from sklearn.metrics import roc_auc_score
    auc_val = roc_auc_score(res["is_exp"].astype(int), scores)
    print(f"\nVal RL²(expanding): {rl2_exp:.4f}")
    print(f"Val AUC:            {auc_val:.4f}")

    # Generate figures
    log_csv = run / "logs" / "train_log.csv"
    if log_csv.exists():
        fig_learning_curves(log_csv, fig_dir / "learning_curves.png")

    fig_pred_examples(res, fig_dir / "pred_examples.png")
    fig_snr_curve(res,     fig_dir / "snr_curve.png")
    fig_param_scatter(res, fig_dir / "param_scatter.png")
    fig_roc(res,           fig_dir / "roc_curve.png")

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
