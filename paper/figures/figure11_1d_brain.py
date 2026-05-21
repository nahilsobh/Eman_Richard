#!/usr/bin/env python3
"""Figure 11 - Train SIREN1d on real BBIR brain voxel lines.

First real-data training in the 1D foundation-model pipeline.  Takes
1D voxel lines extracted from the BBIR UDel 3D MRE volumes (via
bbir_ingest_1d.py), trains the same SIREN1d architecture as Figs 2/5/9
with masked relative-L^2 against NLI ground-truth G, and evaluates on
held-out subjects.

If this works (held-out RL^2 << 0.5), the foundation-model direction is
viable: the 1D operator handles real brain MRE despite the 1D
approximation of a 3D wave equation.

Output:
  fig11_1d_brain.pt
  fig11_1d_brain_history.json
  fig11_1d_brain_summary.json
  fig11_1d_brain_loss.png
  fig11_1d_brain_predict.png
"""
from __future__ import annotations

import json, time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from figure5_siren_compare import SIREN1d


SRC_DEFAULT = "/projects/bfid/sobh/data/bbir_udel_1d_lines.h5"
G_SCALE = 5000.0


def load_h5_subset(src_path: str, keep_subjects: set[str],
                   freqs: list[int] | None = None,
                   cap: int | None = None, seed: int = 0,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                              np.ndarray, np.ndarray]:
    """Load a subset of the BBIR 1D HDF5 into memory.

    Returns
    -------
    X    : (N, 2, L) float32  -- complex displacement, ||u||_inf=1 per sample
    Y    : (N, 2, L) float32  -- complex G / G_scale
    mask : (N,   L)  bool     -- brain mask
    subj : (N,) str          -- subject id per line
    freq : (N,) int          -- driving frequency per line (Hz)
    """
    with h5py.File(src_path, "r") as h:
        sbj = np.array(h["meta/subject"][:], dtype="U32")
        frq = np.array(h["meta/freq"][:],    dtype=np.int32)
        keep_mask = np.isin(sbj, np.array(list(keep_subjects), dtype="U32"))
        if freqs is not None:
            keep_mask &= np.isin(frq, np.array(list(freqs), dtype=np.int32))
        idx = np.where(keep_mask)[0]
        if cap is not None and len(idx) > cap:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, cap, replace=False))
        # Random access via fancy indexing on h5py is OK for a few thousand
        X_raw = np.array(h["X"][idx],    dtype=np.float32)
        Y_raw = np.array(h["Y"][idx],    dtype=np.float32)
        mask  = np.array(h["mask"][idx], dtype=np.bool_)
        sbj   = sbj[idx]
        frq   = frq[idx]
    # Per-sample normalisation so ||u||_inf = 1
    amax = np.abs(X_raw).reshape(len(idx), -1).max(axis=1)
    amax = np.where(amax > 0, amax, 1.0)
    X = X_raw / amax[:, None, None]
    Y = Y_raw / G_SCALE
    return (torch.from_numpy(X), torch.from_numpy(Y),
            torch.from_numpy(mask), sbj, frq)


def masked_mse_1d(Y_pred: torch.Tensor, Y_true: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Per-pixel MSE on (B, 2, L) prediction inside the (B, L) brain mask."""
    m = mask.unsqueeze(1).float()
    diff = (Y_pred - Y_true) * m
    return (diff ** 2).sum() / m.sum().clamp(min=1.0)


def masked_rl2_1d(Y_pred: torch.Tensor, Y_true: torch.Tensor,
                  mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Mean of per-sample masked relative L^2 (used only as a metric)."""
    m = mask.unsqueeze(1).float()
    diff = (Y_pred - Y_true) * m
    tgt  =  Y_true * m
    num  = torch.linalg.vector_norm(diff.flatten(1), dim=-1)
    den  = torch.linalg.vector_norm(tgt.flatten(1),  dim=-1).clamp_min(eps)
    return (num / den).mean()


def subject_split(src_path: str, val_frac: float = 0.10,
                   test_frac: float = 0.10, seed: int = 0,
                   ) -> tuple[list[str], list[str], list[str]]:
    with h5py.File(src_path, "r") as h:
        all_sbj = sorted(set(h["meta/subject"][:].astype("U32").tolist()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_sbj))
    n_val  = max(1, int(round(len(all_sbj) * val_frac)))
    n_test = max(1, int(round(len(all_sbj) * test_frac)))
    val   = [all_sbj[i] for i in perm[:n_val]]
    test  = [all_sbj[i] for i in perm[n_val:n_val + n_test]]
    train = [all_sbj[i] for i in perm[n_val + n_test:]]
    return sorted(train), sorted(val), sorted(test)


def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",       default=SRC_DEFAULT)
    ap.add_argument("--n_train",   type=int, default=5000)
    ap.add_argument("--n_val",     type=int, default=1000)
    ap.add_argument("--n_test",    type=int, default=1000)
    ap.add_argument("--epochs",    type=int, default=20)
    ap.add_argument("--batch",     type=int, default=64)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--width",     type=int, default=32)
    ap.add_argument("--kernel",    type=int, default=11)
    ap.add_argument("--n_blocks",  type=int, default=4)
    ap.add_argument("--threads",   type=int, default=16)
    ap.add_argument("--seed",      type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    figs = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  threads: {torch.get_num_threads()}")

    train_subj, val_subj, test_subj = subject_split(args.src, seed=args.seed)
    print(f"Subject split: train={len(train_subj)}, val={len(val_subj)}, "
          f"test={len(test_subj)}")

    t0 = time.time()
    print("Loading 1D lines into memory ...")
    X_tr, Y_tr, M_tr, _, _ = load_h5_subset(args.src, set(train_subj),
                                              cap=args.n_train, seed=args.seed)
    X_va, Y_va, M_va, _, _ = load_h5_subset(args.src, set(val_subj),
                                              cap=args.n_val, seed=args.seed + 1)
    X_te, Y_te, M_te, sb_te, fr_te = load_h5_subset(
        args.src, set(test_subj), cap=args.n_test, seed=args.seed + 2)
    print(f"  train: {tuple(X_tr.shape)}  val: {tuple(X_va.shape)}  "
          f"test: {tuple(X_te.shape)}   ({time.time() - t0:.1f}s)")

    train_dl = DataLoader(TensorDataset(X_tr, Y_tr, M_tr),
                           batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl   = DataLoader(TensorDataset(X_va, Y_va, M_va),
                           batch_size=args.batch, shuffle=False)
    test_dl  = DataLoader(TensorDataset(X_te, Y_te, M_te),
                           batch_size=args.batch, shuffle=False)

    model = SIREN1d(in_ch=2, out_ch=2, width=args.width,
                    kernel_size=args.kernel, n_blocks=args.n_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SIREN1d: width={args.width}, kernel={args.kernel}, "
          f"blocks={args.n_blocks},  {n_params:,} params")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-5)

    def eval_loader(dl):
        model.eval()
        mse_s, rl2_s, n = 0.0, 0.0, 0
        with torch.no_grad():
            for X, Y, m in dl:
                X = X.to(device); Y = Y.to(device); m = m.to(device)
                Yp = model(X)
                mse_s += float(masked_mse_1d(Yp, Y, m).item()) * len(X)
                rl2_s += float(masked_rl2_1d(Yp, Y, m).item()) * len(X)
                n += len(X)
        return mse_s / max(n, 1), rl2_s / max(n, 1)

    history = {"train_mse": [], "train_rl2": [],
               "val_mse":   [], "val_rl2":   []}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        mse_s, rl2_s, n = 0.0, 0.0, 0
        for X, Y, m in train_dl:
            X = X.to(device); Y = Y.to(device); m = m.to(device)
            Yp = model(X)
            loss = masked_mse_1d(Yp, Y, m)               # MSE is the training objective
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                rl2 = masked_rl2_1d(Yp, Y, m)
            mse_s += float(loss.item())   * len(X)
            rl2_s += float(rl2.item())    * len(X)
            n += len(X)
        sched.step()
        train_mse, train_rl2 = mse_s / n, rl2_s / n
        val_mse,   val_rl2   = eval_loader(val_dl)
        history["train_mse"].append(train_mse)
        history["train_rl2"].append(train_rl2)
        history["val_mse"].append(val_mse)
        history["val_rl2"].append(val_rl2)
        print(f"  ep {ep:3d}  train MSE={train_mse:.4f}  RL2={train_rl2:.4f}  "
              f"|  val MSE={val_mse:.4f}  RL2={val_rl2:.4f}  "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    test_mse, test_rl2 = eval_loader(test_dl)
    test_loss = test_rl2
    print(f"\nFinal test masked RL2 on {len(X_te)} held-out subject lines: "
          f"{test_loss:.4f}")

    # Save
    torch.save(model.state_dict(), figs / "fig11_1d_brain.pt")
    json.dump(history, open(figs / "fig11_1d_brain_history.json", "w"), indent=2)
    json.dump({
        "n_params": n_params,
        "test_rl2": test_loss,
        "train_subjects": train_subj,
        "val_subjects":   val_subj,
        "test_subjects":  test_subj,
        "n_train_used":   int(X_tr.shape[0]),
        "n_val_used":     int(X_va.shape[0]),
        "n_test_used":    int(X_te.shape[0]),
    }, open(figs / "fig11_1d_brain_summary.json", "w"), indent=2)

    # Loss plot: two panels (MSE training objective + RL² metric)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ep_axis = np.arange(1, args.epochs + 1)
    axes[0].plot(ep_axis, history["train_mse"], color="steelblue", lw=1.6, label="train")
    axes[0].plot(ep_axis, history["val_mse"],   color="tomato",    lw=1.6, label="val")
    axes[0].axhline(test_mse, color="seagreen", ls=":", lw=1.3,
                     label=f"test MSE = {test_mse:.4f}")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("masked MSE on G/G_scale")
    axes[0].set_yscale("log"); axes[0].set_title("Training objective")
    axes[0].legend(loc="upper right", fontsize=9); axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(ep_axis, history["train_rl2"], color="steelblue", lw=1.6, label="train")
    axes[1].plot(ep_axis, history["val_rl2"],   color="tomato",    lw=1.6, label="val")
    axes[1].axhline(test_rl2, color="seagreen", ls=":", lw=1.3,
                     label=f"test RL2 = {test_rl2:.4f}")
    axes[1].axhline(0.36, color="black", ls="--", lw=1.0, alpha=0.5,
                     label="constant-mean baseline = 0.36")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel(r"masked RL$^2$ on G")
    axes[1].set_title("Reported metric")
    axes[1].legend(loc="upper right", fontsize=9); axes[1].grid(alpha=0.3)

    fig.suptitle(f"SIREN1d on BBIR UDel 1D voxel lines  ({n_params/1e3:.0f}k params, "
                  f"{len(X_tr)} train lines)", fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(figs / "fig11_1d_brain_loss.png", dpi=160, bbox_inches="tight")
    print(f"Wrote {figs / 'fig11_1d_brain_loss.png'}")

    # Prediction figure: 6 held-out test lines
    rng = np.random.default_rng(args.seed + 11)
    idx = rng.choice(len(X_te), 6, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.0), sharex=True, sharey=False)
    L = X_te.shape[-1]
    x_axis = np.arange(L)
    model.eval()
    for k, ax in enumerate(axes.flat):
        i = int(idx[k])
        X = X_te[i].unsqueeze(0).to(device)
        with torch.no_grad():
            Yp = model(X).cpu().numpy()[0] * G_SCALE
        Yt = Y_te[i].numpy() * G_SCALE
        m  = M_te[i].numpy().astype(bool)
        Yt_re = np.where(m, Yt[0] / 1000, np.nan)
        Yt_im = np.where(m, Yt[1] / 1000, np.nan)
        Yp_re = np.where(m, Yp[0] / 1000, np.nan)
        Yp_im = np.where(m, Yp[1] / 1000, np.nan)
        ax.plot(x_axis, Yt_re, color="black", lw=1.6, label=r"true Re$\,G$")
        ax.plot(x_axis, Yp_re, color="steelblue", lw=1.4, ls="--",
                 label=r"pred Re$\,G$")
        ax.plot(x_axis, Yt_im, color="gray", lw=1.0, alpha=0.6)
        ax.plot(x_axis, Yp_im, color="tomato", lw=1.4, ls="--",
                 label=r"pred Im$\,G$")
        ax.set_title(f"{sb_te[i]} {fr_te[i]} Hz", fontsize=9)
        ax.set_xlabel("voxel along line"); ax.set_ylabel("G [kPa]")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Held-out 1D voxel-line predictions on real brain MRE",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(figs / "fig11_1d_brain_predict.png", dpi=160, bbox_inches="tight")
    print(f"Wrote {figs / 'fig11_1d_brain_predict.png'}")


if __name__ == "__main__":
    main()
