#!/usr/bin/env python3
"""Figure 10 - Train a 2D FNO on the BBIR UDel slices.

First real-data supervised training in the foundation-model pipeline.
Uses NLI-computed complex G as labels (the only available source of
ground truth for brain MRE).  Subject-wise 80/10/10 train/val/test
split.

Training set: 17,535 slices x 82 subjects x 3 frequencies, masked
brain-only RL2 loss.  At 80x80 resolution, 2D FNO with width=48
modes=16 blocks=4 has ~1M parameters.

Outputs:
  paper/figures/fig10_bbir_fno2d.pt
  paper/figures/fig10_bbir_history.json
  paper/figures/fig10_bbir_loss.png
  paper/figures/fig10_bbir_predict.png
  paper/figures/fig10_bbir_summary.json
"""
from __future__ import annotations

import json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from bbir_dataset import (BBIRSliceDataset, train_val_test_split_by_subject,
                            masked_relative_l2, G_SCALE_DEFAULT)
from fno2d_brain   import FNO2dBrain, SIREN2dBrain


def evaluate(model, loader, device):
    model.eval()
    s, n = 0.0, 0
    with torch.no_grad():
        for X, Y, m in loader:
            X = X.to(device); Y = Y.to(device); m = m.to(device)
            Yp = model(X)
            l = masked_relative_l2(Yp, Y, m)
            s += float(l.item()) * len(X); n += len(X)
    return s / max(n, 1)


def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5",       default="/projects/bfid/sobh/data/bbir_udel_2d_slices.h5")
    ap.add_argument("--epochs",   type=int, default=15)
    ap.add_argument("--batch",    type=int, default=32)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--width",    type=int, default=48)
    ap.add_argument("--modes",    type=int, default=16)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--arch",     choices=["fno", "siren"], default="fno")
    ap.add_argument("--kernel",   type=int, default=7,
                    help="SIREN kernel size (FNO ignores)")
    ap.add_argument("--n_train",  type=int, default=3000,
                    help="Number of training slices to load in memory")
    ap.add_argument("--threads",  type=int, default=16)
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    figs = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  threads: {torch.get_num_threads()}")

    split = train_val_test_split_by_subject(args.h5, val_frac=0.10,
                                              test_frac=0.10, seed=args.seed)
    print(f"Splits: train={len(split.train)}  val={len(split.val)}  "
          f"test={len(split.test)} subjects")

    # Preload selected slices into memory for fast iteration -- per-batch
    # H5py reads (compressed) were the bottleneck on CPU.  We further
    # subsample the training set so an epoch fits in a few seconds.
    def _preload(subjects: list[str], cap: int | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds = BBIRSliceDataset(args.h5, subjects=subjects)
        idx = list(range(len(ds)))
        if cap is not None and cap < len(idx):
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(idx), cap, replace=False).tolist()
        Xs, Ys, Ms = [], [], []
        for i in idx:
            X, Y, M = ds[i]
            Xs.append(X); Ys.append(Y); Ms.append(M)
        ds.close()
        return torch.stack(Xs), torch.stack(Ys), torch.stack(Ms)

    print("Preloading training slices into memory ...")
    t0 = time.time()
    X_tr, Y_tr, M_tr = _preload(split.train, cap=args.n_train)
    print(f"  train: {tuple(X_tr.shape)}  ({time.time()-t0:.1f}s)")
    X_va, Y_va, M_va = _preload(split.val,   cap=500)
    print(f"  val  : {tuple(X_va.shape)}")
    X_te, Y_te, M_te = _preload(split.test,  cap=500)
    print(f"  test : {tuple(X_te.shape)}")

    from torch.utils.data import TensorDataset
    train_dl = DataLoader(TensorDataset(X_tr, Y_tr, M_tr),
                           batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl   = DataLoader(TensorDataset(X_va, Y_va, M_va),
                           batch_size=args.batch, shuffle=False)
    test_dl  = DataLoader(TensorDataset(X_te, Y_te, M_te),
                           batch_size=args.batch, shuffle=False)

    if args.arch == "fno":
        model = FNO2dBrain(in_ch=2, out_ch=2, width=args.width,
                            modes1=args.modes, modes2=args.modes,
                            n_blocks=args.n_blocks).to(device)
        cfg = f"FNO2dBrain: width={args.width}, modes={args.modes}, blocks={args.n_blocks}"
    else:
        model = SIREN2dBrain(in_ch=2, out_ch=2, width=args.width,
                              kernel_size=args.kernel,
                              n_blocks=args.n_blocks).to(device)
        cfg = f"SIREN2dBrain: width={args.width}, kernel={args.kernel}, blocks={args.n_blocks}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg},  {n_params:,} params")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-5)

    history = {"train": [], "val": []}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for X, Y, m in train_dl:
            X = X.to(device); Y = Y.to(device); m = m.to(device)
            Yp = model(X)
            loss = masked_relative_l2(Yp, Y, m)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * len(X); n += len(X)
        sched.step()
        train_loss = tot / n
        val_loss = evaluate(model, val_dl, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"  ep {ep:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    test_loss = evaluate(model, test_dl, device)
    print(f"\nFinal test masked RL2 on {len(test_ds)} held-out subject slices: "
          f"{test_loss:.4f}")

    torch.save(model.state_dict(), figs / "fig10_bbir_fno2d.pt")
    json.dump(history,    open(figs / "fig10_bbir_history.json", "w"), indent=2)
    json.dump(
        {"n_params": n_params, "test_rl2": test_loss,
         "train_subjects": split.train, "val_subjects": split.val,
         "test_subjects":  split.test,
         "n_train_used": int(X_tr.shape[0]),
         "n_val_used":   int(X_va.shape[0]),
         "n_test_used":  int(X_te.shape[0])},
        open(figs / "fig10_bbir_summary.json", "w"), indent=2)

    # Loss plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.4))
    ep_axis = np.arange(1, args.epochs + 1)
    ax.plot(ep_axis, history["train"], color="steelblue", lw=1.6, label="train")
    ax.plot(ep_axis, history["val"],   color="tomato",    lw=1.6, label="val")
    ax.axhline(test_loss, color="seagreen", ls=":", lw=1.3, label=f"test = {test_loss:.4f}")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"masked RL$^2$ on G")
    ax.set_yscale("log")
    ax.set_title(f"2D FNO on BBIR UDel  ({n_params/1e6:.2f}M params, "
                  f"{len(train_ds)} train slices)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(figs / "fig10_bbir_loss.png", dpi=160, bbox_inches="tight")
    print(f"Wrote {figs / 'fig10_bbir_loss.png'}")

    # Prediction grid on 4 held-out test samples (from preloaded tensors)
    model.eval()
    rng = np.random.default_rng(args.seed + 7)
    idx = rng.choice(len(X_te), 4, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))
    for r, i in enumerate(idx):
        X, Y, m = X_te[int(i)], Y_te[int(i)], M_te[int(i)]
        with torch.no_grad():
            Yp = model(X.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()
        Y    = Y.numpy(); m = m.numpy()[0]
        Gtre = Y[0] * G_SCALE_DEFAULT / 1000.0   # kPa
        Gtim = Y[1] * G_SCALE_DEFAULT / 1000.0
        Gpre = Yp[0] * G_SCALE_DEFAULT / 1000.0
        Gpim = Yp[1] * G_SCALE_DEFAULT / 1000.0
        Gtre = np.where(m > 0.5, Gtre, np.nan)
        Gtim = np.where(m > 0.5, Gtim, np.nan)
        Gpre = np.where(m > 0.5, Gpre, np.nan)
        Gpim = np.where(m > 0.5, Gpim, np.nan)
        for c, (img, title, vmax) in enumerate([
            (Gtre, "true Re G [kPa]",   6),
            (Gpre, "pred Re G [kPa]",   6),
            (Gtim, "true Im G [kPa]",   2),
            (Gpim, "pred Im G [kPa]",   2),
        ]):
            ax = axes[r, c]
            im = ax.imshow(img, cmap="viridis", vmin=0, vmax=vmax)
            if r == 0:
                ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("BBIR UDel test predictions (subject-disjoint from train)",
                  fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(figs / "fig10_bbir_predict.png", dpi=160, bbox_inches="tight")
    print(f"Wrote {figs / 'fig10_bbir_predict.png'}")


if __name__ == "__main__":
    main()
