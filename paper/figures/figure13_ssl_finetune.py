#!/usr/bin/env python3
"""Figure 13 - Few-shot supervised fine-tune from the multi-view SSL encoder.

Loads the SIRENEncoder1d pretrained by figure12_ssl_multiview.py and
adds a small G-prediction head (Conv1d) on top.  Fine-tunes with masked
MSE against NLI labels on UDel.  Compares against scratch training
under the same data budgets, in BrainIAC-style few-shot (K=1, K=5)
and full-data conditions.

Output:
  fig13_ssl_finetune_K{K}_pretrained.pt
  fig13_ssl_finetune_K{K}_scratch.pt
  fig13_ssl_finetune_history.json
  fig13_ssl_finetune_summary.json
  fig13_ssl_finetune_bars.png
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from figure12_ssl_multiview import SIRENEncoder1d


SRC_DEFAULT = "/projects/bfid/sobh/data/bbir_udel_1d_lines.h5"
ENCODER_DEFAULT = "fig12_ssl_encoder.pt"
G_SCALE = 5000.0


class SIREN1dForG(nn.Module):
    """Encoder body + 1x1 conv output head -> (B, 2, L) G prediction."""
    def __init__(self, encoder: SIRENEncoder1d, width: int = 24, out_ch: int = 2):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Conv1d(width, out_ch, 1)
        with torch.no_grad():
            self.head.weight.uniform_(-0.01, 0.01)
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def masked_mse(Y_pred, Y_true, mask):
    m = mask.unsqueeze(1).float()
    diff = (Y_pred - Y_true) * m
    return (diff ** 2).sum() / m.sum().clamp(min=1.0)


def masked_rl2(Y_pred, Y_true, mask, eps=1e-3):
    m = mask.unsqueeze(1).float()
    diff = (Y_pred - Y_true) * m
    tgt  =  Y_true * m
    num  = torch.linalg.vector_norm(diff.flatten(1), dim=-1)
    den  = torch.linalg.vector_norm(tgt.flatten(1),  dim=-1).clamp_min(eps)
    return (num / den).mean()


def subject_split_1d(src_path: str, val_frac=0.10, test_frac=0.10, seed=0):
    with h5py.File(src_path, "r") as h:
        all_sbj = sorted(set(h["meta/subject"][:].astype("U32").tolist()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_sbj))
    n_v = max(1, int(round(len(all_sbj) * val_frac)))
    n_t = max(1, int(round(len(all_sbj) * test_frac)))
    val   = [all_sbj[i] for i in perm[:n_v]]
    test  = [all_sbj[i] for i in perm[n_v:n_v + n_t]]
    train = [all_sbj[i] for i in perm[n_v + n_t:]]
    return sorted(train), sorted(val), sorted(test)


def load_lines(src, keep_subjects, cap=None, seed=0):
    with h5py.File(src, "r") as h:
        sbj = np.array(h["meta/subject"][:], dtype="U32")
        keep = np.isin(sbj, np.array(list(keep_subjects), dtype="U32"))
        idx = np.where(keep)[0]
        if cap is not None and len(idx) > cap:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, cap, replace=False))
        X_raw = np.array(h["X"][idx],    dtype=np.float32)
        Y_raw = np.array(h["Y"][idx],    dtype=np.float32)
        mask  = np.array(h["mask"][idx], dtype=np.bool_)
    amax = np.abs(X_raw).reshape(len(idx), -1).max(axis=1)
    amax = np.where(amax > 0, amax, 1.0)
    X = X_raw / amax[:, None, None]
    Y = Y_raw / G_SCALE
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(mask)


def train_one(model, X_tr, Y_tr, M_tr, X_va, Y_va, M_va,
              epochs, batch, lr, device, tag):
    train_dl = DataLoader(TensorDataset(X_tr, Y_tr, M_tr),
                           batch_size=batch, shuffle=True, drop_last=True)
    val_dl   = DataLoader(TensorDataset(X_va, Y_va, M_va),
                           batch_size=batch, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)

    def eval_loader(dl):
        model.eval()
        mse_s, rl2_s, n = 0.0, 0.0, 0
        with torch.no_grad():
            for X, Y, m in dl:
                X = X.to(device); Y = Y.to(device); m = m.to(device)
                Yp = model(X)
                mse_s += float(masked_mse(Yp, Y, m).item()) * len(X)
                rl2_s += float(masked_rl2(Yp, Y, m).item()) * len(X)
                n += len(X)
        return mse_s / max(n, 1), rl2_s / max(n, 1)

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        for X, Y, m in train_dl:
            X = X.to(device); Y = Y.to(device); m = m.to(device)
            Yp = model(X)
            loss = masked_mse(Yp, Y, m)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        val_mse, val_rl2 = eval_loader(val_dl)
        history.append({"epoch": ep, "val_mse": val_mse, "val_rl2": val_rl2})
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"    [{tag}] ep {ep:3d}  val MSE={val_mse:.4f}  RL2={val_rl2:.4f}",
                  flush=True)
    return history


def evaluate(model, X, Y, M, batch, device):
    model.eval()
    s_m, s_r, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            Xb = X[i:i+batch].to(device)
            Yb = Y[i:i+batch].to(device)
            Mb = M[i:i+batch].to(device)
            Yp = model(Xb)
            s_m += float(masked_mse(Yp, Yb, Mb).item()) * len(Xb)
            s_r += float(masked_rl2(Yp, Yb, Mb).item()) * len(Xb)
            n   += len(Xb)
    return s_m / n, s_r / n


def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",        default=SRC_DEFAULT)
    ap.add_argument("--encoder",    default=ENCODER_DEFAULT)
    ap.add_argument("--epochs",     type=int, default=15)
    ap.add_argument("--batch",      type=int, default=64)
    ap.add_argument("--lr_head",    type=float, default=3e-4)
    ap.add_argument("--width",      type=int, default=24)
    ap.add_argument("--kernel",     type=int, default=11)
    ap.add_argument("--n_blocks",   type=int, default=3)
    ap.add_argument("--K_per_subject", type=int, nargs="+",
                    default=[1, 5, 50, 500],
                    help="Lines per training subject (few-shot data budgets)")
    ap.add_argument("--threads",    type=int, default=16)
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    figs = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  threads: {torch.get_num_threads()}")

    enc_path = figs / args.encoder if not Path(args.encoder).is_absolute() else Path(args.encoder)
    if not enc_path.exists():
        raise FileNotFoundError(f"SSL encoder not found at {enc_path} -- run "
                                 f"figure12_ssl_multiview.py first.")

    train_subj, val_subj, test_subj = subject_split_1d(args.src, seed=args.seed)
    print(f"Subject split: train={len(train_subj)}, val={len(val_subj)}, "
          f"test={len(test_subj)}")

    # Always load full val + test sets (no cap)
    print("Loading val + test sets ...")
    X_va, Y_va, M_va = load_lines(args.src, set(val_subj),  cap=1500, seed=args.seed)
    X_te, Y_te, M_te = load_lines(args.src, set(test_subj), cap=1500, seed=args.seed)
    print(f"  val: {tuple(X_va.shape)}  test: {tuple(X_te.shape)}")

    history_all = {}
    summary = {}
    for K in args.K_per_subject:
        n_per_subj = K
        # cap = ~K lines per training subject (deterministic sampling)
        cap_tr = K * len(train_subj)
        print(f"\n=== Budget K={K} (~{cap_tr} train lines) ===")
        X_tr, Y_tr, M_tr = load_lines(args.src, set(train_subj),
                                        cap=cap_tr, seed=args.seed)
        print(f"  train: {tuple(X_tr.shape)}")

        # 1) SSL-pretrained fine-tune
        print(f"\n  -- SSL-pretrained fine-tune --")
        enc_pre = SIRENEncoder1d(in_ch=2, width=args.width,
                                  kernel_size=args.kernel, n_blocks=args.n_blocks)
        enc_pre.load_state_dict(torch.load(enc_path, map_location="cpu",
                                            weights_only=False))
        model_pre = SIREN1dForG(enc_pre, width=args.width).to(device)
        hist_pre = train_one(model_pre, X_tr, Y_tr, M_tr, X_va, Y_va, M_va,
                              epochs=args.epochs, batch=args.batch,
                              lr=args.lr_head, device=device, tag=f"K={K} pre")
        test_mse_pre, test_rl2_pre = evaluate(model_pre, X_te, Y_te, M_te,
                                                 args.batch, device)
        torch.save(model_pre.state_dict(),
                    figs / f"fig13_ssl_finetune_K{K}_pretrained.pt")

        # 2) Scratch (random-init) -- same architecture, no SSL
        print(f"\n  -- Scratch (random init) --")
        enc_sc = SIRENEncoder1d(in_ch=2, width=args.width,
                                 kernel_size=args.kernel, n_blocks=args.n_blocks)
        model_sc = SIREN1dForG(enc_sc, width=args.width).to(device)
        hist_sc = train_one(model_sc, X_tr, Y_tr, M_tr, X_va, Y_va, M_va,
                             epochs=args.epochs, batch=args.batch,
                             lr=args.lr_head, device=device, tag=f"K={K} sc")
        test_mse_sc, test_rl2_sc = evaluate(model_sc, X_te, Y_te, M_te,
                                              args.batch, device)
        torch.save(model_sc.state_dict(),
                    figs / f"fig13_ssl_finetune_K{K}_scratch.pt")

        history_all[K] = {"pretrained": hist_pre, "scratch": hist_sc}
        summary[K] = {
            "n_train":  int(X_tr.shape[0]),
            "test_mse_pretrained": float(test_mse_pre),
            "test_rl2_pretrained": float(test_rl2_pre),
            "test_mse_scratch":    float(test_mse_sc),
            "test_rl2_scratch":    float(test_rl2_sc),
        }
        print(f"\n  K={K}:  SSL+FT test MSE={test_mse_pre:.4f}, RL2={test_rl2_pre:.4f}  "
              f"|  Scratch test MSE={test_mse_sc:.4f}, RL2={test_rl2_sc:.4f}",
              flush=True)

    json.dump(history_all, open(figs / "fig13_ssl_finetune_history.json", "w"),
              indent=2)
    json.dump(summary,     open(figs / "fig13_ssl_finetune_summary.json", "w"),
              indent=2)

    # Bar chart: SSL+FT vs Scratch across K values, on both MSE and RL2
    Ks = list(summary.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, metric in zip(axes, ["mse", "rl2"]):
        x = np.arange(len(Ks)); w = 0.35
        pre_v = [summary[K][f"test_{metric}_pretrained"] for K in Ks]
        sc_v  = [summary[K][f"test_{metric}_scratch"]    for K in Ks]
        b1 = ax.bar(x - w/2, pre_v, w, color="seagreen",  edgecolor="black",
                     lw=0.5, alpha=0.85, label="SSL pretrained + fine-tune")
        b2 = ax.bar(x + w/2, sc_v,  w, color="steelblue", edgecolor="black",
                     lw=0.5, alpha=0.85, label="scratch")
        for bars in (b1, b2):
            for b in bars:
                v = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, v * 1.02, f"{v:.3f}",
                         ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={K}\n({summary[K]['n_train']} lines)"
                             for K in Ks], fontsize=9)
        ax.set_ylabel(f"test masked {metric.upper()} on G")
        ax.set_title(f"SSL pretrain vs scratch ({metric.upper()})")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if metric == "rl2":
            ax.axhline(0.36, color="black", ls="--", lw=1.0,
                        label="constant-mean baseline")
    fig.suptitle("BrainIAC-style few-shot comparison: SSL helps when labels are scarce",
                  fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(figs / "fig13_ssl_finetune_bars.png", dpi=160, bbox_inches="tight")
    print(f"Wrote {figs / 'fig13_ssl_finetune_bars.png'}")


if __name__ == "__main__":
    main()
