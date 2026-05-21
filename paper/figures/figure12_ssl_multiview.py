#!/usr/bin/env python3
"""Figure 12 - Multi-view contrastive SSL pretraining on BBIR 1D voxel pairs.

For each voxel in the brain we have two 1D views (x-line, y-line)
through the same tissue point.  The unknown G(voxel) is identical for
both, so a useful representation should map the two views to the same
embedding.  This is the MRE analogue of BrainIAC's SimCLR-on-3D-patches
recipe, with x/y line-pair views replacing augmented patch pairs.

Loss: SimCLR-style InfoNCE (normalized temperature-scaled cross-entropy)
over a batch of 2N views (N pairs).  The positive of view i is its
paired counterpart; all 2N-2 other views are negatives.

Architecture: a small encoder that maps (2, L) -> embedding (D,), plus
a 2-layer MLP projector to a contrastive space (D_proj,).  The encoder
body is shared between the two views (Siamese setup).  After SSL the
projector is discarded and the encoder is used as initialisation for
downstream supervised G prediction (figure13_ssl_finetune.py).

Output:
  fig12_ssl_encoder.pt        (encoder state-dict only)
  fig12_ssl_full.pt           (encoder + projector + opt for resume)
  fig12_ssl_history.json
  fig12_ssl_loss.png
  fig12_ssl_summary.json
"""
from __future__ import annotations

import json, math, time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SRC_DEFAULT = "/projects/bfid/sobh/data/bbir_udel_multiview.h5"


# ── Encoder: SIREN body + global mean pool ────────────────────────────────────

class SineConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 11,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        pad = kernel_size // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad)
        with torch.no_grad():
            fan_in = c_in * kernel_size
            if is_first:
                self.conv.weight.uniform_(-1.0 / fan_in, 1.0 / fan_in)
            else:
                b = math.sqrt(6.0 / fan_in) / omega_0
                self.conv.weight.uniform_(-b, b)
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.conv(x))


class SIRENEncoder1d(nn.Module):
    """SIREN body that maps (B, 2, L) to a feature volume (B, width, L).

    Same hyperparameters as the SIREN1d body used in Figs 5/6/9 -- so
    weights can be loaded directly into the downstream model.
    """
    def __init__(self, in_ch: int = 2, width: int = 24,
                 kernel_size: int = 11, n_blocks: int = 3,
                 first_omega_0: float = 30.0, hidden_omega_0: float = 30.0):
        super().__init__()
        self.first = SineConv1d(in_ch + 1, width, 1, is_first=True,
                                  omega_0=first_omega_0)
        self.hidden = nn.ModuleList([
            SineConv1d(width, width, kernel_size=kernel_size,
                        is_first=False, omega_0=hidden_omega_0)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, N = x.shape
        grid = torch.linspace(0, 1, N, device=x.device).view(1, 1, N).expand(B, 1, N)
        h = torch.cat([x, grid], dim=1)
        h = self.first(h)
        for layer in self.hidden:
            h = layer(h)
        return h


class ContrastiveModel(nn.Module):
    """Encoder + projection head for SimCLR-style training."""
    def __init__(self, encoder: SIRENEncoder1d, embed_dim: int = 24,
                 proj_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)                  # (B, W, L)
        return feat.mean(dim=-1)                # global pool over L -> (B, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.representation(x))   # (B, proj_dim)
        return F.normalize(z, dim=-1)           # SimCLR unit-sphere convention


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1
              ) -> torch.Tensor:
    """SimCLR InfoNCE.  z1, z2 are L2-normalised (B, D).

    Concatenates to a (2B, D) batch; positive of i is the paired view.
    """
    B = z1.shape[0]
    Z = torch.cat([z1, z2], dim=0)              # (2B, D)
    sim = (Z @ Z.t()) / tau                     # (2B, 2B)
    # mask out self-similarities
    mask = torch.eye(2 * B, device=Z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    # positive indices: pair i is z1_i and z2_i  ->  partner of i in [0..B) is i+B,
    # partner of i in [B..2B) is i-B
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(Z.device)
    return F.cross_entropy(sim, pos_idx)


# ── Data loading (in-memory) ──────────────────────────────────────────────────

def load_multiview(src: str, n_train: int, n_val: int, seed: int = 0
                    ) -> tuple[torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor]:
    with h5py.File(src, "r") as h:
        N = h["X_x"].shape[0]
        sbj = np.array(h["meta/subject"][:], dtype="U32")
        # subject-disjoint split (same scheme as Fig 11)
        all_sbj = sorted(set(sbj.tolist()))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_sbj))
        n_v = max(1, int(round(len(all_sbj) * 0.10)))
        n_t = max(1, int(round(len(all_sbj) * 0.10)))
        val_set   = set(all_sbj[i] for i in perm[:n_v])
        test_set  = set(all_sbj[i] for i in perm[n_v:n_v + n_t])
        train_set = set(all_sbj) - val_set - test_set
        tr_mask = np.isin(sbj, list(train_set))
        va_mask = np.isin(sbj, list(val_set))
        idx_tr_all = np.where(tr_mask)[0]
        idx_va_all = np.where(va_mask)[0]
        if len(idx_tr_all) > n_train:
            idx_tr = np.sort(rng.choice(idx_tr_all, n_train, replace=False))
        else:
            idx_tr = idx_tr_all
        if len(idx_va_all) > n_val:
            idx_va = np.sort(rng.choice(idx_va_all, n_val, replace=False))
        else:
            idx_va = idx_va_all
        Xx_tr = np.array(h["X_x"][idx_tr], dtype=np.float32)
        Xy_tr = np.array(h["X_y"][idx_tr], dtype=np.float32)
        Xx_va = np.array(h["X_x"][idx_va], dtype=np.float32)
        Xy_va = np.array(h["X_y"][idx_va], dtype=np.float32)
    # per-sample max-amplitude normalisation
    def normalise(X):
        m = np.abs(X).reshape(X.shape[0], -1).max(axis=1)
        m = np.where(m > 0, m, 1.0)
        return X / m[:, None, None]
    return (torch.from_numpy(normalise(Xx_tr)),
            torch.from_numpy(normalise(Xy_tr)),
            torch.from_numpy(normalise(Xx_va)),
            torch.from_numpy(normalise(Xy_va)))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",       default=SRC_DEFAULT)
    ap.add_argument("--n_train",   type=int, default=20000)
    ap.add_argument("--n_val",     type=int, default=2000)
    ap.add_argument("--epochs",    type=int, default=15)
    ap.add_argument("--batch",     type=int, default=128)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--width",     type=int, default=24)
    ap.add_argument("--kernel",    type=int, default=11)
    ap.add_argument("--n_blocks",  type=int, default=3)
    ap.add_argument("--proj_dim",  type=int, default=64)
    ap.add_argument("--tau",       type=float, default=0.1)
    ap.add_argument("--threads",   type=int, default=16)
    ap.add_argument("--seed",      type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    figs = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  threads: {torch.get_num_threads()}")

    print(f"Loading multi-view pairs from {args.src} ...")
    t0 = time.time()
    Xx_tr, Xy_tr, Xx_va, Xy_va = load_multiview(args.src,
                                                  args.n_train, args.n_val,
                                                  seed=args.seed)
    print(f"  train pairs: {tuple(Xx_tr.shape)}  val pairs: {tuple(Xx_va.shape)}  "
          f"({time.time()-t0:.1f}s)")

    encoder = SIRENEncoder1d(in_ch=2, width=args.width,
                              kernel_size=args.kernel, n_blocks=args.n_blocks)
    model   = ContrastiveModel(encoder, embed_dim=args.width,
                                proj_dim=args.proj_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ContrastiveModel: {n_params:,} params  "
          f"(encoder={sum(p.numel() for p in encoder.parameters()):,})")

    train_dl = DataLoader(TensorDataset(Xx_tr, Xy_tr),
                           batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl   = DataLoader(TensorDataset(Xx_va, Xy_va),
                           batch_size=args.batch, shuffle=False, drop_last=True)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-5)

    def eval_loop():
        model.eval()
        s, n = 0.0, 0
        with torch.no_grad():
            for xx, xy in val_dl:
                xx = xx.to(device); xy = xy.to(device)
                zx = model(xx); zy = model(xy)
                l = info_nce(zx, zy, tau=args.tau)
                s += float(l.item()) * len(xx); n += len(xx)
        return s / max(n, 1)

    history = {"train": [], "val": []}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xx, xy in train_dl:
            xx = xx.to(device); xy = xy.to(device)
            zx = model(xx); zy = model(xy)
            loss = info_nce(zx, zy, tau=args.tau)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * len(xx); n += len(xx)
        sched.step()
        train_loss = tot / max(n, 1)
        val_loss   = eval_loop()
        history["train"].append(train_loss); history["val"].append(val_loss)
        print(f"  ep {ep:3d}  train InfoNCE={train_loss:.4f}  "
              f"val InfoNCE={val_loss:.4f}  (elapsed {time.time()-t0:.0f}s)",
              flush=True)

    # Save encoder state-dict for downstream fine-tuning
    torch.save(encoder.state_dict(), figs / "fig12_ssl_encoder.pt")
    torch.save({"encoder": encoder.state_dict(),
                "projector": model.proj.state_dict(),
                "args": vars(args)},
                figs / "fig12_ssl_full.pt")
    json.dump(history, open(figs / "fig12_ssl_history.json", "w"), indent=2)
    json.dump({"n_params": n_params,
                "n_train_used": int(len(Xx_tr)),
                "n_val_used":   int(len(Xx_va)),
                "final_train_nce": history["train"][-1],
                "final_val_nce":   history["val"][-1]},
                open(figs / "fig12_ssl_summary.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.4))
    ep_axis = np.arange(1, args.epochs + 1)
    ax.plot(ep_axis, history["train"], color="steelblue", lw=1.6, label="train")
    ax.plot(ep_axis, history["val"],   color="tomato",    lw=1.6, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title(f"SimCLR multi-view SSL on BBIR (x-line, y-line) pairs  "
                  f"({n_params/1e3:.0f}k params)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "fig12_ssl_loss.png", dpi=160, bbox_inches="tight")
    print(f"Wrote {figs / 'fig12_ssl_loss.png'}")


if __name__ == "__main__":
    main()
