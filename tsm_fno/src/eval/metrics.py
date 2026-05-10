"""Numpy evaluation metrics for the TSM-FNO."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def relative_l2_np(pred: np.ndarray, target: np.ndarray,
                    eps: float = 1e-6) -> np.ndarray:
    num = np.linalg.norm(pred - target, axis=(-2, -1))
    den = np.maximum(np.linalg.norm(target, axis=(-2, -1)), eps)
    return num / den


def ssim_np(pred: np.ndarray, target: np.ndarray,
            C1: float = 1e-4, C2: float = 9e-4) -> np.ndarray:
    out = np.empty(len(pred), dtype=np.float32)
    for i in range(len(pred)):
        p, t = pred[i], target[i]
        mu_p, mu_t = p.mean(), t.mean()
        sig_p, sig_t = p.var(), t.var()
        sig_pt = ((p - mu_p) * (t - mu_t)).mean()
        out[i] = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
                 ((mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2))
    return out


def ring_rl2(eps_pred: np.ndarray, eps_true: np.ndarray,
             ring_mask: np.ndarray, eps_floor: float = 1e-3) -> np.ndarray:
    """Per-sample relative-L² evaluated only inside the perilesional shell."""
    ring = ring_mask.astype(bool)
    out = np.empty(len(eps_pred), dtype=np.float32)
    for i in range(len(eps_pred)):
        m = ring[i]
        if m.sum() == 0:
            out[i] = 0.0; continue
        diff = eps_pred[i][m] - eps_true[i][m]
        num = np.linalg.norm(diff)
        den = max(np.linalg.norm(eps_true[i][m]), eps_floor)
        out[i] = num / den
    return out


def expansion_auc(eps_pred: np.ndarray, ring_mask: np.ndarray,
                   is_expanding: np.ndarray) -> float:
    """ROC-AUC for max(eps_pred in ring) as expanding/control classifier."""
    score = np.empty(len(eps_pred), dtype=np.float32)
    for i in range(len(eps_pred)):
        m = ring_mask[i].astype(bool)
        score[i] = float(eps_pred[i][m].max()) if m.sum() else 0.0
    if len(np.unique(is_expanding)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(is_expanding.astype(int), score))
    except Exception:
        return float("nan")


def estimate_tau(G_t0, G_t1, eps_t0, eps_t1,
                  delta_t_days: float, shell_mask) -> float | None:
    """Single-exponential fit of perilesional ε decay over two time points."""
    e0 = float(eps_t0[shell_mask].mean())
    e1 = float(eps_t1[shell_mask].mean())
    if e0 < 1e-4 or e1 < 1e-4:
        return None
    tau = -delta_t_days / np.log(e1 / e0 + 1e-10)
    return max(tau, 0.0)


def recover_acoustoelastic_constant(
    G_pred: np.ndarray, eps_pred: np.ndarray,
    G_bg: np.ndarray, ring: np.ndarray,
    A_true: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample least-squares fit of  G_pred − G_bg ≈ A · ε_pred · G_bg in ring.

    Returns (A_recovered, A_relative_error_or_nan) where the second value is
    np.nan if no ground truth was passed.
    """
    n = len(G_pred)
    A_rec = np.empty(n, dtype=np.float32)
    A_err = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        m = ring[i].astype(bool)
        if m.sum() < 5 or G_bg[i] <= 0:
            A_rec[i] = np.nan
            continue
        x = (eps_pred[i][m] * G_bg[i]).reshape(-1, 1)
        y = (G_pred[i][m] - G_bg[i])
        if np.linalg.norm(x) < 1e-9:
            A_rec[i] = np.nan
            continue
        A_fit = float(np.linalg.lstsq(x, y, rcond=None)[0][0])
        A_rec[i] = A_fit
        if A_true is not None and abs(A_true[i]) > 1e-6:
            A_err[i] = abs(A_fit - A_true[i]) / abs(A_true[i])
    return A_rec, A_err


def stratified_table(rl2_G, rl2_eps, ring_rl2,
                      pressure, contrast, size_mm,
                      path: str | Path = None) -> str:
    """Print and optionally write a stratified accuracy table."""
    p_bins  = [(0, 0.5), (500, 2000), (2000, 5000), (5000, 8500)]
    c_bins  = [(1, 3), (3, 7), (7, 20)]
    s_bins  = [(0, 10), (10, 20), (20, 50)]

    lines = []
    def _row(name, lo, hi, mask):
        n = int(mask.sum())
        if n == 0:
            return f"  {name}: {lo}-{hi}  N=0"
        return (f"  {name}: {lo}-{hi}  N={n:5d}  "
                f"RL2_G={rl2_G[mask].mean():.4f}  "
                f"RL2_eps={rl2_eps[mask].mean():.4f}  "
                f"ring_RL2={ring_rl2[mask].mean():.4f}")

    lines.append("== By expansion pressure (Pa) ==")
    for lo, hi in p_bins:
        m = (pressure >= lo) & (pressure < hi)
        lines.append(_row("p", lo, hi, m))
    lines.append("\n== By stiffness contrast G_lesion / G_bg ==")
    for lo, hi in c_bins:
        m = (contrast >= lo) & (contrast < hi)
        lines.append(_row("c", lo, hi, m))
    lines.append("\n== By equivalent radius (mm) ==")
    for lo, hi in s_bins:
        m = (size_mm >= lo) & (size_mm < hi)
        lines.append(_row("a", lo, hi, m))

    out = "\n".join(lines)
    print(out)
    if path is not None:
        Path(path).write_text(out + "\n")
    return out
