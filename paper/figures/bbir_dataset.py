"""PyTorch Dataset wrapper for the BBIR UDel 2D slice HDF5.

Loads ``bbir_udel_2d_slices.h5`` (produced by ``bbir_ingest.py``) as a
PyTorch ``Dataset`` and exposes per-sample tensors compatible with the
2D inversion architecture.

Per-sample tensors:

    X    : (2, 80, 80)  float32  -- (Re u, Im u) normalised so ||u||_inf = 1
    Y    : (2, 80, 80)  float32  -- (Re G, Im G) / G_scale  (Pa / G_scale)
    mask : (1, 80, 80)  float32  -- brain mask (1 inside, 0 outside)
    info : dict                  -- subject, freq_hz, z_idx, G_scale

The loss can be masked using ``mask`` so non-brain pixels do not
contribute. The default G_scale of 5000 Pa matches the 1D pipeline and
keeps Y in O(1) for stable training.

A split helper (``train_val_test_split_by_subject``) ensures train/val/test
splits respect subject boundaries -- no subject is split across splits,
so the network never sees a slice from a subject whose other slices it
trained on.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


G_SCALE_DEFAULT = 5000.0    # Pa, matches 1D pipeline


@dataclass
class BBIRSubjectSplit:
    """Holds subject IDs for the three splits."""
    train: list[str]
    val:   list[str]
    test:  list[str]


def train_val_test_split_by_subject(h5_path: Path, val_frac: float = 0.10,
                                     test_frac: float = 0.10, seed: int = 0
                                     ) -> BBIRSubjectSplit:
    """Deterministic split that keeps each subject's slices in one split."""
    with h5py.File(h5_path, "r") as h:
        subjects = np.array(h["meta/subject"][:], dtype="U32")
    uniq = sorted(set(subjects.tolist()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_val  = max(1, int(round(len(uniq) * val_frac)))
    n_test = max(1, int(round(len(uniq) * test_frac)))
    val_ids   = [uniq[i] for i in perm[:n_val]]
    test_ids  = [uniq[i] for i in perm[n_val:n_val + n_test]]
    train_ids = [uniq[i] for i in perm[n_val + n_test:]]
    return BBIRSubjectSplit(train=sorted(train_ids),
                             val=sorted(val_ids),
                             test=sorted(test_ids))


class BBIRSliceDataset(Dataset):
    """2D BBIR slice dataset backed by an HDF5 file.

    Parameters
    ----------
    h5_path     : Path to the HDF5 file produced by bbir_ingest.py.
    subjects    : Optional list of subject IDs (UTF strings) to include.
                  If None, includes all subjects in the file.
    freqs       : Optional list of MRE frequencies (Hz) to include. If
                  None, includes all frequencies in the file.
    G_scale     : Normalisation constant applied to G (Pa).
    return_info : If True, ``__getitem__`` returns a 4-tuple including
                  per-sample metadata; otherwise the 3-tuple
                  (X, Y, mask).
    """
    def __init__(self, h5_path: Path,
                 subjects: list[str] | None = None,
                 freqs: list[int] | None = None,
                 G_scale: float = G_SCALE_DEFAULT,
                 return_info: bool = False):
        self.h5_path     = Path(h5_path)
        self.G_scale     = float(G_scale)
        self.return_info = return_info
        # Build index by reading metadata; do NOT keep the file handle open
        # (h5py file handles do not survive forking in DataLoader workers
        # unless created lazily per worker).
        with h5py.File(self.h5_path, "r") as h:
            sbj  = np.array(h["meta/subject"][:], dtype="U32")
            frq  = np.array(h["meta/freq"][:],    dtype=np.int32)
            zix  = np.array(h["meta/z_idx"][:],   dtype=np.int32)
        keep = np.ones(len(sbj), dtype=bool)
        if subjects is not None:
            keep &= np.isin(sbj, np.array(list(subjects), dtype="U32"))
        if freqs is not None:
            keep &= np.isin(frq, np.array(list(freqs), dtype=np.int32))
        self.indices = np.where(keep)[0]
        self.subjects_all = sbj
        self.freqs_all    = frq
        self.zix_all      = zix
        self._h: h5py.File | None = None

    def _open(self) -> h5py.File:
        if self._h is None:
            self._h = h5py.File(self.h5_path, "r")
        return self._h

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        h = self._open()
        X      = np.array(h["X"][idx],      dtype=np.float32)      # (2,80,80)
        Y_re   = np.array(h["Y_G_re"][idx], dtype=np.float32)
        Y_im   = np.array(h["Y_G_im"][idx], dtype=np.float32)
        mask_a = np.array(h["mask"][idx],   dtype=np.bool_)

        # Per-sample displacement normalisation so the network sees ||u||_inf=1.
        amax = float(np.max(np.abs(X)))
        if amax > 0:
            X = X / amax

        # Stack G into a 2-channel tensor and normalise to G_scale.
        Y = np.stack([Y_re, Y_im]) / self.G_scale                # (2,80,80)

        X_t    = torch.from_numpy(X)
        Y_t    = torch.from_numpy(Y)
        mask_t = torch.from_numpy(mask_a).float().unsqueeze(0)   # (1,80,80)

        if not self.return_info:
            return X_t, Y_t, mask_t

        info = {
            "subject": str(self.subjects_all[idx]),
            "freq_hz": int(self.freqs_all[idx]),
            "z_idx":   int(self.zix_all[idx]),
            "G_scale": self.G_scale,
            "amax_u":  amax,
        }
        return X_t, Y_t, mask_t, info

    def close(self):
        if self._h is not None:
            self._h.close()
            self._h = None


def masked_relative_l2(Y_pred: torch.Tensor, Y_true: torch.Tensor,
                        mask: torch.Tensor, eps: float = 1e-3
                        ) -> torch.Tensor:
    """Per-sample RL^2 evaluated only inside the brain mask.

    Y_pred, Y_true : (B, 2, H, W)
    mask           : (B, 1, H, W) -- 1 inside brain, 0 outside.
    """
    m = mask
    diff = (Y_pred - Y_true) * m
    tgt  = Y_true * m
    num  = torch.linalg.vector_norm(diff.flatten(1), dim=-1)
    den  = torch.linalg.vector_norm(tgt.flatten(1),  dim=-1).clamp_min(eps)
    return (num / den).mean()
