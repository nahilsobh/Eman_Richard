"""HDF5 dataset class for 1D eigenstrain training pairs."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset


class EigenstrainDataset1D(Dataset):
    """Load (X, Y) pairs from an HDF5 file. Data is cached in RAM."""

    def __init__(self, h5_path: str | Path, split: str = "train",
                 train_frac: float = 0.9):
        path = Path(h5_path)
        with h5py.File(path, "r") as f:
            n = f["X"].shape[0]
            split_idx = int(n * train_frac)
            if split == "train":
                sl = slice(0, split_idx)
            else:
                sl = slice(split_idx, n)

            self.X            = torch.from_numpy(f["X"][sl])
            self.eps_true     = torch.from_numpy(f["Y_eps_true"][sl])
            self.eps_analytic = torch.from_numpy(f["Y_eps_analytic"][sl])
            self.is_expanding = torch.from_numpy(
                f["meta/is_expanding"][sl].astype(np.float32))
            self.E_bg         = torch.from_numpy(f["meta/E_bg"][sl])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i: int) -> dict:
        return {
            "X":            self.X[i],
            "eps_true":     self.eps_true[i],
            "eps_analytic": self.eps_analytic[i],
            "is_expanding": self.is_expanding[i],
            "E_bg":         self.E_bg[i],
        }
