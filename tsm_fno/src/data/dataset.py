"""HDF5-backed PyTorch dataset with TSM-aware augmentation.

The augmentations are vector-aware: when we 90°-rotate or h-flip an
image, the displacement components (u_x = Re(u), u_y = Im(u)) must be
rotated consistently with the spatial axes — otherwise the wave-direction
information is corrupted.

(Note: in our scalar Phase-0 solver, Re/Im are not true vector components
but phase-quadrature — so spatial rotation alone is not strictly the same
as a vector rotation. We provide both augmentation paths and default to
the *spatial-only* path, which is provably safe. Vector-aware rotation is
exposed via `augment_vector=True` for users who upgrade to a vector
solver in future work.)
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TSMDataset(Dataset):
    """Reads X / Y_G / Y_epsilon / Y_ring / meta from an HDF5 file."""

    def __init__(self, h5_path: str | Path, augment: bool = False,
                 augment_vector: bool = False, seed: int = 0):
        self.path = Path(h5_path)
        self.augment = bool(augment)
        self.augment_vector = bool(augment_vector)
        self.rng = np.random.default_rng(seed)
        self._h5 = None
        with h5py.File(self.path, "r") as f:
            self.n = int(f["X"].shape[0])
            self.shape_X = tuple(f["X"].shape[1:])

    def _ensure_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        self._ensure_h5()
        f = self._h5
        X = np.asarray(f["X"][idx], dtype=np.float32)
        G = np.asarray(f["Y_G"][idx], dtype=np.float32)
        eps = np.asarray(f["Y_epsilon"][idx], dtype=np.float32)
        ring = np.asarray(f["Y_ring"][idx], dtype=np.bool_)
        meta = {
            "p":            float(f["meta/p"][idx]),
            "A":            float(f["meta/A"][idx]),
            "G_bg":         float(f["meta/G_bg"][idx]),
            "G_lesion":     float(f["meta/G_lesion"][idx]),
            "is_expanding": bool(f["meta/is_expanding"][idx]),
        }

        if self.augment:
            X, G, eps, ring = self._augment(X, G, eps, ring)

        return {
            "X":            torch.from_numpy(X),
            "G":            torch.from_numpy(G),
            "epsilon":      torch.from_numpy(eps),
            "ring":         torch.from_numpy(ring.astype(np.float32)),
            "G_bg":         torch.tensor(meta["G_bg"], dtype=torch.float32),
            "is_expanding": torch.tensor(meta["is_expanding"],
                                           dtype=torch.float32),
        }

    # ── augmentation primitives ─────────────────────────────

    def _augment(self, X, G, eps, ring):
        # 1. Random rotation k ∈ {0, 1, 2, 3}
        k = int(self.rng.integers(0, 4))
        X = self._rot90_X(X, k)
        G = np.rot90(G, k=k).copy()
        eps = np.rot90(eps, k=k).copy()
        ring = np.rot90(ring, k=k).copy()

        # 2. Random horizontal flip
        if self.rng.random() < 0.5:
            X = self._flip_X(X)
            G = np.flip(G, axis=-1).copy()
            eps = np.flip(eps, axis=-1).copy()
            ring = np.flip(ring, axis=-1).copy()

        # 3. Random source-phase rotation on wave channels
        phase = float(self.rng.uniform(0.0, 2 * np.pi))
        c = np.cos(phase); s = np.sin(phase)
        re60, im60, re120, im120 = X[0], X[1], X[2], X[3]
        X[0] = c * re60  - s * im60
        X[1] = s * re60  + c * im60
        X[2] = c * re120 - s * im120
        X[3] = s * re120 + c * im120

        # 4. Random global G scale ±15%
        scale = float(self.rng.uniform(0.85, 1.15))
        G = (G * scale).astype(np.float32)
        # The Lamé prior channel (4) is dimensionless and unaffected by global G
        # scale; the eps target is also unaffected.

        return X, G, eps, ring

    def _rot90_X(self, X: np.ndarray, k: int) -> np.ndarray:
        """Spatial rotation. If augment_vector, rotate (u_x, u_y) consistently."""
        out = np.rot90(X, k=k, axes=(-2, -1)).copy()
        if self.augment_vector:
            # Treat (Re, Im) as (u_x, u_y) for each frequency pair
            for f0, f1 in [(0, 1), (2, 3)]:
                ux = out[f0].copy()
                uy = out[f1].copy()
                for _ in range(k):
                    ux, uy = -uy, ux       # 90° CCW: (ux, uy) → (-uy, ux)
                out[f0] = ux
                out[f1] = uy
        return out

    def _flip_X(self, X: np.ndarray) -> np.ndarray:
        out = np.flip(X, axis=-1).copy()
        if self.augment_vector:
            for f0, f1 in [(0, 1), (2, 3)]:
                out[f0] = -out[f0]
        return out
