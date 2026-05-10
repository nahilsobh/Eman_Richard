#!/usr/bin/env python3
import os
import sys
import h5py
import numpy as np

import sys
path = sys.argv[1] if len(sys.argv) > 1 else "data/mre_v2_50000.h5"

f = h5py.File(path, "r")

# ── Shape checks ──────────────────────────────────────────
expected_C = f["X"].shape[1]   # 2 for v2 (60 Hz only), 4 for v1 (60 + 120 Hz)
assert expected_C in (2, 4), f"Unexpected channel count {expected_C}"
assert f["X"].shape == (50000, expected_C, 64, 64), f"Wrong X shape: {f['X'].shape}"
assert f["Y"].shape == (50000, 64, 64),    f"Wrong Y shape: {f['Y'].shape}"

# ── Streaming range + diversity checks ────────────────────
CHUNK = 1000
n = f["X"].shape[0]

X_finite = True;  Y_finite = True
Y_min = np.inf;   Y_max = -np.inf
G_stds = []
re_vals = [];  im_vals = []
sample_hashes = {}

for start in range(0, n, CHUNK):
    sl = slice(start, start + CHUNK)
    Xc = f["X"][sl]
    Yc = f["Y"][sl]

    if not np.isfinite(Xc).all(): X_finite = False
    if not np.isfinite(Yc).all(): Y_finite = False
    Y_min = min(Y_min, float(Yc.min()))
    Y_max = max(Y_max, float(Yc.max()))
    G_stds.append(Yc.std(axis=(-2, -1)))

    # Sub-sample for correlation and hash checks
    for i in range(0, len(Xc), 500):
        re_vals.append(Xc[i, 0].ravel())
        im_vals.append(Xc[i, 1].ravel())
        h_val = hash(Yc[i].tobytes())
        assert h_val not in sample_hashes, \
            f"Duplicate sample at {start+i} and {sample_hashes[h_val]}"
        sample_hashes[h_val] = start + i

assert X_finite, "X contains NaN or Inf"
assert Y_finite, "Y contains NaN or Inf"
assert Y_min >= 500,    f"Stiffness below physical floor: {Y_min:.1f} Pa"
assert Y_max <= 50_000, f"Stiffness above physical ceiling: {Y_max:.1f} Pa"

G_std_mean = np.concatenate(G_stds).mean()
assert G_std_mean > 500, f"Suspiciously uniform G maps: mean std = {G_std_mean:.1f} Pa"

re_all = np.concatenate(re_vals)
im_all = np.concatenate(im_vals)
corr = np.corrcoef(re_all, im_all)[0, 1]
assert abs(corr) < 0.99, f"Re and Im channels nearly identical (corr={corr:.4f}) — solver bug"

print("✓ Dataset integrity: PASS")
print(f"  Samples    : {n:,}")
print(f"  G range    : {Y_min:.0f} – {Y_max:.0f} Pa")
print(f"  G std mean : {G_std_mean:.0f} Pa  (diversity check)")
print(f"  Re/Im corr : {corr:.4f}  (channel independence check)")
print(f"  File size  : {os.path.getsize(path)/1e9:.2f} GB")
