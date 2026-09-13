"""Convergence check: FDM u_r should approach Lamé as cube size → ∞.

Compare FDM u_r at fixed r_check = 8 vx (surface + a bit) as we grow
the surrounding cube size. If the FDM is correct, u_r should approach
the infinite-matrix Lamé value as the cube grows.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solver.elastostatic_fd_3d import elastostatic_solve_3d


def measure_ur(N, dx, a0_vx, a_vx, r_check_vx):
    center = (N // 2, N // 2, N // 2)
    du_r = (a_vx - a0_vx) * dx  # radial displacement at balloon surface

    dirichlet = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                dxv = i - center[0]
                dyv = j - center[1]
                dzv = k - center[2]
                r_vx = math.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
                if r_vx <= a0_vx and r_vx > 1e-6:
                    scale = (r_vx / a0_vx) * du_r / r_vx
                    dirichlet.append((i, j, k, 0, scale * dxv))
                    dirichlet.append((i, j, k, 1, scale * dyv))
                    dirichlet.append((i, j, k, 2, scale * dzv))

    u = elastostatic_solve_3d(
        N, N, N, dx, 1000.0, 1.0e6,
        fixed_faces=("-x", "+x", "-y", "+y", "-z", "+z"),
        mirror_faces=(),
        dirichlet=dirichlet,
    )

    # Average u_r at r ≈ r_check_vx
    dr = 0.5
    ur_vals = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                dxv = i - center[0]
                dyv = j - center[1]
                dzv = k - center[2]
                r_vx = math.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
                if abs(r_vx - r_check_vx) < dr and r_vx > 1:
                    u_here = u[i, j, k, :]
                    r_hat = np.array([dxv, dyv, dzv]) / r_vx
                    ur_vals.append(float(np.dot(u_here, r_hat)))

    return float(np.mean(ur_vals))


def lame_ur(a0_vx, a_vx, r_vx, dx):
    a  = a_vx  * dx
    a0 = a0_vx * dx
    r  = r_vx  * dx
    R  = (r ** 3 - a0 ** 3 + a ** 3) ** (1.0 / 3.0)
    return R - r


if __name__ == "__main__":
    dx = 0.003
    a0_vx = 8.0
    a_vx  = 10.0
    r_check = 12.0

    ur_lame_val = lame_ur(a0_vx, a_vx, r_check, dx)
    print(f"Lamé infinite-matrix u_r at r={r_check} vx: {ur_lame_val*1e3:.3f} mm")
    print()

    # All-faces-fixed cube.
    for N in [24, 32, 48, 64]:
        ur = measure_ur(N, dx, a0_vx, a_vx, r_check)
        err = (ur - ur_lame_val) / ur_lame_val * 100
        print(f"  N = {N:3d} (cube = {N * dx * 100:5.1f} cm): "
              f"u_r = {ur*1e3:.3f} mm  ({err:+6.1f}% vs Lamé)")
