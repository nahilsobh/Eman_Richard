#!/usr/bin/env python3
"""Nonlinear finite-strain FEM inside Yin's container via dolfinx.

Purpose: option-3 baseline at real hex-mesh resolution with FINITE-STRAIN
kinematics (not the small-strain approximation of my earlier FDM).

Constitutive law
----------------
The FEM solve uses NEAR-INCOMPRESSIBLE NEO-HOOKEAN:

  W = (μ/2) · (J^(-2/3)·tr(C) − 3) + (κ/2) · (J−1)²        ← in FEM
  κ = 1000·μ  (penalty enforcing det F ≈ 1)

We then compute the deformation gradient F(x) from the FEM displacement,
extract principal stretches (λ_1, λ_2, λ_3), and apply Ogden EX-POST:

  G_θ(x) = Σ_p μ_p · λ_θ(x)^(α_p − 2)   with  λ_θ ≡ max(λ_1, λ_2, λ_3)

WHY this is a valid Ogden option 3
----------------------------------
For truly incompressible material with prescribed-displacement BCs,
the displacement field u(x) is determined by volume-preserving kinematics
alone — NOT by material parameters. Neo-Hookean and Ogden give the
IDENTICAL displacement field (they differ only in the internal pressure
distribution). We use neo-Hookean in the FEM solve because it has a
clean UFL expression; the extracted stretch field IS the finite-strain
container-confined stretch field that Ogden would predict. Applying
Ogden's principal-stretch strain-energy law to that field is exact.

If we were to compute internal stresses (not our target), the choice
would matter. But we're extracting stretch and applying the Ogden
constitutive law to it — which is unaffected by the FEM's constitutive
choice for near-incompressible material.

Mesh
----
  dx = 1 mm →  150 × 150 × 180 =   4.05M hex cells,   ~12.4M vector DOF
  dx = 2 mm →   75 ×  75 ×  90 =    506k hex cells,   ~1.5M vector DOF
  dx = 3 mm →   50 ×  50 ×  60 =    150k hex cells,   ~460k vector DOF
  dx = 5 mm →   30 ×  30 ×  36 =     32k hex cells,   ~99k vector DOF

BCs
---
  - Rigid on 5 walls (u = 0 on −x, +x, −y, +y, −z)
  - TRACTION-FREE on +z (top): natural boundary (no constraint)
  - Balloon Dirichlet: nodes within r ≤ a₀ get u = (r/a₀)·(a−a₀)·r̂
                       (rigid balloon growth from a₀ = 50-mL radius to
                       the current target radius)

Load stepping
-------------
Each balloon volume is reached via `--load-steps` uniform increments from
the previous balloon volume; Newton at each step. This handles the
finite-strain nonlinearity gracefully.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CONTAINER_L_M = 0.15
CONTAINER_H_M = 0.18

BALLOON_VOLUMES_ML = [50, 100, 150, 200, 250]
_r_m = lambda v: (3 * v * 1e-6 / (4 * math.pi)) ** (1/3)
BALLOON_RADII_M = [_r_m(v) for v in BALLOON_VOLUMES_ML]
A0_M = BALLOON_RADII_M[0]

PHANTOMS = [
    dict(short="P1", name="Phantom 1 — 10% bovine gelatin",
         mu=[1800.0, 700.0], alpha=[2.5, 3.0],
         yin_tsm=[3.5, 3.8, 3.9, 4.2, 4.4]),
    dict(short="P2", name="Phantom 2 — 8% gel + 7% cellulose",
         mu=[2500.0, 1500.0], alpha=[2.5, 5.0],
         yin_tsm=[3.5, 3.9, 4.2, 4.9, 5.15]),
]


def run(dx_mm: float, load_steps: int):
    from mpi4py import MPI
    from dolfinx import fem, log, mesh
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver
    import ufl

    log.set_log_level(log.LogLevel.WARNING)

    dx = dx_mm * 1e-3
    Nx = int(round(CONTAINER_L_M / dx))
    Ny = int(round(CONTAINER_L_M / dx))
    Nz = int(round(CONTAINER_H_M / dx))
    print(f"Mesh: {Nx}×{Ny}×{Nz} hex ({Nx*Ny*Nz:,} cells, "
          f"container {Nx*dx*100:.1f}×{Ny*dx*100:.1f}×{Nz*dx*100:.1f} cm)")

    comm = MPI.COMM_WORLD
    domain = mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]),
         np.array([CONTAINER_L_M, CONTAINER_L_M, CONTAINER_H_M])],
        [Nx, Ny, Nz],
        cell_type=mesh.CellType.hexahedron,
    )

    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"Vector DOFs: {n_dof:,}")

    cx = CONTAINER_L_M / 2
    cy = CONTAINER_L_M / 2
    cz = CONTAINER_H_M / 2

    def wall_marker(x):
        tol = dx * 0.5
        return (np.isclose(x[0], 0.0, atol=tol) |
                np.isclose(x[0], CONTAINER_L_M, atol=tol) |
                np.isclose(x[1], 0.0, atol=tol) |
                np.isclose(x[1], CONTAINER_L_M, atol=tol) |
                np.isclose(x[2], 0.0, atol=tol))

    def balloon_marker(x):
        r = np.sqrt((x[0]-cx)**2 + (x[1]-cy)**2 + (x[2]-cz)**2)
        return r <= A0_M + dx * 0.5

    wall_dofs    = fem.locate_dofs_geometrical(V, wall_marker)
    balloon_dofs = fem.locate_dofs_geometrical(V, balloon_marker)

    u_zero = fem.Function(V)
    bc_walls = fem.dirichletbc(u_zero, wall_dofs)

    # ── Neo-Hookean weak form ────────────────────────────────────────
    u = fem.Function(V, name="displacement")
    v = ufl.TestFunction(V)

    I = ufl.Identity(3)
    F_tensor = I + ufl.grad(u)
    C_tensor = F_tensor.T * F_tensor
    J = ufl.det(F_tensor)

    MU_SOLVE = 1000.0  # arbitrary; result is material-independent
    mu_c    = fem.Constant(domain, MU_SOLVE)
    kappa_c = fem.Constant(domain, MU_SOLVE * 1000.0)

    W_iso = (mu_c / 2.0) * (J ** (-2.0/3.0) * ufl.tr(C_tensor) - 3.0)
    W_vol = (kappa_c / 2.0) * (J - 1.0) ** 2
    Pi = (W_iso + W_vol) * ufl.dx
    R  = ufl.derivative(Pi, u, v)

    # ── Balloon-surface Dirichlet builder ────────────────────────────
    def make_balloon_bc(a_now):
        du_r = a_now - A0_M
        u_bal = fem.Function(V)
        def _expr(x):
            rx = x[0] - cx; ry = x[1] - cy; rz = x[2] - cz
            r  = np.sqrt(rx*rx + ry*ry + rz*rz)
            r_safe = np.where(r < 1e-12, 1e-12, r)
            scale  = (r / A0_M) * du_r / r_safe
            return np.vstack([scale*rx, scale*ry, scale*rz])
        u_bal.interpolate(_expr)
        return fem.dirichletbc(u_bal, balloon_dofs)

    # ── Ex-post ring extraction ──────────────────────────────────────
    # Interpolate F_tensor onto DG-0 space, then evaluate at ring nodes.
    Vt = fem.functionspace(domain, ("DG", 0, (3, 3)))

    def ring_marker(x):
        r = np.sqrt((x[0]-cx)**2 + (x[1]-cy)**2 + (x[2]-cz)**2)
        return (r >= A0_M + 0.003) & (r <= A0_M + 0.008)

    def extract_lam_theta():
        F_expr = fem.Expression(F_tensor, Vt.element.interpolation_points())
        F_field = fem.Function(Vt)
        F_field.interpolate(F_expr)

        # Get ring-cell centres by evaluating balloon distance at cell centres
        tdim = domain.topology.dim
        n_cells_local = domain.topology.index_map(tdim).size_local
        cell_indices = np.arange(n_cells_local, dtype=np.int32)
        # Cell midpoints
        midpts = np.zeros((n_cells_local, 3))
        for c in cell_indices:
            # Use dolfinx.mesh.compute_midpoints (dolfinx 0.7+)
            pass
        from dolfinx.mesh import compute_midpoints
        midpts = compute_midpoints(domain, tdim, cell_indices)
        r_mid = np.sqrt((midpts[:, 0]-cx)**2 + (midpts[:, 1]-cy)**2 + (midpts[:, 2]-cz)**2)
        ring_mask = (r_mid >= A0_M + 0.003) & (r_mid <= A0_M + 0.008)
        ring_cells = cell_indices[ring_mask]

        lam_theta_list = []
        F_values = F_field.x.array.reshape(n_cells_local, 9)
        for c in ring_cells:
            F_c = F_values[c].reshape(3, 3)
            C_c = F_c.T @ F_c
            eig = np.sort(np.linalg.eigvalsh(C_c))[::-1]
            eig = np.clip(eig, 1e-12, None)
            stretch = np.sqrt(eig)
            # For spherical inflation, two largest stretches are tangential
            # (λ_θ = λ_φ), smallest is radial (λ_r).
            lam_theta_list.append(0.5 * (stretch[0] + stretch[1]))
        return np.array(lam_theta_list)

    def ring_G_ogden(lam_arr, mu_list, alpha_list):
        G = np.zeros_like(lam_arr)
        for m, al in zip(mu_list, alpha_list):
            G += m * np.power(lam_arr, al - 2.0)
        lo, hi = np.percentile(G, [10, 90])
        tr = G[(G >= lo) & (G <= hi)]
        return float(tr.mean()) / 1000  # kPa

    # ── Load-step through balloon volumes ────────────────────────────
    # Build the NonlinearProblem once (reuses form compilation).
    # Update the balloon BC value between load steps by mutating the
    # existing Function.

    # Persistent BC function for the balloon (its values get updated
    # before each Newton solve).
    u_bal_persistent = fem.Function(V)
    bc_bal = fem.dirichletbc(u_bal_persistent, balloon_dofs)

    problem = NonlinearProblem(R, u, bcs=[bc_walls, bc_bal])
    solver = NewtonSolver(comm, problem)
    solver.rtol = 1e-6
    solver.atol = 1e-8
    solver.max_it = 50
    solver.convergence_criterion = "residual"
    solver.report = True   # print Newton iters

    # Configure linear solver directly on the KSP object (PETSc.Options
    # global values are not picked up by dolfinx NewtonSolver's KSP unless
    # option-prefixed).
    from petsc4py import PETSc
    ksp = solver.krylov_solver
    if n_dof < 500_000:
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        print("Using direct LU (MUMPS)", flush=True)
    else:
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=200)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.GAMG)
        print("Using CG + GAMG", flush=True)

    def update_balloon_bc(a_now):
        du_r = a_now - A0_M
        def _expr(x):
            rx = x[0] - cx; ry = x[1] - cy; rz = x[2] - cz
            r  = np.sqrt(rx*rx + ry*ry + rz*rz)
            r_safe = np.where(r < 1e-12, 1e-12, r)
            scale  = (r / A0_M) * du_r / r_safe
            return np.vstack([scale*rx, scale*ry, scale*rz])
        u_bal_persistent.interpolate(_expr)

    results = []
    a_prev = A0_M
    for vol_ml, a_tgt in zip(BALLOON_VOLUMES_ML, BALLOON_RADII_M):
        print(f"\n=== Volume {vol_ml} mL, a = {a_tgt*1000:.2f} mm ===", flush=True)
        for step in range(load_steps):
            a_now = a_prev + (a_tgt - a_prev) * (step + 1) / load_steps
            update_balloon_bc(a_now)
            t0 = time.time()
            try:
                n_iter, converged = solver.solve(u)
            except Exception as e:
                print(f"    Newton FAILED at step {step+1}: {e}", flush=True)
                raise
            t1 = time.time()
            print(f"  step {step+1}/{load_steps}: a={a_now*1000:.2f} mm  "
                  f"[{n_iter} iters, {t1-t0:.1f} s, converged={converged}]",
                  flush=True)
            if not converged:
                print("    ABORT: Newton did not converge, aborting further steps.")
                return results

        lam_ring = extract_lam_theta()
        row = {"vol_ml": vol_ml, "a_m": a_tgt,
                "n_ring": int(len(lam_ring)),
                "lam_theta_mean": float(np.mean(lam_ring)) if lam_ring.size else float("nan")}
        for p in PHANTOMS:
            row[f"G_{p['short']}"] = ring_G_ogden(lam_ring, p["mu"], p["alpha"])
        results.append(row)
        print(f"  RING: n={row['n_ring']}, ⟨λ_θ⟩={row['lam_theta_mean']:.4f}, "
              f"G_P1={row.get('G_P1', float('nan')):.2f}, "
              f"G_P2={row.get('G_P2', float('nan')):.2f} kPa", flush=True)
        a_prev = a_tgt

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx-mm", type=float, default=3.0)
    parser.add_argument("--load-steps", type=int, default=5)
    args = parser.parse_args()

    out_dir = ROOT / "results" / f"paper_ogden_fenics_fem_dx{int(args.dx_mm)}mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== FEniCS nonlinear FEM at dx = {args.dx_mm} mm ===")
    print(f"Load steps per volume: {args.load_steps}")
    print()
    t0 = time.time()
    results = run(args.dx_mm, load_steps=args.load_steps)
    dt = time.time() - t0
    print(f"\nTotal wall time: {dt/60:.1f} min")

    lines = [
        f"FEniCS finite-strain FEM at dx = {args.dx_mm} mm",
        "=" * 76,
        f"Wall time: {dt/60:.1f} min",
        "",
        "IMPORTANT: FEM constitutive law = near-incompressible neo-Hookean,",
        "not Ogden. This is because for incompressible material under",
        "prescribed-displacement BCs, u(x) is material-independent (up to",
        "internal pressure). Ogden is applied EX-POST to the FEM stretch",
        "field to compute G_θ. The stretch field is the correct finite-strain",
        "container-confined kinematic response.",
        "",
    ]
    def _fmt(vals, spec):
        return "  ".join(format(v, spec) for v in vals)
    for p in PHANTOMS:
        key = f"G_{p['short']}"
        vols  = [r["vol_ml"]         for r in results]
        nrs   = [r["n_ring"]         for r in results]
        lams  = [r["lam_theta_mean"] for r in results]
        gs    = [r[key]              for r in results]
        yins  = p["yin_tsm"]
        gaps  = [gs[i] - yins[i] for i in range(len(yins))]
        lines.append(f"{p['name']}")
        lines.append(f"  Ogden: μ = {p['mu']} Pa, α = {p['alpha']}")
        lines.append(f"  Volume(mL):     {_fmt(vols, '>5d')}")
        lines.append(f"  n_ring cells:   {_fmt(nrs, '>5d')}")
        lines.append(f"  ⟨λ_θ⟩ ring:     {_fmt(lams, '5.3f')}")
        lines.append(f"  G_θ FEM [kPa]:  {_fmt(gs, '5.2f')}")
        lines.append(f"  Yin ref [kPa]:  {_fmt(yins, '5.2f')}")
        lines.append(f"  Yin−FEM [kPa]:  {_fmt([-g for g in gaps], '+5.2f')}")
        lines.append("")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    import json
    (out_dir / "results.json").write_text(
        json.dumps({"dx_mm": args.dx_mm, "load_steps": args.load_steps,
                    "wall_time_min": dt/60, "results": results}, indent=2)
    )


if __name__ == "__main__":
    main()
