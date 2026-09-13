#!/usr/bin/env python3
"""Nonlinear finite-strain FEM inside Yin's container — 1/4 SYMMETRY model.

Same physics as `paper_ogden_fenics_fem.py` but exploits the two mirror
planes (x=0 and y=0) to reduce the mesh by 4×. This makes the ideal
1 mm resolution reachable (3.1M vector DOFs vs 12.4M full).

Symmetry planes
---------------
The container is a 15×15×18 cm box with the balloon at its centre and
z-asymmetric BCs (bottom fixed, top free). This gives two mirror
symmetries (x=centre, y=centre) but NO z symmetry.

Modeled domain in this script:  [0, L/2] × [0, L/2] × [0, H]
Balloon centre at corner (0, 0, H/2). The balloon is spherical, so
the modeled region contains 1/4 of the full balloon.

BCs on the modeled domain
-------------------------
  x = 0      → symmetry, u_x = 0    (partial-component Dirichlet)
  x = L/2    → rigid wall, u = 0
  y = 0      → symmetry, u_y = 0    (partial-component Dirichlet)
  y = L/2    → rigid wall, u = 0
  z = 0      → rigid wall, u = 0    (bottom of container)
  z = H      → free                 (natural, no BC applied)
  balloon-interior nodes → prescribed u = (r/a₀)·(a−a₀)·r̂

Mesh
----
  dx = 1 mm → 75×75×180 =   1.01M hex, ~3.1M vector DOFs  ✓ tractable
  dx = 2 mm → 38×38×90  =    130k hex, ~400k vector DOFs
  dx = 3 mm → 25×25×60  =     37k hex, ~115k vector DOFs

Constitutive law
----------------
FEM solve uses near-incompressible neo-Hookean (see main script docstring
for the rationale — displacement is material-independent under
incompressibility + prescribed-displacement BCs). Ogden is applied
ex-post to the FEM stretch field.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


HALF_L = 0.075   # half-width of container in x, y (15 cm total → 7.5 cm modelled)
FULL_H = 0.18    # full container height 18 cm

_SOLVER_CHOICE = "cg-gamg"  # default; overridden in main()

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
    from petsc4py import PETSc
    import ufl

    # INFO shows Newton iteration residuals — critical for diagnosing
    # convergence problems.
    log.set_log_level(log.LogLevel.INFO)

    dx = dx_mm * 1e-3
    Nx = int(round(HALF_L / dx))
    Ny = int(round(HALF_L / dx))
    Nz = int(round(FULL_H  / dx))
    print(f"Quarter mesh: {Nx}×{Ny}×{Nz} hex ({Nx*Ny*Nz:,} cells)")
    print(f"Domain: {Nx*dx*100:.1f}×{Ny*dx*100:.1f}×{Nz*dx*100:.1f} cm "
          f"(quarter of {2*Nx*dx*100:.1f}×{2*Ny*dx*100:.1f}×{Nz*dx*100:.1f} cm container)")

    comm = MPI.COMM_WORLD
    domain = mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]),
         np.array([HALF_L, HALF_L, FULL_H])],
        [Nx, Ny, Nz],
        cell_type=mesh.CellType.hexahedron,
    )

    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"Vector DOFs (quarter): {n_dof:,}", flush=True)

    # Balloon center: at corner (0, 0, H/2)
    cz = FULL_H / 2

    # ── Boundary markers ─────────────────────────────────────────────
    tol = dx * 0.5
    def x0(x):    return np.isclose(x[0], 0.0,    atol=tol)   # symmetry x=0
    def xL(x):    return np.isclose(x[0], HALF_L, atol=tol)   # rigid wall
    def y0(x):    return np.isclose(x[1], 0.0,    atol=tol)   # symmetry y=0
    def yL(x):    return np.isclose(x[1], HALF_L, atol=tol)   # rigid wall
    def z0(x):    return np.isclose(x[2], 0.0,    atol=tol)   # rigid bottom

    def wall_rigid_marker(x):    # all 3 rigid walls (xL, yL, z0)
        return xL(x) | yL(x) | z0(x)

    def balloon_marker(x):
        r = np.sqrt(x[0]**2 + x[1]**2 + (x[2]-cz)**2)
        return r <= A0_M + dx * 0.5

    # ── Dirichlet BCs ────────────────────────────────────────────────
    # (1) Rigid walls: u=0 on all 3 components
    wall_dofs = fem.locate_dofs_geometrical(V, wall_rigid_marker)
    u_zero = fem.Function(V)
    bc_walls = fem.dirichletbc(u_zero, wall_dofs)

    # (2) Symmetry planes: only NORMAL component clamped
    # Sub-space accessors: V.sub(0) is u_x, V.sub(1) is u_y, V.sub(2) is u_z
    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()

    def locate_dofs_subspace(V_full, sub, marker):
        # Returns (parent_dofs, sub_dofs)
        return fem.locate_dofs_geometrical((V_full.sub(sub), V_full.sub(sub).collapse()[0]),
                                             marker)

    # For x=0 plane, u_x = 0
    dofs_x0_pair = fem.locate_dofs_geometrical(
        (V.sub(0), Vx), x0)
    u_zero_x = fem.Function(Vx)
    bc_x0 = fem.dirichletbc(u_zero_x, dofs_x0_pair, V.sub(0))

    # For y=0 plane, u_y = 0
    dofs_y0_pair = fem.locate_dofs_geometrical(
        (V.sub(1), Vy), y0)
    u_zero_y = fem.Function(Vy)
    bc_y0 = fem.dirichletbc(u_zero_y, dofs_y0_pair, V.sub(1))

    # (3) Balloon Dirichlet — will be updated per load step
    balloon_dofs = fem.locate_dofs_geometrical(V, balloon_marker)
    u_bal_persistent = fem.Function(V)
    bc_bal = fem.dirichletbc(u_bal_persistent, balloon_dofs)

    print(f"Boundary DOFs: rigid_walls={len(wall_dofs)}, "
          f"x0_symm={len(dofs_x0_pair[0])}, y0_symm={len(dofs_y0_pair[0])}, "
          f"balloon={len(balloon_dofs)}", flush=True)

    # ── Neo-Hookean weak form (full domain) ──────────────────────────
    u = fem.Function(V, name="displacement")
    v = ufl.TestFunction(V)

    I = ufl.Identity(3)
    F_tensor = I + ufl.grad(u)
    C_tensor = F_tensor.T * F_tensor
    J = ufl.det(F_tensor)

    MU_SOLVE = 1000.0
    # κ/μ = 10 gives ν ≈ 0.478. Not fully incompressible but numerically
    # much easier at large deformation. Displacement will be slightly off
    # from true incompressible, but ring stretch is dominated by kinematics
    # not by exact volume conservation, so the effect on G_θ is small.
    mu_c    = fem.Constant(domain, MU_SOLVE)
    kappa_c = fem.Constant(domain, MU_SOLVE * 10.0)

    W_iso = (mu_c / 2.0) * (J ** (-2.0/3.0) * ufl.tr(C_tensor) - 3.0)
    W_vol = (kappa_c / 2.0) * (J - 1.0) ** 2
    Pi = (W_iso + W_vol) * ufl.dx      # full-domain integral (tissue-everywhere)
    R  = ufl.derivative(Pi, u, v)

    problem = NonlinearProblem(R, u, bcs=[bc_walls, bc_x0, bc_y0, bc_bal])
    solver = NewtonSolver(comm, problem)
    solver.rtol = 1e-6
    solver.atol = 1e-8
    solver.max_it = 50
    solver.convergence_criterion = "residual"
    solver.report = True

    ksp = solver.krylov_solver
    solver_choice = _SOLVER_CHOICE
    if solver_choice == "lu":
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        print("Using direct LU (MUMPS)", flush=True)
    else:
        # CG + GAMG for large problems. Elasticity is SPD → CG is fine.
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.GAMG)
        # GAMG configured for elasticity — use smoothed aggregation, sensible
        # coarsening and inner smoother.
        petsc_opts = PETSc.Options()
        petsc_opts.setValue("pc_gamg_type", "agg")
        petsc_opts.setValue("pc_gamg_agg_nsmooths", "1")
        petsc_opts.setValue("pc_gamg_threshold", "0.02")
        petsc_opts.setValue("pc_gamg_coarse_eq_limit", "2000")
        petsc_opts.setValue("mg_levels_ksp_type", "chebyshev")
        petsc_opts.setValue("mg_levels_pc_type", "jacobi")
        petsc_opts.setValue("mg_levels_ksp_chebyshev_esteig_steps", "10")
        petsc_opts.setValue("mg_coarse_ksp_type", "preonly")
        petsc_opts.setValue("mg_coarse_pc_type", "lu")
        petsc_opts.setValue("mg_coarse_pc_factor_mat_solver_type", "mumps")
        ksp.setFromOptions()
        print("Using CG + GAMG (iterative, chebyshev/jacobi smoother)",
              flush=True)

    # ── Balloon-BC updater ───────────────────────────────────────────
    def update_balloon_bc(a_now):
        du_r = a_now - A0_M
        def _expr(x):
            rx = x[0];  ry = x[1];  rz = x[2] - cz
            r  = np.sqrt(rx*rx + ry*ry + rz*rz)
            r_safe = np.where(r < 1e-12, 1e-12, r)
            scale  = (r / A0_M) * du_r / r_safe
            return np.vstack([scale*rx, scale*ry, scale*rz])
        u_bal_persistent.interpolate(_expr)

    # ── Lamé infinite-matrix initial guess ────────────────────────────
    # For prescribed balloon inflation from a₀ to a in infinite incompressible
    # matrix, radial deformation is:  R(r₀) = (r₀³ − a₀³ + a³)^(1/3)
    # displacement:                    u_r(r₀) = R(r₀) − r₀
    # For r₀ ≤ a₀ (balloon interior), use "rigid balloon growth":
    #                                  u_r(r₀) = (r₀ / a₀) · (a − a₀)
    # This initial guess satisfies incompressibility everywhere and
    # equilibrium in the far field; only the container BCs perturb it.
    def lame_initial_guess(a_now):
        def _expr(x):
            rx = x[0]; ry = x[1]; rz = x[2] - cz
            r  = np.sqrt(rx*rx + ry*ry + rz*rz)
            r_safe = np.where(r < 1e-12, 1e-12, r)
            # Interior: rigid growth  u_r = (r/a₀)·(a−a₀)
            # Exterior: Lamé          u_r = R(r) − r
            R_ext = np.cbrt(r_safe**3 - A0_M**3 + a_now**3)
            u_r = np.where(r_safe <= A0_M,
                           (r_safe / A0_M) * (a_now - A0_M),
                           R_ext - r_safe)
            scale = u_r / r_safe
            return np.vstack([scale*rx, scale*ry, scale*rz])
        u.interpolate(_expr)

    # ── Ring extraction (from the quarter-domain) ────────────────────
    Vt = fem.functionspace(domain, ("DG", 0, (3, 3)))
    from dolfinx.mesh import compute_midpoints

    def extract_lam_theta():
        F_expr = fem.Expression(F_tensor, Vt.element.interpolation_points())
        F_field = fem.Function(Vt)
        F_field.interpolate(F_expr)
        tdim = domain.topology.dim
        n_cells_local = domain.topology.index_map(tdim).size_local
        cell_indices = np.arange(n_cells_local, dtype=np.int32)
        midpts = compute_midpoints(domain, tdim, cell_indices)
        r_mid = np.sqrt(midpts[:, 0]**2 + midpts[:, 1]**2 +
                          (midpts[:, 2]-cz)**2)
        ring_mask = (r_mid >= A0_M + 0.003) & (r_mid <= A0_M + 0.008)
        ring_cells = cell_indices[ring_mask]
        F_values = F_field.x.array.reshape(n_cells_local, 9)
        lam_theta_list = []
        for c in ring_cells:
            F_c = F_values[c].reshape(3, 3)
            C_c = F_c.T @ F_c
            eig = np.sort(np.linalg.eigvalsh(C_c))[::-1]
            eig = np.clip(eig, 1e-12, None)
            stretch = np.sqrt(eig)
            # Two largest = tangential; smallest = radial.
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
    # Standard load-stepped Newton (no warm start — Lamé caused ill-
    # conditioning at Dirichlet walls). Previous state u carries over
    # between load steps and volumes, providing a natural warm start.
    results = []
    step_results = []
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
                return {"results": results, "step_results": step_results}
            t1 = time.time()
            print(f"  step {step+1}/{load_steps}: a={a_now*1000:.2f} mm  "
                  f"[{n_iter} iters, {t1-t0:.1f} s, converged={converged}]",
                  flush=True)
            if not converged:
                print("    ABORT: Newton did not converge, stopping.", flush=True)
                return {"results": results, "step_results": step_results}
            step_lam = extract_lam_theta()
            step_row = {"target_vol_ml": vol_ml, "step": step + 1,
                        "a_m": a_now,
                        "lam_theta_mean": float(np.mean(step_lam)) if step_lam.size else float("nan")}
            for p in PHANTOMS:
                step_row[f"G_{p['short']}"] = ring_G_ogden(step_lam, p["mu"], p["alpha"])
            step_results.append(step_row)

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

    return {"results": results, "step_results": step_results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx-mm", type=float, default=3.0)
    ap.add_argument("--load-steps", type=int, default=5)
    ap.add_argument("--solver", choices=("lu", "cg-gamg"), default="cg-gamg",
                     help="linear solver: 'lu' (MUMPS direct) or 'cg-gamg' (iterative)")
    args = ap.parse_args()
    global _SOLVER_CHOICE
    _SOLVER_CHOICE = args.solver

    out_dir = ROOT / "results" / f"paper_ogden_fenics_fem_qtr_dx{int(args.dx_mm)}mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== FEniCS 1/4-symmetry FEM at dx = {args.dx_mm} mm ===")
    print(f"Load steps per volume: {args.load_steps}")
    print()
    t0 = time.time()
    out = run(args.dx_mm, load_steps=args.load_steps)
    results = out.get("results", [])
    step_results = out.get("step_results", [])
    dt = time.time() - t0
    print(f"\nTotal wall time: {dt/60:.1f} min")
    print(f"Completed volumes: {len(results)}, total load steps: {len(step_results)}")

    def _fmt(vals, spec):
        return "  ".join(format(v, spec) for v in vals)

    lines = [
        f"FEniCS 1/4-symmetry finite-strain FEM at dx = {args.dx_mm} mm",
        "=" * 76,
        f"Wall time: {dt/60:.1f} min",
        f"Mirror planes: x = 0 (u_x = 0), y = 0 (u_y = 0)",
        "Constitutive law in FEM: near-incompressible neo-Hookean.",
        "Ogden applied ex-post to the FEM stretch field.",
        "",
    ]
    for p in PHANTOMS:
        key = f"G_{p['short']}"
        vols  = [r["vol_ml"]         for r in results]
        nrs   = [r["n_ring"]         for r in results]
        lams  = [r["lam_theta_mean"] for r in results]
        gs    = [r[key]              for r in results]
        yins  = p["yin_tsm"][:len(vols)]
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

    (out_dir / "results.json").write_text(
        json.dumps({"dx_mm": args.dx_mm, "load_steps": args.load_steps,
                    "symmetry": "1/4 (x, y mirror planes)",
                    "wall_time_min": dt/60,
                    "results": results,
                    "step_results": step_results}, indent=2))


if __name__ == "__main__":
    main()
