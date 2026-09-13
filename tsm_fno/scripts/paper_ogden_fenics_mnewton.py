#!/usr/bin/env python3
"""Modified Newton solve: Lamé initial guess + fixed tangent from that guess.

Same 1/4-symmetry, near-incompressible neo-Hookean formulation as
paper_ogden_fenics_fem_quarter.py, but replaces the full Newton (which
re-factors the Jacobian each iteration and blows up at large deformation
with MUMPS error 76) with modified Newton (fixed Jacobian).

Method per Yin volume:
  1. Set u to the analytical Lamé infinite-matrix displacement field.
     Enforce Dirichlet BCs (walls zeroed, balloon nodes at prescribed
     values).  This is a well-conditioned kinematic state that satisfies
     incompressibility everywhere.
  2. Assemble the Jacobian J₀ = ∂R/∂u ONCE at this state.
  3. LU-factor J₀ once (MUMPS).
  4. Iterate:
       Rᵢ = residual at uᵢ, with BC lifting
       Δu = −J₀⁻¹ Rᵢ
       uᵢ₊₁ = uᵢ + Δu
     until ‖R‖ / ‖R₀‖ < rtol.

Trade-offs vs full Newton:
  +  MUMPS factorization runs ONCE per volume, not 3-6 times.
  +  Jacobian is evaluated at a well-defined kinematic state (Lamé),
     avoiding the ill-conditioning that Newton hits at large deformation.
  −  Convergence is LINEAR, so may need 10-30 iterations (vs 3-6 for
     quadratic full Newton).  Each iteration is O(N²) for the solve
     though (already factorised) so it's cheap.
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


HALF_L = 0.075
FULL_H = 0.18

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


def run(dx_mm: float, max_it: int, rtol: float):
    from mpi4py import MPI
    from dolfinx import fem, log, mesh
    from dolfinx.fem import petsc as fem_petsc
    from petsc4py import PETSc
    import ufl

    log.set_log_level(log.LogLevel.WARNING)   # too noisy at INFO

    dx = dx_mm * 1e-3
    Nx = int(round(HALF_L / dx))
    Ny = int(round(HALF_L / dx))
    Nz = int(round(FULL_H  / dx))
    print(f"Quarter mesh: {Nx}×{Ny}×{Nz} hex ({Nx*Ny*Nz:,} cells)", flush=True)

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

    cz = FULL_H / 2

    tol_geom = dx * 0.5
    def x0m(x): return np.isclose(x[0], 0.0,    atol=tol_geom)
    def xL(x):  return np.isclose(x[0], HALF_L, atol=tol_geom)
    def y0m(x): return np.isclose(x[1], 0.0,    atol=tol_geom)
    def yL(x):  return np.isclose(x[1], HALF_L, atol=tol_geom)
    def z0(x):  return np.isclose(x[2], 0.0,    atol=tol_geom)

    def wall_rigid_marker(x):
        return xL(x) | yL(x) | z0(x)

    def balloon_marker(x):
        r = np.sqrt(x[0]**2 + x[1]**2 + (x[2]-cz)**2)
        return r <= A0_M + tol_geom

    wall_dofs    = fem.locate_dofs_geometrical(V, wall_rigid_marker)
    balloon_dofs = fem.locate_dofs_geometrical(V, balloon_marker)
    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    dofs_x0 = fem.locate_dofs_geometrical((V.sub(0), Vx), x0m)
    dofs_y0 = fem.locate_dofs_geometrical((V.sub(1), Vy), y0m)

    u_zero    = fem.Function(V)
    u_zero_x  = fem.Function(Vx)
    u_zero_y  = fem.Function(Vy)
    u_bal     = fem.Function(V)

    bc_walls  = fem.dirichletbc(u_zero,   wall_dofs)
    bc_x0     = fem.dirichletbc(u_zero_x, dofs_x0, V.sub(0))
    bc_y0     = fem.dirichletbc(u_zero_y, dofs_y0, V.sub(1))
    bc_bal    = fem.dirichletbc(u_bal,    balloon_dofs)
    bcs = [bc_walls, bc_x0, bc_y0, bc_bal]

    print(f"BCs: walls={len(wall_dofs)}, x-sym={len(dofs_x0[0])}, "
          f"y-sym={len(dofs_y0[0])}, balloon={len(balloon_dofs)}", flush=True)

    u = fem.Function(V, name="displacement")
    v = ufl.TestFunction(V)
    du_trial = ufl.TrialFunction(V)

    I = ufl.Identity(3)
    F_ = I + ufl.grad(u)
    C_ = F_.T * F_
    J  = ufl.det(F_)

    MU_SOLVE = 1000.0
    mu_c    = fem.Constant(domain, MU_SOLVE)
    kappa_c = fem.Constant(domain, MU_SOLVE * 100.0)   # ν ≈ 0.495

    W_iso = (mu_c / 2.0) * (J ** (-2.0/3.0) * ufl.tr(C_) - 3.0)
    W_vol = (kappa_c / 2.0) * (J - 1.0) ** 2
    Pi = (W_iso + W_vol) * ufl.dx
    R_ufl = ufl.derivative(Pi, u, v)
    J_ufl = ufl.derivative(R_ufl, u, du_trial)

    R_form = fem.form(R_ufl)
    J_form = fem.form(J_ufl)

    # ── Balloon & Lamé updaters ──────────────────────────────────────
    def update_balloon_bc(a_now):
        du_r = a_now - A0_M
        def _expr(x):
            rx = x[0]; ry = x[1]; rz = x[2] - cz
            r  = np.sqrt(rx*rx + ry*ry + rz*rz)
            r_safe = np.where(r < 1e-12, 1e-12, r)
            scale = (r / A0_M) * du_r / r_safe
            return np.vstack([scale*rx, scale*ry, scale*rz])
        u_bal.interpolate(_expr)

    def lame_initial_guess(a_now):
        """Set u to the Lamé infinite-matrix displacement field."""
        def _expr(x):
            rx = x[0]; ry = x[1]; rz = x[2] - cz
            r  = np.sqrt(rx*rx + ry*ry + rz*rz)
            r_safe = np.where(r < 1e-12, 1e-12, r)
            R_ext = np.cbrt(r_safe**3 - A0_M**3 + a_now**3)
            u_r = np.where(r_safe <= A0_M,
                           (r_safe / A0_M) * (a_now - A0_M),
                           R_ext - r_safe)
            scale = u_r / r_safe
            return np.vstack([scale*rx, scale*ry, scale*rz])
        u.interpolate(_expr)

    # ── Ring extraction ──────────────────────────────────────────────
    Vt = fem.functionspace(domain, ("DG", 0, (3, 3)))
    from dolfinx.mesh import compute_midpoints

    def extract_lam_theta():
        F_field = fem.Function(Vt)
        F_field.interpolate(fem.Expression(F_, Vt.element.interpolation_points()))
        tdim = domain.topology.dim
        nc = domain.topology.index_map(tdim).size_local
        cells = np.arange(nc, dtype=np.int32)
        mid = compute_midpoints(domain, tdim, cells)
        r_mid = np.sqrt(mid[:, 0]**2 + mid[:, 1]**2 + (mid[:, 2]-cz)**2)
        ring = (r_mid >= A0_M + 0.003) & (r_mid <= A0_M + 0.008)
        ring_cells = cells[ring]
        F_vals = F_field.x.array.reshape(nc, 9)
        lam_theta = []
        for c in ring_cells:
            F_c = F_vals[c].reshape(3, 3)
            C_c = F_c.T @ F_c
            eig = np.sort(np.linalg.eigvalsh(C_c))[::-1]
            eig = np.clip(eig, 1e-12, None)
            s = np.sqrt(eig)
            lam_theta.append(0.5 * (s[0] + s[1]))
        return np.array(lam_theta)

    def ring_G_ogden(lam, mu_list, alpha_list):
        G = np.zeros_like(lam)
        for m, a in zip(mu_list, alpha_list):
            G += m * np.power(lam, a - 2.0)
        lo, hi = np.percentile(G, [10, 90])
        tr = G[(G >= lo) & (G <= hi)]
        return float(tr.mean()) / 1000

    # ── Modified Newton core ─────────────────────────────────────────
    def _residual_norm():
        """Compute the current residual norm (with BC lifting)."""
        with b_scratch.localForm() as b_loc:
            b_loc.set(0.0)
        fem_petsc.assemble_vector(b_scratch, R_form)
        fem_petsc.apply_lifting(b_scratch, [J_form], bcs=[bcs],
                                 x0=[u.x.petsc_vec], alpha=-1.0)
        b_scratch.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                                mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b_scratch, bcs, u.x.petsc_vec, alpha=-1.0)
        return b_scratch.norm()

    A_global = None
    b_scratch = None

    def modified_newton_solve(a_tgt, max_it, rtol_local):
        """Modified Newton with fixed tangent from Lamé + backtracking line search."""
        nonlocal A_global, b_scratch

        # 1) Set balloon BC and Lamé initial u
        update_balloon_bc(a_tgt)
        lame_initial_guess(a_tgt)
        fem_petsc.set_bc(u.x.petsc_vec, bcs)
        u.x.scatter_forward()

        # 2) Assemble Jacobian ONCE at this Lamé state
        if A_global is None:
            A_global = fem_petsc.create_matrix(J_form)
            b_scratch = fem_petsc.create_vector(R_form)
        A_global.zeroEntries()
        fem_petsc.assemble_matrix(A_global, J_form, bcs=bcs)
        A_global.assemble()
        print(f"    J₀ assembled, nnz = {A_global.getInfo()['nz_used']:.0f}",
              flush=True)

        # 3) LU factorise once
        ksp = PETSc.KSP().create(domain.comm)
        ksp.setOperators(A_global)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        t_fac0 = time.time()
        ksp.setUp()
        t_fac1 = time.time()
        print(f"    MUMPS factor: {t_fac1-t_fac0:.1f} s", flush=True)

        # 4) Iterate with backtracking line search
        du_v = A_global.createVecLeft()
        u_prev = u.x.petsc_vec.copy()  # for line search rollback

        r0 = _residual_norm()
        r0 = max(r0, 1e-30)
        print(f"    mNewton iter 0: |r|={r0:.3e}, rel=1.000e+00", flush=True)
        if r0 < 1e-10:
            return 0, True

        for it in range(1, max_it + 1):
            # Assemble residual b at current u (again)
            with b_scratch.localForm() as b_loc:
                b_loc.set(0.0)
            fem_petsc.assemble_vector(b_scratch, R_form)
            fem_petsc.apply_lifting(b_scratch, [J_form], bcs=[bcs],
                                     x0=[u.x.petsc_vec], alpha=-1.0)
            b_scratch.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                                    mode=PETSc.ScatterMode.REVERSE)
            fem_petsc.set_bc(b_scratch, bcs, u.x.petsc_vec, alpha=-1.0)
            r_current = b_scratch.norm()

            # Solve fixed system: A · Δu = b (residual)
            ksp.solve(b_scratch, du_v)

            # Save current u for potential rollback
            u_prev.copy(u.x.petsc_vec)

            # Backtracking line search on α ∈ {1, 0.5, 0.25, 0.125, 0.0625}
            alpha = 1.0
            for _ in range(6):
                u_prev.copy(u.x.petsc_vec)
                u.x.petsc_vec.axpy(-alpha, du_v)
                u.x.scatter_forward()
                r_trial = _residual_norm()
                if np.isfinite(r_trial) and r_trial < r_current:
                    break
                # rollback and shrink
                u_prev.copy(u.x.petsc_vec)
                u.x.scatter_forward()
                alpha *= 0.5
            else:
                print(f"    mNewton iter {it}: line search failed "
                      f"(all α gave increase or NaN)", flush=True)
                return it, False

            r_new = r_trial
            r_rel = r_new / r0
            print(f"    mNewton iter {it}: |r|={r_new:.3e}, rel={r_rel:.3e}, "
                  f"α={alpha:.3f}", flush=True)
            if r_new < 1e-10 or r_rel < rtol_local:
                return it, True

        return max_it, False

    # ── Main loop over Yin volumes ────────────────────────────────────
    results = []
    for vol_ml, a_tgt in zip(BALLOON_VOLUMES_ML, BALLOON_RADII_M):
        print(f"\n=== Volume {vol_ml} mL, a = {a_tgt*1000:.2f} mm ===",
              flush=True)
        t0 = time.time()
        try:
            n_iter, converged = modified_newton_solve(a_tgt, max_it, rtol)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            return results
        t1 = time.time()
        print(f"  → {n_iter} iters, {t1-t0:.1f} s, converged={converged}",
              flush=True)
        if not converged:
            print("    ABORT: not converged, stopping.", flush=True)
            return results

        lam_ring = extract_lam_theta()
        row = {"vol_ml": vol_ml, "a_m": a_tgt,
               "n_ring": int(len(lam_ring)),
               "lam_theta_mean": float(np.mean(lam_ring)) if lam_ring.size else float("nan")}
        for p in PHANTOMS:
            row[f"G_{p['short']}"] = ring_G_ogden(lam_ring, p["mu"], p["alpha"])
        results.append(row)
        print(f"  RING: n={row['n_ring']}, ⟨λ_θ⟩={row['lam_theta_mean']:.4f}, "
              f"G_P1={row['G_P1']:.2f}, G_P2={row['G_P2']:.2f} kPa",
              flush=True)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx-mm", type=float, default=3.0)
    ap.add_argument("--max-it", type=int, default=100)
    ap.add_argument("--rtol", type=float, default=1e-6)
    args = ap.parse_args()

    out_dir = ROOT / "results" / f"paper_ogden_fenics_mnewton_dx{int(args.dx_mm)}mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Modified Newton FEM at dx = {args.dx_mm} mm ===")
    print(f"max_it={args.max_it}, rtol={args.rtol}")
    print()
    t0 = time.time()
    results = run(args.dx_mm, args.max_it, args.rtol)
    dt = time.time() - t0
    print(f"\nTotal wall time: {dt/60:.1f} min")
    print(f"Completed volumes: {len(results)}/5")

    def _fmt(vals, spec):
        return "  ".join(format(v, spec) for v in vals)

    lines = [
        f"Modified Newton FEM at dx = {args.dx_mm} mm",
        "=" * 72,
        f"Wall time: {dt/60:.1f} min   |   Volumes completed: {len(results)}/5",
        "",
        "Method: Lamé initial guess + Jacobian assembled ONCE, LU-factored",
        "        once, reused across all iterations (modified Newton).",
        "",
    ]
    for p in PHANTOMS:
        key = f"G_{p['short']}"
        vols  = [r["vol_ml"]         for r in results]
        lams  = [r["lam_theta_mean"] for r in results]
        gs    = [r[key]              for r in results]
        yins  = p["yin_tsm"][:len(vols)]
        lines.append(f"{p['name']}")
        lines.append(f"  Ogden: μ = {p['mu']} Pa, α = {p['alpha']}")
        lines.append(f"  Volume(mL):     {_fmt(vols, '>5d')}")
        lines.append(f"  ⟨λ_θ⟩ ring:     {_fmt(lams, '5.3f')}")
        lines.append(f"  G_θ FEM [kPa]:  {_fmt(gs, '5.2f')}")
        lines.append(f"  Yin ref [kPa]:  {_fmt(yins, '5.2f')}")
        gaps = [yins[i] - gs[i] for i in range(len(yins))]
        lines.append(f"  Yin−FEM [kPa]:  {_fmt(gaps, '+5.2f')}")
        lines.append("")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    (out_dir / "results.json").write_text(
        json.dumps({"dx_mm": args.dx_mm, "max_it": args.max_it,
                    "rtol": args.rtol, "wall_time_min": dt/60,
                    "results": results}, indent=2))


if __name__ == "__main__":
    main()
