#!/usr/bin/env python3
"""Minimal FEniCS test: small inflation in a coarse mesh.

Verifies:
  - Mesh generation works
  - BCs (walls + balloon Dirichlet) are set correctly
  - Newton converges for a small deformation
"""
import math
import sys
import time

import numpy as np
from mpi4py import MPI
from dolfinx import fem, log, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
import ufl


log.set_log_level(log.LogLevel.INFO)   # verbose Newton output

# Small container, coarse mesh: 3 cm × 3 cm × 4 cm at 3 mm resolution
L, H = 0.03, 0.04
Nx, Ny, Nz = 10, 10, 13
dx = L / Nx

comm = MPI.COMM_WORLD
domain = mesh.create_box(
    comm,
    [np.array([0.0, 0.0, 0.0]), np.array([L, L, H])],
    [Nx, Ny, Nz],
    cell_type=mesh.CellType.hexahedron,
)
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
print(f"Mesh: {Nx}×{Ny}×{Nz}, DOFs = {n_dof}", flush=True)

cx, cy, cz = L/2, L/2, H/2
a0 = 0.008  # 8 mm initial balloon radius

def wall_marker(x):
    tol = dx * 0.5
    return (np.isclose(x[0], 0.0, atol=tol) | np.isclose(x[0], L, atol=tol) |
            np.isclose(x[1], 0.0, atol=tol) | np.isclose(x[1], L, atol=tol) |
            np.isclose(x[2], 0.0, atol=tol))

def balloon_marker(x):
    r = np.sqrt((x[0]-cx)**2 + (x[1]-cy)**2 + (x[2]-cz)**2)
    return r <= a0 + dx * 0.5

wall_dofs = fem.locate_dofs_geometrical(V, wall_marker)
bal_dofs  = fem.locate_dofs_geometrical(V, balloon_marker)
print(f"Wall DOFs: {len(wall_dofs)}, Balloon DOFs: {len(bal_dofs)}", flush=True)

u_zero = fem.Function(V)
bc_walls = fem.dirichletbc(u_zero, wall_dofs)

u_bal = fem.Function(V)
def bal_expr(x, du_r):
    rx = x[0] - cx; ry = x[1] - cy; rz = x[2] - cz
    r = np.sqrt(rx*rx + ry*ry + rz*rz)
    r_safe = np.where(r < 1e-12, 1e-12, r)
    scale = (r / a0) * du_r / r_safe
    return np.vstack([scale*rx, scale*ry, scale*rz])

# Small inflation: 5% at balloon surface (0.4 mm out of 8 mm)
DU = 0.0004
u_bal.interpolate(lambda x: bal_expr(x, DU))
bc_bal = fem.dirichletbc(u_bal, bal_dofs)

u = fem.Function(V, name="u")
v = ufl.TestFunction(V)

I = ufl.Identity(3)
F_ = I + ufl.grad(u)
C_ = F_.T * F_
J  = ufl.det(F_)

MU  = 1000.0
KAP = 1000 * MU   # near-incompressible
mu_c = fem.Constant(domain, MU)
kap_c = fem.Constant(domain, KAP)
W_iso = (mu_c / 2.0) * (J ** (-2.0/3.0) * ufl.tr(C_) - 3.0)
W_vol = (kap_c / 2.0) * (J - 1.0) ** 2
Pi = (W_iso + W_vol) * ufl.dx
R  = ufl.derivative(Pi, u, v)

problem = NonlinearProblem(R, u, bcs=[bc_walls, bc_bal])
solver = NewtonSolver(comm, problem)
solver.rtol = 1e-6
solver.atol = 1e-8
solver.max_it = 30
solver.report = True

PETSc.Options().setValue("ksp_type", "preonly")
PETSc.Options().setValue("pc_type", "lu")
PETSc.Options().setValue("pc_factor_mat_solver_type", "mumps")
ksp = solver.krylov_solver
ksp.setFromOptions()

print("Starting Newton solve...", flush=True)
t0 = time.time()
n_iter, converged = solver.solve(u)
t1 = time.time()
print(f"Newton: {n_iter} iters, converged={converged}, {t1-t0:.2f} s", flush=True)

# Sanity: extract u_r at a point near balloon
sample_r = a0 + 0.002  # 2 mm outside
sample_pt = np.array([cx + sample_r, cy, cz]).reshape(1, 3)
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points
tree = bb_tree(domain, domain.topology.dim)
cand = compute_collisions_points(tree, sample_pt)
coll = compute_colliding_cells(domain, cand, sample_pt)
cell = coll.array[0] if len(coll.array) else -1
if cell >= 0:
    val = u.eval(sample_pt, np.array([cell], dtype=np.int32))
    print(f"u at ({sample_r*1000:.1f} mm out): {val*1e3} mm", flush=True)
