"""Export the perilesional stiffness distribution for the MRE pipeline.

The medium is NOT isotropic.  Acoustoelastic prestrain makes it locally transversely
isotropic about the radial direction:

    mu_theta(x) = mu0 f(I1) lam_theta^2     tangential propagation   (stiff)
    mu_r(x)     = mu0 f(I1) lam_theta^-4    radial propagation       (soft)
    mu_app(D)   = mu0 f(I1) (lam_r^2 cos^2 D + lam_theta^2 sin^2 D)

with D the angle between the propagation direction and the local radial direction
n(x) = x/|x|.  That directional dependence IS the TSM signal: feeding a single scalar
field to an isotropic solver gives mu_TSM == mu_conv and therefore zero contrast.

Three ways to consume this, in increasing fidelity:
  (a) scalar, isotropic   -- use mu_conv (the direction average).  Drop-in for the
      existing helmholtz_solve(G, ...), but produces no TSM/conventional contrast.
  (b) direction-resolved  -- call field_for_direction(n) once per DF direction and
      solve the isotropic problem each time.  Reproduces the TSM construction with
      an unmodified scalar solver.  This is the cheap route to a synthetic ring.
  (c) anisotropic solver  -- use mu_theta, mu_r and the radial field n directly.
"""
import numpy as np, os
from sce_models import (P1, P2, cavity_radius, lam_theta, I1, f_stiffen,
                        V_STATES, VOX, ROI_WID)

FOV_N = 80                     # 80^3 at 3 mm = 24 cm, matching the acquisition
RHO   = 1000.0                 # kg/m^3, gel ~ water

def grid(n=FOV_N, dx=VOX):
    c = (np.arange(n) - (n - 1) / 2.0) * dx        # cm, centred on the cavity
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return X, Y, Z, np.sqrt(X**2 + Y**2 + Z**2)

def fields(V, mat, n=FOV_N, dx=VOX):
    """Principal moduli, radial direction field and cavity mask at volume V."""
    a = cavity_radius(V)
    X, Y, Z, R = grid(n, dx)
    gel = R >= a                                    # cavity carries no shear modulus
    Rs = np.where(gel, R, a)
    lt = lam_theta(Rs, a)
    f  = f_stiffen(lt, mat)
    mu_t = np.where(gel, mat["mu0"] * f * lt**2,  np.nan)      # tangential
    mu_r = np.where(gel, mat["mu0"] * f * lt**-4, np.nan)      # radial
    with np.errstate(invalid="ignore", divide="ignore"):
        nx, ny, nz = X / R, Y / R, Z / R                       # local radial unit vector
    mu_conv = (mu_r + 2 * mu_t) / 3.0                          # isotropic direction average
    return dict(mu_theta=mu_t, mu_r=mu_r, mu_conv=mu_conv,
                n_x=nx, n_y=ny, n_z=nz, gel=gel, a_cm=a, dx_cm=dx)

def field_for_direction(fl, d):
    """Scalar modulus seen by a plane wave propagating along unit vector d.

    mu(D) = mu_r cos^2 D + mu_theta sin^2 D,  cos D = d . n(x).
    Feed this to an isotropic solver, one call per DF direction (route (b))."""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    cosD = fl["n_x"]*d[0] + fl["n_y"]*d[1] + fl["n_z"]*d[2]
    return fl["mu_r"]*cosD**2 + fl["mu_theta"]*(1.0 - cosD**2)

def dodecahedral_directions():
    """20 directions on a dodecahedral set, as used by the source study's filter bank."""
    p = (1 + 5**0.5) / 2
    v = [(s1, s2, s3) for s1 in (1,-1) for s2 in (1,-1) for s3 in (1,-1)]
    v += [(0, s1/p, s2*p) for s1 in (1,-1) for s2 in (1,-1)]
    v += [(s1/p, s2*p, 0) for s1 in (1,-1) for s2 in (1,-1)]
    v += [(s1*p, 0, s2/p) for s1 in (1,-1) for s2 in (1,-1)]
    a = np.array(v, float)
    return a / np.linalg.norm(a, axis=1, keepdims=True)

def export(outdir=None, nifti=True):
    """Write to <script dir>/stiffness_export unless told otherwise."""
    if outdir is None:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stiffness_export")
    os.makedirs(outdir, exist_ok=True)
    written = []
    for tag, mat in (("P1", P1), ("P2", P2)):
        for V in V_STATES:
            fl = fields(V, mat)
            base = f"{tag}_{int(V)}mL"
            np.savez_compressed(
                os.path.join(outdir, base + ".npz"),
                mu_theta=fl["mu_theta"].astype(np.float32),
                mu_r=fl["mu_r"].astype(np.float32),
                mu_conv=fl["mu_conv"].astype(np.float32),
                n_x=fl["n_x"].astype(np.float32), n_y=fl["n_y"].astype(np.float32),
                n_z=fl["n_z"].astype(np.float32), gel=fl["gel"],
                a_cm=fl["a_cm"], dx_cm=fl["dx_cm"], volume_mL=V, rho=RHO,
                mu0_kPa=mat["mu0"], phantom=tag, shell_mm=ROI_WID*10,
                note="mu in kPa; NaN inside the cavity; n is the local radial unit vector")
            written.append(base + ".npz")
            if nifti:
                try:
                    import nibabel as nib
                    aff = np.diag([fl["dx_cm"]*10, fl["dx_cm"]*10, fl["dx_cm"]*10, 1.0])
                    for k in ("mu_theta", "mu_r", "mu_conv"):
                        nib.save(nib.Nifti1Image(np.nan_to_num(fl[k]).astype(np.float32), aff),
                                 os.path.join(outdir, f"{base}_{k}.nii.gz"))
                except ImportError:
                    pass
    return written

if __name__ == "__main__":
    w = export()
    print(f"wrote {len(w)} .npz volumes (+ NIfTI) to stiffness_export/\n")
    fl = fields(250., P2)
    print("Phantom 2 at 250 mL, 80^3 grid at 3 mm:")
    g = fl["gel"]
    print(f"   cavity radius        {fl['a_cm']:.2f} cm")
    print(f"   mu_theta  {np.nanmin(fl['mu_theta']):6.2f} - {np.nanmax(fl['mu_theta']):6.2f} kPa")
    print(f"   mu_r      {np.nanmin(fl['mu_r']):6.2f} - {np.nanmax(fl['mu_r']):6.2f} kPa")
    print(f"   anisotropy mu_theta/mu_r at the cavity: "
          f"{np.nanmax(fl['mu_theta'])/np.nanmin(fl['mu_r']):.1f}")
    d = dodecahedral_directions()
    print(f"\n   {len(d)} DF directions; modulus seen along the first one:")
    Gd = field_for_direction(fl, d[0])
    print(f"      {np.nanmin(Gd):.2f} - {np.nanmax(Gd):.2f} kPa "
          f"(isotropic mu_conv: {np.nanmin(fl['mu_conv']):.2f} - {np.nanmax(fl['mu_conv']):.2f})")
