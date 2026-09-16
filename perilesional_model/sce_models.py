"""Spherical cavity expansion kinematics + per-phantom acoustoelastic moduli.

Reference state = the 50 mL cast state (gel solidified around the balloon at 50 mL),
so lambda_theta == 1 everywhere at V = 50 mL.
"""
import numpy as np

V_STATES = np.array([50., 100., 150., 200., 250.])   # mL
VOX      = 0.3                                        # cm, 3 mm acquisition voxel
ROI_OFF  = 0.0                                        # shell starts at the cavity edge: no standoff
ROI_WID  = 4*VOX                                      # cm, 12 mm shell = exactly 4 acquisition voxels
DR_ROI   = ROI_WID                                    # kept for callers that expect a thickness
A_REF    = (3*50./(4*np.pi))**(1/3.)                  # cm, reference cavity radius (50 mL)

# Both phantoms are reported as neo-Hookean (f == 1).  mu0 is not free: lam_theta == 1
# at the 50 mL cast state pins it to the measured baseline of the digitised Fig. 6 series.
# On this shell the data require f_req < 1 for both phantoms, which no strain-stiffening
# law can supply, so a fibre term is not carried -- see paper Sec. 3.
P1 = dict(name="Phantom 1 (gelatin, neo-Hookean)", mu0=3.53)
# P2 carries a fibre term identified from the P1-normalised ratio, in which the
# kinematics and any measurement bias common to the two phantoms cancel:
#   f_P2(g) = 1 + c g^m,  g = I1 - 3,  fitted so the SHELL AVERAGE matches.
P2 = dict(name="Phantom 2 (gelatin+cellulose, fibre-reinforced)",
          mu0=3.34, c=0.2728, m=0.5791)

def cavity_radius(V):
    """Current cavity radius a (cm) for injected volume V (mL)."""
    return (3*V/(4*np.pi))**(1/3.)

def lam_theta(r, a, A=A_REF):
    """Hoop stretch at current radius r, for cavity at a. r^3 = R^3 + (a^3 - A^3)."""
    r = np.asarray(r, float)
    R3 = r**3 - (a**3 - A**3)
    R3 = np.maximum(R3, 1e-12)
    return r / R3**(1/3.)

def I1(lt):
    """First invariant, incompressible spherical: lam_r = lt^-2."""
    return lt**-4 + 2*lt**2

def f_stiffen(lt, mat):
    """Constitutive stiffening factor f = (2/mu0) dW/dI1."""
    lt = np.asarray(lt, float)
    if "c" in mat:                                     # power-law fibre recruitment
        g = np.maximum(I1(lt) - 3.0, 0.0)
        return 1.0 + mat["c"]*g**mat["m"]
    if "finf" in mat:                                  # bounded Hill recruitment
        g = np.maximum(I1(lt) - 3.0, 0.0)
        return 1.0 + (mat["finf"]-1.0)*g**mat["n"]/(g**mat["n"] + mat["gr"]**mat["n"])
    return np.ones_like(lt)                            # neo-Hookean

def mu_app_tt(r, a, mat):
    """Tangential apparent shear modulus mu_0 f(I1) lam_theta^2 (kPa)."""
    lt = lam_theta(r, a)
    return mat["mu0"] * f_stiffen(lt, mat) * lt**2

def roi_mean_pointwise(V, mat, dr=ROI_WID, n=4001, weight="shell", off=ROI_OFF):
    """Spherical-shell average over a <= r <= a+dr.

    Geometry: a 12 mm shell seated directly on the cavity edge, no standoff --
    exactly four 3 mm acquisition voxels, consistent with the ~11 mm measured from
    the drawn contours in the published elastograms."""
    a = cavity_radius(V)
    r = np.linspace(a+off, a+off+dr, n)
    mu = mu_app_tt(r, a, mat)
    w = r**2 if weight == "shell" else r          # spherical shell (default) vs 2-D annulus
    return np.trapz(mu*w, r)/np.trapz(w, r)

if __name__ == "__main__":
    print(f"A_ref = {A_REF:.3f} cm")
    for mat, tag in ((P1,"P1"), (P2,"P2")):
        print(f"\n{mat['name']}")
        for V in V_STATES:
            a = cavity_radius(V)
            lt_in  = lam_theta(a, a)
            lt_out = lam_theta(a+DR_ROI, a)
            print(f"  V={V:5.0f} mL  a={a:.2f} cm  lam_th {lt_in:.3f}->{lt_out:.3f}  "
                  f"mu_ROI(slice)={roi_mean_pointwise(V,mat):6.2f}  "
                  f"mu_ROI(shell)={roi_mean_pointwise(V,mat,weight='shell'):6.2f} kPa")
