# Claude Code Prompt — 1D FNO Eigenstrain Inversion with Analytical Validation
#
# Paste this verbatim into a Claude Code session from an empty project directory.
# Builds the complete 1D pipeline: analytical solutions → FD forward solver →
# FNO training → rigorous validation against closed-form ground truth.
# No prior code assumed. Build everything from scratch.

---

```
You are building a 1D physics validation pipeline for the eigenstrain
inversion problem arising in Tissue Strain Mapping (TSM) via MR Elastography.

The physical setting: a 1D elastic domain with a pressurized inclusion
(tumor analog). The expanding inclusion generates an eigenstrain field
ε*(x) that perturbs local wave speed. MRE measures displacement u(x).
The FNO must recover ε*(x) from u(x).

The critical feature of this project: every FNO prediction is validated
against CLOSED-FORM ANALYTICAL SOLUTIONS. This is the scientific
contribution — proving the FNO achieves theoretically optimal accuracy
and establishing SNR-limited error bounds that carry forward to all
higher-dimensional extensions.

No plastic strain. The strain decomposition is:
    ε = εᵉ + ε*    (elastic + eigenstrain only)

Build everything in the order specified. Verify each module with
numerical checks before proceeding. Do not move to the next module
until all tests pass.

---

## Project structure

eigenstrain_1d/
├── README.md
├── environment.yml
├── src/
│   ├── __init__.py
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── static_solution.py      # closed-form static equilibrium
│   │   ├── wave_solution.py        # exact Helmholtz solution (transfer matrix)
│   │   ├── inversion_formula.py    # k-space eigenstrain inversion
│   │   ├── nonlocal_solution.py    # nonlocal kernel analytic results
│   │   └── viscoelastic.py         # relaxation time analytics
│   ├── solver/
│   │   ├── __init__.py
│   │   ├── forward_1d.py           # 1D finite-difference Helmholtz solver
│   │   └── nonlocal_fd.py          # nonlocal FD solver for verification
│   ├── model/
│   │   ├── __init__.py
│   │   ├── spectral_conv_1d.py     # 1D spectral convolution layer
│   │   ├── fno_1d.py               # 1D FNO: u(x) → ε*(x)
│   │   └── losses_1d.py            # all loss functions with physics terms
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator_1d.py         # 1D training pair generator
│   │   └── dataset_1d.py           # HDF5 dataset class
│   └── validation/
│       ├── __init__.py
│       ├── validate_analytics.py   # FNO vs analytical formula comparison
│       ├── error_bounds.py         # theoretical SNR-limited error bounds
│       └── figures.py              # all publication-quality figures
├── scripts/
│   ├── generate_dataset.py         # CLI: generate N training pairs
│   ├── train_1d.py                 # CLI: train FNO
│   └── validate_1d.py              # CLI: full validation suite
└── tests/
    ├── test_analytics.py           # verify every closed-form formula
    ├── test_solver.py              # verify FD solver against analytics
    ├── test_fno_1d.py              # model shapes, ranges, gradients
    └── test_validation.py          # validate the validation pipeline itself

---

## PHYSICAL SETUP — READ THIS CAREFULLY

Domain: x ∈ [0, 2L], periodic boundary conditions
    L    = 0.10 m      (10 cm domain half-length)
    N    = 256         (grid points, power of 2 for FFT)
    dx   = 2L/N        (spatial resolution ≈ 0.78 mm)

Lesion (inclusion): centered at x=L, half-width a
    a    ~ Uniform(0.005, 0.020) m    (5–20 mm radius)

Material parameters:
    rho       = 1000.0   kg/m³     (tissue density)
    E_bg      ~ Uniform(800, 3000)  Pa   (background Young's modulus)
    E_lesion  ~ Uniform(3000, 15000) Pa  (lesion core modulus)
    A_coeff   ~ Uniform(-8.0, -2.0)      (acoustoelastic constant, dimensionless)
    eps0      ~ Uniform(0.005, 0.08)     (eigenstrain magnitude in lesion)
                = 0 for control cases (20% probability)

MRE drive frequencies:
    freq1 = 60.0  Hz
    freq2 = 120.0 Hz
    omega1 = 2*pi*freq1
    omega2 = 2*pi*freq2

Viscoelastic parameters (for relaxation validation):
    G_inf  = E_bg * 0.7          (equilibrium modulus, 70% of instantaneous)
    G1     = E_bg * 0.2          (fast relaxation component)
    G2     = E_bg * 0.1          (slow relaxation component)
    tau1   ~ Uniform(0.5, 5.0)   days (fast)
    tau2   ~ Uniform(5.0, 30.0)  days (slow)

Nonlocal kernel:
    alpha(r) = (1/(2*ell)) * exp(-r/ell)
    alpha_hat(k) = 1 / (1 + k²*ell²)    (Fourier transform, exact)
    ell    ~ Uniform(0.003, 0.012) m     (3–12 mm interaction length)

---

## MODULE 1: Analytical solutions (src/analytics/)

This is the scientific core. Every formula must be derived carefully
and tested numerically. These are the ground truth that the FNO is
validated against.

### 1a. Static equilibrium (src/analytics/static_solution.py)

Physical setting: quasi-static, no inertia. ε = εᵉ + ε*, σ = E(x)·εᵉ.
Equilibrium: dσ/dx = 0  →  σ = σ_bar (constant everywhere, 1D result).
Periodic BC: ∫_0^{2L} ε dx = 0 (no net strain).

DERIVATION:
    ε_total = σ_bar/E(x) + ε*(x)     [elastic + eigenstrain]
    ∫_0^{2L} (σ_bar/E(x) + ε*(x)) dx = 0
    σ_bar · ∫_0^{2L} 1/E(x) dx = -∫_0^{2L} ε*(x) dx

    Let C_compliance = ∫_0^{2L} 1/E(x) dx
                     = 2a/E_lesion + (2L-2a)/E_bg    [for step-function E]

    Let E_star_integral = ∫_0^{2L} ε*(x) dx
                        = 2a * eps0                   [for box eigenstrain]

    CLOSED FORM:
    σ_bar = -E_star_integral / C_compliance
           = -(2a * eps0) / [2a/E_lesion + (2L-2a)/E_bg]
           = -(eps0 * E_lesion * E_bg) / [E_bg + (L/a - 1)*E_lesion]

    As L → ∞:   σ_bar → -(eps0 * E_bg * a) / (L-a) → 0
    This proves: the perilesional stress VANISHES in 1D infinite medium.
    The stiffening ring requires 2D/3D geometry.

    The displacement field u_static(x):
    Inside lesion [L-a, L+a]:
        du/dx = σ_bar/E_lesion + eps0
        u(x) = (σ_bar/E_lesion + eps0)(x - L) + u(L)    [by symmetry u(L)=0]

    Outside lesion:
        du/dx = σ_bar/E_bg
        u(x) = σ_bar/E_bg * (x - L)    [for x > L+a]

    Implement as:
    def static_solution(N, dx, a, L, E_bg, E_lesion, eps0):
        Returns: sigma_bar (scalar), u_static (N,), E_field (N,)

    Verify: ∫u_static dx ≈ 0 (to machine precision for periodic BC)
    Verify: du/dx - eps0*mask - sigma_bar/E_field ≈ 0 everywhere

### 1b. Exact wave solution (src/analytics/wave_solution.py)

Time-harmonic displacement: u(x,t) = Re[û(x)exp(-iωt)]
Governing equation: -ρω²û = d/dx[E_eff(x) dû/dx]
where E_eff(x) = E(x) + A_coeff * σ_static(x)    [acoustoelastic]

For piecewise constant E_eff (two values: E1 in lesion, E2 outside):
    Define k1 = ω*sqrt(ρ/E1),  k2 = ω*sqrt(ρ/E2)

    General solution in each region:
    Inside   [L-a, L+a]:  û(x) = C1*exp(ik1*x) + D1*exp(-ik1*x)
    Outside  (other):      û(x) = C2*exp(ik2*x) + D2*exp(-ik2*x)

TRANSFER MATRIX METHOD for periodic domain:
    At each interface (x=L-a, x=L+a), enforce:
        û is continuous
        E_eff * dû/dx is continuous  (stress continuity)

    Periodic BC: û(0) = û(2L), E_eff*û'(0) = E_eff*û'(2L)

    This is a 4×4 linear system for [C1, D1, C2, D2].
    Add source term as plane wave S*exp(ik2*x) incident from left.

    Implement as:
    def transfer_matrix_solution(N, dx, a, L, E_bg, E_lesion,
                                  A_coeff, sigma_bar, rho, freq):
        Returns: u_hat (N,) complex array

    The source amplitude is set to give unit maximum displacement.
    Add noise at specified SNR before returning.

    VERIFY by substituting into the FD discretization of the wave equation.
    Residual ||FD(u_hat) + rho*omega^2*u_hat|| / ||u_hat|| < 1e-6

### 1c. Eigenstrain inversion formula (src/analytics/inversion_formula.py)

This is the main analytical result. Given measured displacement û(x)
at frequency ω, recover ε*(x) exactly (in the noise-free case).

DERIVATION in Fourier space (periodic domain, N grid points):
    DFT of wave equation:
    -ρω²·Û(k) = -(ik) · FT[E_eff · dû/dx](k)

    For HOMOGENEOUS E (E_eff = E everywhere, no lesion variation):
    -ρω²·Û(k) = -k²·E·(Û(k) - iε̂*(k)/k)    [see note below]

    Note: FT[dε*/dx] = ik·ε̂*(k), so FT[ε*(x)] = ε̂*(k)
          FT[∂u/∂x - ε*] = ik·Û(k) - ε̂*(k)
          FT[E·(∂u/∂x - ε*)] = E·(ik·Û(k) - ε̂*(k))
          FT[d/dx(E·(ε-ε*))] = ik·E·(ik·Û(k) - ε̂*(k)) = -k²·E·Û(k) - ik·E·ε̂*(k)

    Wave equation in Fourier space:
    -ρω²·Û(k) = -k²·E·Û(k) - ik·E·ε̂*(k)

    Solving for ε̂*(k):
    ik·E·ε̂*(k) = (k² - ρω²/E)·E·Û(k) - ... 
    
    MORE CAREFULLY:
    -ρω²·Û(k) = ik · FT[E·(ε - ε*)](k)
              = ik · (E·ik·Û(k) - E·ε̂*(k))
              = -k²·E·Û(k) - ik·E·ε̂*(k)

    Therefore:
    ik·E·ε̂*(k) = ρω²·Û(k) - k²·E·Û(k)
    ε̂*(k) = Û(k) · (ρω² - k²·E) / (ik·E)
           = Û(k)/ik · (ρω²/E - k²) / (-1)     ... SIGN CHECK:
           = -Û(k)/ik · (ρω²/E - k²)
           = Û(k)/ik · (k² - ρω²/E)

    FINAL CLOSED FORM (homogeneous medium, local):
    ε̂*(k) = (ik·Û(k)) · [1 - ρω²/(k²·E)]
           = ε̂(k) · [1 - ρω²/(k²·E)]     for k ≠ 0

    where ε̂(k) = ik·Û(k) is the Fourier transform of the measured strain.

    Physical interpretation:
    - First term ε̂(k): the measured total strain
    - Correction [1 - ρω²/(k²·E)]: removes the inertial contribution
      At low k (long wavelength): correction ≈ -ρω²/(k²·E) dominates
      At high k (short wavelength): correction → 1 (strain = eigenstrain)

    FOR THE NONLOCAL CASE (kernel α̂(k) = 1/(1+k²ℓ²)):
    ε̂*(k) = ε̂(k) · [1 - ρω²/(k²·α̂(k)·E)]
           = ε̂(k) · [1 - ρω²·(1+k²ℓ²)/(k²·E)]

    The nonlocal correction adds (ρω²ℓ²/E) to the inertial term,
    effectively broadening the recovery kernel.

    Implement as:
    def inversion_formula(u_hat_array, dx, E, rho, freq, ell=0.0):
        Inputs:
            u_hat_array: complex (N,) measured displacement at one frequency
            dx:          grid spacing [m]
            E:           Young's modulus [Pa] (homogeneous approximation)
            rho:         density [kg/m³]
            freq:        drive frequency [Hz]
            ell:         nonlocal length scale [m] (0 = local case)
        Returns:
            eps_star:    real (N,) recovered eigenstrain field

        Implementation:
            omega = 2*pi*freq
            k = fftfreq(N, d=dx) * 2*pi     # wavenumbers
            U_hat = fft(u_hat_array)          # k-space displacement
            eps_hat = 1j * k * U_hat          # k-space strain

            alpha_hat = 1.0 / (1.0 + k**2 * ell**2)  # nonlocal kernel
            alpha_hat[0] = 1.0                         # DC component

            # Inversion formula
            correction = 1.0 - rho * omega**2 / (k**2 * alpha_hat * E + 1e-30)
            correction[0] = 0.0    # DC: ε* has zero mean (periodic domain)

            eps_star_hat = eps_hat * correction
            eps_star = ifft(eps_star_hat).real
            return eps_star

    CRITICAL NUMERICAL ISSUE: k=0 gives division by zero.
    Set correction[0] = 0: the mean eigenstrain is not recoverable
    from wave data alone (a fundamental limitation of the method).
    The FNO should also predict ε* with zero mean.

    VERIFY: Apply to transfer_matrix_solution output (noise-free).
    Recovered ε* should match ground truth to < 1e-6 relative error.

### 1d. Nonlocal static solution (src/analytics/nonlocal_solution.py)

For the exponential kernel, the integral equation
    σ(x) = ∫(1/2ℓ)exp(-|x-ξ|/ℓ)·E·(ε(ξ)-ε*(ξ))dξ
is equivalent to the differential equation:
    σ - ℓ²·d²σ/dx² = E·(ε - ε*)

With equilibrium dσ/dx = 0 → σ = σ_bar (constant), this becomes:
    σ_bar = E·(ε(x) - ε*(x))    for all x ??? 

    NO — this is only true for homogeneous E. For piecewise E:
    σ_bar - ℓ²·σ_bar'' = σ_bar (since σ_bar is constant, σ_bar'' = 0)
    So: σ_bar = E(x)·(ε(x) - ε*(x)) at every point individually.

    This means: ε_elastic(x) = σ_bar/E(x) everywhere.
    Consistent with local solution — the nonlocal kernel does not affect
    the static solution for constant stress.

    The nonlocal effect enters only in the WAVE EQUATION where stress
    varies in space. Document this finding explicitly.

    For the wave problem with nonlocal E_eff and periodic domain:
    The conversion trick gives:
    E_eff(x)·ε(x) - ℓ²·d²/dx²[E_eff(x)·ε(x)] = local_E_eff(x)·ε(x)
    ...this doesn't simplify cleanly for heterogeneous E_eff.

    Use Fourier space directly: σ̂(k) = α̂(k)·E·(ε̂(k)-ε̂*(k))
    which is exact and gives the inversion formula in 1c.

    Implement:
    def nonlocal_stress_fourier(eps_field, eps_star_field, E, ell, dx):
        Computes nonlocal stress field in real space via Fourier convolution.
        Verify: matches direct numerical integration to < 1e-8 relative error.

### 1e. Viscoelastic relaxation (src/analytics/viscoelastic.py)

Prony series relaxation modulus:
    G(t) = G_inf + G1*exp(-t/tau1) + G2*exp(-t/tau2)    [Pa]

The acoustoelastically-detectable pre-stress at time t after growth stops:
    sigma_static(t) = sigma_0 * G(t) / G(0)

where sigma_0 is the static solution at t=0 (from static_solution.py)
and G(0) = G_inf + G1 + G2 = E_bg (instantaneous modulus).

Wave speed perturbation at time t:
    delta_c(t)/c0 = A_coeff * sigma_static(t) / (2 * E_bg)
                  = A_coeff * sigma_0 * G(t) / (2 * E_bg * G(0))

For two MRE measurements at times t1, t2:
    delta_c(t2)/delta_c(t1) = G(t2)/G(t1)

For single-exponential (G1 only, G2=0):
    CLOSED FORM for tau estimation:
    tau = -(t2-t1) / ln(delta_c(t2)/delta_c(t1))

For two-exponential, solve numerically. Implement both.

Functions:
    def G_relaxation(t, G_inf, G1, tau1, G2=0, tau2=None):
        Returns G(t) [Pa]

    def delta_c_over_c0(t, sigma_0, E_bg, A_coeff, G_inf, G1, tau1,
                        G2=0, tau2=None):
        Returns relative wave speed perturbation at time t

    def estimate_tau_analytical(delta_c_t1, delta_c_t2, dt_days):
        Single-exponential tau estimation. Returns tau [days].
        Returns None if delta_c ratio ≤ 0 (unphysical).

    def estimate_tau_two_exp(delta_c_times, delta_c_values, dt_days_array):
        Two-exponential tau estimation by nonlinear least squares.
        Returns (tau1, tau2, G1_frac, G2_frac).

---

## MODULE 2: Forward solver (src/solver/forward_1d.py)

1D time-harmonic Helmholtz equation:
    -rho*omega²*u = d/dx[E_eff(x) * du/dx] + f(x)

Source: f(x) = F_source * delta(x - x_source) where x_source = 0 (left end)
Actually for periodic domain: use a smooth source distributed near x=0.

IMPLEMENTATION:
Use second-order finite differences on periodic domain.
The FD stiffness matrix for -d/dx[E_eff*du/dx]:
    (FD matrix)_{ij}: standard second-order staggered-grid scheme
    E at half-points: E_{j+1/2} = (E_j + E_{j+1})/2

Assemble:
    A = FD_stiffness_matrix - rho*omega² * Identity
    b = source_vector

Solve: u = spsolve(A, b)

Add random-phase source:
    source_phase ~ Uniform(0, 2pi)
    source_amplitude ~ Uniform(0.5, 2.0)
    This prevents the FNO from learning boundary artifacts.

Add noise at SNR_dB ~ Uniform(15, 30):
    noise_std = |u|.max() * 10^(-SNR/20) / sqrt(2)
    u_noisy = u + noise_std * (randn(N) + 1j*randn(N))

Function:
    def helmholtz_solve_1d(N, dx, E_eff, rho, freq,
                           source_phase=0.0, source_amplitude=1.0,
                           snr_db=None, rng=None):
        Returns: u_complex (N,) complex array

VERIFY against transfer_matrix_solution:
    For homogeneous E_eff, both should agree to < 1e-4 relative error.
    For piecewise E_eff, FD solution converges to transfer matrix as N→∞.

---

## MODULE 3: Full training pair generator (src/data/generator_1d.py)

Function: make_1d_pair(N=256, dx, rng)

Step 1: Sample parameters
    E_bg      = rng.uniform(800, 3000)
    E_lesion  = rng.uniform(3000, 15000)
    a         = rng.uniform(0.005, 0.020)   [m]
    eps0      = rng.uniform(0.005, 0.08) if rng.random() > 0.2 else 0.0
    A_coeff   = rng.uniform(-8.0, -2.0)
    ell       = rng.uniform(0.003, 0.012)
    snr_db    = rng.uniform(15, 30)

Step 2: Build E(x) field
    E_field = np.where(lesion_mask, E_lesion, E_bg)
    lesion_mask: bool (N,) True in [L-a, L+a]

Step 3: Compute static solution
    sigma_bar, u_static = static_solution(...)

Step 4: Build acoustoelastic E_eff(x)
    sigma_static_field = sigma_bar * np.ones(N)
    E_eff = E_field + A_coeff * sigma_static_field

Step 5: Solve wave equation at two frequencies
    u_60  = helmholtz_solve_1d(..., freq=60,  E_eff=E_eff, snr_db=snr_db)
    u_120 = helmholtz_solve_1d(..., freq=120, E_eff=E_eff, snr_db=snr_db)

Step 6: Compute ground truth eigenstrain
    eps_star_true = np.where(lesion_mask, eps0, 0.0)   [box function]

Step 7: Compute analytical inversion (GROUND TRUTH REFERENCE)
    eps_star_analytic_60  = inversion_formula(u_60,  dx, E_bg, rho, 60,  ell)
    eps_star_analytic_120 = inversion_formula(u_120, dx, E_bg, rho, 120, ell)
    eps_star_analytic_avg = (eps_star_analytic_60 + eps_star_analytic_120) / 2

Step 8: Build FNO input tensor X and labels Y
    X: float32 (5, N)
        Ch 0: Re(u_60)
        Ch 1: Im(u_60)
        Ch 2: Re(u_120)
        Ch 3: Im(u_120)
        Ch 4: Lamé prior = sigma_bar * (a_eq/r)^1 * lesion_boundary_indicator
               where r = distance from lesion center (clipped to avoid /0)
               In 1D: Lamé prior = sigma_bar for x in lesion, decaying outside

    Y: dict
        'eps_star_true':     float32 (N,) — ground truth box eigenstrain
        'eps_star_analytic': float32 (N,) — analytical inversion result
        'sigma_bar':         float  — static stress
        'A_coeff':           float  — acoustoelastic constant
        'E_bg':              float  — background modulus
        'E_lesion':          float  — lesion modulus
        'eps0':              float  — eigenstrain magnitude (0 for control)
        'ell':               float  — nonlocal length scale
        'snr_db':            float  — noise level
        'is_expanding':      bool   — eps0 > 0

HDF5 format (data/1d_pairs_{N_samples}.h5):
    /X                  float32  (n_samples, 5, N)
    /Y_eps_true         float32  (n_samples, N)
    /Y_eps_analytic     float32  (n_samples, N)
    /meta/sigma_bar     float32  (n_samples,)
    /meta/A_coeff       float32  (n_samples,)
    /meta/E_bg          float32  (n_samples,)
    /meta/eps0          float32  (n_samples,)
    /meta/ell           float32  (n_samples,)
    /meta/snr_db        float32  (n_samples,)
    /meta/is_expanding  bool     (n_samples,)

Generate: 20,000 training samples (takes ~15 min CPU)
          2,000 validation samples
          1,000 test samples (held out, analytical formula applied to all)

---

## MODULE 4: 1D FNO model (src/model/)

### 4a. 1D Spectral convolution (src/model/spectral_conv_1d.py)

class SpectralConv1d(nn.Module):
    """
    1D Fourier layer. Applies learned complex weight matrix
    to retained Fourier modes of the input.

    Key difference from 2D: rfft instead of rfft2
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes,
                               dtype=torch.cfloat))

    def forward(self, x):
        # x: (B, C, N)
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, norm='ortho')    # (B, C, N//2+1)

        out_ft = torch.zeros(B, self.out_channels, N//2+1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            'bim,iom->bom', x_ft[:, :, :self.modes], self.weights)

        return torch.fft.irfft(out_ft, n=N, norm='ortho')  # (B, C, N)

### 4b. 1D FNO (src/model/fno_1d.py)

class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator for eigenstrain inversion.

    Input:  u(x) displacement field — 5 channels, N points
    Output: ε*(x) eigenstrain field — 1 channel, N points

    Architecture:
      Lift → N_layers Fourier blocks → Project → softplus

    Modes: retain lowest `modes` Fourier modes (default 32 for N=256)
    Width: channel width in trunk (default 64)
    """

    def __init__(self, modes=32, width=64, n_layers=4, in_channels=5,
                 eps_max=0.12):
        super().__init__()
        self.eps_max = eps_max

        # Lift: (in_channels + 1 grid coord) → width
        self.fc_in = nn.Linear(in_channels + 1, width)

        # Fourier blocks
        self.spectral_convs = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.bypass_convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)])
        self.activations = nn.ModuleList([
            nn.GELU() if i < n_layers-1 else nn.Identity()
            for i in range(n_layers)])

        # Project: width → 128 → 1
        self.fc_out1 = nn.Linear(width, 128)
        self.fc_out2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 5, N)
        B, C, N = x.shape

        # Add coordinate channel
        grid = torch.linspace(0, 1, N, device=x.device)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, 1, N)
        x = torch.cat([x, grid], dim=1)            # (B, 6, N)

        # Lift: permute for Linear layer
        x = x.permute(0, 2, 1)                    # (B, N, 6)
        x = self.fc_in(x)                          # (B, N, width)
        x = x.permute(0, 2, 1)                    # (B, width, N)

        # Fourier blocks
        for spec, bypass, act in zip(self.spectral_convs,
                                     self.bypass_convs,
                                     self.activations):
            x = act(spec(x) + bypass(x))

        # Project
        x = x.permute(0, 2, 1)                    # (B, N, width)
        x = F.gelu(self.fc_out1(x))               # (B, N, 128)
        x = self.fc_out2(x).squeeze(-1)           # (B, N)

        # Enforce ε* ≥ 0 and bounded by eps_max
        x = torch.sigmoid(x) * self.eps_max       # (B, N) in [0, eps_max]
        return x

### 4c. Loss functions (src/model/losses_1d.py)

ALL losses compare against BOTH ground truth labels:
    eps_true:     the box eigenstrain (what we want to recover)
    eps_analytic: the analytical inversion formula result (optimal baseline)

This dual comparison is the key scientific contribution: the FNO
should match eps_analytic (the theoretically optimal reconstruction)
while also approximating eps_true (the smooth physical ground truth).

def relative_l2(pred, true, eps_floor=1e-6):
    norm_diff = torch.norm(pred - true, dim=-1)
    norm_true = torch.norm(true, dim=-1)
    return (norm_diff / (norm_true + eps_floor)).mean()

def lesion_weighted_l2(pred, true, lesion_mask, w_lesion=0.7, w_bg=0.3):
    """More weight inside and near lesion boundary."""
    l_lesion = relative_l2(pred * lesion_mask, true * lesion_mask)
    l_bg     = relative_l2(pred * ~lesion_mask, true * ~lesion_mask)
    return w_lesion * l_lesion + w_bg * l_bg

def acoustoelastic_consistency_loss(eps_pred, E_bg, A_coeff, sigma_bar,
                                     E_eff_pred):
    """
    Enforce: E_eff = E_bg + A_coeff * sigma_bar
    where sigma_bar is determined by eps_pred via the analytical formula.
    """
    sigma_pred = compute_sigma_bar_from_eps(eps_pred, E_bg)
    E_eff_from_eps = E_bg + A_coeff * sigma_pred
    return F.mse_loss(E_eff_pred, E_eff_from_eps)

def inversion_consistency_loss(eps_pred, u_input, E_bg, rho, freq1, freq2, dx):
    """
    Enforce that eps_pred is consistent with the analytical inversion formula.
    The analytical formula applied to u_input should give eps_pred.
    This is the physics-informed loss.
    """
    u_60  = u_input[:, 0] + 1j * u_input[:, 1]   # complex displacement
    u_120 = u_input[:, 2] + 1j * u_input[:, 3]
    eps_analytic_60  = inversion_formula_batch(u_60,  dx, E_bg, rho, freq1)
    eps_analytic_120 = inversion_formula_batch(u_120, dx, E_bg, rho, freq2)
    eps_analytic = (eps_analytic_60 + eps_analytic_120) / 2
    return relative_l2(eps_pred, eps_analytic)

class TSMLoss1D(nn.Module):
    def __init__(self, lambda_true=1.0, lambda_analytic=0.5,
                 lambda_pde=0.1, lambda_expand=0.2):
        """
        Total loss:
          L = lambda_true     * RL²(eps_pred, eps_true)         [vs physics]
            + lambda_analytic * RL²(eps_pred, eps_analytic)     [vs analytics]
            + lambda_pde      * inversion_consistency_loss(...)  [PDE residual]
            + lambda_expand   * BCE(expand_pred, expand_true)   [detection]
        """

---

## MODULE 5: Training (scripts/train_1d.py)

Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
Scheduler: CosineAnnealingLR, T_max=200, eta_min=1e-5
Epochs: 200 (1D trains much faster than 2D)
Batch size: 64
Gradient clip: max_norm=1.0

Data split: 18,000 train / 2,000 val from the 20k dataset.
Test set (1,000 samples) held out entirely, used only in validation suite.

Log every epoch:
    train_loss (total + all components)
    val_RL2_vs_true      (FNO vs ground truth eigenstrain)
    val_RL2_vs_analytic  (FNO vs analytical formula — THE KEY METRIC)
    val_SSIM             (spatial accuracy)
    val_expand_AUC       (expanding vs non-expanding AUC)

Save best checkpoint on val_RL2_vs_analytic.

CLI:
    python scripts/train_1d.py \
        --data_path data/1d_pairs_20000.h5 \
        --run_dir   runs/fno_1d \
        --epochs    200 \
        --modes     32 \
        --width     64 \
        --n_layers  4 \
        --lr        1e-3

---

## MODULE 6: Validation suite (src/validation/)

This is the scientific heart of the project. Five validation levels,
each with a specific pass criterion.

### V1. Analytical formula accuracy (validate_analytics.py)
Compare FNO to analytical inversion formula on test set:

    FNO_RL2_vs_analytic:  mean RL²(eps_FNO, eps_analytic)    [TARGET < 0.08]
    FNO_RL2_vs_true:      mean RL²(eps_FNO, eps_true)        [TARGET < 0.12]
    Analytic_RL2_vs_true: mean RL²(eps_analytic, eps_true)   [BASELINE]

The gap (FNO_RL2 - Analytic_RL2) is the excess error from learning.
Target: gap < 0.05 — the FNO should nearly match the optimal formula.

### V2. SNR-limited error bounds (error_bounds.py)
Theoretical minimum achievable RL² as a function of SNR:

    DERIVATION:
    The analytical formula amplifies noise at low k:
        correction(k) = 1 - ρω²/(k²·E)    grows as k → 0

    For noise Δu ~ Normal(0, σ_noise):
        Δε*(k) = ik·Δu·correction(k)
        |Δε*(k)|² ≈ k²·σ_noise²·|correction(k)|²

    Integrating over k:
        ||Δε*||² ≈ σ_noise² · Σ_k k²·(1 - ρω²/(k²E))²

    RL²_min(SNR) = σ_noise / (||eps_true|| · SNR_factor)

    Compute this bound numerically for SNR ∈ [10, 35] dB.
    Plot FNO RL² curve vs theoretical bound.

    FNO should approach the bound to within a factor of 2.
    If FNO RL² >> bound: room for improvement via more data or better architecture.
    If FNO RL² ≈ bound: model is Cramér-Rao optimal — cannot improve without
                         more measurements or lower noise.

### V3. Parameter recovery accuracy (validate_analytics.py)
From FNO-predicted ε*(x), recover physical parameters:

    a) Acoustoelastic constant A:
       A_recovered = (E_eff - E_bg) / sigma_bar  in lesion region
       Compare to ground truth A_coeff
       Plot: A_FNO vs A_true, report R² and RMSE

    b) Static stress σ_bar:
       sigma_recovered = analytical_static_solution(eps_FNO, E_bg)
       Compare to ground truth sigma_bar
       Plot: sigma_FNO vs sigma_true

    c) Eigenstrain magnitude ε₀:
       eps0_recovered = eps_FNO[lesion_region].mean()
       Compare to ground truth eps0
       Plot: eps0_FNO vs eps0_true

    d) Lesion half-width a:
       a_recovered = half-width of region where eps_FNO > eps0/2
       Compare to ground truth a
       Plot: a_FNO vs a_true

    PASS CRITERIA:
       R² > 0.90 for all four parameters
       RMSE < 20% of parameter range for all four

### V4. Relaxation time recovery (validate_analytics.py)
Using serial pair simulation:
    Generate pairs at t=0 and t=Δt (Δt ∈ [1, 7] days)
    Apply FNO to each: eps_pred_t0, eps_pred_t1
    Compute delta_c from eps_pred via acoustoelastic formula
    Apply analytical tau formula: tau_FNO = estimate_tau_analytical(...)
    Compare to ground truth tau

    PASS CRITERION: |tau_FNO - tau_true| / tau_true < 0.20 (20% relative error)

### V5. Control case specificity (validate_analytics.py)
For non-expanding (eps0=0) cases:
    FNO should predict ε*≈0 everywhere (high specificity)
    False positive rate: fraction of control cases where
        max(eps_FNO[lesion_region]) > 0.005   [detection threshold]
    TARGET: false positive rate < 0.10

### Validation figure (src/validation/figures.py)
Generate all publication-quality figures:

Figure 1: Single-case example (4 panels)
    Panel 1: Input Re(u_60) and Re(u_120) — two frequency channels
    Panel 2: Ground truth ε*(x) vs analytical formula vs FNO prediction
    Panel 3: E_eff(x) and residual of wave equation
    Panel 4: Recovery error distribution across test set

Figure 2: SNR error bound curve
    x-axis: SNR [dB]
    y-axis: RL²
    Lines:  theoretical bound, FNO, analytical formula

Figure 3: Parameter recovery scatter plots (2×2)
    A_coeff, σ_bar, ε₀, a — FNO vs ground truth

Figure 4: Expanding vs non-expanding ROC curve
    AUC annotation, operating point at 10% FPR

Figure 5: Frequency dependence
    RL² vs drive frequency ∈ [40, 80, 120, 200] Hz
    Shows optimal frequency for eigenstrain recovery

Save all figures to results/figures/ as PDF (for publication) + PNG.

---

## MODULE 7: Tests (tests/)

### test_analytics.py
Run before anything else. All must pass to machine precision.

    test_static_periodic_bc:
        Verify ∫ε dx = 0 for static solution
        Verify σ = constant everywhere
        Verify σ_bar formula with explicit numbers:
            E_bg=2000, E_lesion=8000, eps0=0.02, a=0.01, L=0.10
            Expected: σ_bar = -(0.02*8000*2000*0.01)/(2000*0.01+8000*0.09)
            Check: abs(σ_bar_computed - σ_bar_expected) < 1e-6

    test_inversion_noisefree:
        Generate transfer matrix solution (no noise)
        Apply inversion formula
        Assert RL²(eps_recovered, eps_true) < 1e-4
        This is the most important test in the entire project

    test_inversion_dc_component:
        Assert eps_recovered.mean() < 1e-10 (DC suppressed)

    test_nonlocal_kernel_fourier:
        Verify alpha_hat(k) = 1/(1+k²ℓ²) matches direct numerical integration
        of (1/2ℓ)exp(-|x|/ℓ) for k ∈ [0, k_max]

    test_relaxation_single_exp:
        Generate delta_c at t=0 and t=5 days
        Apply estimate_tau_analytical
        Assert |tau_recovered - tau_true| / tau_true < 0.01

    test_static_infinite_limit:
        For L = 10*a (near infinite): sigma_bar < 0.05 * E_bg * eps0
        Confirms perilesional stress vanishes in 1D infinite medium

### test_solver.py
    test_fd_vs_transfer_matrix_homogeneous:
        Homogeneous E_eff: FD and transfer matrix agree to < 1e-4 RL²

    test_fd_convergence:
        Piecewise E_eff: FD error halves as N doubles (second-order convergence)

    test_wave_speed_correct:
        Measure phase velocity from FD solution
        Assert |c_measured - omega/k1| / (omega/k1) < 0.01

### test_fno_1d.py
    test_output_shape:    (4, 256) input → (4, 256) output
    test_output_range:    output ∈ [0, eps_max] always
    test_gradient_flows:  loss.backward() completes without NaN
    test_zero_input:      u=0 → eps*≈0 (no displacement → no eigenstrain)

### test_validation.py
    test_rl2_metric:      RL²(x, x) = 0, RL²(x, 0) = 1
    test_snr_bound_monotone: bound decreases as SNR increases
    test_tau_formula:     estimate_tau consistent with exponential decay

---

## Implementation order

1.  Create directories + environment.yml
    Dependencies: numpy, scipy, torch, h5py, matplotlib, tqdm, pytest

2.  Write src/analytics/static_solution.py
    Immediately test with hardcoded numbers vs hand calculation:
        E_bg=2000, E_lesion=8000, eps0=0.02, a=0.01, L=0.10
        sigma_bar should be ≈ -22.4 Pa (compute by hand to verify)

3.  Write src/analytics/wave_solution.py (transfer matrix)
    Test: for homogeneous E=2000, N=256, verify wave equation residual < 1e-8

4.  Write src/analytics/inversion_formula.py
    CRITICAL TEST: apply to noise-free transfer matrix output
    RL² should be < 1e-4 before proceeding

5.  Write src/analytics/nonlocal_solution.py + viscoelastic.py
    Test each with pytest tests/test_analytics.py — all must pass

6.  Write src/solver/forward_1d.py
    Test: FD vs transfer matrix for homogeneous E, must agree to 1e-4 RL²
    pytest tests/test_solver.py — all must pass

7.  Write src/data/generator_1d.py
    Generate 5 pairs. Visualize all channels. Check:
        - Re(u_60) shows wave pattern
        - eps_analytic visually resembles eps_true (smoothed box)
        - lesion region visible in Ch 4 (Lamé prior)

8.  Write src/data/dataset_1d.py + scripts/generate_dataset.py
    Generate 1000 test pairs first. Run check_dataset.py (see below).

9.  Write src/model/spectral_conv_1d.py + fno_1d.py
    pytest tests/test_fno_1d.py — all must pass

10. Write src/model/losses_1d.py
    Test each loss component individually with known tensors

11. Write scripts/train_1d.py
    Train for 5 epochs on 1000 samples. Verify:
        - Loss is finite and decreasing
        - val_RL2_vs_analytic is finite

12. Generate full dataset: 20,000 train + 1,000 test
13. Train for 200 epochs
14. Write src/validation/ modules
15. Run full validation suite: python scripts/validate_1d.py
16. Generate all figures
17. Write README.md
18. Print directory tree

---

## Pre-training dataset check

Run after generating test set (1,000 samples), before training:

    python - << 'EOF'
    import h5py, numpy as np

    f = h5py.File('data/1d_pairs_test.h5', 'r')

    # Shape check
    assert f['X'].shape == (1000, 5, 256)
    assert f['Y_eps_true'].shape == (1000, 256)

    # Physics check 1: expanding cases have nonzero eigenstrain
    expanding = f['meta/is_expanding'][:]
    eps_max = f['Y_eps_true'][:].max(axis=-1)
    assert eps_max[expanding].mean() > 0.01, "Expanding cases have no eigenstrain"
    assert eps_max[~expanding].max() < 1e-8, "Control cases have nonzero eigenstrain"
    print("Physics check 1 PASS")

    # Physics check 2: analytical inversion is close to true eigenstrain
    eps_true    = f['Y_eps_true'][expanding]
    eps_analytic = f['Y_eps_analytic'][expanding]
    rl2_analytic = np.linalg.norm(eps_analytic - eps_true, axis=-1) / (
                   np.linalg.norm(eps_true, axis=-1) + 1e-8)
    print(f"Analytical RL² (expanding only): {rl2_analytic.mean():.4f}")
    assert rl2_analytic.mean() < 0.30, "Analytical inversion too inaccurate"
    print("Physics check 2 PASS")

    # Physics check 3: wave equation satisfied
    # (checked implicitly via FD solver test)

    # SNR check
    snr_vals = f['meta/snr_db'][:]
    assert snr_vals.min() >= 14.5 and snr_vals.max() <= 30.5
    print(f"SNR range: {snr_vals.min():.1f} – {snr_vals.max():.1f} dB PASS")

    print(f"\nDataset ready. {f['X'].shape[0]} samples.")
    EOF

Only proceed to training if all four checks pass.

---

## The key scientific result to report

At the end of validation, compute and print this summary table:

    ╔══════════════════════════════════════════════════════════════════╗
    ║           1D EIGENSTRAIN INVERSION — VALIDATION SUMMARY          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Metric                          FNO        Analytical  Bound   ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  RL² vs eps_true                 X.XXX      X.XXX       —       ║
    ║  RL² vs eps_analytic             X.XXX      0.000       —       ║
    ║  RL² gap (FNO - analytic)        X.XXX      —           —       ║
    ║  SNR-limited bound (25dB avg)    —          —           X.XXX   ║
    ║  AUC (expanding detection)       X.XXX      —           —       ║
    ║  False positive rate             X.XXX      —           —       ║
    ║  tau recovery error              X.XXX      —           —       ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  PASS if:  FNO RL² < 0.12                                       ║
    ║            gap < 0.05                                            ║
    ║            AUC > 0.90                                            ║
    ║            FPR < 0.10                                            ║
    ║            tau error < 0.20                                      ║
    ╚══════════════════════════════════════════════════════════════════╝

This table is the complete scientific result. It answers:
    1. Can the FNO recover eigenstrain from 1D wave data? (RL² vs true)
    2. Is it as good as the optimal analytical formula? (RL² gap)
    3. Is it operating near the SNR-limited floor? (vs bound)
    4. Can it distinguish expanding from stable lesions? (AUC, FPR)
    5. Can it estimate relaxation time from serial data? (tau error)

If all criteria pass: the 1D problem is solved. Proceed to 2D extension.
If any criterion fails: report which one and why before stopping.
```
