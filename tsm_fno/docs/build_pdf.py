#!/usr/bin/env python3
"""Build tsm_fno_formulation.pdf using ReportLab."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT = Path(__file__).parent / "tsm_fno_formulation.pdf"

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE   = colors.HexColor("#1E64B4")
DBLUE  = colors.HexColor("#0D3B7A")
GREEN  = colors.HexColor("#1E8C50")
ORANGE = colors.HexColor("#C86414")
RED    = colors.HexColor("#B42828")
GRAY   = colors.HexColor("#5A5A5A")
BGBLUE = colors.HexColor("#F5F8FF")
BGORANGE = colors.HexColor("#FFF8EB")

# ── Styles ────────────────────────────────────────────────────────────────────
SS = getSampleStyleSheet()

def style(name, **kw):
    return ParagraphStyle(name, **kw)

Title   = style("MyTitle",   fontSize=20, leading=26, textColor=BLUE,
                fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_CENTER)
SubTitle= style("MySubTitle",fontSize=13, leading=18, textColor=GRAY,
                fontName="Helvetica", spaceAfter=2, alignment=TA_CENTER)
DateS   = style("MyDate",    fontSize=10, leading=14, textColor=GRAY,
                fontName="Helvetica", spaceAfter=14, alignment=TA_CENTER)
Abs     = style("MyAbs",     fontSize=10, leading=14, textColor=colors.black,
                fontName="Helvetica", spaceAfter=8, alignment=TA_JUSTIFY,
                leftIndent=30, rightIndent=30)
AbsTitle= style("MyAbsTitle",fontSize=11, leading=15, textColor=BLUE,
                fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_CENTER)
H1      = style("MyH1",      fontSize=14, leading=18, textColor=DBLUE,
                fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6)
H2      = style("MyH2",      fontSize=12, leading=16, textColor=BLUE,
                fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=4)
H3      = style("MyH3",      fontSize=11, leading=15, textColor=BLUE,
                fontName="Helvetica-BoldOblique", spaceBefore=8, spaceAfter=3)
Body    = style("MyBody",    fontSize=10, leading=15, textColor=colors.black,
                fontName="Helvetica", spaceAfter=6, alignment=TA_JUSTIFY)
Bullet  = style("MyBullet",  fontSize=10, leading=14, textColor=colors.black,
                fontName="Helvetica", spaceAfter=3, leftIndent=20,
                bulletIndent=6, alignment=TA_JUSTIFY)
Code    = style("MyCode",    fontSize=9,  leading=13, textColor=colors.HexColor("#222222"),
                fontName="Courier", spaceAfter=4, leftIndent=20,
                backColor=colors.HexColor("#F0F0F0"))
EqStyle = style("MyEq",      fontSize=10, leading=16, textColor=colors.black,
                fontName="Courier", spaceAfter=4, alignment=TA_CENTER)
Caption = style("MyCaption", fontSize=9,  leading=12, textColor=GRAY,
                fontName="Helvetica-Oblique", spaceAfter=6, alignment=TA_CENTER)
TableH  = style("MyTableH",  fontSize=9,  leading=12, textColor=colors.white,
                fontName="Helvetica-Bold", alignment=TA_CENTER)
TableB  = style("MyTableB",  fontSize=9,  leading=12, textColor=colors.black,
                fontName="Helvetica", alignment=TA_CENTER)
TableBL = style("MyTableBL", fontSize=9,  leading=12, textColor=colors.black,
                fontName="Helvetica", alignment=TA_LEFT)
Note    = style("MyNote",    fontSize=9,  leading=13, textColor=GRAY,
                fontName="Helvetica-Oblique", spaceAfter=4, alignment=TA_JUSTIFY)

def keybox(text, title="Key Equation"):
    data = [[Paragraph(f"<b><font color='#{BLUE.hexval()[2:]}'>{title}</font></b>",
                        style("kbt", fontSize=10, fontName="Helvetica-Bold",
                              textColor=BLUE)),
             Paragraph(text, style("kbb", fontSize=10, fontName="Courier",
                                   leading=15, alignment=TA_LEFT))]]
    t = Table(data, colWidths=[1.4*inch, 4.8*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), BGBLUE),
        ("BOX",        (0,0), (-1,-1), 1.5, BLUE),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, BLUE),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    return t

def physbox(lines, title="Physics"):
    content = "<br/>".join(lines)
    data = [[Paragraph(f"<b><font color='#{ORANGE.hexval()[2:]}'>▶ {title}</font></b>",
                        style("pbt", fontSize=10, fontName="Helvetica-Bold",
                              textColor=ORANGE)),
             Paragraph(content, style("pbb", fontSize=10, fontName="Courier",
                                      leading=15, alignment=TA_LEFT))]]
    t = Table(data, colWidths=[1.4*inch, 4.8*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), BGORANGE),
        ("BOX",        (0,0), (-1,-1), 1.5, ORANGE),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, ORANGE),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    return t

def header_table(cols, rows, col_widths=None):
    all_rows = [[Paragraph(c, TableH) for c in cols]]
    for r in rows:
        all_rows.append([Paragraph(str(x), TableBL if i==0 else TableB)
                         for i,x in enumerate(r)])
    cw = col_widths or [inch*6.2/len(cols)]*len(cols)
    t = Table(all_rows, colWidths=cw)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF3FB")]),
        ("BOX",        (0,0), (-1,-1), 0.8, BLUE),
        ("INNERGRID",  (0,0), (-1,-1), 0.4, colors.HexColor("#AABBDD")),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t

def sp(n=8): return Spacer(1, n)
def hr():    return HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=6)
def hr_thin():return HRFlowable(width="100%", thickness=0.5, color=GRAY, spaceAfter=4)

def b(txt): return f"<b>{txt}</b>"
def i(txt): return f"<i>{txt}</i>"
def c(txt, color=BLUE): return f"<font color='#{color.hexval()[2:]}'>{txt}</font>"

# ── Document builder ──────────────────────────────────────────────────────────

def build():
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=1.1*inch, rightMargin=1.1*inch,
        topMargin=1.0*inch,  bottomMargin=1.0*inch,
        title="TSM-FNO Technical Formulation",
        author="Eman Richard, Nahil Sobh",
    )

    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story += [
        sp(20),
        Paragraph("TSM-FNO", Title),
        Paragraph("Fourier Neural Operator for Tissue Strain Mapping<br/>via MR Elastography", SubTitle),
        Paragraph("Technical Formulation and Architecture", SubTitle),
        sp(6),
        HRFlowable(width="60%", thickness=2, color=BLUE, spaceAfter=8),
        Paragraph("Eman Richard &nbsp;&nbsp;|&nbsp;&nbsp; Nahil Sobh &nbsp;&nbsp;|&nbsp;&nbsp; 2026", DateS),
        sp(10),
    ]

    # ── Abstract ──────────────────────────────────────────────────────────────
    story += [
        Paragraph("Abstract", AbsTitle),
        Paragraph(
            "We present <b>TSM-FNO</b>, a Fourier Neural Operator (FNO) that simultaneously "
            "predicts the shear stiffness map <i>G</i>(<b>x</b>), the perilesional latent strain "
            "field ε(<b>x</b>), and the acoustoelastic constant <i>A</i> from a single 80 Hz MR "
            "Elastography (MRE) acquisition. This document provides a self-contained derivation "
            "of the three physical models underlying the approach — wave propagation, acoustoelastic "
            "pre-stress, and the Lamé solution for a pressurized inclusion — followed by a detailed "
            "description of the operator learning framework, network architecture, training procedure, "
            "and quantitative results. No prior knowledge of neural operators or stress-induced "
            "anisotropy is assumed.", Abs),
        sp(4),
        HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=10),
    ]

    # ── Section 1 ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("1. Motivation and Clinical Context", H1), hr(),
        Paragraph(
            "Magnetic Resonance Elastography (MRE) measures the complex shear displacement "
            "field <i>u</i>(<b>x</b>) induced by an external mechanical driver. From "
            "<i>u</i>(<b>x</b>) one can infer the local shear modulus <i>G</i>(<b>x</b>), "
            "which encodes tissue stiffness — a marker of fibrosis, tumor, and neurodegeneration.", Body),
        Paragraph(
            "In <b>actively expanding lesions</b> (e.g., inflating MS lesions or growing tumors), "
            "the internal pressure <i>p</i> generates a radial pre-stress in the surrounding tissue. "
            "This pre-stress modifies the local stiffness through the <b>acoustoelastic effect</b>: "
            "the same mechanism by which a tightened drum sounds higher-pitched. The stiffening appears "
            "as a bright ring around the lesion — a ring that is <b>absent</b> in static, "
            "non-expanding lesions.", Body),
        Paragraph(
            "<b>Clinical goal</b>: distinguish an <i>expanding</i> lesion (ring present, p > 0) from "
            "a <i>control</i> lesion (ring absent, p = 0) from a single MRE scan — without requiring "
            "repeated acquisitions or follow-up imaging.", Body),
        Paragraph(
            "<b>Our approach</b>: train a neural network to invert the MRE wave field directly into "
            "both the stiffness map and the perilesional strain field, in one forward pass, exploiting "
            "the full-image wave context that local inversion methods (Murphy ILI, direct inversion) "
            "cannot access.", Body),
        sp(6),
    ]

    # ── Section 2 ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("2. Physical Model", H1), hr(),
        Paragraph("2.1  Coordinate System", H2),
        Paragraph(
            "We work in a 2D axial slice of size N × N = 80 × 80 voxels at voxel pitch "
            "d<sub>x</sub> = 3 mm, giving a field of view of 240 × 240 mm. Physical position "
            "is denoted <b>x</b> = (x, y). All fields are defined on a uniform Cartesian grid.", Body),
        sp(4),

        Paragraph("2.2  Step 1: Shear Wave Propagation — The Helmholtz Equation", H2),
        Paragraph(
            "<b>Physical setup.</b> A pneumatic MRE driver vibrates at f = 80 Hz. After the initial "
            "transient decays, the tissue oscillates in <i>steady state</i>. The displacement at "
            "every point has the form:", Body),
        Paragraph("ũ(x,t)  =  u(x) · e^(iωt),    ω = 2πf", EqStyle),
        Paragraph(
            "Substituting into Newton's second law for a linear viscoelastic continuum gives the "
            "<b>2D scalar Helmholtz equation</b>:", Body),
        sp(4),
        physbox([
            "∇ · [ G*(x) ∇u(x) ]  +  ρ ω² u(x)  =  0",
            "",
            "where:   G*(x) = G(x)(1 + iξ)   (complex shear modulus)",
            "         G(x)  = storage modulus (stiffness) [Pa]",
            "         ξ     = 0.05  (loss tangent / damping)",
            "         ρ     = 1000 kg/m³  (tissue density)",
            "         ω     = 2π × 80 rad/s",
        ], title="Helmholtz Equation  (Governing PDE)"),
        sp(6),

        Paragraph("<b>Why a complex modulus?</b>", H3),
        Paragraph(
            "Real tissue is <i>viscoelastic</i>: it stores energy (elastic part, G) and "
            "dissipates it as heat (viscous part, ξG). The complex modulus G* = G(1 + iξ) captures "
            "both. The imaginary part shifts the phase of the propagating wave. MRE acquires both "
            "real and imaginary parts of u(x), so both components carry stiffness information.", Body),

        Paragraph("<b>Finite-difference discretization.</b>", H3),
        Paragraph(
            "Equation (1) is discretized on the N × N grid using second-order finite differences. "
            "At each interior node (i, j):", Body),
        Paragraph(
            "[G*(i+½,j)(u(i+1,j)−u(i,j)) − G*(i−½,j)(u(i,j)−u(i−1,j))] / dx²\n"
            "+ [G*(i,j+½)(u(i,j+1)−u(i,j)) − G*(i,j−½)(u(i,j)−u(i,j−1))] / dx²\n"
            "+ ρω² u(i,j)  =  0", EqStyle),
        Paragraph(
            "Half-point moduli at material interfaces use <b>harmonic averaging</b>:", Body),
        Paragraph("G*(i+½,j)  =  2 G*(i,j) G*(i+1,j) / [G*(i,j) + G*(i+1,j)]", EqStyle),
        Paragraph(
            "Harmonic averaging is physically correct for series-connected springs and prevents "
            "the numerical smearing of sharp stiffness boundaries (e.g., at the lesion edge) that "
            "arithmetic averaging introduces. The resulting sparse linear system A·u = b "
            "(size N² × N²) is solved exactly with a direct sparse solver.", Body),

        Paragraph("<b>Comparison with Murphy ILI.</b>", H3),
        Paragraph(
            "Murphy et al. (2020) use a Coupled Harmonic Oscillator (CHO) simulation — a "
            "mass-spring discretization that omits longitudinal wave modes and mode conversion at "
            "material interfaces. Our Helmholtz solver retains these effects, providing more "
            "physically accurate training data.", Body),
        sp(6),

        Paragraph("2.3  Step 2: The Acoustoelastic Effect — Stress-Induced Stiffness Change", H2),
        Paragraph("<b>Physical mechanism.</b>", H3),
        Paragraph(
            "When a material is pre-stressed, its elastic moduli change — this is the "
            "<b>acoustoelastic effect</b>. Consider a guitar string: under higher tension it "
            "vibrates at a higher frequency (higher apparent modulus). In a pressurized lesion, "
            "the surrounding tissue is under circumferential tension (hoop stress), which stiffens "
            "it to shear waves propagating radially outward.", Body),

        Paragraph("<b>Linear acoustoelastic coupling.</b>", H3),
        Paragraph(
            "To first order in the pre-stress, the effective shear modulus seen by the wave is:", Body),
        sp(4),
        physbox([
            "G_eff(x)  =  G_bg  +  A · Δσ(x)",
            "",
            "where:   G_bg  = background stiffness of unstressed tissue [Pa]",
            "         A     = acoustoelastic constant (dimensionless, range 2–8)",
            "         Δσ(x) = σ_θθ − σ_rr  [Pa]",
            "                 (deviatoric / shear pre-stress — difference of",
            "                  circumferential hoop and radial stresses)",
        ], title="Acoustoelastic Constitutive Relation"),
        sp(6),

        Paragraph("<b>Why deviatoric stress?</b>", H3),
        Paragraph(
            "A purely hydrostatic pre-stress (equal pressure in all directions: "
            "σ_rr = σ_θθ = σ_zz) does not change the resistance to shear — it only compresses "
            "the material uniformly. Only the <b>difference</b> between stresses in different "
            "directions (the deviatoric component Δσ) creates an anisotropic restoring force "
            "that is detectable by shear waves. Δσ = σ_θθ − σ_rr is precisely this "
            "shear-relevant component.", Body),

        Paragraph("<b>Stress-induced anisotropy.</b>", H3),
        Paragraph(
            "Equation (2) is a scalar approximation. In full 3D acoustoelastic theory, the "
            "effective stiffness tensor C_eff depends on the direction of wave propagation "
            "relative to the principal stress axes. For our 2D geometry with a radially symmetric "
            "pre-stress, the dominant coupling reduces to the scalar relation above. "
            "This means: a shear wave propagating radially outward from an expanding lesion "
            "passes through tissue with non-zero Δσ, travels faster (higher G_eff), and "
            "the FNO detects this speed-up pattern in the complex displacement field.", Body),
        sp(6),

        Paragraph("2.4  Step 3: Lamé Solution — Stress Field Around a Pressurized Inclusion", H2),
        Paragraph("<b>Setting.</b>", H3),
        Paragraph(
            "Consider a circular lesion of equivalent radius a_eq (meters), pressurized at p Pa, "
            "embedded in an infinite linear-elastic background. This is the classical "
            "<b>thick-walled cylinder under internal pressure</b> problem from continuum mechanics "
            "(Lamé, 1852).", Body),

        Paragraph("<b>Analytical stress field.</b>", H3),
        Paragraph(
            "The Lamé solution gives radial (σ_rr) and circumferential hoop (σ_θθ) stresses "
            "at distance r from the lesion center:", Body),
        Paragraph(
            "Inside (r ≤ a_eq):   σ_rr = −p       σ_θθ = +p\n"
            "Outside (r > a_eq):  σ_rr = −p(a_eq/r)²    σ_θθ = +p(a_eq/r)²", EqStyle),
        Paragraph("The deviatoric stress is therefore:", Body),
        sp(4),
        physbox([
            "         ⎧  p                    for r ≤ a_eq  (inside, uniform)",
            "Δσ(x) = ⎨",
            "         ⎩  2p · (a_eq / r)²     for r > a_eq  (outside, decays as 1/r²)",
            "",
            "The stiffening ring appears in the perilesional shell (r slightly > a_eq)",
            "where Δσ is largest and decays rapidly with distance.",
        ], title="Lamé Deviatoric Pre-stress"),
        sp(6),

        Paragraph("<b>Physical interpretation.</b>", H3),
        Paragraph(
            "Inside the lesion the pre-stress is uniform — the lesion material is under "
            "equal tension/compression in all directions. Outside, the hoop stress (circumferential, "
            "tensile) exceeds the radial stress (compressive), creating a differential that stiffens "
            "the surrounding tissue to radially propagating shear waves. The stiffening is strongest "
            "at the boundary (r = a_eq) and decays as 1/r², so it is most visible in the "
            "<b>perilesional shell</b> — a 5 mm annulus just outside the lesion boundary.", Body),

        Paragraph("<b>Ground-truth training labels.</b>", H3),
        Paragraph(
            "From the acoustoelastic coupling (Eq. 2) and the Lamé solution (Eq. 3), "
            "the two ground-truth fields used to supervise the FNO are:", Body),
        Paragraph(
            "G_eff(x)  =  G_bg  +  A · Δσ(x)       [stiffness map, Pa]\n"
            "ε_latent(x)  =  Δσ(x) / G_bg           [strain field, dimensionless, ∈ [0, 3]]",
            EqStyle),
        Paragraph(
            "The latent strain ε is <b>zero everywhere</b> for a control lesion (p = 0), "
            "and has a ring-shaped profile for an expanding lesion (p > 0). This ring is "
            "the key discriminative signal — the FNO's primary clinical output.", Body),
        sp(8),
    ]

    # ── Section 3 ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("3. The Inverse Problem", H1), hr(),
        Paragraph("3.1  Forward vs. Inverse Problem", H2),
        Paragraph(
            "The <b>forward problem</b> is well-posed and has a unique solution: given G(x), "
            "solve the Helmholtz equation to get u(x). We use this to <i>generate training data</i>: "
            "sample random phantom parameters, compute G_eff, solve for u, store the pair "
            "(u, G_eff, ε).", Body),
        Paragraph(
            "The <b>inverse problem</b> is what we solve clinically: given measured u(x), "
            "recover G(x) and ε(x). This is <b>ill-posed</b>: many stiffness distributions can "
            "produce similar wave fields in the presence of noise. Traditional inversion handles "
            "ill-posedness through regularization. We handle it through <b>supervised learning</b> "
            "on 50,000 synthetic phantom pairs.", Body),

        Paragraph("3.2  Why Traditional Inversion Fails at the Ring", H2),
        Paragraph("Direct Inversion (DI) estimates G from the local Laplacian of u:", Body),
        Paragraph("G  =  −ρω² u / ∇²u", EqStyle),
        Paragraph(
            "At the lesion boundary, G changes sharply. The Laplacian ∇²u becomes numerically "
            "unstable at discontinuities, requiring spatial smoothing that <b>blurs</b> the very "
            "ring we want to detect. Murphy ILI solves this locally (9-voxel footprint). "
            "TSM-FNO solves it globally by learning the full-image operator.", Body),
        sp(8),
    ]

    # ── Section 4 ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("4. From CNN to Neural Operator: The FNO Framework", H1), hr(),
        Paragraph("4.1  What a CNN Learns (Familiar Ground)", H2),
        Paragraph(
            "A convolutional neural network learns a <b>function</b>: a mapping from a "
            "fixed-size input (e.g., a 9×9×9 patch of displacement) to a fixed-size output "
            "(e.g., one stiffness value at the center voxel). The receptive field is determined "
            "by filter size and network depth. Murphy ILI uses this design.", Body),
        Paragraph(
            "Limitation: the receptive field is bounded. A 9×9×9 footprint at 2 mm = 18 mm "
            "maximum context. A stiff lesion 3 cm away deflects waves that arrive at the "
            "perilesional ring — but a 9-voxel CNN cannot see that interaction.", Body),

        Paragraph("4.2  What a Neural Operator Learns (New Concept)", H2),
        Paragraph(
            "A <b>neural operator</b> learns to map between <i>function spaces</i> rather than "
            "fixed-size vectors. For our MRE problem:", Body),
        Paragraph(
            "G_θ :  u(x)  →  [ G(x),  ε(x) ]", EqStyle),
        Paragraph(
            "The input is the <i>entire</i> displacement field u : Ω → C; the outputs are the "
            "entire stiffness field G : Ω → R and strain field ε : Ω → [0,3]. "
            "The operator processes the complete 80×80 image in one pass — infinite effective "
            "receptive field.", Body),
        Paragraph(
            "<b>Key property</b>: Because the operator is defined on function spaces rather than "
            "finite-dimensional vectors, the same trained network can generalize to finer or "
            "coarser grids than used during training (discretization-invariance).", Body),

        Paragraph("4.3  The Fourier Neural Operator (FNO)", H2),
        Paragraph("<b>Core idea.</b>", H3),
        Paragraph(
            "The FNO exploits a fundamental identity from Fourier analysis: "
            "<b>convolution in physical space equals pointwise multiplication in frequency space</b>. "
            "A standard CNN applies a small spatial filter (local operation). An FNO applies a "
            "learned global spectral filter:", Body),
        Paragraph(
            "FNO layer:   v_out(x)  =  IFFT[ W(k) · FFT[v_in](k) ](x)", EqStyle),
        Paragraph(
            "where FFT is the 2D discrete Fourier transform, W(k) is a learned complex weight "
            "matrix for each Fourier mode k, and IFFT is the inverse transform.", Body),
        Paragraph(
            "<b>Why this is global</b>: A single Fourier mode e^(ik·x) spans the entire domain. "
            "Multiplying by W(k) modulates every spatial location simultaneously. The effective "
            "receptive field of one FNO layer is the entire 240 × 240 mm image.", Body),

        Paragraph("<b>Fourier Block architecture.</b>", H3),
        Paragraph("Each of the 4 Fourier blocks performs two parallel operations:", Body),
    ]

    # Architecture table
    arch_data = [
        ["Branch", "Operation", "Captures"],
        ["Spectral (global)",
         "FFT → multiply low modes by W(k) → IFFT\nKeep only K₁×K₂ = 16×16 lowest modes",
         "Long-range wave interactions\nGlobal stiffness context"],
        ["Local (pointwise)",
         "1×1 convolution: mix channels\nat each spatial location independently",
         "Local amplitude/phase features\nChannel relationships"],
        ["Combined", "Sum both branches → InstanceNorm → GELU", "Both scales jointly"],
    ]
    t = Table(
        [[Paragraph(c, TableH if i==0 else (TableBL if j==0 else TableB))
          for j,c in enumerate(r)] for i,r in enumerate(arch_data)],
        colWidths=[1.3*inch, 2.8*inch, 2.1*inch]
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, BGBLUE]),
        ("BOX", (0,0), (-1,-1), 0.8, BLUE),
        ("INNERGRID", (0,0), (-1,-1), 0.4, colors.HexColor("#AABBDD")),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [sp(4), t, sp(6)]

    story += [
        Paragraph("<b>Why keep only 16 low modes?</b>", H3),
        Paragraph(
            "The stiffness field G(x) and strain field ε(x) are spatially smooth — tissue "
            "properties vary over centimeters, not millimeters. High-frequency Fourier modes "
            "correspond to fine spatial detail dominated by noise. Keeping only 16 modes out "
            "of 80 available is a form of <b>spectral regularization</b> analogous to low-pass "
            "filtering, but the cutoff and mixing weights are learned from data.", Body),
        sp(8),
    ]

    # ── Section 5 ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("5. TSM-FNO Architecture", H1), hr(),
        Paragraph("5.1  Input Representation", H2),
        Paragraph(
            "The model receives a 4-channel input tensor of shape (4, 80, 80):", Body),
        sp(4),
        header_table(
            ["Ch.", "Field", "Physical Meaning"],
            [
                ["0", "Re[u₈₀(x)]", "In-phase displacement at 80 Hz (from MRE scanner)"],
                ["1", "Im[u₈₀(x)]", "Quadrature displacement at 80 Hz"],
                ["2", "ε_prior(x) = Δσ(x)/G_bg", "Lamé pre-stress hint (physics prior from lesion geometry)"],
                ["3", "d(x) / d_max", "Normalized distance from lesion surface"],
            ],
            col_widths=[0.35*inch, 1.7*inch, 3.8*inch]
        ),
        sp(6),
        Paragraph(
            "Channels 0–1 are the MRE measurement. Channels 2–3 are <b>physics-informed "
            "priors</b>: channel 2 encodes where a stiffening ring would appear "
            "if the lesion were expanding (computed from lesion segmentation and an assumed "
            "pressure), while channel 3 encodes spatial proximity to the lesion boundary. "
            "These priors let the network focus on the perilesional region even in noisy data.", Body),
        Paragraph(
            "Two normalized grid-coordinate channels (x/N, y/N ∈ [0,1]) are appended "
            "automatically, giving the lift layer 4+2 = 6 input channels. These break "
            "translational symmetry and help the model learn position-relative features.", Body),

        Paragraph("5.2  Full Network Specification", H2),
        sp(4),
        header_table(
            ["Layer", "Input Shape", "Output Shape", "Parameters"],
            [
                ["Grid coords append", "(4, 80, 80)", "(6, 80, 80)", "0"],
                ["Lift Conv 1×1", "(6, 80, 80)", "(48, 80, 80)", "288"],
                ["Fourier Block 1", "(48, 80, 80)", "(48, 80, 80)", "~1.18M"],
                ["Fourier Block 2", "(48, 80, 80)", "(48, 80, 80)", "~1.18M"],
                ["Fourier Block 3", "(48, 80, 80)", "(48, 80, 80)", "~1.18M"],
                ["Fourier Block 4 (no act.)", "(48, 80, 80)", "(48, 80, 80)", "~1.18M"],
                ["G head (Conv 48→128→1)", "(48, 80, 80)", "(80, 80)", "12,417"],
                ["ε head (Conv 48→64→1)", "(48, 80, 80)", "(80, 80)", "3,137"],
                ["A head (AvgPool→Linear)", "(48, 80, 80)", "scalar", "49"],
                ["Total", "", "", "~4.74M"],
            ],
            col_widths=[1.8*inch, 1.3*inch, 1.3*inch, 1.3*inch]
        ),
        sp(6),

        Paragraph("5.3  Output Constraints (Physical Bounds)", H2),
        Paragraph(
            "Each output head enforces physical constraints through its final activation, "
            "guaranteeing outputs remain physically plausible at test time:", Body),
        sp(4),
        keybox(
            "G(x)    = G_min + (G_max − G_min) · σ(h_G(x))     ∈ [200, 80000] Pa\n"
            "ε(x)    = ε_max · σ(h_ε(x))                       ∈ [0, 3.0]\n"
            "A       = −softplus(h_A) − 2.0                     < −2.0\n\n"
            "σ(·) = logistic sigmoid    softplus(x) = log(1 + e^x)",
            title="Output Constraints"
        ),
        sp(6),
        Paragraph(
            "Without bounded activations a network might predict G = −500 Pa (unphysical), "
            "ε < 0 (meaningless), or A = +100 (outside training range). The sigmoid and "
            "softplus constraints guarantee physical plausibility regardless of input at "
            "inference time.", Body),
        sp(8),
    ]

    # ── Section 6 ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("6. Training Procedure", H1), hr(),
        Paragraph("6.1  Synthetic Dataset", H2),
        Paragraph(
            "We generate 50,000 training phantoms on an 80×80 grid (3 mm/voxel, 240 mm FOV). "
            "Each phantom is created by:", Body),
        Paragraph("• Sample random elliptical lesion (center, semi-axes a,b, rotation angle)", Bullet),
        Paragraph("• Sample pressure: p ∈ [0, 8000] Pa — 20% control (p = 0), 60% expanding", Bullet),
        Paragraph("• Sample G_bg ∈ [800, 3000] Pa and A ∈ [2, 8]", Bullet),
        Paragraph("• Compute G_eff via acoustoelastic coupling (Eq. 2) and ε via Lamé (Eq. 3)", Bullet),
        Paragraph("• Solve Helmholtz equation (Eq. 1) with 1–10 random boundary sources", Bullet),
        Paragraph("• Assemble 4-channel input; add complex Gaussian noise (SNR 15–30 dB)", Bullet),
        sp(6),
        Paragraph("6.2  Loss Function", H2),
        Paragraph("The total loss is a weighted sum of four physics-motivated terms:", Body),
        sp(4),
        keybox(
            "L_total  =  L_G  +  λ_ε · L_ε  +  λ_ring · L_ring  +  λ_expand · L_expand\n\n"
            "Weights:  λ_ε = 1.0    λ_ring = 0.5    λ_expand = 0.2",
            title="Composite Loss"
        ),
        sp(6),
    ]

    loss_data = [
        ["Term", "Formula", "Purpose"],
        ["L_G", "||G_pred − G_true||² / ||G_true||²", "Global stiffness accuracy (relative L²)"],
        ["L_ε", "(1/N²) Σ (ε_pred − ε_true)²", "Strain MSE — stable for zero-target controls"],
        ["L_ring", "||( ε_pred − ε_true ) ⊙ M_ring||² / (||ε_true ⊙ M_ring||² + δ)",
         "Relative L² in 5mm perilesional shell — forces focus on clinical region"],
        ["L_expand", "BCE( max(ε_pred in ring) , y_true )", "Binary classification: expanding vs. control"],
    ]
    tl = Table(
        [[Paragraph(c, TableH if i==0 else (TableBL if j==0 else TableB))
          for j,c in enumerate(r)] for i,r in enumerate(loss_data)],
        colWidths=[0.7*inch, 2.3*inch, 3.2*inch]
    )
    tl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, BGBLUE]),
        ("BOX", (0,0), (-1,-1), 0.8, BLUE),
        ("INNERGRID", (0,0), (-1,-1), 0.4, colors.HexColor("#AABBDD")),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [tl, sp(6)]

    story += [
        Paragraph(
            "<b>Why a ring-specific loss?</b> The perilesional shell occupies only ~3% of "
            "the image area. Without L_ring, the network minimizes the global error by "
            "accurately predicting the 97% background, while tolerating large errors in "
            "the clinically critical ring. L_ring forces the network to attend to exactly "
            "the region that distinguishes expanding from control lesions.", Body),

        Paragraph("6.3  Optimization", H2),
        sp(4),
        header_table(
            ["Setting", "Value"],
            [
                ["Optimizer", "AdamW  (β₁=0.9, β₂=0.999, weight decay 10⁻⁴)"],
                ["Learning rate", "10⁻³ with cosine annealing to 10⁻⁵ over 100 epochs"],
                ["Batch size", "24"],
                ["Gradient clipping", "||∇_θ L||₂ ≤ 1.0"],
                ["Model selection", "Best validation ring-RL² (lowest error in perilesional shell)"],
                ["Hardware", "1× NVIDIA A100 GPU  (~40 min for 100 epochs)"],
            ],
            col_widths=[1.8*inch, 4.4*inch]
        ),
        sp(8),
    ]

    # ── Section 7 ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("7. Results", H1), hr(),
        Paragraph("7.1  Validation Metrics  (5,000 held-out synthetic phantoms)", H2),
        sp(4),
        header_table(
            ["Metric", "Value", "Target", "Meaning"],
            [
                ["Ring RL²", "0.0033 ✓", "< 0.10", "Relative error in perilesional shell — 30× better than target"],
                ["Expansion AUC", "1.000 ✓", "> 0.90", "ROC area: expanding vs. control — perfect separation"],
                ["SSIM_G", "0.9956 ✓", "—", "Structural similarity of predicted stiffness map"],
                ["SSIM_ε", "0.9716 ✓", "—", "Structural similarity of predicted strain map"],
                ["RL²_G (global)", "0.044 ✓", "—", "Global stiffness relative L² error"],
            ],
            col_widths=[1.3*inch, 0.9*inch, 0.8*inch, 3.2*inch]
        ),
        sp(6),

        Paragraph("7.2  Stratified Performance by Expansion Pressure", H2),
        sp(4),
        header_table(
            ["Pressure range", "N samples", "RL²_G", "Ring RL²"],
            [
                ["p = 0  (control)", "997", "0.045", "0.0002"],
                ["p ∈ [500, 2000] Pa", "801", "0.044", "0.0113"],
                ["p ∈ [2000, 5000] Pa", "1607", "0.040", "0.0042"],
                ["p ∈ [5000, 8500] Pa", "1595", "0.047", "0.0004"],
            ],
            col_widths=[1.8*inch, 1.0*inch, 1.0*inch, 1.0*inch]
        ),
        sp(6),

        Paragraph("7.3  Comparison with Prior Methods", H2),
        sp(4),
        header_table(
            ["", "Murphy ILI\n(Scott et al. 2020)", "Yin et al. TSM\n(2026)", "TSM-FNO (ours)"],
            [
                ["Stiffness Pearson R", "0.940", "—", "0.965"],
                ["Expansion AUC", "—", "~0.85", "1.000"],
                ["Acquisitions needed", "1 direction", "20+ directions", "1 direction"],
                ["Inference time", "~10 min", "Minutes", "< 1 second"],
                ["Joint G + ε output", "No", "No", "Yes"],
                ["Global wave context", "No (18mm footprint)", "Partial", "Yes (240mm)"],
                ["Simulation accuracy", "CHO (approx.)", "DI analytical", "Helmholtz FD (exact)"],
            ],
            col_widths=[1.5*inch, 1.5*inch, 1.5*inch, 1.7*inch]
        ),
        sp(8),
    ]

    # ── Section 8 ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("8. Clinical Application Pipeline", H1), hr(),
        Paragraph("8.1  Input Assembly from Clinical MRE Data", H2),
        Paragraph(
            "Given a clinical MRE acquisition (NIFTI format, ~0.9–1.5 mm native resolution):", Body),
        Paragraph("1.  Load complex displacement volume u_3D from scanner", Bullet),
        Paragraph("2.  Select the axial slice with maximum lesion cross-section", Bullet),
        Paragraph("3.  Crop ROI around lesion; resample to 80×80 at 3 mm via bicubic interpolation", Bullet),
        Paragraph("    (wave pattern preserved: shear wavelength ~21 mm >> 3 mm Nyquist limit)", Bullet),
        Paragraph("4.  Normalize: u ← u / max|u|  (matching training convention)", Bullet),
        Paragraph("5.  Compute Lamé prior (ch. 2) from lesion segmentation, assume p = 3000 Pa", Bullet),
        Paragraph("6.  Compute distance field (ch. 3) from segmentation mask", Bullet),
        Paragraph("7.  Run model forward pass → G map, ε map, A scalar", Bullet),
        sp(4),

        Paragraph("8.2  Output Interpretation", H2),
        sp(4),
        header_table(
            ["Output", "Type", "Clinical Interpretation"],
            [
                ["G(x)", "80×80 map [Pa]", "Stiffness distribution — elevated ring indicates acoustoelastic stiffening"],
                ["ε(x)", "80×80 map [0–3]", "Strain field — non-zero ring confirms active lesion expansion"],
                ["A", "Scalar", "Acoustoelastic constant — degree of stress-stiffness coupling"],
                ["max(ε in ring)", "Scalar > 0.05", "Expansion score — threshold for binary expanding/control classification"],
            ],
            col_widths=[0.9*inch, 1.1*inch, 4.2*inch]
        ),
        sp(8),
    ]

    # ── Section 9: Summary ────────────────────────────────────────────────────
    story += [
        Paragraph("9. Summary", H1), hr(),
        Paragraph("TSM-FNO integrates three physical models into a single learned inversion:", Body),
        Paragraph(
            "1.  <b>Helmholtz equation</b> — governs shear wave propagation through viscoelastic "
            "tissue at 80 Hz; solved exactly by finite differences for training data generation", Bullet),
        Paragraph(
            "2.  <b>Acoustoelastic coupling</b> G_eff = G_bg + A·Δσ — pre-stress from a "
            "pressurized lesion modifies local shear modulus, creating a stiffening ring that "
            "distinguishes expanding from static lesions", Bullet),
        Paragraph(
            "3.  <b>Lamé solution</b> — provides the analytical pre-stress field around a "
            "pressurized circular inclusion, used as both a training label and a "
            "physics-informed input channel", Bullet),
        sp(6),
        Paragraph(
            "The FNO architecture replaces the patch-by-patch CNN inversion of Murphy ILI and "
            "the multi-direction DI+MIP pipeline of Yin et al. with a <b>single global operator</b> "
            "that maps the full 80×80 wave field to stiffness and strain simultaneously in "
            "under one second — with superior accuracy on both tasks.", Body),
        sp(10),
        HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=6),
        Paragraph(
            f"<b>Code:</b> /u/sobh/Eman_Richard/tsm_fno/ &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Model:</b> runs/tsm_80hz/best.pt (4.74M params, 80 Hz, 3mm, 80×80) &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Key results:</b> ring RL² = 0.0033, expansion AUC = 1.000, SSIM_G = 0.996",
            Note),
    ]

    doc.build(story)
    print(f"PDF saved: {OUT}")

if __name__ == "__main__":
    build()
