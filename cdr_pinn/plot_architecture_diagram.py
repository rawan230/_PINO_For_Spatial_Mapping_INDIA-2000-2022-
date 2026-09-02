"""Publication-style schematic of the CDR-PINN architecture: PINO/FNO backbone
(one-step operator G_theta: u_t -> u_{t+1}) + the three physics heads (D_net,
advection, rho_net) that assemble the CDR PDE residual, plus the four-term
adaptive loss. Purely illustrative (box-and-arrow), but every number on it is
read directly from the real model/training code, not invented:
  - model.py: FNO2d(width=32, modes_h=modes_w=16, n_layers=4, GELU),
    DiffusivityHead/ReactionHead (hidden=12, Tanh MLPs), AdvectionHead (single
    learned scalar).
  - preprocessing.py: covariate_stack() channel order (7 covariates + u_t = 8
    input channels), physics_covariates() (which fields feed which head).
  - train_standard_protocol.py: WIDTH=32, N_LAYERS=4, MODES=16,
    n_static_channels=7.
  - losses.py: AdaptiveLossBalancer (Wang, Teng & Perdikaris 2021) over the
    four loss groups (data / pde / bc / ic).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_architecture_diagram.png"

C_INPUT = "#e8eef5"
C_BACKBONE = "#c9dcf0"
C_SPECTRAL = "#4472a8"
C_PHYSICS = "#dcefdc"
C_PHYSICS_EDGE = "#4a8a4a"
C_PDE = "#fbe8d6"
C_PDE_EDGE = "#c97b2e"
C_LOSS = "#f3dede"
C_LOSS_EDGE = "#b0433f"
C_CFG = "#f2f2f2"
C_CFG_EDGE = "#777777"
C_TEXT = "#1c1c1c"

FIG_W, FIG_H = 20.0, 15.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")


def box(x, y, w, h, text, fc, ec="#333333", lw=1.3, fontsize=9, fontweight="normal", zorder=2):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.09",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, color=C_TEXT, zorder=zorder + 1, linespacing=1.45)
    return b


def arrow(x0, y0, x1, y1, color="#333333", lw=1.5, connstyle="arc3,rad=0.0"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                         linewidth=lw, color=color, connectionstyle=connstyle, zorder=1.5)
    ax.add_patch(a)


CX = 8.55  # main-flow horizontal center

# ---------------------------------------------------------------------- #
# Title
# ---------------------------------------------------------------------- #
ax.text(CX, 14.55, "CDR-PINN Architecture", ha="center", fontsize=19, fontweight="bold", color=C_TEXT)
ax.text(CX, 14.1, "Physics-informed neural operator (PINO/FNO backbone) + convection-diffusion-reaction PDE constraint",
        ha="center", fontsize=11, color="#444444", style="italic")

# ---------------------------------------------------------------------- #
# Row: Input channels
# ---------------------------------------------------------------------- #
ax.text(1.0, 13.35, "Input: 256\u00d7256 grid, 8 channels (7 covariates + $u_t$)", fontsize=10.5, fontweight="bold", color=C_TEXT)
in_y, in_h = 12.35, 0.8
names = ["$u_t$\n(current-month\nfire logit)", "ndvi_f1", "ndvi_anomaly$_t$", "forest_frac",
         "dryness$_t$", "slope", "dist_roads", "elevation"]
n_in = len(names)
in_w = 1.72
in_gap = 0.12
in_total = n_in * in_w + (n_in - 1) * in_gap
in_x0 = CX - in_total / 2
for i, name in enumerate(names):
    box(in_x0 + i * (in_w + in_gap), in_y, in_w, in_h, name, C_INPUT, ec="#7d93ab", fontsize=8.6)
arrow(CX, in_y, CX, in_y - 0.55, color="#7d93ab")

# ---------------------------------------------------------------------- #
# Lifting layer
# ---------------------------------------------------------------------- #
lift_y = in_y - 1.15
box(CX - 1.85, lift_y, 3.7, 0.6, "Lift: 1\u00d71 Conv2d,  8 \u2192 32 channels", C_BACKBONE, ec="#2f5788", fontsize=10, fontweight="bold")
arrow(CX, lift_y, CX, lift_y - 0.45, color="#555555")

# ---------------------------------------------------------------------- #
# 4x FNO Fourier blocks
# ---------------------------------------------------------------------- #
fno_y = lift_y - 2.0
fno_h = 1.55
fno_w = 3.3
gap = 0.3
n_blocks = 4
total_w = n_blocks * fno_w + (n_blocks - 1) * gap
start_x = CX - total_w / 2

for i in range(n_blocks):
    bx = start_x + i * (fno_w + gap)
    outer = FancyBboxPatch((bx, fno_y), fno_w, fno_h, boxstyle="round,pad=0.06,rounding_size=0.1",
                            linewidth=1.6, edgecolor="#2f5788", facecolor=C_BACKBONE, zorder=2)
    ax.add_patch(outer)
    ax.text(bx + fno_w / 2, fno_y + fno_h - 0.2, f"FNO Block {i + 1}", ha="center", fontsize=9.3, fontweight="bold", color=C_TEXT, zorder=3)
    box(bx + 0.15, fno_y + 0.5, 1.45, 0.68, "SpectralConv2d\n16\u00d716 Fourier\nmodes (learned)", "white", ec=C_SPECTRAL, fontsize=6.7, zorder=3)
    box(bx + fno_w - 1.6, fno_y + 0.5, 1.45, 0.68, "1\u00d71 Conv2d\n(skip path)", "white", ec="#888888", fontsize=7.3, zorder=3)
    ax.text(bx + fno_w / 2, fno_y + 0.58, "+", ha="center", fontsize=14, fontweight="bold", color=C_TEXT, zorder=3)
    ax.text(bx + fno_w / 2, fno_y + 0.1, "GELU", ha="center", fontsize=8.2, style="italic", color="#444444", zorder=3)
    if i > 0:
        px = start_x + (i - 1) * (fno_w + gap) + fno_w
        arrow(px, fno_y + fno_h / 2, bx, fno_y + fno_h / 2, color="#555555")

ax.text(CX, fno_y - 0.32, "width = 32 channels throughout  |  4 stacked Fourier layers (L = 4)", ha="center", fontsize=9.5, color="#444444")

# ---------------------------------------------------------------------- #
# Projection + output
# ---------------------------------------------------------------------- #
proj_y = fno_y - 1.15
arrow(CX, fno_y, CX, proj_y + 0.6, color="#555555")
box(CX - 3.9, proj_y, 3.55, 0.6, "1\u00d71 Conv (32\u219232) + GELU", C_BACKBONE, ec="#2f5788", fontsize=9.2)
box(CX + 0.35, proj_y, 3.55, 0.6, "1\u00d71 Conv (32\u21921)", C_BACKBONE, ec="#2f5788", fontsize=9.2)
arrow(CX - 0.35, proj_y + 0.3, CX + 0.35, proj_y + 0.3, color="#555555")

u_out_y = proj_y - 1.0
arrow(CX + 2.1, proj_y, CX + 2.1, u_out_y + 0.55, color="#555555")
box(CX - 2.4, u_out_y, 4.8, 0.55, "$u_{t+1}$  (next-month fire logit)", "#fff7d6", ec="#a88a1f", fontsize=10, fontweight="bold")

# ---------------------------------------------------------------------- #
# Physics heads (own row, parallel branch)
# ---------------------------------------------------------------------- #
heads_title_y = u_out_y - 0.55
ax.text(start_x, heads_title_y, "Physics heads  (read covariates directly, bypassing the FNO)",
        fontsize=10.5, fontweight="bold", color=C_TEXT)

heads_y = heads_title_y - 2.15
heads_h = 1.85
head_w = 3.3
head_gap = 0.3
head_total = n_blocks * head_w + (n_blocks - 1) * head_gap  # reuse FNO span (4 slots -> use 3 wider boxes below)

hw = (total_w - 2 * gap) / 3  # 3 heads spanning the same total width as the 4 FNO blocks
hx0 = start_x
head_boxes_x = [hx0, hx0 + hw + gap, hx0 + 2 * (hw + gap)]

box(head_boxes_x[0], heads_y, hw, heads_h,
    "DiffusivityHead\nMLP([ndvi_f1, forest_frac]): 2\u219212\u219212\u21921, Tanh\n"
    "$z = z_{struct} -$ softplus($w$)$\\cdot$ndvi_anomaly$_t$\n"
    "$D(x,y,t) = $ softplus($z$)",
    C_PHYSICS, ec=C_PHYSICS_EDGE, fontsize=8.3)

box(head_boxes_x[1], heads_y, hw, heads_h,
    "AdvectionHead\n$v(x,y) = $ softplus($c$) $\\cdot \\nabla E(x,y)$\n"
    "(elevation gradient; terrain-driven,\nupslope by construction)",
    C_PHYSICS, ec=C_PHYSICS_EDGE, fontsize=8.3)

box(head_boxes_x[2], heads_y, hw, heads_h,
    "ReactionHead\nMLP([dryness, ndvi_f1, slope, dist_roads]):\n4\u219212\u219212\u21921, Tanh\n"
    "$\\rho(x,y,t) = $ softplus(MLP output)\n"
    "$R = \\rho\\,\\sigma(u)(1-\\sigma(u))$  (Fisher-KPP form)",
    C_PHYSICS, ec=C_PHYSICS_EDGE, fontsize=8.1)

# ---------------------------------------------------------------------- #
# PDE residual assembly box
# ---------------------------------------------------------------------- #
pde_y = heads_y - 1.55
box(start_x, pde_y, total_w, 1.05,
    "CDR PDE residual:   $r = \\dfrac{\\partial u}{\\partial t} - D\\,\\nabla^2 u + v\\cdot\\nabla u - \\rho\\,\\sigma(u)(1-\\sigma(u))$\n"
    "spatial operators evaluated with spherical spectral differentiation (Neumann whole-sample-symmetric extension)",
    C_PDE, ec=C_PDE_EDGE, fontsize=9.8, fontweight="bold")

for hbx in head_boxes_x:
    arrow(hbx + hw / 2, heads_y, hbx + hw / 2, pde_y + 1.05, color=C_PHYSICS_EDGE, lw=1.2)
arrow(CX, u_out_y, CX + total_w / 2 - 0.6, pde_y + 1.05, color="#a88a1f", lw=1.3, connstyle="arc3,rad=-0.25")
ax.text(CX + total_w / 2 - 1.7, u_out_y - 0.55, "$u_t,\\,u_{t+1}$", fontsize=9, color="#a88a1f", style="italic")

# ---------------------------------------------------------------------- #
# Loss assembly box
# ---------------------------------------------------------------------- #
loss_y = pde_y - 2.15
box(start_x, loss_y, total_w, 1.85,
    "Total loss:   $\\mathcal{L} = w_{data}\\mathcal{L}_{data} + w_{data}\\mathcal{L}_{terminal} + w_{pde}\\mathcal{L}_{pde} + w_{bc}\\mathcal{L}_{bc} + w_{ic}\\mathcal{L}_{ic}$\n"
    "$\\mathcal{L}_{data}$: monthly BCE, pos-weighted ($\\sim$2.3% positive rate)   |   $\\mathcal{L}_{terminal}$: LSE-pooled ($\\tau$=5) trajectory BCE vs. fire_ever label\n"
    "$\\mathcal{L}_{pde} = \\mathbb{E}[r^2]$   |   $\\mathcal{L}_{bc}$: homogeneous Neumann (boundary $|\\nabla u|^2$)   |   $\\mathcal{L}_{ic} = \\mathbb{E}[u_{t=0}^2]$\n"
    "weights $w_\\bullet$: gradient-norm-balanced (Wang, Teng and Perdikaris 2021), rescaled every 5 steps",
    C_LOSS, ec=C_LOSS_EDGE, fontsize=9.4)
arrow(CX - total_w / 2 + 1.2, pde_y, CX - total_w / 2 + 1.2, loss_y + 1.85, color=C_PDE_EDGE, lw=1.4)
arrow(CX + 2.5, u_out_y, CX + total_w / 2 - 1.2, loss_y + 1.85, color="#a88a1f", lw=1.2, connstyle="arc3,rad=-0.3")

# ---------------------------------------------------------------------- #
# Config panel (right margin, spans input row down to physics-heads row)
# ---------------------------------------------------------------------- #
cfg_x = start_x + total_w + 0.6
cfg_w = FIG_W - cfg_x - 0.3
cfg_y = heads_y
cfg_h = (in_y + in_h) - heads_y
box(cfg_x, cfg_y, cfg_w, cfg_h,
    "Training configuration\n\n"
    "\u2013 Operator: one-step-ahead\n  $G_\\theta(u_t, a_t) \\to u_{t+1}$\n"
    "\u2013 Grid: 256\u00d7256 (India,\n  68.2\u201397.4\u00b0E, 6.75\u201337.09\u00b0N)\n"
    "\u2013 Optimizer: AdamW\n  (weight_decay=0, validated)\n"
    "\u2013 LR schedule: ReduceLROnPlateau\n  (loss-monitored)\n"
    "\u2013 Split: 65/15/20 train/val/test\n  (seed=42)\n"
    "\u2013 Early stopping: val ROC-AUC,\n  patience=4, checked every\n  5 epochs\n"
    "\u2013 Rollout scoring: LSE-pool\n  ($\\tau$=5) over full trajectory,\n  matches fire_ever semantics",
    C_CFG, ec=C_CFG_EDGE, fontsize=8.6)

fig.savefig(OUT_PATH, dpi=160, facecolor="white", bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
