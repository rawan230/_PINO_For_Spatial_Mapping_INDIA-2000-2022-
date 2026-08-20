"""
CDR-PINN model: FNO/PINO backbone + three physics heads (D_net, advection,
rho_net) + the full CDR PDE residual, per CDR_PINN_Final_Design_STEP_D.md.

Architecture defaults (width=64, modes~12-20, L=4, GELU) follow the PINO paper's
own reported defaults (Li et al. 2023, Appendix A) as a starting point.
"""
import torch
import torch.nn as nn
import numpy as np

from spectral_ops import (
    fft_derivative_1d, spherical_gradient, spherical_laplacian,
    neumann_periodic_extend, neumann_periodic_crop, EARTH_RADIUS_KM,
)


# --------------------------------------------------------------------------- #
# FNO backbone
# --------------------------------------------------------------------------- #

class SpectralConv2d(nn.Module):
    """Fourier convolution layer (PINO paper Definition 3 / Li et al. 2023): a
    learned, truncated-mode linear operator applied in Fourier space."""

    def __init__(self, in_ch, out_ch, modes_h, modes_w):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes_h, self.modes_w = modes_h, modes_w
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_h, modes_w, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        mh, mw = min(self.modes_h, H), min(self.modes_w, W // 2 + 1)
        out_ft[:, :, :mh, :mw] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :mh, :mw], self.weight[:, :, :mh, :mw]
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, width, modes_h, modes_w):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_h, modes_w)
        self.skip = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.skip(x))


class FNO2d(nn.Module):
    """Lifting -> L Fourier blocks -> projection, the standard PINO/FNO pattern
    (PINO paper Definition 1). Operates on a grid-shaped input (B, C_in, H, W),
    outputs (B, C_out, H, W)."""

    def __init__(self, in_ch, out_ch=1, width=64, modes_h=16, modes_w=16, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes_h, modes_w) for _ in range(n_layers)])
        self.proj1 = nn.Conv2d(width, width, kernel_size=1)
        self.proj2 = nn.Conv2d(width, out_ch, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = self.act(self.proj1(x))
        return self.proj2(x)


# --------------------------------------------------------------------------- #
# Physics heads (CDR_PINN_Final_Design_STEP_D.md Section 1)
# --------------------------------------------------------------------------- #

class DiffusivityHead(nn.Module):
    """D(x,y,t) = softplus(D_net([NDVI_F1, forest_frac]) - softplus(w_raw)*NDVI_anomaly).
    STEP C resolution: forest_frac folded into D_net's input alongside NDVI_F1."""

    def __init__(self, hidden=12):
        super().__init__()
        self.d_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.w_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, ndvi_f1, forest_frac, ndvi_anomaly):
        # all inputs: (B, H, W)
        stacked = torch.stack([ndvi_f1, forest_frac], dim=-1)  # (B,H,W,2)
        z_struct = self.d_net(stacked).squeeze(-1)             # (B,H,W)
        z = z_struct - torch.nn.functional.softplus(self.w_raw) * ndvi_anomaly
        return torch.nn.functional.softplus(z)  # D(x,y,t), (B,H,W)


class AdvectionHead(nn.Module):
    """v(x,y) = softplus(c_raw) * grad(E)(x,y) -- terrain-driven, upslope by
    construction (CDR_PINN_Advection_Design.md Section 1)."""

    def __init__(self):
        super().__init__()
        self.c_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, grad_e_x, grad_e_y):
        c_adv = torch.nn.functional.softplus(self.c_raw)
        return c_adv * grad_e_x, c_adv * grad_e_y


class ReactionHead(nn.Module):
    """rho(x,y,t) = softplus(rho_net([dryness, NDVI_F1, slope, dist_roads])),
    Fisher-KPP form R = rho * sigmoid(u) * (1 - sigmoid(u))
    (CDR_PINN_Reaction_Design.md Sections 1-2)."""

    def __init__(self, hidden=12):
        super().__init__()
        self.rho_net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def rho(self, dryness, ndvi_f1, slope, dist_roads):
        stacked = torch.stack([dryness, ndvi_f1, slope, dist_roads], dim=-1)
        z = self.rho_net(stacked).squeeze(-1)
        return torch.nn.functional.softplus(z)

    def forward(self, u, dryness, ndvi_f1, slope, dist_roads):
        rho = self.rho(dryness, ndvi_f1, slope, dist_roads)
        s = torch.sigmoid(u)
        return rho * s * (1 - s)


# --------------------------------------------------------------------------- #
# Full CDR-PINN
# --------------------------------------------------------------------------- #

class CDRPINN(nn.Module):
    """One-step-ahead operator G_theta: (u_t, a_t) -> u_{t+1}, per
    CDR_PINN_Final_Design_STEP_D.md Section 3. `a_t` is the stacked static +
    monthly-varying covariate grid; `u_t` is appended as an extra input channel."""

    def __init__(self, n_static_channels, width=64, modes_h=16, modes_w=16, n_layers=4):
        super().__init__()
        # +1 channel for u_t itself (the state being evolved)
        self.operator = FNO2d(in_ch=n_static_channels + 1, out_ch=1,
                               width=width, modes_h=modes_h, modes_w=modes_w, n_layers=n_layers)
        self.D_head = DiffusivityHead()
        self.v_head = AdvectionHead()
        self.rho_head = ReactionHead()

    def forward(self, u_t, covariate_stack):
        """u_t: (B,1,H,W). covariate_stack: (B,C,H,W). Returns u_{t+1}: (B,1,H,W)."""
        x = torch.cat([u_t, covariate_stack], dim=1)
        return self.operator(x)

    def pde_residual(self, u_t, u_next, dt_months, covariates: dict, lat_rad_1d,
                      dlon_rad, dlat_rad, R=EARTH_RADIUS_KM,
                      use_diffusion=True, use_advection=True, use_reaction=True):
        """Assemble the single combined CDR residual:
        r = du/dt - D*lap(u) + v.grad(u) - rho*sigma(u)(1-sigma(u))
        Time derivative via simple forward difference between consecutive months
        (the per-month operator's own natural time discretization); spatial
        derivatives via the verified spherical spectral operators, evaluated with
        a Neumann whole-sample-symmetric extension on both spatial axes so the
        FFT sees an array that's actually periodic (spectral_ops.py Test 5).

        The use_* flags implement the term-ablation study
        (CDR_PINN_Final_Design_STEP_D.md Section 5 / Methodology Section 8):
        diffusion-only, diffusion+advection, and full-CDR configurations all call
        this same method, differing only in which terms contribute to the
        residual actually optimized -- the physics heads (D/v/rho) are still
        computed and returned either way, so a disabled term's own diagnostic
        output remains inspectable even when it isn't part of the loss."""
        u = u_t.squeeze(1)          # (B,H,W)
        u_np1 = u_next.squeeze(1)
        du_dt = (u_np1 - u) / dt_months

        def extend2d(field):
            f = neumann_periodic_extend(field, axis=-1)
            f = neumann_periodic_extend(f, axis=-2)
            return f

        def crop2d(field, H, W):
            f = neumann_periodic_crop(field, axis=-2, n_original=H)
            f = neumann_periodic_crop(f, axis=-1, n_original=W)
            return f

        H, W = u.shape[-2:]
        u_mid = 0.5 * (u + u_np1)  # evaluate spatial operators at the midpoint (Crank-Nicolson-style)
        u_ext = extend2d(u_mid)
        n_ext_h = u_ext.shape[-2]
        lat_ext = torch.cat([
            lat_rad_1d,
            lat_rad_1d.flip(0)[1:-1] if lat_rad_1d.numel() > 2 else lat_rad_1d.flip(0)
        ])[:n_ext_h].to(u.device)

        lap_ext = spherical_laplacian(u_ext, lat_ext, dlon_rad, dlat_rad, R=R)
        lap = crop2d(lap_ext, H, W)

        du_dx_ext, du_dy_ext = spherical_gradient(u_ext, lat_ext, dlon_rad, dlat_rad, R=R)
        du_dx = crop2d(du_dx_ext, H, W)
        du_dy = crop2d(du_dy_ext, H, W)

        D = self.D_head(covariates["ndvi_f1"], covariates["forest_frac"], covariates["ndvi_anomaly"])
        vx, vy = self.v_head(covariates["grad_e_x"], covariates["grad_e_y"])
        R_term = self.rho_head(u_mid, covariates["dryness"], covariates["ndvi_f1"],
                                covariates["slope"], covariates["dist_roads"])

        diffusion_term = D * lap if use_diffusion else torch.zeros_like(lap)
        advection_term = (vx * du_dx + vy * du_dy) if use_advection else torch.zeros_like(lap)
        reaction_term = R_term if use_reaction else torch.zeros_like(R_term)

        residual = du_dt - diffusion_term + advection_term - reaction_term
        return residual, {"D": D, "vx": vx, "vy": vy, "reaction": R_term, "laplacian": lap}
