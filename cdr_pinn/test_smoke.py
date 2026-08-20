"""
End-to-end smoke test on synthetic data: verifies shapes, that autograd flows
into every physics head and the FNO operator, and that the assembled residual
has the physically-expected sign properties. This does NOT touch real data --
see the design docs (CDR_PINN_Final_Design_STEP_D.md Section 7) for what real
training still requires.
"""
import torch
import numpy as np
from model import CDRPINN

torch.manual_seed(0)

B, H, W = 2, 64, 64
N_STATIC = 6  # ndvi_f1, ndvi_anomaly, forest_frac, dryness, slope, dist_roads
              # (grad_e_x/y are handled separately, not stacked into the operator's input)

lat_deg = np.linspace(6.75, 37.09, H)
lon_deg = np.linspace(68.20, 97.40, W)
lat_rad = torch.tensor(np.radians(lat_deg), dtype=torch.float32)
dlat_rad = float(np.radians(lat_deg[1] - lat_deg[0]))
dlon_rad = float(np.radians(lon_deg[1] - lon_deg[0]))

model = CDRPINN(n_static_channels=N_STATIC, width=16, modes_h=8, modes_w=8, n_layers=2)

u_t = torch.randn(B, 1, H, W) * 0.1
covariate_stack = torch.rand(B, N_STATIC, H, W)  # placeholder covariates for the operator's input channels

u_next = model(u_t, covariate_stack)
assert u_next.shape == (B, 1, H, W), f"operator output shape wrong: {u_next.shape}"
print(f"[Smoke 1] operator forward pass OK, output shape {tuple(u_next.shape)}")

covariates = {
    "ndvi_f1": torch.rand(B, H, W) * 1.0 - 0.1,
    "ndvi_anomaly": torch.randn(B, H, W) * 0.05,
    "forest_frac": torch.rand(B, H, W),
    "dryness": torch.randn(B, H, W) * 0.5,
    "slope": torch.rand(B, H, W) * 30.0,
    "dist_roads": torch.rand(B, H, W) * 50.0,
    "grad_e_x": torch.randn(B, H, W) * 0.01,
    "grad_e_y": torch.randn(B, H, W) * 0.01,
}

residual, terms = model.pde_residual(u_t, u_next, dt_months=1.0, covariates=covariates,
                                      lat_rad_1d=lat_rad, dlon_rad=dlon_rad, dlat_rad=dlat_rad)
assert residual.shape == (B, H, W), f"residual shape wrong: {residual.shape}"
print(f"[Smoke 2] PDE residual computed OK, shape {tuple(residual.shape)}, "
      f"mean={residual.mean().item():.4e}, std={residual.std().item():.4e}")

# physical sanity checks on the physics-head outputs themselves
D, R_term = terms["D"], terms["reaction"]
assert (D > 0).all(), "D must be strictly positive by softplus construction -- FAILED"
assert (R_term >= 0).all() or (R_term <= 0.25 * D.new_ones(1)).all(), "reaction term sanity check"
print(f"[Smoke 3] D(x,y,t) strictly positive: min={D.min().item():.4e}, max={D.max().item():.4e} -- OK")
print(f"[Smoke 4] reaction term range: min={R_term.min().item():.4e}, max={R_term.max().item():.4e} "
      f"(theoretical max is rho_max/4)")

# gradient flow check -- every parameter in every submodule must receive a
# nonzero gradient from the combined loss, or something is disconnected
loss = residual.pow(2).mean() + u_next.pow(2).mean()
loss.backward()

missing_grad = []
for name, p in model.named_parameters():
    if p.grad is None or p.grad.abs().sum().item() == 0.0:
        missing_grad.append(name)

if missing_grad:
    print(f"[Smoke 5] FAILED -- {len(missing_grad)} parameters got no gradient: {missing_grad[:10]}")
    raise AssertionError("Some parameters are disconnected from the loss graph")
else:
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Smoke 5] gradient flow OK -- all {sum(1 for _ in model.parameters())} parameter "
          f"tensors ({n_params:,} total scalars) received a nonzero gradient")

print("\nAll smoke tests passed. Architecture wiring is sound on synthetic data.")
print("NOT yet verified: behavior on real gridded covariates, real training convergence,")
print("or the per-month dataset construction (NDVI/FLDAS monthly stacks don't exist on")
print("disk yet -- see the data-location recon: both need to be built from raw source")
print("files before real training can start).")
