"""
Loss terms and adaptive weighting for the CDR-PINN, per
CDR_PINN_Final_Design_STEP_D.md Sections 3.2 and 4.

Four loss groups only (data / pde / bc / ic) -- the PDE residual is ALWAYS a
single combined quantity (diffusion+advection+reaction assembled in
model.CDRPINN.pde_residual), never three separately-weighted per-mechanism
terms; that design choice was made explicitly to match the PINO paper's own
Eq. 4 loss formula and avoid unjustified manual sub-term tuning.
"""
import torch
import torch.nn.functional as F

from spectral_ops import neumann_periodic_extend, neumann_periodic_crop, spherical_gradient, EARTH_RADIUS_KM


def lse_pool(x: torch.Tensor, dim: int, tau: float = 5.0) -> torch.Tensor:
    """Smooth-max (log-sum-exp) pooling over `dim` -- CDR_PINN_Final_Design_STEP_D.md
    Section 3.2: approaches true max as tau->inf, matches the semantics of a
    whole-record 'did this pixel ever burn' label better than a mean would."""
    n = x.shape[dim]
    return (1.0 / tau) * (torch.logsumexp(tau * x, dim=dim) - torch.log(torch.tensor(float(n), device=x.device)))


def data_loss_monthly(pred_u, fire_indicator, valid_mask, pos_weight=None):
    """BCE between sigma(u_t) and the sparse monthly fire indicator, masked to
    valid (non-NaN, in-India) pixels only.

    pos_weight matters a great deal here: the real monthly fire-positive rate is
    only ~2.3% (measured directly from the data). An unweighted BCE's gradient is
    dominated by the 97.7% easy negatives, and was empirically observed (first
    ablation run, diffusion_only config) to let the model collapse onto the
    trivial constant-field solution that satisfies a diffusion-only PDE residual
    almost perfectly (residual -> ~1e-7) while carrying zero discriminative
    information (BCE loss stuck at ln(2), i.e. sigma(u)=0.5 everywhere, held-out
    ROC-AUC=0.53, barely above chance). This is not a hypothetical failure mode --
    it is what the first real run actually did. pos_weight (inverse-frequency,
    matching this project's own RF `class_weight='balanced'` convention in Step 7)
    is the fix: it up-weights the rare positive class's contribution to the
    gradient enough to compete with the physics loss's pull toward the trivial
    solution."""
    s_logit = pred_u  # BCEWithLogitsLoss expects raw logits, not post-sigmoid
    if pos_weight is not None:
        pw = torch.tensor(pos_weight, device=pred_u.device, dtype=pred_u.dtype)
        loss = F.binary_cross_entropy_with_logits(
            s_logit[valid_mask], fire_indicator[valid_mask], pos_weight=pw, reduction="mean")
    else:
        loss = F.binary_cross_entropy_with_logits(s_logit[valid_mask], fire_indicator[valid_mask], reduction="mean")
    return loss


def data_loss_terminal(u_trajectory, fire_ever_frac, valid_mask, tau=5.0):
    """LSE-pooled sigma(u) over the time dimension, compared against the
    already-validated Step 6/7 fire_ever (fractional, post-resampling) label."""
    s = torch.sigmoid(u_trajectory)          # (T,H,W) or (B,T,H,W)
    pooled = lse_pool(s, dim=0 if s.dim() == 3 else 1, tau=tau)
    return F.binary_cross_entropy(pooled[valid_mask], fire_ever_frac[valid_mask], reduction="mean")


def pde_loss(residual, valid_mask):
    return residual[valid_mask].pow(2).mean()


def bc_loss(u_field, lat_rad_1d, dlon_rad, dlat_rad, boundary_mask, R=EARTH_RADIUS_KM):
    """Homogeneous Neumann: penalize the squared normal derivative at boundary
    pixels. `boundary_mask` marks the India/non-India edge pixels (a ring
    around the in-India valid region, not the rectangular grid edge)."""
    u_ext_lon = neumann_periodic_extend(u_field, axis=-1)
    u_ext = neumann_periodic_extend(u_ext_lon, axis=-2)
    n_h = u_ext.shape[-2]
    lat_ext = torch.cat([lat_rad_1d, lat_rad_1d.flip(0)[1:-1]])[:n_h].to(u_field.device)
    dudx_ext, dudy_ext = spherical_gradient(u_ext, lat_ext, dlon_rad, dlat_rad, R=R)
    H, W = u_field.shape[-2:]
    dudx = neumann_periodic_crop(neumann_periodic_crop(dudx_ext, -2, H), -1, W)
    dudy = neumann_periodic_crop(neumann_periodic_crop(dudy_ext, -2, H), -1, W)
    grad_mag_sq = dudx.pow(2) + dudy.pow(2)  # proxy for |du/dn|^2 at the boundary ring
    return grad_mag_sq[boundary_mask].mean() if boundary_mask.any() else grad_mag_sq.new_zeros(())


def ic_loss(u_at_t0):
    return u_at_t0.pow(2).mean()


class AdaptiveLossBalancer:
    """Gradient-norm-balanced loss weighting (Wang, Teng & Perdikaris 2021),
    already cited in this project's own Step 8 methodology -- reused here
    rather than fixed hand-picked weights. Rescales each term's weight every
    `update_every` steps so the terms' gradient norms (w.r.t. shared model
    parameters) match on average."""

    def __init__(self, names, update_every=1, ema=0.9):
        self.names = list(names)
        self.weights = {n: 1.0 for n in self.names}
        self.update_every = update_every
        self.ema = ema
        self._step = 0

    def combine(self, losses: dict, model_params):
        total = sum(self.weights[n] * losses[n] for n in self.names)
        self._step += 1
        if self._step % self.update_every == 0:
            grad_norms = {}
            for n in self.names:
                if losses[n].requires_grad:
                    grads = torch.autograd.grad(losses[n], model_params, retain_graph=True, allow_unused=True)
                    # .abs() before squaring: some params (SpectralConv2d.weight) are
                    # complex-valued, and autograd.grad returns a COMPLEX gradient for
                    # those (the standard Wirtinger convention) -- squaring a complex
                    # tensor directly (g.pow(2)) stays complex and silently corrupts
                    # every downstream weight into a complex Python number a few steps
                    # later. .abs() gives the correct real magnitude for both real and
                    # complex tensors.
                    sq = sum((g.abs().pow(2).sum() for g in grads if g is not None))
                    grad_norms[n] = (sq.sqrt() + 1e-8).item()
                else:
                    grad_norms[n] = 1e-8
            mean_norm = sum(grad_norms.values()) / len(grad_norms)
            for n in self.names:
                new_w = mean_norm / (grad_norms[n] + 1e-8)
                self.weights[n] = self.ema * self.weights[n] + (1 - self.ema) * new_w
        return total, dict(self.weights)
