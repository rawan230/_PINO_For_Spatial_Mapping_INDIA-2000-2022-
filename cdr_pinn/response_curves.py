"""
Biswas-et-al.-style response-curve analysis (their Figs. 8/9) for CDR-PINN: sweep
one covariate across its realistic range while holding all others at their
domain-mean constant value, record the model's predicted probability at each sweep
point. Matches their own MaxEnt marginal-response methodology exactly (a synthetic
sweep over a representative "average" location, not a per-pixel map) -- inference
only, same already-trained full_cdr checkpoint, no retraining.
"""
import json
import numpy as np
import torch

from model import CDRPINN
from train import build_masks
from build_monthly_stacks import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, TARGET_H, TARGET_W

DATA_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_monthly_stacks.npz"
CKPT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_full_cdr.pt"
OUT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_response_curves.json"
N_SWEEP = 15
N_MONTHS_EVAL = 24  # one representative year's worth of monthly steps is enough for a mean-field sweep

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

d = np.load(DATA_PATH)
H, W = TARGET_H, TARGET_W
ndvi_f1_np = d["ndvi_f1"]
valid_np, boundary_np, train_np, test_np = build_masks(ndvi_f1_np)


def mean_of(arr, is_3d=False):
    a = arr[0] if is_3d else arr
    return float(np.nanmean(a[valid_np]))


MEANS = {
    "ndvi_f1": mean_of(ndvi_f1_np),
    "ndvi_anomaly": mean_of(d["ndvi_anomaly"], is_3d=True),
    "forest_frac": mean_of(d["forest_frac"]),
    "dryness": mean_of(d["dryness_proxy"], is_3d=True),
    "slope": mean_of(d["slope"]),
    "dist_roads": mean_of(d["dist_roads"]),
    "elevation": mean_of(d["elevation"]),
}
RANGES = {
    "elevation": (float(np.nanpercentile(d["elevation"][valid_np], 1)), float(np.nanpercentile(d["elevation"][valid_np], 99))),
    "ndvi_f1": (-0.1, 0.9),
    "slope": (0.0, float(np.nanpercentile(d["slope"][valid_np], 99))),
    "dist_roads": (0.0, float(np.nanpercentile(d["dist_roads"][valid_np], 99))),
    "dryness": (float(np.nanpercentile(d["dryness_proxy"][0][valid_np], 1)), float(np.nanpercentile(d["dryness_proxy"][0][valid_np], 99))),
    "forest_frac": (0.0, 1.0),
}

ckpt = torch.load(CKPT_PATH, map_location=device)
model = CDRPINN(n_static_channels=7, width=32, modes_h=16, modes_w=16, n_layers=4).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Checkpoint loaded.")


def constant_field(value):
    return torch.full((H, W), value, dtype=torch.float32, device=device)


def covariate_stack_const(overrides):
    fields = {k: constant_field(v) for k, v in MEANS.items()}
    fields.update(overrides)
    return torch.stack([
        fields["ndvi_f1"], fields["ndvi_anomaly"], fields["forest_frac"], fields["dryness"],
        fields["slope"], fields["dist_roads"], fields["elevation"],
    ], dim=0).unsqueeze(0)


def mean_field_prediction(overrides):
    with torch.no_grad():
        u = torch.zeros(1, 1, H, W, device=device)
        for _ in range(N_MONTHS_EVAL):
            u = model(u, covariate_stack_const(overrides))
        s = torch.sigmoid(u).mean().item()  # mean predicted probability over the (constant) field
    return s


results = {}
for cov_name, (lo, hi) in RANGES.items():
    sweep_vals = np.linspace(lo, hi, N_SWEEP)
    probs = []
    for v in sweep_vals:
        p = mean_field_prediction({cov_name: constant_field(float(v))})
        probs.append(p)
    results[cov_name] = {"sweep_values": sweep_vals.tolist(), "predicted_probability": probs}
    print(f"[{cov_name}] range [{lo:.3g}, {hi:.3g}] -> prob range "
          f"[{min(probs):.4f}, {max(probs):.4f}] (delta={max(probs)-min(probs):.4f})")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_PATH}")

print("\nRanked by response-curve range (largest swing = strongest marginal effect):")
ranked = sorted(results.items(), key=lambda kv: -(max(kv[1]["predicted_probability"]) - min(kv[1]["predicted_probability"])))
for name, r in ranked:
    delta = max(r["predicted_probability"]) - min(r["predicted_probability"])
    print(f"  {name}: delta={delta:.4f}")
