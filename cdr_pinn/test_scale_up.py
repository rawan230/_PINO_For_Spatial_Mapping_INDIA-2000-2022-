"""
One targeted test: is the Track-A accuracy gap to RF/MaxEnt explained by
under-parameterization/under-training, not the physics formulation itself?
Reuses run_validation_tracks.train_and_eval verbatim (same verified code, same
LSE-pool-consistent evaluation) with a larger width and longer epoch budget,
on the identical Track-A-style split, seed, and full_cdr term configuration.
"""
import json
import torch
import run_validation_tracks as rvt

rvt.WIDTH = 64      # was 32 -- 4x spectral-conv parameter count
rvt.N_LAYERS = 4
rvt.MODES = 16
EPOCHS = 150         # was 80

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | scale-up test: width={rvt.WIDTH}, epochs={EPOCHS}")
ctx = rvt.load_data(device)

import numpy as np
rng = np.random.RandomState(rvt.SEED)
valid_idx = np.argwhere(ctx["valid_np"])
n_test = int(len(valid_idx) * 0.2)
perm = rng.permutation(len(valid_idx))
test_idx, train_idx = valid_idx[perm[:n_test]], valid_idx[perm[n_test:]]
train_mask = np.zeros_like(ctx["valid_np"]); train_mask[train_idx[:, 0], train_idx[:, 1]] = True
test_mask = np.zeros_like(ctx["valid_np"]); test_mask[test_idx[:, 0], test_idx[:, 1]] = True

result = rvt.train_and_eval(ctx, device, train_mask, test_mask, tag="scaleup_w64_e150",
                             epochs=EPOCHS, use_physics=True)

if device == "cuda":
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e6:.1f} MB")

out_path = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_scaleup_result.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_path}")
print(f"\nComparison: width=32/80ep AUC=0.9406 (original) vs width=64/{EPOCHS}ep AUC={result['roc_auc']:.4f}")
