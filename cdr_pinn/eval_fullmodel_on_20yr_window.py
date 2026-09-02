"""
The other half of the 22yr-vs-20yr ablation (see train_20yr_subset.py for the first
half). Loads the ALREADY-TRAINED full-period checkpoint (cdr_pinn_full_cdr_standard_
protocol.pt, trained on all 266 months) and re-scores it -- no retraining -- using
ONLY the 2001-2020 (240-month) slice of its own rollout trajectory, against the same
recomputed fire_ever_2001_2020 label and the same seed=42 test pixels train_20yr_
subset.py uses.

This isolates training-data volume as the single variable in the comparison:
  - This script:              model trained on 22 years, evaluated on a 20-year target
  - train_20yr_subset.py:     model trained on 20 years, evaluated on the same 20-year target
Same test pixels, same label, same LSE-pool (tau=5.0) aggregation -- the only
difference is whether the model saw 2000/2021/2022 during training.
"""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from model import CDRPINN
from losses import lse_pool
from preprocessing import build_masks_3way, load_tensors, covariate_stack, DATA_PATH, TARGET_H, TARGET_W

CKPT_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data"
FULL_CKPT_PATH = f"{CKPT_DIR}/cdr_pinn_full_cdr_standard_protocol.pt"
WIDTH, N_LAYERS, MODES = 32, 4, 16

WINDOW_START_IDX = 2    # 2001-01
WINDOW_END_IDX = 242    # exclusive, 2020-12 = idx 241


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tensors, fire_ever_binary_full_np, ndvi_f1_np, n_months = load_tensors(device)
    d = np.load(DATA_PATH)
    months = d["months"]

    fire_indicator_window = tensors["fire_indicator"][WINDOW_START_IDX:WINDOW_END_IDX]
    fire_ever_binary_window_np = fire_indicator_window.max(dim=0).values.cpu().numpy()
    print(f"2001-2020 fire-ever pixels: {int((fire_ever_binary_window_np > 0).sum()):,} "
          f"(full 22yr record: {int((fire_ever_binary_full_np > 0).sum()):,})")

    valid_np, boundary_np, train_np, val_np, test_np = build_masks_3way(ndvi_f1_np)
    print(f"Test pixels (identical seed=42 split): {test_np.sum()}")

    ckpt = torch.load(FULL_CKPT_PATH, map_location=device, weights_only=False)
    model = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded full-period checkpoint: {FULL_CKPT_PATH}")
    print(f"  (that run's own full-period test AUC was {ckpt['result']['test_roc_auc']:.4f}, for reference)")

    H, W = TARGET_H, TARGET_W
    with torch.no_grad():
        u = torch.zeros(1, 1, H, W, device=device)
        scores = []
        for ti in range(n_months - 1):
            u_next = model(u, covariate_stack(tensors, ti))
            scores.append(torch.sigmoid(u_next.squeeze(0).squeeze(0)))
            u = u_next

    # scores[ti] is the prediction for month (ti+1) of the full array. Restrict to
    # predictions whose target month index falls in [WINDOW_START_IDX, WINDOW_END_IDX).
    window_score_idx = [ti for ti in range(len(scores)) if WINDOW_START_IDX <= ti + 1 < WINDOW_END_IDX]
    print(f"Restricting the full model's own rollout to {len(window_score_idx)} months "
          f"(months {months[window_score_idx[0]+1]} .. {months[window_score_idx[-1]+1]})")
    assert len(window_score_idx) == 240

    window_scores = torch.stack([scores[i] for i in window_score_idx], dim=0)
    pooled_window = lse_pool(window_scores, dim=0, tau=5.0)

    def score(mask_np, y_true_np):
        m_np = valid_np & mask_np
        y_true = y_true_np[m_np]
        y_score = pooled_window.cpu().numpy()[m_np]
        ok = ~np.isnan(y_score) & ~np.isnan(y_true)
        y_true, y_score = y_true[ok], y_score[ok]
        return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)

    val_auc, val_ap = score(val_np, fire_ever_binary_window_np)
    test_auc, test_ap = score(test_np, fire_ever_binary_window_np)
    print(f"[full-22yr-model, restricted to 2001-2020 target] VAL:  ROC-AUC={val_auc:.4f}, AP={val_ap:.4f}")
    print(f"[full-22yr-model, restricted to 2001-2020 target] TEST: ROC-AUC={test_auc:.4f}, AP={test_ap:.4f}")

    result = {
        "description": "Full 22-year-trained CDR-PINN, evaluated ONLY on the 2001-2020 slice of its own "
                        "rollout trajectory, against a fire_ever label recomputed for that same window -- "
                        "no retraining. Companion to cdr_20yr_2001_2020_subset_result.json for the "
                        "22yr-vs-20yr training-data-volume ablation.",
        "source_checkpoint": FULL_CKPT_PATH,
        "n_fire_ever_pixels_window": int((fire_ever_binary_window_np > 0).sum()),
        "val_roc_auc": float(val_auc), "val_ap": float(val_ap),
        "test_roc_auc": float(test_auc), "test_ap": float(test_ap),
    }
    out_path = f"{CKPT_DIR}/cdr_pinn_full_model_on_20yr_window_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
