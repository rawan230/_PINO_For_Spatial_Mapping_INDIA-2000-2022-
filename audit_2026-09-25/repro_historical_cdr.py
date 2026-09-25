"""
AUDIT 2026-09-24 -- pure numerical reproduction of the historical CDR-PINN results.

Runs the project's own, unmodified training code (train_standard_protocol.main and
run_validation_tracks' B1/B2/B3 functions) with ONLY the output directory redirected
to results/cdr_pino/reproduction/, so no historical checkpoint or JSON is overwritten.
No hyperparameter, split, seed, or loss is changed. Purpose: test whether the
historical headline numbers (Track A test AUC 0.9398, B1 0.7510+/-0.0182,
B2 0.6187+/-0.0680, B3 0.8960) are numerically reproducible from the saved stacks.

Environment: cdr_pinn_env (Python 3.11.15, torch 2.11.0+cu128) -- the only env on
this machine with CUDA torch; base anaconda3 has CPU-only torch 2.12.1.
"""
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = r"D:\FOREST FIRE MAPPING(INDIA)"
CDR_DIR = os.path.join(ROOT, "Physics_Informed_FireRisk_Model", "cdr_pinn")
HIST_DATA = os.path.join(ROOT, "Physics_Informed_FireRisk_Model", "CDR_PINN_Data")
OUT = os.path.join(ROOT, "results", "cdr_pino", "reproduction")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, CDR_DIR)
os.chdir(CDR_DIR)


def commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


META = {
    "audit_date": "2026-09-24",
    "code_commit_Physics_Informed_FireRisk_Model": commit(os.path.join(ROOT, "Physics_Informed_FireRisk_Model")),
    "data": os.path.join(HIST_DATA, "cdr_pinn_monthly_stacks.npz"),
    "python": sys.version.split()[0],
}


def track_a():
    import torch
    import train_standard_protocol as tsp
    META["torch"] = torch.__version__
    # historical validated weight-decay result is an INPUT here -- copy, don't recompute
    shutil.copy(os.path.join(HIST_DATA, "cdr_pinn_weight_decay_search.json"), OUT)
    tsp.CKPT_DIR = OUT
    t0 = time.time()
    tsp.main()
    print(f"[repro] Track A done in {time.time()-t0:.1f}s")


def tracks_b():
    import torch
    import run_validation_tracks as rvt
    device = "cuda"
    ctx = rvt.load_data(device)
    res = {}
    for name, fn in [("B1", rvt.run_track_b1), ("B2", rvt.run_track_b2), ("B3", rvt.run_track_b3)]:
        t0 = time.time()
        res[name] = fn(ctx, device)
        print(f"[repro] {name} done in {time.time()-t0:.1f}s")
        with open(os.path.join(OUT, "repro_validation_tracks_b1_b2_b3.json"), "w") as f:
            json.dump({"meta": META, "results": res}, f, indent=2)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("a", "all"):
        track_a()
    if which in ("b", "all"):
        tracks_b()
    with open(os.path.join(OUT, "repro_meta.json"), "w") as f:
        json.dump(META, f, indent=2)
