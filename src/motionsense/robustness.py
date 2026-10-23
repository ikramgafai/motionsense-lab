"""Test-time Gaussian noise robustness for already-trained runs.

No model is retrained or re-selected here; this only re-evaluates the final
models from ``run_grid`` on perturbed copies of the (standardised) test set.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from . import data as D
from .experiment import prepare
from .models import SmallCNN, flatten
from .noise import add_noise, channel_mask
from .train import predict
from .utils import load_json, save_json


def _load_predictor(cfg: dict, run_dir: Path, rec: dict):
    if rec["model"] == "cnn":
        m = cfg["cnn"]
        model = SmallCNN(rec["n_channels"], widths=tuple(m["widths"]),
                         kernels=tuple(m["kernels"]), dropout=m["dropout"])
        model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
        model.eval()
        return lambda X: predict(model, X)[0]
    w = np.load(run_dir / "logreg.npz")
    coef, intercept, classes = w["coef"], w["intercept"], w["classes"]
    return lambda X: classes[np.argmax(flatten(X) @ coef.T + intercept, axis=1)]


def run_robustness(cfg: dict) -> list[dict]:
    torch.set_num_threads(cfg["training"]["train_threads"])
    rcfg = cfg["robustness"]
    results_dir = Path(cfg["results_dir"])
    train, test = D.load(Path(cfg["data_dir"]))
    rows = []
    for condition in cfg["conditions"]:
        p = prepare(cfg, train, test, condition)
        for run_dir in sorted((results_dir / "runs").glob(f"*_{condition}_s*")):
            rec = load_json(run_dir / "metrics.json")
            if rec["condition"] != condition:
                continue  # "*_gyro_s*" also matches acc_gyro / acc_raw3_gyro runs
            pred_fn = _load_predictor(cfg, run_dir, rec)
            for target in rcfg["targets"]:
                mask = channel_mask(condition, target)
                for sigma in rcfg["sigmas"]:
                    f1s = []
                    for rep in range(rcfg["repeats"] if sigma > 0 else 1):
                        rng = np.random.default_rng([rcfg["seed"], rec["seed"], rep])
                        Xn = add_noise(p["X_te"], sigma, mask, rng) if sigma > 0 else p["X_te"]
                        f1s.append(f1_score(p["y_te"], pred_fn(Xn), average="macro"))
                    rows.append({"model": rec["model"], "condition": condition,
                                 "seed": rec["seed"], "target": target, "sigma": sigma,
                                 "noised_channels": int(mask.sum()),
                                 "macro_f1": float(np.mean(f1s)), "repeats": len(f1s)})
            print(f"[robustness] {rec['run_id']} done")

    with open(results_dir / "robustness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_json(rows, results_dir / "robustness.json")
    return rows
