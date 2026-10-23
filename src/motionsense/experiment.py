"""Run the sensor-ablation grid: {LogReg, CNN} x sensor conditions x seeds.

Protocol (see RESEARCH_REPORT.md):
  * Official UCI train/test split is subject-disjoint and kept as is.
  * Validation = a seeded subset of the 21 training subjects (window-level
    disjoint by construction, since every window of a subject goes one way).
  * The channel scaler is fitted on the training partition only.
  * Model selection (LogReg C, CNN early-stopping epoch) uses validation only.
  * The test set is touched once per final model, after selection.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from . import data as D
from .evaluate import classification_metrics, per_subject_metrics
from .models import flatten, logreg_param_count, make_logreg, torch_param_count
from .preprocess import ChannelScaler
from .train import predict, train_cnn
from .utils import environment_info, save_json, set_seed


def prepare(cfg: dict, train: D.HARSplit, test: D.HARSplit, condition: str) -> dict:
    """Select channels, split train/val by subject, fit scaler on train only."""
    tr_mask, val_mask = D.subject_split(train.subjects, cfg["split"]["n_val_subjects"],
                                        cfg["split"]["seed"])
    Xtr = D.select_channels(train.X[tr_mask], condition)
    Xval = D.select_channels(train.X[val_mask], condition)
    Xte = D.select_channels(test.X, condition)
    scaler = ChannelScaler().fit(Xtr)
    return {
        "X_tr": scaler.transform(Xtr), "y_tr": train.y[tr_mask], "s_tr": train.subjects[tr_mask],
        "X_val": scaler.transform(Xval), "y_val": train.y[val_mask],
        "s_val": train.subjects[val_mask],
        "X_te": scaler.transform(Xte), "y_te": test.y, "s_te": test.subjects,
        "scaler": scaler,
    }


def _split_record(p: dict) -> dict:
    return {"train_subjects": np.unique(p["s_tr"]).tolist(),
            "val_subjects": np.unique(p["s_val"]).tolist(),
            "test_subjects": np.unique(p["s_te"]).tolist(),
            "n_train": len(p["y_tr"]), "n_val": len(p["y_val"]), "n_test": len(p["y_te"])}


def run_logreg(cfg: dict, p: dict, condition: str, seed: int, out: Path) -> dict:
    lr_cfg = cfg["logreg"]
    Xtr, Xval, Xte = flatten(p["X_tr"]), flatten(p["X_val"]), flatten(p["X_te"])
    sweep = []
    best = None
    t0 = time.perf_counter()
    for C in lr_cfg["C_grid"]:
        clf = make_logreg(C, lr_cfg["max_iter"], seed).fit(Xtr, p["y_tr"])
        f1 = f1_score(p["y_val"], clf.predict(Xval), average="macro")
        sweep.append({"C": C, "val_macro_f1": float(f1), "n_iter": int(clf.n_iter_.max())})
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    train_seconds = time.perf_counter() - t0
    best_f1, best_C, clf = best

    y_pred = clf.predict(Xte)
    np.savez(out / "logreg.npz", coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_)
    return {
        "model": "logreg", "condition": condition, "seed": seed,
        "n_channels": p["X_tr"].shape[1], "params": logreg_param_count(clf),
        "selected": {"C": best_C}, "val_sweep": sweep, "best_val_macro_f1": float(best_f1),
        "train_seconds": train_seconds,
        "test": classification_metrics(p["y_te"], y_pred),
        "test_per_subject": per_subject_metrics(p["y_te"], y_pred, p["s_te"]),
    }


def run_cnn(cfg: dict, p: dict, condition: str, seed: int, out: Path) -> dict:
    set_seed(seed)
    model, history, summary = train_cnn(p["X_tr"], p["y_tr"], p["X_val"], p["y_val"],
                                        cfg, seed, cfg["training"].get("log_every", 0))
    y_pred, _ = predict(model, p["X_te"])
    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    return {
        "model": "cnn", "condition": condition, "seed": seed,
        "n_channels": p["X_tr"].shape[1], "params": torch_param_count(model),
        **summary,
        "test": classification_metrics(p["y_te"], y_pred),
        "test_per_subject": per_subject_metrics(p["y_te"], y_pred, p["s_te"]),
    }


def run_grid(cfg: dict) -> list[dict]:
    torch.set_num_threads(cfg["training"]["train_threads"])
    results_dir = Path(cfg["results_dir"])
    runs_dir = results_dir / "runs"
    train, test = D.load(Path(cfg["data_dir"]))
    env = environment_info()
    save_json({"config": cfg, "environment": env}, results_dir / "experiment_config.json")

    records = []
    for condition in cfg["conditions"]:
        p = prepare(cfg, train, test, condition)
        split = _split_record(p)
        for model_name in cfg["models"]:
            seeds = cfg["logreg"]["seeds"] if model_name == "logreg" else cfg["seeds"]
            for seed in seeds:
                run_id = f"{model_name}_{condition}_s{seed}"
                out = runs_dir / run_id
                out.mkdir(parents=True, exist_ok=True)
                print(f"[{run_id}] channels={p['X_tr'].shape[1]} "
                      f"train/val/test={split['n_train']}/{split['n_val']}/{split['n_test']}")
                fn = run_logreg if model_name == "logreg" else run_cnn
                rec = fn(cfg, p, condition, seed, out)
                rec["run_id"] = run_id
                rec["split"] = split
                rec["scaler"] = p["scaler"].to_dict()
                save_json(rec, out / "metrics.json")
                print(f"    test acc={rec['test']['accuracy']:.4f} "
                      f"macroF1={rec['test']['macro_f1']:.4f} params={rec['params']} "
                      f"best_epoch={rec.get('best_epoch', '-')}")
                records.append(rec)

    write_tables(records, results_dir)
    return records


def write_tables(records: list[dict], results_dir: Path) -> None:
    """Per-run CSV and aggregated (mean/std over seeds) CSV + JSON.

    Latency columns are NaN until the separate ``benchmark`` stage has run.
    """
    rows = []
    for r in records:
        lat = r.get("latency", {})
        nan = float("nan")
        rows.append({
            "run_id": r["run_id"], "model": r["model"], "condition": r["condition"],
            "seed": r["seed"], "n_channels": r["n_channels"], "params": r["params"],
            "val_macro_f1": round(r["best_val_macro_f1"], 5),
            "test_accuracy": round(r["test"]["accuracy"], 5),
            "test_macro_f1": round(r["test"]["macro_f1"], 5),
            "latency_median_ms": round(lat.get("model_1t", {}).get("median_ms", nan), 5),
            "latency_p95_ms": round(lat.get("model_1t", {}).get("p95_ms", nan), 5),
            "e2e_latency_median_ms": round(lat.get("e2e_1t", {}).get("median_ms", nan), 5),
            "e2e_latency_p95_ms": round(lat.get("e2e_1t", {}).get("p95_ms", nan), 5),
            "best_epoch": r.get("best_epoch", ""), "train_seconds": round(r["train_seconds"], 2),
        })
    with open(results_dir / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    keys = sorted({(r["model"], r["condition"]) for r in rows},
                  key=lambda k: (k[0], k[1]))
    for model, cond in keys:
        g = [r for r in rows if r["model"] == model and r["condition"] == cond]
        entry = {"model": model, "condition": cond, "n_seeds": len(g),
                 "n_channels": g[0]["n_channels"], "params": g[0]["params"]}
        for k in ["val_macro_f1", "test_accuracy", "test_macro_f1",
                  "latency_median_ms", "latency_p95_ms",
                  "e2e_latency_median_ms", "e2e_latency_p95_ms"]:
            v = np.array([r[k] for r in g], dtype=float)
            entry[f"{k}_mean"] = round(float(v.mean()), 5)
            entry[f"{k}_std"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 5)
        summary.append(entry)
    with open(results_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    save_json(summary, results_dir / "summary.json")
