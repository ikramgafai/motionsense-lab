"""Independent audit of saved results: leakage, metric correctness, reproducibility.

Usage:
    python scripts/audit_results.py --config configs/default.yaml [--retrain cnn_acc_gyro_s1]

Every check re-derives its quantity from raw data or saved artifacts instead
of trusting the numbers written by the training code. Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from motionsense import data as D
from motionsense.experiment import prepare
from motionsense.robustness import _load_predictor
from motionsense.utils import load_config, load_json

FAIL = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--retrain", default=None, help="run_id to retrain and compare")
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = Path(cfg["results_dir"])
    train, test = D.load(Path(cfg["data_dir"]))
    off_train, off_test = set(train.subjects.tolist()), set(test.subjects.tolist())

    print("[1] official split")
    check(not off_train & off_test, "official train/test subjects are disjoint")
    for s in sorted(off_test):
        n_cls = len(np.unique(test.y[test.subjects == s]))
        check(n_cls == 6, f"test subject {s} has all 6 activities (per-subject macro-F1 valid)")

    runs = {p.parent.name: load_json(p) for p in sorted((results / "runs").glob("*/metrics.json"))}
    print(f"[2] per-run checks ({len(runs)} runs)")
    preps = {}
    for rid, r in runs.items():
        sp = r["split"]
        tr, va, te = set(sp["train_subjects"]), set(sp["val_subjects"]), set(sp["test_subjects"])
        check(not tr & va and not tr & te and not va & te, f"{rid}: train/val/test subjects disjoint")
        check(va <= off_train and te == off_test, f"{rid}: val drawn from official train only")
        check(sp["n_train"] + sp["n_val"] == len(train.y) and sp["n_test"] == len(test.y),
              f"{rid}: window counts add up")

        # Scaler must equal statistics of the TRAIN partition only.
        cond = r["condition"]
        if cond not in preps:
            preps[cond] = prepare(cfg, train, test, cond)
        mask = np.isin(train.subjects, sorted(tr))
        Xtr = D.select_channels(train.X[mask], cond)
        check(np.allclose(r["scaler"]["mean"], Xtr.mean(axis=(0, 2)), atol=1e-5) and
              np.allclose(r["scaler"]["std"], Xtr.std(axis=(0, 2)), atol=1e-5),
              f"{rid}: scaler == train-partition statistics")
        Xall = D.select_channels(train.X, cond)
        check(not np.allclose(r["scaler"]["std"], Xall.std(axis=(0, 2)), atol=1e-5),
              f"{rid}: scaler differs from train+val statistics (val not used)")

        cm = np.array(r["test"]["confusion"])
        check(cm.sum() == len(test.y), f"{rid}: confusion sums to n_test")
        check(abs(np.trace(cm) / cm.sum() - r["test"]["accuracy"]) < 1e-9,
              f"{rid}: accuracy recomputed from confusion")
        check(abs(macro_f1_from_confusion(cm) - r["test"]["macro_f1"]) < 1e-9,
              f"{rid}: macro-F1 recomputed from confusion")

        pred = _load_predictor(cfg, results / "runs" / rid, r)(preps[cond]["X_te"])
        f1 = f1_score(test.y, pred, average="macro")
        check(abs(f1 - r["test"]["macro_f1"]) < 1e-9, f"{rid}: reloaded model reproduces test F1")

        if r["model"] == "logreg":
            max_iter = cfg["logreg"]["max_iter"]
            check(all(s["n_iter"] < max_iter for s in r["val_sweep"]),
                  f"{rid}: lbfgs converged for every C (n_iter < {max_iter})")
            check(r["selected"]["C"] == max(r["val_sweep"], key=lambda s: s["val_macro_f1"])["C"],
                  f"{rid}: selected C = argmax of validation sweep")

        if r["model"] == "cnn":
            with open(results / "runs" / rid / "history.csv", encoding="utf-8") as f:
                h = list(csv.DictReader(f))
            vals = [float(x["val_macro_f1"]) for x in h]
            check(abs(max(vals) - r["best_val_macro_f1"]) < 1e-12 and
                  int(np.argmax(vals)) + 1 == r["best_epoch"],
                  f"{rid}: best epoch = argmax of validation curve")
            check(len(h) == min(cfg["training"]["max_epochs"],
                                r["best_epoch"] + cfg["training"]["patience"]),
                  f"{rid}: stopped exactly `patience` epochs after best (or at max)")

    print("[3] summary table recomputed from runs.csv")
    with open(results / "runs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for e in load_json(results / "summary.json"):
        v = [float(r["test_macro_f1"]) for r in rows
             if r["model"] == e["model"] and r["condition"] == e["condition"]]
        check(abs(np.mean(v) - e["test_macro_f1_mean"]) < 1e-4 and len(v) == e["n_seeds"],
              f"summary {e['model']}/{e['condition']}: mean over {len(v)} runs")

    if args.retrain:
        import torch

        from motionsense.experiment import run_cnn
        print(f"[4] determinism: retraining {args.retrain}")
        r = runs[args.retrain]
        torch.set_num_threads(cfg["training"]["train_threads"])
        out = Path(cfg["results_dir"]) / "_audit_retrain"
        out.mkdir(parents=True, exist_ok=True)
        new = run_cnn(cfg, preps[r["condition"]], r["condition"], r["seed"], out)
        check(new["best_epoch"] == r["best_epoch"], "retrain: same best epoch")
        check(abs(new["test"]["macro_f1"] - r["test"]["macro_f1"]) < 1e-12,
              f"retrain: identical test macro-F1 ({new['test']['macro_f1']:.6f})")

    print(f"\n{len(FAIL)} failed checks")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
