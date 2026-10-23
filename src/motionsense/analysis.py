"""Paired comparisons between sensor conditions.

The unit of generalisation is the *test subject* (n=9), not the window: windows
from one subject are strongly correlated (and overlap by 50%), so window-level
tests would overstate certainty. We therefore report

  * seed-level differences in overall test macro-F1 (same seeds in both arms),
  * per-subject differences (macro-F1 averaged over seeds), with a Wilcoxon
    signed-rank test and a subject-level bootstrap 95% CI of the mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .data import ACTIVITIES
from .utils import load_json, save_json

COMPARISONS = [
    ("acc_gyro", "acc", "fusion vs accelerometer (9 vs 6 ch)"),
    ("acc_gyro", "gyro", "fusion vs gyroscope (9 vs 3 ch)"),
    ("acc_raw3_gyro", "acc_raw3", "total_acc+gyro vs total_acc (6 vs 3 ch)"),
    ("acc_raw3", "gyro", "total_acc vs gyro (3 vs 3 ch)"),
    ("acc", "acc_raw3", "adding derived body_acc (6 vs 3 ch)"),
]


def _load_runs(results_dir: Path) -> list[dict]:
    return [load_json(p) for p in sorted((results_dir / "runs").glob("*/metrics.json"))]


def _bootstrap_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(runs: list[dict], model: str, a: str, b: str, n_boot: int = 10000) -> dict | None:
    ra = {r["seed"]: r for r in runs if r["model"] == model and r["condition"] == a}
    rb = {r["seed"]: r for r in runs if r["model"] == model and r["condition"] == b}
    seeds = sorted(set(ra) & set(rb))
    if not seeds:
        return None
    seed_diff = np.array([ra[s]["test"]["macro_f1"] - rb[s]["test"]["macro_f1"] for s in seeds])

    subjects = sorted(ra[seeds[0]]["test_per_subject"], key=int)

    def subj_mean(rs):
        return np.array([np.mean([rs[s]["test_per_subject"][k]["macro_f1"] for s in seeds])
                         for k in subjects])

    subj_diff = subj_mean(ra) - subj_mean(rb)
    try:
        p = float(wilcoxon(subj_diff).pvalue)
    except ValueError:  # all differences zero
        p = 1.0
    per_class = {c: float(np.mean([ra[s]["test"]["per_class_f1"][c] -
                                   rb[s]["test"]["per_class_f1"][c] for s in seeds]))
                 for c in ACTIVITIES}
    return {
        "model": model, "a": a, "b": b, "n_seeds": len(seeds),
        "seed_macro_f1_diff_mean": float(seed_diff.mean()),
        "seed_macro_f1_diff_std": float(seed_diff.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "seeds_a_better": int((seed_diff > 0).sum()),
        "n_subjects": len(subjects),
        "subject_diff": {k: float(d) for k, d in zip(subjects, subj_diff)},
        "subjects_a_better": int((subj_diff > 0).sum()),
        "subject_diff_mean": float(subj_diff.mean()),
        "subject_diff_bootstrap_ci95": _bootstrap_ci(subj_diff, n_boot, seed=0),
        "wilcoxon_p_two_sided": p,
        "per_class_f1_diff_mean": per_class,
    }


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj.tolist()


def run_analysis(cfg: dict) -> list[dict]:
    results_dir = Path(cfg["results_dir"])
    runs = _load_runs(results_dir)
    out = []
    for model in cfg["models"]:
        for a, b, label in COMPARISONS:
            res = compare(runs, model, a, b)
            if res is not None:
                res["label"] = label
                out.append(res)
                lo, hi = res["subject_diff_bootstrap_ci95"]
                print(f"{model:6s} {label:45s} seed_dF1={res['seed_macro_f1_diff_mean']:+.4f} "
                      f"subj_d={res['subject_diff_mean']:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
                      f"p={res['wilcoxon_p_two_sided']:.4f} "
                      f"subj a>b {res['subjects_a_better']}/{res['n_subjects']}")
    # Multiple-comparison correction within each model family.
    for model in cfg["models"]:
        fam = [r for r in out if r["model"] == model]
        for r, padj in zip(fam, holm([r["wilcoxon_p_two_sided"] for r in fam])):
            r["wilcoxon_p_holm"] = padj
    save_json(out, results_dir / "comparisons.json")
    if (results_dir / "runs").exists():
        subject_confusions(cfg)
    return out


def subject_confusions(cfg: dict, models=("cnn",), conditions=("acc", "gyro", "acc_gyro")
                       ) -> dict:
    """Per-test-subject confusion matrices (summed over seeds) from saved models."""
    import torch
    from sklearn.metrics import confusion_matrix

    from . import data as D
    from .experiment import prepare
    from .robustness import _load_predictor

    torch.set_num_threads(cfg["training"]["train_threads"])
    results_dir = Path(cfg["results_dir"])
    train, test = D.load(Path(cfg["data_dir"]))
    out = {}
    for cond in conditions:
        if cond not in cfg["conditions"]:
            continue
        p = prepare(cfg, train, test, cond)
        for model in models:
            if model not in cfg["models"]:
                continue
            cms = {}
            for rd in sorted((results_dir / "runs").glob(f"{model}_{cond}_s*")):
                rec = load_json(rd / "metrics.json")
                if rec["condition"] != cond:
                    continue
                pred = _load_predictor(cfg, rd, rec)(p["X_te"])
                for s in np.unique(p["s_te"]):
                    m = p["s_te"] == s
                    cm = confusion_matrix(p["y_te"][m], pred[m], labels=range(len(ACTIVITIES)))
                    cms[int(s)] = cms.get(int(s), 0) + cm
            out[f"{model}_{cond}"] = {s: cm.tolist() for s, cm in cms.items()}
    save_json({"activities": ACTIVITIES, "confusions": out},
              results_dir / "subject_confusions.json")
    return out
