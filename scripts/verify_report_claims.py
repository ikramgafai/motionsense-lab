"""Check that numbers quoted in README.md / RESEARCH_REPORT.md match saved results.

Each claim recomputes a value from ``results/`` and asserts that its formatted
string occurs in the named document. This catches stale or mistyped numbers
after any edit to the documents or a re-run of the experiments.

Usage:  python scripts/verify_report_claims.py
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SUP = RES / "supplementary"
RAW = {"report": (ROOT / "RESEARCH_REPORT.md").read_text(encoding="utf-8"),
       "readme": (ROOT / "README.md").read_text(encoding="utf-8")}
# Line wrapping must not matter: compare with all whitespace runs collapsed.
DOCS = {k: " ".join(v.split()) for k, v in RAW.items()}
FAILED: list[str] = []
N_OK = 0


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def claim(doc: str, text: str, what: str) -> None:
    global N_OK
    if " ".join(text.split()) in DOCS[doc]:
        N_OK += 1
    else:
        FAILED.append(f"[{doc}] {what}: expected to find {text!r}")


def f3(x):
    return f"{x:.3f}"


def summary(path: Path) -> dict:
    return {(e["model"], e["condition"]): e for e in load(path)}


def runs(path: Path) -> list[dict]:
    return [load(p) for p in sorted((path / "runs").glob("*/metrics.json"))]


def comps(path: Path) -> dict:
    return {(c["model"], c["a"], c["b"]): c for c in load(path)}


def exact_wilcoxon(d) -> float:
    d = np.asarray(d, dtype=float)
    nz = d[d != 0]
    r = rankdata(np.abs(nz))
    w = r[nz > 0].sum()
    mu = r.sum() / 2
    dist = np.array([(r * np.array(s)).sum() for s in itertools.product([0, 1], repeat=len(nz))])
    return float(np.mean(np.abs(dist - mu) >= abs(w - mu) - 1e-12))


S = summary(RES / "summary.json")
R = runs(RES)
CMP = comps(RES / "comparisons.json")
CONDS = ["acc", "gyro", "acc_gyro", "acc_raw3", "acc_raw3_gyro"]

# --- Main results tables -------------------------------------------------
for c in CONDS:
    e = S[("cnn", c)]
    cnn_runs = [r for r in R if r["model"] == "cnn" and r["condition"] == c]
    t4m = np.mean([r["latency"]["model_4t"]["median_ms"] for r in cnn_runs])
    t4p = np.mean([r["latency"]["model_4t"]["p95_ms"] for r in cnn_runs])
    row = (f"| {e['params']:,} | {f3(e['val_macro_f1_mean'])} | "
           f"{f3(e['test_accuracy_mean'])} ± {f3(e['test_accuracy_std'])} | "
           f"**{f3(e['test_macro_f1_mean'])} ± {f3(e['test_macro_f1_std'])}** | "
           f"{f3(e['latency_median_ms_mean'])} / {f3(e['latency_p95_ms_mean'])} | "
           f"{f3(t4m)} / {f3(t4p)} | "
           f"{f3(e['e2e_latency_median_ms_mean'])} / {f3(e['e2e_latency_p95_ms_mean'])} |")
    claim("report", row, f"CNN table row {c}")
    claim("readme", f"{f3(e['test_macro_f1_mean'])} ± {f3(e['test_macro_f1_std'])}",
          f"README CNN macro-F1 {c}")

    e = S[("logreg", c)]
    lr = next(r for r in R if r["model"] == "logreg" and r["condition"] == c)
    npm, npp = lr["latency"]["numpy_1t"]["median_ms"], lr["latency"]["numpy_1t"]["p95_ms"]
    row = (f"| {e['params']:,} | {lr['selected']['C']:g} | {f3(e['val_macro_f1_mean'])} | "
           f"{f3(e['test_accuracy_mean'])} | **{f3(e['test_macro_f1_mean'])}** | "
           f"{f3(e['latency_median_ms_mean'])} / {f3(e['latency_p95_ms_mean'])} | "
           f"{npm:.4f} / {npp:.4f} |")
    claim("report", row, f"LogReg table row {c}")
    claim("readme", f"| {f3(e['test_macro_f1_mean'])} |", f"README LogReg macro-F1 {c}")

# --- Paired comparisons table --------------------------------------------
for (a, b) in [("acc_gyro", "acc"), ("acc_gyro", "gyro"), ("acc_raw3_gyro", "acc_raw3"),
               ("acc_raw3", "gyro"), ("acc", "acc_raw3")]:
    c = CMP[("cnn", a, b)]
    lo, hi = c["subject_diff_bootstrap_ci95"]
    sgn = lambda x: f"{x:+.4f}".replace("-", "−")  # noqa: E731
    row = (f"| {sgn(c['seed_macro_f1_diff_mean'])} ± {c['seed_macro_f1_diff_std']:.4f} "
           f"({c['seeds_a_better']}/{c['n_seeds']}) | {sgn(c['subject_diff_mean'])} "
           f"[{sgn(lo)}, {sgn(hi)}] | {c['subjects_a_better']}/{c['n_subjects']} | "
           f"{c['wilcoxon_p_two_sided']:.3f} | {c['wilcoxon_p_holm']:.3f} |")
    claim("report", row, f"comparison row {a} vs {b}")
    p = exact_wilcoxon(list(c["subject_diff"].values()))
    if abs(p - c["wilcoxon_p_two_sided"]) > 1e-9:
        FAILED.append(f"Wilcoxon p for {a} vs {b}: stored {c['wilcoxon_p_two_sided']} "
                      f"!= exact {p}")

fa = CMP[("cnn", "acc_gyro", "acc")]
claim("report", f"Wilcoxon p-value is {fa['wilcoxon_p_two_sided']:.2f} (Holm-adjusted "
      f"{fa['wilcoxon_p_holm']:.2f})", "headline p-values")
zeros = [k for k, v in fa["subject_diff"].items() if v == 0]
for z in zeros:
    claim("report", f"S{z} scores", "subject with zero fusion-vs-acc difference is disclosed")
fg = CMP[("cnn", "acc_gyro", "gyro")]
claim("report", f"Holm p = {fg['wilcoxon_p_holm']:.3f}", "fusion vs gyro Holm p")

for (a, b, txt) in [("acc_gyro", "acc", "acc+gyro vs acc +"),
                    ("acc_gyro", "gyro", "acc+gyro vs gyro +")]:
    c = CMP[("logreg", a, b)]
    claim("report", f"{txt}{c['seed_macro_f1_diff_mean']:.3f}", f"LogReg {a} vs {b}")

# --- Efficiency -------------------------------------------------------------
cnn_runs = [r for r in R if r["model"] == "cnn"]
meds = [r["latency"]["model_1t"]["median_ms"] for r in cnn_runs]
claim("report", f"({min(meds):.2f}–{max(meds):.2f} ms)", "per-run median latency spread")
cm = [S[("cnn", c)]["latency_median_ms_mean"] for c in CONDS]
claim("report", f"({min(cm):.2f}–{max(cm):.2f} ms means)", "between-condition latency means")
cp = [S[("cnn", c)]["latency_p95_ms_mean"] for c in CONDS]
claim("report", f"{min(cp):.2f}–{max(cp):.2f} ms p95", "CNN p95 range (report)")
claim("readme", f"{min(cp):.2f}–{max(cp):.2f} ms", "CNN p95 range (readme)")
d_params = S[("cnn", "acc_gyro")]["params"] - S[("cnn", "acc")]["params"]
claim("report", f"+{d_params} parameters", "fusion parameter overhead")
lr_sk = [S[("logreg", c)]["latency_median_ms_mean"] for c in CONDS]
lr_np = [next(r for r in R if r["model"] == "logreg" and r["condition"] == c)
         ["latency"]["numpy_1t"]["median_ms"] for c in CONDS]
ratios = [s / n for s, n in zip(lr_sk, lr_np)]
claim("report", f"(~{min(lr_sk):.2f}–{max(lr_sk):.2f} ms)", "sklearn predict range")
claim("report", f"about {min(ratios):.0f}–{max(ratios):.0f}× slower", "sklearn overhead ratio")

# --- Robustness table -------------------------------------------------------
rob = defaultdict(list)
with open(RES / "robustness.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rob[(r["model"], r["condition"], r["target"], float(r["sigma"]))].append(
            float(r["macro_f1"]))
names = {"acc": "acc", "gyro": "gyro", "acc_gyro": "acc+gyro", "acc_raw3": "total_acc",
         "acc_raw3_gyro": "total_acc+gyro"}
for target, conds in [("all", CONDS), ("gyro", ["acc_gyro", "acc_raw3_gyro"])]:
    for c in conds:
        vals = [f3(np.mean(rob[("cnn", c, target, s)])) for s in [0.0, 0.1, 0.25, 0.5, 1.0]]
        rows = [ln for ln in RAW["report"].splitlines() if ln.startswith("|")
                and all(v in ln.replace("**", "") for v in vals)]
        if not rows:
            FAILED.append(f"robustness row {target}/{c}: no table row with {vals}")
        else:
            N_OK += 1

# --- Failure analysis -------------------------------------------------------
sc = load(RES / "subject_confusions.json")["confusions"]


def total_cm(key):
    return np.sum([np.array(v) for v in sc[key].values()], axis=0)


for key, pct in [("cnn_acc", 21), ("cnn_acc_gyro", 19)]:
    m = total_cm(key)
    got = round(100 * m[3, 4] / m[3].sum())
    if got != pct:
        FAILED.append(f"{key} sit->stand: report says {pct} %, data {got} %")
    else:
        N_OK += 1
g = total_cm("cnn_gyro")
claim("report", f"laying recall\n   {g[5, 5] / g[5].sum():.2f}; {round(100 * g[5, 3] / g[5].sum())} %",
      "gyro-only laying recall / laying->sitting")
s = sc["cnn_acc_gyro"]
claim("report", f"{s['2'][3][4]} of {sum(s['2'][3])} sitting", "S2 sit->stand counts")
claim("report", f"{sc['cnn_acc']['9'][1][0]} of {sum(sc['cnn_acc']['9'][1])} upstairs",
      "S9 acc-only upstairs->walking")
claim("report", f"({s['9'][1][0]} of {sum(s['9'][1])} with fusion)", "S9 fusion upstairs->walking")
claim("report", f"({s['9'][4][3]}\n    of {sum(s['9'][4])})", "S9 standing->sitting")
claim("report", f"{s['10'][5][1]} of {sum(s['10'][5])} laying", "S10 laying->upstairs")
ps = {k: np.mean([r["test_per_subject"][k]["macro_f1"] for r in R
                  if r["model"] == "cnn" and r["condition"] == "acc_gyro"])
      for k in R[0]["test_per_subject"]}
claim("report", f"0.742\n  for S9 and 0.591 for S10", "S9/S10 macro-F1")
if not (abs(ps["9"] - 0.742) < 5e-4 and abs(ps["10"] - 0.591) < 5e-4):
    FAILED.append(f"S9/S10 values differ: {ps['9']:.4f}, {ps['10']:.4f}")
early = sorted(r["run_id"] for r in R if r["model"] == "cnn" and r["best_epoch"] <= 3)
claim("report", f"in {len(early)} of 25 main", "count of runs stopped by epoch 3")

# --- Supplementary ----------------------------------------------------------
S2 = summary(SUP / "val_split_seed1" / "summary.json")
for c in CONDS:
    e = S2[("cnn", c)]
    claim("report", f"{names[c]}: {f3(e['test_macro_f1_mean'])} ± {f3(e['test_macro_f1_std'])}",
          f"supplementary B {c}")
c2 = comps(SUP / "val_split_seed1" / "comparisons.json")[("cnn", "acc_gyro", "acc")]
claim("report", f"+{c2['seed_macro_f1_diff_mean']:.4f} ± {c2['seed_macro_f1_diff_std']:.4f}",
      "supplementary B fusion vs acc seed delta")
claim("report", f"Holm p = {c2['wilcoxon_p_holm']:.2f}", "supplementary B Holm p")
SA = summary(SUP / "logreg_extended_C" / "summary.json")
deltas = [abs(S[("logreg", c)]["test_macro_f1_mean"] - SA[("logreg", c)]["test_macro_f1_mean"])
          for c in CONDS]
claim("report", f"most {max(deltas):.3f}", "supplementary A max change")

# --- README-specific claims -------------------------------------------------
sgn = lambda x: f"{x:+.4f}".replace("-", "−")  # noqa: E731
for (a, b) in [("acc_gyro", "acc"), ("acc_gyro", "gyro"), ("acc_raw3_gyro", "acc_raw3"),
               ("acc_raw3", "gyro")]:
    c = CMP[("cnn", a, b)]
    lo, hi = c["subject_diff_bootstrap_ci95"]
    ties = sum(1 for v in c["subject_diff"].values() if v == 0)
    better = f"{c['subjects_a_better']}/{c['n_subjects']}" + (f" ({ties} tie)" if ties else "")
    claim("readme", f"| {sgn(c['seed_macro_f1_diff_mean'])} | {c['seeds_a_better']}/"
          f"{c['n_seeds']} | {better} | [{sgn(lo)}, {sgn(hi)}] | "
          f"{c['wilcoxon_p_two_sided']:.3f} | {c['wilcoxon_p_holm']:.3f} |",
          f"README comparison row {a} vs {b}")
for c in CONDS:
    e = S[("cnn", c)]
    claim("readme", f"| {e['params']:,} | {f3(e['latency_median_ms_mean'])} / "
          f"{f3(e['latency_p95_ms_mean'])} |", f"README CNN params/latency {c}")
    claim("readme", f"{f3(e['test_accuracy_mean'])} ± {f3(e['test_accuracy_std'])}",
          f"README CNN accuracy {c}")
claim("readme", f"+{fg['seed_macro_f1_diff_mean']:.3f} macro-F1", "README fusion vs gyro delta")
claim("readme", f"(p = {fg['wilcoxon_p_holm']:.3f})", "README fusion vs gyro Holm p")
claim("readme", f"(+{fa['seed_macro_f1_diff_mean']:.3f}; +{c2['seed_macro_f1_diff_mean']:.3f}",
      "README fusion vs acc deltas")
claim("readme", f"(Wilcoxon p = {fa['wilcoxon_p_two_sided']:.2f}, Holm {fa['wilcoxon_p_holm']:.2f})",
      "README fusion vs acc p")
pct = 100 * d_params / S[("cnn", "acc")]["params"]
claim("readme", f"{d_params} parameters (+{pct:.1f} %)", "README parameter overhead")
g1 = np.mean(rob[("cnn", "acc_gyro", "gyro", 1.0)])
claim("readme", f"({f3(g1)} vs {f3(S[('cnn', 'acc')]['test_macro_f1_mean'])})",
      "README gyro-noise crossover")
claim("readme", f"from\n{f3(S[('cnn', 'acc_gyro')]['test_macro_f1_mean'])} to {f3(g1)} at σ = 1",
      "README gyro-noise degradation")
claim("readme", f"({min(meds):.2f}–{max(meds):.2f} ms medians)", "README latency spread")
sds = [S[("cnn", c)]["test_macro_f1_std"] for c in ("acc", "acc_gyro")]
claim("readme", f"({min(sds):.3f}–{max(sds):.3f})", "README seed SD range")
claim("readme", f"Holm p = {c2['wilcoxon_p_holm']:.2f}", "README supplementary B Holm p")
vals = sorted(ps.values())
claim("readme", f"ranges from {vals[0]:.2f} to {vals[-1]:.2f}", "README per-subject range")
claim("readme", f"≥ {vals[2]:.3f} for\n  7 of 9 test subjects but {ps['9']:.3f} for S9 and "
      f"{ps['10']:.3f} for S10", "README per-subject failures")
claim("readme", f"{s['10'][5][1]} of {sum(s['10'][5])} laying", "README S10")
claim("readme", f"{s['2'][3][4]} of {sum(s['2'][3])} sitting", "README S2")
claim("readme", f"epoch 1–3 in {len(early)} of 25", "README early stopping count")

# --- Audit log --------------------------------------------------------------
audit = (RES / "audit_log.txt").read_text(encoding="utf-8")
n_ok = audit.count("  ok   ")
claim("report", f"{n_ok} checks", "audit check count")
if "0 failed checks" not in audit:
    FAILED.append("audit_log.txt does not report 0 failed checks")

print(f"{N_OK} claims verified, {len(FAILED)} failed")
for f in FAILED:
    print("  FAIL", f)
sys.exit(1 if FAILED else 0)
