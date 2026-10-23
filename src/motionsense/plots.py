"""Static figures for the report. Everything is read from results/*.json|csv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .data import ACTIVITIES  # noqa: E402
from .utils import load_json  # noqa: E402

# Categorical slots (fixed order) and a one-hue sequential ramp.
MODEL_COLORS = {"logreg": "#2a78d6", "cnn": "#eb6834"}
COND_COLORS = {"acc": "#2a78d6", "gyro": "#eb6834", "acc_gyro": "#1baf7a"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#6da7ec",
                                                     "#256abf", "#0d366b"])
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_LABEL = {"logreg": "LogReg (flattened)", "cnn": "1D CNN"}
COND_LABEL = {"acc": "acc\n(6 ch)", "gyro": "gyro\n(3 ch)", "acc_gyro": "acc+gyro\n(9 ch)",
              "acc_raw3": "total_acc\n(3 ch)", "acc_raw3_gyro": "total_acc+gyro\n(6 ch)"}
SHORT_ACT = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _runs(results_dir: Path) -> list[dict]:
    return [load_json(p) for p in sorted((results_dir / "runs").glob("*/metrics.json"))]


def comparison_chart(runs, conditions, models, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    x = np.arange(len(conditions))
    width = 0.38
    for j, model in enumerate(models):
        means, stds = [], []
        for i, c in enumerate(conditions):
            v = np.array([r["test"]["macro_f1"] for r in runs
                          if r["model"] == model and r["condition"] == c])
            means.append(v.mean())
            stds.append(v.std(ddof=1) if len(v) > 1 else 0)
            xs = x[i] + (j - 0.5) * width
            ax.scatter(np.full(len(v), xs), v, s=10, color=TEXT, zorder=4, linewidths=0)
        xs = x + (j - 0.5) * width
        ax.bar(xs, means, width - 0.04, color=MODEL_COLORS[model], label=MODEL_LABEL[model],
               zorder=2, edgecolor="white", linewidth=1)
        ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=TEXT, elinewidth=1, capsize=3,
                    zorder=3)
        for xi, m, sd in zip(xs, means, stds):
            ax.text(xi, m + sd + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=7,
                    color=TEXT)
    ax.set_xticks(x, [COND_LABEL[c] for c in conditions])
    lo = min(0.3, ax.get_ylim()[0])
    ax.set_ylim(lo, 1.02)
    ax.set_ylabel("Test macro-F1 (9 unseen subjects)", color=TEXT, fontsize=9)
    ax.set_title("Sensor ablation, test macro-F1 on 9 unseen subjects. CNN: mean ± 1 SD over "
                 "5 seeds (dots = seeds). LogReg: one deterministic run.",
                 fontsize=9, color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion_grid(runs, model, conditions, path: Path):
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.0 * len(conditions), 4.0), dpi=150)
    for ax, c in zip(np.atleast_1d(axes), conditions):
        cms = [np.array(r["test"]["confusion"]) for r in runs
               if r["model"] == model and r["condition"] == c]
        cm = np.sum(cms, axis=0).astype(float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(ACTIVITIES)):
            for j in range(len(ACTIVITIES)):
                v = cmn[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.55 else TEXT)
        ax.set_xticks(range(6), SHORT_ACT, fontsize=8)
        ax.set_yticks(range(6), SHORT_ACT, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8, color=MUTED)
        ax.set_ylabel("True", fontsize=8, color=MUTED)
        ax.set_title(f"{COND_LABEL[c].replace(chr(10), ' ')}  (n_seeds={len(cms)})",
                     fontsize=9, color=TEXT)
    fig.suptitle(f"{MODEL_LABEL[model]} — test confusion, row-normalised, summed over runs",
                 fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def efficiency_chart(summary, path: Path):
    """One row per (model, condition): latency median..p95 range and parameter count."""
    order = ["acc", "gyro", "acc_gyro", "acc_raw3", "acc_raw3_gyro"]
    rows = sorted(summary, key=lambda e: (list(MODEL_COLORS).index(e["model"]),
                                         order.index(e["condition"])))
    rows = rows[::-1]  # first row at the top
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(rows) + 1.4), dpi=150,
                             sharey=True, gridspec_kw={"width_ratios": [1.4, 1]})
    for yi, e in zip(y, rows):
        col = MODEL_COLORS[e["model"]]
        med, p95 = e["latency_median_ms_mean"], e["latency_p95_ms_mean"]
        axes[0].plot([med, p95], [yi, yi], color=col, linewidth=2, solid_capstyle="round")
        axes[0].scatter([med], [yi], s=40, color=col, edgecolor="white", linewidth=1.2, zorder=3)
        axes[0].text(p95, yi, f"  {med:.3f} / {p95:.3f}", va="center", fontsize=7, color=MUTED)
        axes[1].barh(yi, e["params"], height=0.6, color=col)
        axes[1].text(e["params"], yi, f" {e['params']:,}", va="center", fontsize=7,
                     color=MUTED)
    labels = [f"{ {'logreg': 'LogReg', 'cnn': 'CNN'}[e['model']]} · "
              f"{COND_LABEL[e['condition']].replace(chr(10), ' ')}" for e in rows]
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].set_xlim(0, max(e["latency_p95_ms_mean"] for e in rows) * 1.45)
    axes[0].set_xlabel("CPU latency per window, batch=1, 1 thread (ms): dot = median, "
                       "line end = p95", fontsize=8, color=TEXT)
    axes[1].set_xlim(0, max(e["params"] for e in rows) * 1.3)
    axes[1].set_xlabel("Trainable parameters", fontsize=8, color=TEXT)
    for ax in axes:
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.suptitle("Inference cost (model only, after warmup). For scale: one observation "
                 "window is 2560 ms.", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def robustness_chart(rows, model, path: Path):
    targets = ["all", "acc", "gyro"]
    titles = {"all": "noise on all channels", "acc": "noise on accelerometer channels",
              "gyro": "noise on gyroscope channels"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150, sharey=True)
    for ax, t in zip(axes, targets):
        for c, col in COND_COLORS.items():
            sub = [r for r in rows if r["model"] == model and r["condition"] == c
                   and r["target"] == t]
            if not sub:
                continue
            sigmas = sorted({r["sigma"] for r in sub})
            m = [np.mean([r["macro_f1"] for r in sub if r["sigma"] == s]) for s in sigmas]
            sd = [np.std([r["macro_f1"] for r in sub if r["sigma"] == s], ddof=1)
                  if sum(r["sigma"] == s for r in sub) > 1 else 0 for s in sigmas]
            ax.plot(sigmas, m, color=col, linewidth=2, marker="o", markersize=5,
                    markeredgecolor="white", label=COND_LABEL[c].replace("\n", " "))
            ax.fill_between(sigmas, np.array(m) - sd, np.array(m) + sd, color=col, alpha=0.15,
                            linewidth=0)
            ax.annotate(COND_LABEL[c].split("\n")[0], (sigmas[-1], m[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7, color=MUTED, va="center")
        ax.set_title(titles[t], fontsize=9, color=TEXT, loc="left")
        ax.set_xlabel("Noise σ (× training-partition channel SD)", fontsize=8, color=TEXT)
        _style(ax)
    axes[0].set_ylabel("Test macro-F1", fontsize=9, color=TEXT)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    n_seeds = len({r["seed"] for r in rows if r["model"] == model})
    spread = "mean ± SD over seeds" if n_seeds > 1 else "single run, mean over noise draws"
    fig.suptitle(f"{MODEL_LABEL[model]} — test-time Gaussian noise ({spread})",
                 fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def per_subject_chart(runs, model, conditions, path: Path):
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    subjects = sorted(runs[0]["test_per_subject"], key=int)
    x = np.arange(len(subjects))
    offs = np.linspace(-0.25, 0.25, len(conditions))
    for o, c in zip(offs, conditions):
        rs = [r for r in runs if r["model"] == model and r["condition"] == c]
        if not rs:
            continue
        m = [np.mean([r["test_per_subject"][s]["macro_f1"] for r in rs]) for s in subjects]
        ax.scatter(x + o, m, s=40, color=COND_COLORS[c], edgecolor="white", linewidth=1.2,
                   label=COND_LABEL[c].replace("\n", " "), zorder=3)
    ax.set_xticks(x, [f"S{s}" for s in subjects])
    ax.set_ylabel("Macro-F1 (mean over seeds)", fontsize=9, color=TEXT)
    ax.set_xlabel("Test subject", fontsize=9, color=TEXT)
    ax.set_title(f"{MODEL_LABEL[model]} — per-subject test macro-F1", fontsize=10, color=TEXT,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right",
              bbox_to_anchor=(1.0, 1.0))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_figures(cfg: dict) -> None:
    results_dir = Path(cfg["results_dir"])
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = _runs(results_dir)
    conditions = [c for c in cfg["conditions"] if c in COND_LABEL]
    main = [c for c in ["acc", "gyro", "acc_gyro"] if c in conditions]
    comparison_chart(runs, conditions, cfg["models"], fig_dir / "comparison_macro_f1.png")
    for model in cfg["models"]:
        confusion_grid(runs, model, main, fig_dir / f"confusion_{model}.png")
        per_subject_chart(runs, model, main, fig_dir / f"per_subject_{model}.png")
    efficiency_chart(load_json(results_dir / "summary.json"), fig_dir / "efficiency.png")
    rob = results_dir / "robustness.json"
    if rob.exists():
        rows = load_json(rob)
        for model in cfg["models"]:
            robustness_chart(rows, model, fig_dir / f"robustness_{model}.png")
    print(f"figures written to {fig_dir}")
