"""CPU inference latency for a single 2.56 s window (batch size 1).

What is measured
  * ``model_*``: the model's forward pass / ``predict`` on one already
    standardised window (CNN: torch forward; LogReg: sklearn ``predict``).
  * ``e2e_1t``: raw window -> channel standardisation -> model -> argmax label.
  * ``numpy_1t`` (LogReg only): the bare matmul + argmax the model needs,
    to separate sklearn's per-call validation overhead from arithmetic.

What is NOT measured: the 2.56 s needed to *observe* a window, sensor I/O,
or OS scheduling in a real app. In a streaming system the decision delay is
dominated by observation (2.56 s per window, or the 1.28 s hop with the
dataset's 50% overlap) plus the inference time measured here.

Protocol: run after all training (no competing load), each configuration is
timed in several rounds with the run order shuffled per round; samples from
all rounds are pooled before taking the median and p95.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from . import data as D
from .models import SmallCNN
from .utils import environment_info, load_json, save_json


def _stats(samples_ns: list[int]) -> dict:
    ms = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean()), "n_samples": int(len(ms))}


def time_fn(fn, warmup: int, iters: int) -> list[int]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return samples


def _cnn_fns(cfg: dict, run_dir: Path, rec: dict, raw: np.ndarray) -> dict:
    m = cfg["cnn"]
    model = SmallCNN(rec["n_channels"], widths=tuple(m["widths"]), kernels=tuple(m["kernels"]),
                     dropout=m["dropout"])
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    model.eval()
    mean = np.asarray(rec["scaler"]["mean"], dtype=np.float32)[:, None]
    std = np.asarray(rec["scaler"]["std"], dtype=np.float32)[:, None]
    x = torch.from_numpy(((raw - mean) / (std + 1e-8))[None].astype(np.float32))

    @torch.inference_mode()
    def model_only():
        model(x)

    @torch.inference_mode()
    def e2e():
        z = torch.from_numpy(((raw - mean) / (std + 1e-8))[None])
        return int(model(z).argmax(1))

    fns = {f"model_{t}t": (model_only, t) for t in cfg["benchmark"]["torch_threads"]}
    fns["e2e_1t"] = (e2e, 1)
    return fns


def _logreg_fns(run_dir: Path, rec: dict, raw: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression

    w = np.load(run_dir / "logreg.npz")
    clf = LogisticRegression()
    clf.coef_, clf.intercept_, clf.classes_ = w["coef"], w["intercept"], w["classes"]
    mean = np.asarray(rec["scaler"]["mean"], dtype=np.float32)[:, None]
    std = np.asarray(rec["scaler"]["std"], dtype=np.float32)[:, None]
    x = ((raw - mean) / (std + 1e-8)).reshape(1, -1)
    W, b = w["coef"], w["intercept"]
    return {
        "model_1t": (lambda: clf.predict(x), 1),
        "numpy_1t": (lambda: np.argmax(x @ W.T + b, axis=1), 1),
        "e2e_1t": (lambda: clf.predict(((raw - mean) / (std + 1e-8)).reshape(1, -1)), 1),
    }


def run_benchmarks(cfg: dict) -> None:
    from .experiment import write_tables

    b = cfg["benchmark"]
    results_dir = Path(cfg["results_dir"])
    run_dirs = sorted(p.parent for p in (results_dir / "runs").glob("*/metrics.json"))
    _, test = D.load(Path(cfg["data_dir"]))

    jobs = []  # (run_dir, key, fn, threads)
    recs = {}
    for rd in run_dirs:
        rec = load_json(rd / "metrics.json")
        recs[rd] = rec
        raw = np.ascontiguousarray(D.select_channels(test.X[:1], rec["condition"])[0])
        fns = _cnn_fns(cfg, rd, rec, raw) if rec["model"] == "cnn" else _logreg_fns(rd, rec, raw)
        jobs += [(rd, k, fn, t) for k, (fn, t) in fns.items()]

    samples: dict[tuple, list[int]] = {(rd, k): [] for rd, k, _, _ in jobs}
    rng = np.random.default_rng(b.get("seed", 0))
    prev_threads = torch.get_num_threads()
    for r in range(b["rounds"]):
        for i in rng.permutation(len(jobs)):
            rd, k, fn, t = jobs[i]
            torch.set_num_threads(t)
            with threadpool_limits(limits=t):
                samples[(rd, k)] += time_fn(fn, b["warmup"], b["iters"])
        print(f"[benchmark] round {r + 1}/{b['rounds']} done")
    torch.set_num_threads(prev_threads)

    for rd, rec in recs.items():
        rec["latency"] = {k: _stats(s) for (d, k), s in samples.items() if d == rd}
        rec["latency"]["protocol"] = {"warmup": b["warmup"], "iters_per_round": b["iters"],
                                      "rounds": b["rounds"], "batch_size": 1}
        save_json(rec, rd / "metrics.json")
    save_json({"benchmark_config": b, "environment": environment_info()},
              results_dir / "benchmark_environment.json")
    write_tables(list(recs.values()), results_dir)
