"""End-to-end run of a tiny config on the real data (skipped without data)."""

from pathlib import Path

import pytest

from motionsense.utils import load_config, load_json

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "uci_har_inertial.npz").exists(),
    reason="run `python -m motionsense download` first",
)


def test_smoke_pipeline(tmp_path):
    from motionsense.analysis import run_analysis
    from motionsense.benchmark import run_benchmarks
    from motionsense.experiment import run_grid
    from motionsense.plots import make_figures
    from motionsense.robustness import run_robustness

    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["data_dir"] = str(ROOT / "data")
    cfg["results_dir"] = str(tmp_path)
    cfg["conditions"] = ["acc", "acc_gyro"]
    cfg["training"]["log_every"] = 0
    cfg["benchmark"]["iters"] = 5

    records = run_grid(cfg)
    assert len(records) == 4
    for r in records:
        assert set(r["split"]["val_subjects"]).isdisjoint(r["split"]["train_subjects"])
        assert set(r["split"]["test_subjects"]).isdisjoint(r["split"]["train_subjects"])
        assert 0.0 <= r["test"]["macro_f1"] <= 1.0
    run_benchmarks(cfg)
    rec = load_json(tmp_path / "runs" / "cnn_acc_s0" / "metrics.json")
    assert rec["latency"]["model_1t"]["median_ms"] > 0
    run_robustness(cfg)
    run_analysis(cfg)
    make_figures(cfg)
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "figures" / "comparison_macro_f1.png").exists()
