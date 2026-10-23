"""Command-line entry point: ``python -m motionsense <command> --config <yaml>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="motionsense")
    ap.add_argument("command", choices=["download", "train", "benchmark", "robustness",
                                        "analyze", "figures", "all"])
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.command in ("download", "all"):
        from .data import load
        train, test = load(Path(cfg["data_dir"]))
        print(f"train {train.X.shape} subjects={len(set(train.subjects))} | "
              f"test {test.X.shape} subjects={len(set(test.subjects))}")
    if args.command in ("train", "all"):
        from .experiment import run_grid
        run_grid(cfg)
    if args.command in ("benchmark", "all"):
        from .benchmark import run_benchmarks
        run_benchmarks(cfg)
    if args.command in ("robustness", "all") and cfg.get("robustness", {}).get("enabled"):
        from .robustness import run_robustness
        run_robustness(cfg)
    if args.command in ("analyze", "all"):
        from .analysis import run_analysis
        run_analysis(cfg)
    if args.command in ("figures", "all"):
        from .plots import make_figures
        make_figures(cfg)


if __name__ == "__main__":
    main()
