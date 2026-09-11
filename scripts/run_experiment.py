#!/usr/bin/env python3
"""Command-line entry point for the tiny unlearning experiment."""

from __future__ import annotations

import argparse
import json

from tiny_unlearning import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hessian-free unlearning on a single-head transformer"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--training-steps", type=int, default=250)
    parser.add_argument("--cg-max-iter", type=int, default=30)
    parser.add_argument("--damping", type=float, default=2e-2)
    parser.add_argument("--unlearn-parameters", choices=["head", "all"], default="head")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        training_steps=args.training_steps,
        cg_max_iter=args.cg_max_iter,
        damping=args.damping,
        unlearn_parameters=args.unlearn_parameters,
    )
    metrics = run_experiment(config, output_path="artifacts/metrics.json")
    if not args.verbose:
        metrics["cg"]["cg_residual_norms"] = [
            metrics["cg"]["cg_residual_norms"][0],
            metrics["cg"]["cg_residual_norms"][-1],
        ]
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
