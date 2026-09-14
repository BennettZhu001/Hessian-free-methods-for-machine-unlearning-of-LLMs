#!/usr/bin/env python3
"""CLI for the distributed 1M-parameter milestone."""

from __future__ import annotations

import argparse
import json

from tiny_unlearning.distributed import cleanup_distributed, initialize_distributed
from tiny_unlearning.scalable_experiment import (
    ScalableExperimentConfig,
    load_scalable_config,
    run_scalable_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--output", default="artifacts/1m_2x5090_metrics.json")
    parser.add_argument(
        "--require-world-size",
        type=int,
        help="Fail instead of silently falling back to fewer processes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = (
        load_scalable_config(args.config) if args.config else ScalableExperimentConfig()
    )
    context = initialize_distributed(require_world_size=args.require_world_size)
    try:
        metrics = run_scalable_experiment(config, context, output_path=args.output)
        if context.is_primary and metrics is not None:
            print(json.dumps(metrics, indent=2))
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
