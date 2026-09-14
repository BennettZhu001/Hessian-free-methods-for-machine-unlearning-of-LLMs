#!/usr/bin/env python3
"""Compare matched-global-batch 1-GPU and 2-GPU result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("one_gpu", type=Path)
    parser.add_argument("two_gpu", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    one = json.loads(args.one_gpu.read_text())
    two = json.loads(args.two_gpu.read_text())
    one_size = one["hardware"]["world_size"]
    two_size = two["hardware"]["world_size"]
    if (one_size, two_size) != (1, 2):
        raise ValueError(f"expected world sizes (1, 2), got {(one_size, two_size)}")
    if one["model"]["total_parameters"] != two["model"]["total_parameters"]:
        raise ValueError("result files use different model sizes")
    one_batch = one["config"]["batch_size_per_gpu"]
    two_batch = two["config"]["batch_size_per_gpu"] * 2
    if one_batch != two_batch:
        raise ValueError("global batch sizes do not match")

    one_rate = one["training"]["full_data"]["tokens_per_second"]
    two_rate = two["training"]["full_data"]["tokens_per_second"]
    speedup = two_rate / one_rate
    print(
        json.dumps(
            {
                "one_gpu_tokens_per_second": one_rate,
                "two_gpu_tokens_per_second": two_rate,
                "speedup": speedup,
                "two_gpu_scaling_efficiency_percent": 100.0 * speedup / 2.0,
                "global_batch_size": one_batch,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
