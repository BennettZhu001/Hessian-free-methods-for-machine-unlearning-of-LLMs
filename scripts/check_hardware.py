#!/usr/bin/env python3
"""Print the CUDA facts that should accompany any reported benchmark."""

from __future__ import annotations

import json

import torch


def main() -> None:
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_gib": round(properties.total_memory / 2**30, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_supported": torch.cuda.is_bf16_supported(index),
            }
        )
    print(
        json.dumps(
            {
                "pytorch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "device_count": torch.cuda.device_count(),
                "devices": devices,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
