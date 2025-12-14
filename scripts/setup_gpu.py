#!/usr/bin/env python
"""Select GPU with most free memory and set CUDA_VISIBLE_DEVICES."""

import os
import re
import subprocess


def get_free_memory_per_gpu() -> dict[int, int]:
    """Parse nvidia-smi to get free memory (MiB) per GPU."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr}")

    free_memory = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        idx, mem = line.split(",")
        free_memory[int(idx.strip())] = int(mem.strip())

    return free_memory


def select_best_gpu() -> int:
    """Return index of GPU with most free memory."""
    free_memory = get_free_memory_per_gpu()
    if not free_memory:
        raise RuntimeError("No GPUs found")
    return max(free_memory, key=free_memory.get)


def setup_cuda_device() -> int:
    """Set CUDA_VISIBLE_DEVICES to GPU with most free memory, return its index."""
    gpu_idx = select_best_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    return gpu_idx


if __name__ == "__main__":
    free_mem = get_free_memory_per_gpu()
    print("Free memory per GPU (MiB):")
    for idx, mem in sorted(free_mem.items()):
        print(f"  GPU {idx}: {mem:,} MiB")

    best = select_best_gpu()
    print(f"\nBest GPU: {best} ({free_mem[best]:,} MiB free)")
    print(f"\nTo use: export CUDA_VISIBLE_DEVICES={best}")

