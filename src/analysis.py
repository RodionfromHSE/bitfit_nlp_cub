import time
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn


@dataclass
class ParameterStats:
    total: int
    trainable: int

    @property
    def percentage(self) -> float:
        return 100.0 * self.trainable / self.total if self.total > 0 else 0.0


def count_parameters(model: nn.Module) -> ParameterStats:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParameterStats(total=total, trainable=trainable)


def get_peak_memory() -> int:
    """Returns peak GPU memory allocated in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0


def reset_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def bytes_to_gb(bytes_val: int) -> float:
    return bytes_val / (1024**3)


@contextmanager
def track_memory():
    """Context manager to track peak memory during a block."""
    reset_memory_stats()
    yield
    peak = get_peak_memory()
    return peak


class Timer:
    def __init__(self):
        self.start_time: float | None = None
        self.elapsed: float = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def format_results_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if "trainable_params" in df.columns and "total_params" in df.columns:
        df["param_percentage"] = df["trainable_params"] / df["total_params"] * 100
    return df


def print_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    stats = count_parameters(model)
    print(f"\n{model_name} Summary:")
    print(f"  Total parameters: {stats.total:,}")
    print(f"  Trainable parameters: {stats.trainable:,}")
    print(f"  Trainable percentage: {stats.percentage:.2f}%")


def print_trainable_parameters(model: nn.Module) -> None:
    """Print all trainable parameter names and their shapes."""
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {tuple(param.shape)}")

