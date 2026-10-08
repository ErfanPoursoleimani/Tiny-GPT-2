"""Timing and memory-reporting helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timer(device: torch.device):
    """Yield a callable returning elapsed seconds (with proper CUDA synchronisation)."""
    sync(device)
    start = time.perf_counter()
    result = {"t": 0.0}

    def elapsed() -> float:
        return result["t"] or (time.perf_counter() - start)

    yield elapsed
    sync(device)
    result["t"] = time.perf_counter() - start


def cpu_rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        import os

        return pages * os.sysconf("SC_PAGE_SIZE") / 1024**2
    except (OSError, ValueError, AttributeError):
        return 0.0


def memory_stats(device: torch.device) -> dict[str, float]:
    """allocated / reserved / peak VRAM in MB (0.0 on CPU) plus process RSS."""
    out = {"alloc_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0, "cpu_rss_mb": cpu_rss_mb()}
    if device.type == "cuda":
        out["alloc_mb"] = torch.cuda.memory_allocated(device) / 1024**2
        out["reserved_mb"] = torch.cuda.memory_reserved(device) / 1024**2
        out["peak_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return out


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
