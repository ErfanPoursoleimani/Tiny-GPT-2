"""Hardware and precision detection."""
from __future__ import annotations

import platform
from dataclasses import dataclass

import torch


@dataclass
class PrecisionInfo:
    device: torch.device
    cuda_available: bool
    bf16_supported: bool
    fp16_supported: bool
    mixed_precision: bool  # autocast enabled
    amp_dtype: torch.dtype | None  # dtype used inside autocast (None => plain fp32)
    use_grad_scaler: bool  # only needed for fp16
    note: str = ""

    @property
    def dtype_name(self) -> str:
        return "float32" if self.amp_dtype is None else str(self.amp_dtype).replace("torch.", "")


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cuda":
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cpu")


def detect_precision(device: torch.device, mixed_precision: bool = True, preferred: str = "bfloat16") -> PrecisionInfo:
    """Pick the safest efficient precision for ``device``.

    Order of preference when mixed precision is requested on CUDA:
      requested dtype -> (bfloat16 -> float16) -> float32.
    BF16 support is *queried* (``torch.cuda.is_bf16_supported``), never assumed from CUDA alone.
    On CPU we default to plain fp32: CPU autocast exists but is rarely faster on laptops
    and complicates numerical comparisons.
    """
    cuda = device.type == "cuda"
    bf16 = fp16 = False
    if cuda:
        try:
            bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:  # older torch
            bf16 = bool(torch.cuda.is_bf16_supported())
        major, minor = torch.cuda.get_device_capability(device)
        fp16 = (major, minor) >= (5, 3)
    if not (mixed_precision and cuda):
        note = "CPU: using float32" if not cuda else "mixed precision disabled in config"
        return PrecisionInfo(device, cuda, bf16, fp16, False, None, False, note)

    order = {"bfloat16": ["bfloat16", "float16"], "float16": ["float16"], "float32": []}[preferred]
    for name in order:
        if name == "bfloat16" and bf16:
            return PrecisionInfo(device, cuda, bf16, fp16, True, torch.bfloat16, False)
        if name == "float16" and fp16:
            note = "fell back from bfloat16 to float16" if preferred == "bfloat16" else ""
            return PrecisionInfo(device, cuda, bf16, fp16, True, torch.float16, True, note)
    return PrecisionInfo(device, cuda, bf16, fp16, False, None, False, "no supported half precision; using float32")


def hardware_summary(device: torch.device) -> dict:
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cpu": platform.processor() or platform.machine(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            gpu=props.name,
            vram_gb=round(props.total_memory / 1024**3, 2),
            compute_capability=f"{props.major}.{props.minor}",
        )
    try:
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
        info["system_ram_gb"] = round(kb / 1024**2, 1)
    except (OSError, ValueError, IndexError):
        pass
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    return info
