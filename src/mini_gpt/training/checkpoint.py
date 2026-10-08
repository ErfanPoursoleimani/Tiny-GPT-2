"""Portable checkpoints: model + optimizer + scheduler + scaler + counters + config + RNG + tokenizer info."""
from __future__ import annotations

import os
import re
from pathlib import Path

import torch

from mini_gpt.config import Config, config_from_dict
from mini_gpt.model.gpt import GPT
from mini_gpt.utils.seed import get_rng_state


def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer,
    scheduler,
    scaler,
    step: int,
    epoch: float,
    best_val_loss: float,
    cfg: Config,
    tokenizer_info: dict | None,
    extra: dict | None = None,
) -> None:
    """Atomic write (tmp file + rename) so an interrupted save never corrupts ``latest.pt``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": cfg.to_dict(),
        "seed": cfg.training.seed,
        "rng_state": get_rng_state(),
        "tokenizer": tokenizer_info,
        "torch_version": torch.__version__,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load onto CPU by default (portable between machines). ``weights_only=False`` because the file holds
    RNG states / config dicts -- only load checkpoints you trust."""
    return torch.load(path, map_location=map_location, weights_only=False)


def config_from_checkpoint(ckpt: dict) -> Config:
    return config_from_dict(ckpt["config"])


def model_from_checkpoint(path: str | Path, device: torch.device) -> tuple[GPT, Config, dict]:
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = config_from_checkpoint(ckpt)
    model = GPT(cfg.model)
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg, ckpt


def prune_step_checkpoints(directory: str | Path, keep_last: int) -> None:
    d = Path(directory)
    files = sorted(
        (p for p in d.glob("step_*.pt") if re.fullmatch(r"step_\d+\.pt", p.name)),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for p in files[: max(0, len(files) - keep_last)]:
        p.unlink()
