"""Logging helpers: console+file logger, JSON metrics, optional TensorBoard writer."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "mini_gpt", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class MetricsLogger:
    """Collects metric rows in memory, writes them to ``metrics.json`` and TensorBoard."""

    def __init__(self, run_dir: Path, tb_dir: Path | None = None, use_tensorboard: bool = True):
        self.run_dir = Path(run_dir)
        self.rows: list[dict] = []
        self.writer = None
        if use_tensorboard and tb_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                Path(tb_dir).mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(tb_dir))
            except Exception as exc:  # tensorboard missing: keep going without it
                logging.getLogger("mini_gpt").warning("TensorBoard disabled: %s", exc)

    def log(self, step: int, **values: float) -> None:
        row = {"step": step, **values}
        self.rows.append(row)

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def save(self, extra: dict | None = None) -> None:
        payload = {"history": self.rows}
        if extra:
            payload.update(extra)
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def create_run_dir(output_dir: str | Path, experiment: str) -> Path:
    """outputs/runs/YYYYMMDD_HHMMSS_<experiment>/ (unique even when two runs start in the same second)."""
    import time

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = Path(output_dir) / "runs" / f"{stamp}_{experiment}"
    run_dir, n = base, 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{n}")
        n += 1
    (run_dir / "samples").mkdir(parents=True)
    return run_dir
