"""Plotting of training diagnostics from the metrics history."""
from __future__ import annotations

from pathlib import Path


def plot_training(history: list[dict], out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = [r for r in history if "train_loss" in r]
    val = [r for r in history if "val_loss" in r]

    def series(rows, key):
        pts = [(r["step"], r[key]) for r in rows if key in r and r[key] is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    specs = [
        ("training_loss.png", "Training loss", "loss (nats)", [("train", train, "train_loss")]),
        ("validation_loss.png", "Validation loss", "loss (nats)", [("validation", val, "val_loss")]),
        ("learning_rate.png", "Learning rate", "lr", [("lr", train, "lr")]),
        ("gradient_norm.png", "Gradient norm (pre-clip)", "norm", [("grad norm", train, "grad_norm")]),
        ("tokens_per_sec.png", "Throughput", "tokens/sec", [("tokens/s", train, "tokens_per_sec")]),
        ("gpu_memory.png", "Memory", "MB", [("allocated", train, "alloc_mb"), ("reserved", train, "reserved_mb"),
                                            ("peak", train, "peak_mb"), ("cpu rss", train, "cpu_rss_mb")]),
    ]
    paths = []
    for fname, title, ylabel, lines in specs:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        drawn = False
        for label, rows, key in lines:
            xs, ys = series(rows, key)
            if xs and any(y != 0 for y in ys):
                ax.plot(xs, ys, label=label, marker="o" if len(xs) < 15 else None)
                drawn = True
        ax.set_title(title); ax.set_xlabel("optimizer step"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
        if drawn and len(lines) > 1:
            ax.legend()
        if not drawn:
            ax.text(0.5, 0.5, "no data (e.g. no GPU)", ha="center", transform=ax.transAxes)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths
