#!/usr/bin/env python
"""Build outputs/research_report.md from recorded run artefacts (nothing is typed in by hand).

    python scripts/make_report.py --run outputs/runs/<run_dir> [--smoke outputs/runs/<tiny_run>] [--benchmark outputs/metrics/benchmark.json]
"""
import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from mini_gpt.config import config_from_dict
from mini_gpt.model.gpt import GPT
from mini_gpt.utils.parameter_count import count_parameters, estimate_flops, human


def load_run(d: Path) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    return {"dir": d, "metrics": m, "summary": m["summary"], "env": json.loads((d / "environment.json").read_text()),
            "cfg": config_from_dict(yaml.safe_load((d / "config.yaml").read_text()))}


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def run_section(title: str, r: dict) -> str:
    c, s, env = r["cfg"], r["summary"], r["env"]
    hist = r["metrics"]["history"]
    tr = [h for h in hist if "train_loss" in h]
    val = [h for h in hist if "val_loss" in h]
    pc = count_parameters(GPT(c.model))
    t = c.training
    tpu = env["effective_tokens_per_update"]
    fl = estimate_flops(c.model, s["tokens_seen"])
    first_val = val[0]
    lines = [f"## {title}", "",
             f"*Run directory:* `{r['dir']}`  \n*Command:* `{env['command']}`  \n*Git commit:* {env.get('git_commit') or 'not a git repository'}", "",
             "### Model", "",
             f"{c.model.num_layers} layers, d={c.model.embedding_dim}, {c.model.num_heads} heads, context {c.model.context_length}, vocab {c.model.vocab_size}, "
             f"dropout {c.model.dropout}, pre-LN, GELU ({c.model.activation}), {c.model.position_encoding} positions, tied embeddings={c.model.tie_embeddings}, "
             f"attention=`{c.model.attention_impl}`.", "",
             f"Parameters: **{pc['total']:,}** (embedding {pc['embedding']:,}; attention {pc['attention']:,}; MLP {pc['mlp']:,}; LayerNorm {pc['layernorm']:,}).", "",
             "### Setup", "",
             f"* Dataset: `{c.data.dataset}` (official splits). Tokenizer: byte-level BPE, vocab {env['tokenizer']['vocab_size']}, trained on the train split only.",
             f"* Optimiser: {t.optimizer}, peak LR {t.learning_rate:g} → {t.min_learning_rate:g} (cosine), warmup {t.warmup_steps}, wd {t.weight_decay}, clip {t.gradient_clip}, seed {t.seed}.",
             f"* Micro-batch {t.batch_size} × accumulation {t.gradient_accumulation_steps} = {t.effective_batch_size} sequences = {tpu:,} tokens/update; {s['steps']} steps.",
             f"* Hardware: {env['hardware'].get('gpu') or 'no GPU (CPU only)'}; CPU cores {env['hardware'].get('cpu_count')}; RAM {env['hardware'].get('system_ram_gb')} GB; "
             f"PyTorch {env['pytorch']}; precision `{env['precision']}`.", "",
             "### Results (measured)", "",
             "| metric | value |", "|---|---|",
             f"| tokens processed | {s['tokens_seen']:,} |",
             f"| wall-clock training time | {s['elapsed_sec']:.0f} s |",
             f"| mean throughput | {s['mean_tokens_per_sec']:.0f} tokens/s |",
             f"| initial val loss (step 0) | {first_val['val_loss']:.3f} (ln V = {math.log(c.model.vocab_size):.3f}) |",
             f"| final train loss (last logged step) | {fmt(s['final_train_loss'])} |",
             f"| best periodic val loss | {fmt(s['best_val_loss'])} |",
             f"| final full-split val loss / PPL / bits per token | {s['final_val']['loss']:.3f} / {s['final_val']['perplexity']:.1f} / {s['final_val']['bits_per_token']:.3f} ({s['final_val']['tokens']:,} tokens) |",
             f"| final full-split test loss / PPL / bits per token | {s['final_test']['loss']:.3f} / {s['final_test']['perplexity']:.1f} / {s['final_test']['bits_per_token']:.3f} ({s['final_test']['tokens']:,} tokens) |",
             f"| peak GPU memory | {'n/a (no GPU in this environment)' if not s['peak_vram_mb'] else str(round(s['peak_vram_mb'])) + ' MB'} |",
             f"| peak process RSS (CPU RAM) | {s['peak_cpu_rss_mb']:.0f} MB |",
             f"| approx. training compute | {human(fl['total_flops'])} FLOPs (model-based estimate) |", "",
             f"Perplexity is per BPE token of this tokenizer (not comparable to word-level WikiText numbers). Plots: `{r['dir']}/plots/`.", ""]
    # val trajectory
    lines += ["Validation trajectory (first `eval_iters` batches of the validation split):", "", "| step | val loss | val PPL |", "|---|---|---|"]
    for v in val:
        lines.append(f"| {v['step']} | {v['val_loss']:.3f} | {v['val_ppl']:.1f} |")
    lines.append("")
    final_sample = (r["dir"] / "samples" / "step_final.txt").read_text().split("\n\n")
    first_sample = (r["dir"] / "samples" / "step_0.txt").read_text().split("\n\n")
    lines += ["### Generation examples (prompt 1 and 2 of the fixed evaluation set)", "", "At initialisation:", "", "```",
              *(first_sample[1:3]), "```", "", "After training:", "", "```", *(final_sample[1:3]), "```", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--benchmark", default="outputs/metrics/benchmark.json")
    ap.add_argument("--extra", default=None, help="markdown file with additional notes appended verbatim (e.g. test results)")
    ap.add_argument("--out", default="outputs/research_report.md")
    args = ap.parse_args()
    parts = ["# Research report: mini GPT-2", "",
             "Generated by `scripts/make_report.py` from recorded artefacts. Every number below was produced by an executed command; "
             "items that could not be executed in this environment are listed under *Not executed*.", ""]
    if args.extra:
        parts += [Path(args.extra).read_text(), ""]
    parts.append(run_section("Main run (default `small` architecture)", load_run(Path(args.run))))
    if args.smoke:
        parts.append(run_section("Smoke run (`tiny` architecture)", load_run(Path(args.smoke))))
    bp = Path(args.benchmark)
    if bp.exists():
        b = json.loads(bp.read_text())
        parts += ["## Benchmark (measured on this machine)", "",
                  f"Hardware: {b['hardware'].get('gpu') or 'CPU only'}, {b['hardware'].get('cpu_count')} core(s). Batch {b['batch_size']}, context {b['context_length']}. "
                  "On CPU, \"mixed\" means bf16 autocast on CPU (informational only; training on CPU uses fp32).", "",
                  "| attention | precision | forward ms | backward ms | tokens/s | peak VRAM MB | RSS MB |", "|---|---|---|---|---|---|---|"]
        for r in b["train_step"]:
            parts.append(f"| {r['attention']} | {r['precision']} | {r['forward_ms']:.0f} | {r['backward_ms']:.0f} | {r['tokens_per_sec']:.0f} | {r['peak_vram_mb']:.0f} | {r['cpu_rss_mb']:.0f} |")
        g = b["generation"]
        parts += ["", "| generation (64 new tokens) | tokens/s |", "|---|---|",
                  f"| without KV cache | {g[0]['tokens_per_sec']:.1f} |", f"| with KV cache | {g[1]['tokens_per_sec']:.1f} |", "",
                  f"KV-cache speed-up: **×{b['kv_cache_speedup']:.2f}**; max |logit difference| cached vs full forward: {b['kv_cache_max_abs_logit_diff']:.1e}. "
                  f"Manual vs SDPA max |logit difference|: {b['manual_vs_sdpa']['max_abs_logit_diff']:.1e}.", ""]
    parts += Path("docs/_report_tail.md").read_text().splitlines(keepends=True) if Path("docs/_report_tail.md").exists() else []
    Path(args.out).write_text("\n".join(p if isinstance(p, str) else p for p in parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
