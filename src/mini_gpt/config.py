"""Configuration dataclasses, YAML loading (with ``base:`` inheritance) and validation."""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 256
    embedding_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    dropout: float = 0.1
    bias: bool = True
    tie_embeddings: bool = True
    activation: str = "gelu"  # "gelu" (exact erf), "gelu_new" (GPT-2 tanh approximation)
    attention_impl: str = "auto"  # auto | manual | sdpa
    position_encoding: str = "learned"  # learned | rope

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_heads

    def validate(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.position_encoding not in ("learned", "rope"):
            raise ValueError(f"position_encoding must be 'learned' or 'rope', got {self.position_encoding!r}")
        if self.position_encoding == "rope" and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        if self.attention_impl not in ("auto", "manual", "sdpa"):
            raise ValueError(f"attention.implementation must be auto|manual|sdpa, got {self.attention_impl!r}")
        if self.activation not in ("gelu", "gelu_new"):
            raise ValueError(f"activation must be gelu|gelu_new, got {self.activation!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("vocab_size", "context_length", "embedding_dim", "num_layers", "num_heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class DataConfig:
    dataset: str = "wikitext2"  # wikitext2 | tinyshakespeare | custom
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    tokenizer_dir: str = "data/tokenizer"
    custom_path: str | None = None  # for dataset == "custom"
    max_train_tokens: int | None = None  # truncate the training stream (smoke tests)
    max_eval_tokens: int | None = None
    num_workers: int = 0


@dataclass
class TokenizerConfig:
    vocab_size: int = 32000
    min_frequency: int = 2


@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_steps: int = 10000
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    optimizer: str = "adamw"  # adamw | sgd (educational comparison)
    eval_interval: int = 250
    eval_iters: int = 50  # micro-batches per periodic evaluation; <=0 means full split
    log_interval: int = 10
    checkpoint_interval: int = 1000
    keep_last_checkpoints: int = 3
    sample_steps: list[int] = field(default_factory=lambda: [0, 1000, 5000])
    sample_max_new_tokens: int = 60
    gradient_checkpointing: bool = False
    seed: int = 42
    checkpoint_dir: str = "checkpoints"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.optimizer not in ("adamw", "sgd"):
            raise ValueError("optimizer must be adamw or sgd")


@dataclass
class PrecisionConfig:
    mixed_precision: bool = True
    dtype: str = "bfloat16"  # preferred: bfloat16 | float16 | float32 (auto-fallback if unsupported)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    experiment_name: str = "run"
    output_dir: str = "outputs"

    def validate(self) -> list[str]:
        """Raise on hard errors; return a list of human-readable warnings."""
        self.model.validate()
        self.training.validate()
        warnings: list[str] = []
        m, t = self.model, self.training
        if m.context_length > 2048:
            warnings.append(f"context_length={m.context_length} is very large for a 6 GB GPU.")
        tokens_per_micro = t.batch_size * m.context_length
        if tokens_per_micro > 16384:
            warnings.append(
                f"batch_size x context_length = {tokens_per_micro} tokens per micro-batch is unusually large for 6 GB VRAM."
            )
        if self.tokenizer.vocab_size != m.vocab_size:
            warnings.append(
                f"tokenizer.vocab_size ({self.tokenizer.vocab_size}) != model.vocab_size ({m.vocab_size}); "
                "the model vocab will be overridden by the trained tokenizer's actual size."
            )
        return warnings

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_base(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base_ref = raw.pop("base", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        return _deep_merge(_load_yaml_with_base(base_path), raw)
    return raw


def _build(cls, values: dict, section: str):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown keys in '{section}': {sorted(unknown)}")
    return cls(**values)


def config_from_dict(raw: dict) -> Config:
    """Build a Config from a (possibly nested, spec-style) dict."""
    raw = copy.deepcopy(raw)
    model = dict(raw.pop("model", {}))
    # Spec-style nested sections are folded into ModelConfig.
    attn = raw.pop("attention", None)
    if attn:
        model["attention_impl"] = attn.get("implementation", model.get("attention_impl", "auto"))
    pe = raw.pop("position_encoding", None)
    if pe:
        model["position_encoding"] = pe.get("type", model.get("position_encoding", "learned"))
    cfg = Config(
        model=_build(ModelConfig, model, "model"),
        data=_build(DataConfig, raw.pop("data", {}), "data"),
        tokenizer=_build(TokenizerConfig, raw.pop("tokenizer", {}), "tokenizer"),
        training=_build(TrainingConfig, raw.pop("training", {}), "training"),
        precision=_build(PrecisionConfig, raw.pop("precision", {}), "precision"),
        experiment_name=raw.pop("experiment_name", "run"),
        output_dir=raw.pop("output_dir", "outputs"),
    )
    if raw:
        raise ValueError(f"Unknown top-level config keys: {sorted(raw)}")
    return cfg


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load a YAML config (resolving ``base:`` chains) and apply optional dotted overrides."""
    raw = _load_yaml_with_base(Path(path).resolve())
    for dotted, value in (overrides or {}).items():
        node = raw
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return config_from_dict(raw)


def apply_overrides_from_cli(items: list[str] | None) -> dict:
    """Parse ``section.key=value`` strings (value parsed as YAML)."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        k, v = item.split("=", 1)
        val = yaml.safe_load(v)
        if isinstance(val, str):  # YAML 1.1 reads "6e-4" (no '.') as a string; coerce numeric-looking strings
            try:
                val = float(val)
            except ValueError:
                pass
        out[k.strip()] = val
    return out
