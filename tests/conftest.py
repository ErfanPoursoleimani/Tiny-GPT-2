import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from mini_gpt.config import ModelConfig


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=100, context_length=16, embedding_dim=32, num_layers=2, num_heads=4, dropout=0.0)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
