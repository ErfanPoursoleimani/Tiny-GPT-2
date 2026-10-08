import copy

import pytest
import torch

from mini_gpt.generation.generate import apply_repetition_penalty, generate, top_k_top_p_filter
from mini_gpt.model.gpt import GPT


@pytest.fixture
def model(tiny_cfg):
    return GPT(tiny_cfg).eval()


def test_generated_length(model, tiny_cfg):
    out = generate(model, torch.randint(0, tiny_cfg.vocab_size, (1, 4)), 7, temperature=1.0, top_k=10)
    assert out.shape == (1, 11)
    assert out.max() < tiny_cfg.vocab_size


def test_greedy_deterministic_and_cache_equivalent(model, tiny_cfg):
    p = torch.randint(0, tiny_cfg.vocab_size, (1, 5))
    a = generate(model, p, 10, greedy=True, use_cache=True)
    b = generate(model, p, 10, greedy=True, use_cache=False)
    c = generate(model, p, 10, greedy=True, use_cache=True)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_seeded_sampling_reproducible(model, tiny_cfg):
    p = torch.randint(0, tiny_cfg.vocab_size, (1, 3))
    a = generate(model, p, 12, temperature=1.0, top_p=0.9, seed=7)
    b = generate(model, p, 12, temperature=1.0, top_p=0.9, seed=7)
    c = generate(model, p, 12, temperature=1.0, top_p=0.9, seed=8)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_generation_beyond_context_window(model, tiny_cfg):
    p = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
    n = tiny_cfg.context_length * 2
    for cache in (True, False):
        out = generate(model, p, n, greedy=True, use_cache=cache)
        assert out.shape == (1, 4 + n)


def test_top_k_filter():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    out = top_k_top_p_filter(logits, top_k=2, top_p=None)
    assert torch.isfinite(out).sum() == 2 and torch.isfinite(out[0, 1]) and torch.isfinite(out[0, 3])


def test_top_p_filter_keeps_nucleus():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.1, 0.05, 0.05]]))
    out = top_k_top_p_filter(logits, top_k=None, top_p=0.75)
    kept = torch.isfinite(out[0]).nonzero().flatten().tolist()
    assert kept == [0, 1]  # 0.5 + 0.3 >= 0.75
    out_all = top_k_top_p_filter(logits, None, 1.0)
    assert torch.isfinite(out_all).all()


def test_repetition_penalty():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = apply_repetition_penalty(logits, torch.tensor([[0, 1]]), 2.0)
    assert out.tolist() == [[1.0, -4.0, 1.0]]


def test_top_k_one_equals_greedy(model, tiny_cfg):
    p = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
    g = generate(model, p, 8, greedy=True)
    k1 = generate(model, p, 8, temperature=1.0, top_k=1, seed=0)
    assert torch.equal(g, k1)
