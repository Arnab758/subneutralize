"""
Unit tests for SubNeutralize Scale-Free Dynamical Governor, Engine, and Text Extraction.
"""

import pytest
import torch
import torch.nn as nn
from subneutralize import (
    ScaleFreeDynamicalGovernor,
    ConsensusEntropyGovernor,
    SubNeutralize,
    GovernedOutput,
    extract_clean_code,
    extract_clean_answer,
    extract_reasoning_and_answer,
)
from subneutralize.engine import sample_next_token


class MockLayer(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class MockTransformer(nn.Module):
    def __init__(self, num_layers=32, hidden_dim=64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockLayer(hidden_dim) for _ in range(num_layers)])
        self.config = nn.Module()
        self.config.hidden_size = hidden_dim


def test_alias_equivalence():
    assert ConsensusEntropyGovernor is ScaleFreeDynamicalGovernor


def test_bottleneck_layer_selection():
    # SubNeutralize automatically selects 50% depth (Layer 16 of 32)
    model = MockTransformer(num_layers=32)
    engine = SubNeutralize(model, tokenizer=None)
    assert engine.target_layer == 16


def test_custom_target_layer():
    model = MockTransformer(num_layers=64)
    engine = SubNeutralize(model, tokenizer=None, target_layer=32)
    assert engine.target_layer == 32


def test_scale_free_momentum_tracking():
    gov = ScaleFreeDynamicalGovernor(
        min_warmup_tokens=5,
        consensus_ratio_threshold=0.85,
        velocity_ceiling=0.15,
        dispersion_ceiling=0.10,
        debounce_tokens=2,
    )
    
    # 1. Step 1: Initial token
    torch.manual_seed(42)
    h0 = torch.randn(1, 64)
    res = gov.update(h0)
    assert res is False
    assert gov.step == 1
    assert gov.ema_velocity is None

    # 2. Step 2-4: Active exploration (high angular variance)
    for _ in range(3):
        h_random = torch.randn(1, 64)
        res = gov.update(h_random)
        assert res is False

    assert gov.step == 4
    assert gov.ema_velocity is not None

    # 3. Step 5+: Converging trajectory into attractor basin (small perturbation)
    h_stable = torch.randn(1, 64)
    reached = False
    for _ in range(12):
        h_next = h_stable + 0.0005 * torch.randn(1, 64)
        if gov.update(h_next):
            reached = True
            break
        h_stable = h_next

    assert reached is True
    assert gov.consensus_token is not None


def test_batched_governor_update():
    gov = ScaleFreeDynamicalGovernor(min_warmup_tokens=2, debounce_tokens=1)
    
    # Pass batch size 4
    h_init = torch.randn(4, 64)
    res1 = gov.update(h_init)
    assert isinstance(res1, torch.Tensor)
    assert res1.shape == (4,)
    assert not res1.any()

    # Pass second step
    h_next = h_init + 0.0001 * torch.randn(4, 64)
    res2 = gov.update(h_next)
    assert isinstance(res2, torch.Tensor)
    assert res2.shape == (4,)


def test_entropy_safety_guard():
    gov = ScaleFreeDynamicalGovernor(
        min_warmup_tokens=2,
        entropy_threshold=1.0,
        debounce_tokens=1
    )
    h0 = torch.randn(1, 64)
    gov.update(h0)

    # Low velocity but very high entropy (uniform logits over 1000 vocab)
    h1 = h0 + 0.0001 * torch.randn(1, 64)
    high_entropy_logits = torch.zeros(1, 1000) # uniform distribution -> entropy = ln(1000) ~ 6.9
    res = gov.update(h1, logits=high_entropy_logits)
    # Should NOT stop because model is in high perplexity / confusion
    assert res is False


def test_extract_clean_code():
    # 1. Markdown block with backticks
    raw_markdown = "Here is the code:\n```python\ndef solution(x):\n    return x * 2\n```\nHope that helps!"
    clean = extract_clean_code(raw_markdown)
    assert clean == "def solution(x):\n    return x * 2"

    # 2. Continuation from prompt fence
    fence_continuation = "```python\ndef run():\n    pass\n```"
    clean_fence = extract_clean_code(fence_continuation)
    assert clean_fence == "def run():\n    pass"

    # 3. Raw python code without backticks
    raw_code = "def add(a, b):\n    return a + b"
    clean_code = extract_clean_code(raw_code)
    assert clean_code == raw_code


def test_extract_reasoning_and_answer():
    full_output = "<think>\nLet's verify step 1.\nStep 1 is correct.\n</think>\nThe answer is 42."
    reasoning, answer = extract_reasoning_and_answer(full_output)
    assert "Let's verify step 1" in reasoning
    assert answer == "The answer is 42."

    # Test clean answer extraction
    clean_ans = extract_clean_answer(full_output)
    assert clean_ans == "The answer is 42."


def test_sample_next_token():
    logits = torch.randn(1, 50)
    # Greedy (temp = 0)
    t_greedy = sample_next_token(logits, temperature=0.0)
    assert t_greedy.item() == torch.argmax(logits, dim=-1).item()

    # Sample with temperature and top_p
    t_sampled = sample_next_token(logits, temperature=0.7, top_p=0.9)
    assert 0 <= t_sampled.item() < 50


def test_as_stopping_criteria():
    model = MockTransformer(num_layers=16, hidden_dim=32)
    engine = SubNeutralize(model, tokenizer=None, warmup_tokens=2)
    criteria = engine.as_stopping_criteria()
    assert callable(criteria)
    
    # Simulate hook trigger and criteria evaluation for single batch
    engine._current_hidden = torch.randn(1, 32)
    res = criteria(torch.tensor([[1, 2]]))
    assert res is False
    
    # Batched criteria evaluation
    engine._current_hidden = torch.randn(2, 32)
    res_batch = criteria(torch.tensor([[1, 2], [3, 4]]))
    assert isinstance(res_batch, torch.Tensor)
    assert res_batch.shape == (2,)

    engine.detach()
    assert engine._hook_handle is None
