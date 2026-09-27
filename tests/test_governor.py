"""
Unit tests for SubNeutralize Scale-Free Dynamical Governor and Code Extraction.
"""

import pytest
import torch
import torch.nn as nn
from subneutralize import (
    ScaleFreeDynamicalGovernor,
    ConsensusEntropyGovernor,
    SubNeutralize,
    extract_clean_code,
)


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
        consensus_ratio_threshold=0.82,
        velocity_ceiling=0.135,
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
    # Feed stabilized vectors to satisfy debounce requirement (debounce_tokens=2)
    h_stable = torch.randn(1, 64)
    reached = False
    for _ in range(10):
        # Stable trajectory -> low cosine distance
        h_next = h_stable + 0.001 * torch.randn(1, 64)
        if gov.update(h_next):
            reached = True
            break
        h_stable = h_next

    assert reached is True
    assert gov.consensus_token is not None


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


def test_as_stopping_criteria():
    model = MockTransformer(num_layers=16, hidden_dim=32)
    engine = SubNeutralize(model, tokenizer=None, warmup_tokens=2)
    criteria = engine.as_stopping_criteria()
    assert callable(criteria)
    
    # Simulate hook trigger and criteria evaluation
    engine._current_hidden = torch.randn(1, 32)
    res = criteria(torch.tensor([[1, 2]]))
    assert res is False
    
    engine.detach()
    assert engine._hook_handle is None

