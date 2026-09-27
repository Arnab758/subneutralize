"""
Unit tests for ConsensusEntropyGovernor logic and threshold scaling.
"""

import torch
import torch.nn as nn
import pytest
from subneutralize.governor import ConsensusEntropyGovernor


class MockDecoderLayer(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return (self.linear(x),)


class MockModel(nn.Module):
    def __init__(self, num_layers=28, hidden_dim=64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockDecoderLayer(hidden_dim) for _ in range(num_layers)])
        self.config = nn.Module()
        self.config.hidden_size = hidden_dim


def test_layer_band_initialization():
    model = MockModel(num_layers=28)
    gov = ConsensusEntropyGovernor(model)
    # Midpoint of 28 is 14 -> band should be [12, 14, 16]
    assert gov.layer_band == [12, 14, 16]


def test_entropy_threshold_scaling():
    model = MockModel(num_layers=16)
    gov = ConsensusEntropyGovernor(
        model, 
        base_threshold=0.06, 
        ref_entropy=1.2, 
        min_threshold=0.025, 
        max_threshold=0.090
    )

    # 1. Low entropy (high certainty) -> threshold shrinks to min_threshold
    low_entropy_logits = torch.zeros(1, 1, 1000)
    low_entropy_logits[0, 0, 42] = 50.0  # Spike on single token
    gov.update_step(low_entropy_logits)
    assert gov.step_threshold == pytest.approx(0.025, rel=1e-2)

    # 2. High entropy (hesitation/diffusion) -> threshold expands to max_threshold
    high_entropy_logits = torch.randn(1, 1, 1000) * 0.01  # Uniform distribution
    gov.update_step(high_entropy_logits)
    assert gov.step_threshold == pytest.approx(0.090, rel=1e-2)
