"""
Unit tests for SubNeutralize v0.2.0 Latent Manifold Governor.
Verifies Welford online statistics, Thin-SVD prompt subspace projection, and Lyapunov potential math.
"""

import math
import torch
import pytest
from subneutralize.manifold import OnlineManifoldStat, PromptSubspaceGovernor


def test_online_manifold_stat():
    stat = OnlineManifoldStat()
    data = [10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0]
    for x in data:
        stat.update(x)

    # Ground truth calculations
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    std = math.sqrt(variance)

    assert abs(stat.mean - mean) < 1e-4
    assert abs(stat.std() - std) < 1e-4
    assert abs(stat.z_score(mean) - 0.0) < 1e-4
    assert abs(stat.z_score(mean + std) - 1.0) < 1e-4


def test_prompt_subspace_governor_orthogonality():
    torch.manual_seed(42)
    seq_len = 48
    hidden_dim = 128
    rank_k = 4

    governor = PromptSubspaceGovernor(rank_k=rank_k)
    H_prompt = torch.randn(seq_len, hidden_dim)

    burn_in, cover_bound = governor.fit_prompt(H_prompt)
    assert burn_in > 0
    assert cover_bound > burn_in
    assert governor.V_bias is not None
    assert governor.V_bias.shape == (hidden_dim, rank_k)

    # Test orthogonal projection: h_perp should be orthogonal to V_bias
    h_t = torch.randn(1, hidden_dim)
    h_perp = governor.project(h_t)

    # Inner product with V_bias columns should be 0 (within float precision)
    projection_overlap = h_perp @ governor.V_bias
    max_overlap = torch.max(torch.abs(projection_overlap)).item()
    assert max_overlap < 1e-5, f"Expected near-zero overlap, got {max_overlap}"


def test_prompt_subspace_governor_unfitted():
    governor = PromptSubspaceGovernor(rank_k=4)
    h_t = torch.randn(1, 64)
    # When unfitted, project should return h_t unmodified
    h_out = governor.project(h_t)
    assert torch.equal(h_t, h_out)
