"""
Unit tests for BiasSubspaceAnalyzer and orthogonal projection operator P_perp.
"""

import torch
import pytest
from subneutralize.subspace import BiasSubspaceAnalyzer


def test_fit_prompt_bias_dimensions():
    analyzer = BiasSubspaceAnalyzer(subspace_dim=4)
    hidden_dim = 128
    samples = 10
    dummy_states = torch.randn(samples, hidden_dim)

    V = analyzer.fit_prompt_bias(dummy_states)
    assert V.shape == (hidden_dim, 4)
    # Check that columns of V are orthonormal
    gram = torch.matmul(V.T, V)
    eye = torch.eye(4)
    assert torch.allclose(gram, eye, atol=1e-5)


def test_orthogonal_projection_idempotency():
    analyzer = BiasSubspaceAnalyzer(subspace_dim=2)
    hidden_dim = 64
    dummy_states = torch.randn(5, hidden_dim)
    analyzer.fit_prompt_bias(dummy_states)

    P = analyzer.get_orthogonal_projector()
    # P_perp should be symmetric: P = P^T
    assert torch.allclose(P, P.T, atol=1e-5)
    # P_perp should be idempotent: P @ P = P
    assert torch.allclose(torch.matmul(P, P), P, atol=1e-5)


def test_orthogonal_nullification():
    analyzer = BiasSubspaceAnalyzer(subspace_dim=3)
    hidden_dim = 64
    dummy_states = torch.randn(6, hidden_dim)
    V = analyzer.fit_prompt_bias(dummy_states)

    # Any vector in span(V) should be nullified by P_perp
    coeff = torch.randn(3, 1)
    v_in_span = torch.matmul(V, coeff).T  # [1, hidden_dim]

    v_clean = analyzer.project_orthogonal(v_in_span)
    assert torch.norm(v_clean) < 1e-4
