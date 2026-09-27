"""
Unit tests for mathematical modules: BiasSubspaceAnalyzer, orthogonal projection, and metrics.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
from src.bias_subspace import BiasSubspaceAnalyzer


def test_bias_subspace_analyzer():
    print("Testing BiasSubspaceAnalyzer...")
    analyzer = BiasSubspaceAnalyzer(subspace_dim=2)
    
    # 1. Test SVD Fitting on synthetic representations [10 samples, 64 hidden_dim]
    dummy_prompts = torch.randn(10, 64)
    analyzer.fit_prompt_bias(dummy_prompts)
    assert analyzer.V_bias is not None
    assert analyzer.V_bias.shape == (64, 2), f"Expected shape (64, 2), got {analyzer.V_bias.shape}"
    
    # Check orthonormality: V^T @ V should be identity [2, 2]
    identity_check = torch.matmul(analyzer.V_bias.T, analyzer.V_bias)
    assert torch.allclose(identity_check, torch.eye(2), atol=1e-5), "V_bias is not orthonormal!"
    print("  [PASS] Orthonormal basis fitting (SVD)")

    # 2. Test Orthogonal Projector: P_perp = I - V @ V^T
    P_perp = analyzer.compute_orthogonal_projection_matrix()
    assert P_perp.shape == (64, 64)
    # P_perp should project V_bias to zero: P_perp @ V_bias == 0
    projected_bias = torch.matmul(P_perp, analyzer.V_bias)
    assert torch.allclose(projected_bias, torch.zeros_like(projected_bias), atol=1e-5), "Projection failed to zero out bias!"
    print("  [PASS] Orthogonal projection operator P_perp nullifies V_bias")

    # 3. Test Trajectory Metrics (Velocity, Curvature, Alignment)
    # Simulate a smooth reasoning trajectory [20 tokens, 64 hidden_dim]
    dummy_trajectory = torch.randn(20, 64)
    metrics = analyzer.compute_trajectory_metrics(dummy_trajectory)
    
    assert "velocity" in metrics
    assert "curvature" in metrics
    assert "bias_alignment" in metrics
    assert len(metrics["velocity"]) == 19
    assert len(metrics["curvature"]) == 18
    assert len(metrics["bias_alignment"]) == 20
    print("  [PASS] Trajectory velocity, curvature, and alignment calculations")

    print("ALL UNIT TESTS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    test_bias_subspace_analyzer()
