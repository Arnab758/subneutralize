"""
Subspace extraction and geometric trajectory dynamics in Transformer residual streams.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


class BiasSubspaceAnalyzer:
    """
    Extracts the Prompt Bias Subspace V_bias and computes kinematic and 
    geometric trajectory metrics (velocity, curvature, bias alignment) 
    across reasoning tokens.
    """

    def __init__(self, subspace_dim: int = 4):
        """
        Args:
            subspace_dim: Dimensionality (rank k) of the orthonormal bias subspace.
        """
        self.k = subspace_dim
        self.V_bias: Optional[torch.Tensor] = None  # Shape: [hidden_dim, k]

    def fit_prompt_bias(self, prompt_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Extracts orthonormal basis V_bias from prompt terminal representations
        using Thin-SVD (Singular Value Decomposition).
        
        Args:
            prompt_hidden_states: Tensor of shape [N_samples, hidden_dim]
                                  or [hidden_dim] representing prompt terminal states.
        Returns:
            V_bias matrix of shape [hidden_dim, k]
        """
        if prompt_hidden_states.ndim == 1:
            # Single prompt vector fallback
            vec = prompt_hidden_states.unsqueeze(0)
        elif prompt_hidden_states.ndim == 2:
            vec = prompt_hidden_states
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got shape {prompt_hidden_states.shape}")

        device = vec.device
        dtype = vec.dtype

        if vec.shape[0] == 1:
            # Rank-1 bias vector
            norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
            normalized = vec / (norm + 1e-12)
            self.V_bias = normalized.T.contiguous()
            return self.V_bias

        # Center representations across prompt variants
        mean_vec = vec.mean(dim=0, keepdim=True)
        centered = (vec - mean_vec).float()

        # Thin-SVD
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        actual_k = min(self.k, Vh.shape[0])
        self.V_bias = Vh[:actual_k, :].T.contiguous().to(device=device, dtype=dtype)
        return self.V_bias

    def get_orthogonal_projector(self) -> torch.Tensor:
        """
        Computes the orthogonal projection operator:
            P_perp = I - V_bias @ V_bias^T
        Returns:
            Tensor of shape [hidden_dim, hidden_dim]
        """
        if self.V_bias is None:
            raise ValueError("V_bias has not been initialized. Call fit_prompt_bias first.")

        d, k = self.V_bias.shape
        device = self.V_bias.device
        dtype = self.V_bias.dtype

        I = torch.eye(d, device=device, dtype=dtype)
        P_perp = I - torch.matmul(self.V_bias, self.V_bias.T)
        return P_perp

    def project_orthogonal(self, h: torch.Tensor) -> torch.Tensor:
        """
        Efficiently applies P_perp to representation vector h without
        explicitly materializing the large [d, d] matrix:
            h_clean = h - (h @ V_bias) @ V_bias^T
        """
        if self.V_bias is None:
            return h

        # Handle [batch, hidden] or [seq, hidden] or [hidden]
        orig_shape = h.shape
        flat_h = h.reshape(-1, orig_shape[-1]).to(self.V_bias.dtype)
        # flat_h: [B, D], V_bias: [D, K]
        proj = torch.matmul(flat_h, self.V_bias)  # [B, K]
        h_bias = torch.matmul(proj, self.V_bias.T)  # [B, D]
        h_clean = flat_h - h_bias
        return h_clean.reshape(orig_shape).to(h.dtype)

    def compute_trajectory_metrics(
        self,
        reasoning_hidden_states: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Computes kinematic and geometric metrics across a sequence of reasoning tokens.
        
        Args:
            reasoning_hidden_states: Tensor of shape [seq_len, hidden_dim]
            
        Returns:
            Dictionary with velocity, curvature, bias_alignment.
        """
        T, D = reasoning_hidden_states.shape
        if T < 2:
            return {
                "velocity": np.array([1.0]),
                "curvature": np.array([0.0]),
                "bias_alignment": np.array([0.0]),
                "sequence_length": T
            }

        h = reasoning_hidden_states.float()

        # 1. Cosine Velocity: v_t = 1 - cos_sim(h_t, h_{t-1})
        h_norm = F.normalize(h, p=2, dim=-1)
        cos_sim = (h_norm[1:] * h_norm[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        velocity = (1.0 - cos_sim).detach().cpu().numpy()

        # 2. Tangent Curvature
        deltas = h[1:] - h[:-1]
        deltas_norm = F.normalize(deltas, p=2, dim=-1)
        cos_theta = (deltas_norm[1:] * deltas_norm[:-1]).sum(dim=-1).clamp(-1.0, 1.0) if T > 2 else torch.tensor([1.0])
        curvature = torch.acos(cos_theta).detach().cpu().numpy()

        # 3. Bias Alignment Energy: || h_t @ V_bias ||^2
        bias_alignment = np.zeros(T)
        if self.V_bias is not None:
            V = self.V_bias.to(h.device).float()
            projections = torch.matmul(h_norm, V)
            bias_energy = (projections ** 2).sum(dim=-1)
            bias_alignment = bias_energy.detach().cpu().numpy()

        return {
            "velocity": velocity,
            "curvature": curvature,
            "bias_alignment": bias_alignment,
            "sequence_length": T
        }
