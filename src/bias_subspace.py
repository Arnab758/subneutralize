"""
Mathematical formulations for extracting prompt bias subspaces (V_bias)
and tracking geometric trajectory dynamics (velocity, curvature, bias alignment)
in Transformer residual streams.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class BiasSubspaceAnalyzer:
    """
    Extracts the First Impression bias subspace V_bias and computes
    kinematic and geometric trajectory metrics across reasoning tokens.
    """

    def __init__(self, subspace_dim: int = 4):
        """
        Args:
            subspace_dim: Dimensionality (rank k) of the orthonormal bias subspace.
        """
        self.k = subspace_dim
        self.V_bias: Optional[torch.Tensor] = None  # Shape: [hidden_dim, k]

    def fit_prompt_bias(self, prompt_hidden_states: torch.Tensor):
        """
        Fits the orthonormal basis V_bias from prompt terminal representations
        using SVD (Singular Value Decomposition).
        
        Args:
            prompt_hidden_states: Tensor of shape [N_samples, hidden_dim]
                                  representing prompt terminal states across variations.
        """
        if prompt_hidden_states.ndim != 2:
            raise ValueError(f"Expected 2D tensor [N_samples, hidden_dim], got {prompt_hidden_states.shape}")

        # Center the representations
        mean_vec = prompt_hidden_states.mean(dim=0, keepdim=True)
        centered = prompt_hidden_states - mean_vec

        # SVD: centered = U @ S @ Vh
        # Vh has shape [hidden_dim, hidden_dim], rows are principal directions
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        
        # Take top k right singular vectors
        actual_k = min(self.k, Vh.shape[0])
        self.V_bias = Vh[:actual_k, :].T.contiguous()  # Shape: [hidden_dim, k]

    def set_bias_vector(self, single_prompt_vec: torch.Tensor):
        """
        Direct initialization when using a single prompt vector as rank-1 bias direction.
        Args:
            single_prompt_vec: Tensor of shape [hidden_dim] or [1, hidden_dim]
        """
        vec = single_prompt_vec.squeeze()
        norm = torch.linalg.norm(vec)
        if norm > 1e-8:
            self.V_bias = (vec / norm).unsqueeze(1)  # Shape: [hidden_dim, 1]
        else:
            self.V_bias = vec.unsqueeze(1)

    def compute_trajectory_metrics(
        self,
        reasoning_hidden_states: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Computes kinematic and geometric metrics across a sequence of reasoning tokens.
        
        Args:
            reasoning_hidden_states: Tensor of shape [seq_len, hidden_dim]
            
        Returns:
            Dictionary containing:
                - velocity: Cosine velocity between adjacent tokens (1 - cos_sim)
                - curvature: Angular change between consecutive velocity vectors
                - bias_alignment: Energy projected onto V_bias (if fitted)
                - length_normalized_drift: Geodesic drift scaled by sqrt(t)
        """
        T, D = reasoning_hidden_states.shape
        if T < 3:
            raise ValueError("Trajectory length must be at least 3 tokens to compute curvature.")

        device = reasoning_hidden_states.device
        h = reasoning_hidden_states.float()

        # 1. Cosine Velocity: v_t = 1 - (h_t . h_{t-1}) / (||h_t|| * ||h_{t-1}||)
        h_norm = F.normalize(h, p=2, dim=-1)
        cos_sim = (h_norm[1:] * h_norm[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        velocity = (1.0 - cos_sim).detach().cpu().numpy()

        # 2. Tangent Displacement Vectors: delta_h_t = h_t - h_{t-1}
        deltas = h[1:] - h[:-1]  # Shape: [T-1, D]
        deltas_norm = F.normalize(deltas, p=2, dim=-1)

        # 3. Local Curvature: angle between consecutive tangent directions
        cos_theta = (deltas_norm[1:] * deltas_norm[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        curvature = torch.acos(cos_theta).detach().cpu().numpy()  # in radians

        # 4. Bias Alignment: || h_t @ V_bias ||^2
        bias_alignment = None
        if self.V_bias is not None:
            V = self.V_bias.to(device).float()
            # Projection: h_norm [T, D] @ V [D, k] -> [T, k]
            projections = torch.matmul(h_norm, V)
            bias_energy = (projections ** 2).sum(dim=-1)
            bias_alignment = bias_energy.detach().cpu().numpy()

        # 5. Length-Normalized Cumulative Drift: ||h_t - h_0|| / sqrt(t + 1)
        h_init = h[0:1, :]
        displacement = torch.linalg.norm(h - h_init, dim=-1)
        time_steps = torch.sqrt(torch.arange(1, T + 1, device=device, dtype=torch.float32))
        length_normalized_drift = (displacement / time_steps).detach().cpu().numpy()

        return {
            "velocity": velocity,
            "curvature": curvature,
            "bias_alignment": bias_alignment,
            "length_normalized_drift": length_normalized_drift,
            "sequence_length": T
        }

    def compute_orthogonal_projection_matrix(self) -> torch.Tensor:
        """
        Computes the orthogonal projector P_perp = I - V_bias @ V_bias^T.
        Returns:
            Tensor of shape [hidden_dim, hidden_dim]
        """
        if self.V_bias is None:
            raise ValueError("V_bias has not been fitted.")
        
        D, k = self.V_bias.shape
        I = torch.eye(D, device=self.V_bias.device, dtype=self.V_bias.dtype)
        P_perp = I - torch.matmul(self.V_bias, self.V_bias.T)
        return P_perp
