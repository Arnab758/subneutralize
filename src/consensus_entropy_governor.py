"""
================================================================================
MODULE: consensus_entropy_governor.py
AUTHORS: Arnab Dutta et al.
TARGET: Production Engine / ICLR / NeurIPS / Y-Combinator W27

DESCRIPTION:
Couples Differential Geometry in the residual stream (Riemannian trajectory
velocity v_t across a dynamic cognitive bottleneck band) with Information
Theory at the language generation head (Instantaneous Shannon Entropy H_t).

MATHEMATICAL GOVERNING LAW:
1. Multi-Layer Consensus:
   v_t^consensus = max_{l in B} v_t^(l)
   where B is the cognitive bottleneck band around the model midpoint.
   Guarantees that generation continues if ANY layer is actively computing.

2. Information-Differential Scaling:
   H_t = -sum_{w in V} p_t(w) log p_t(w)
   epsilon_t(H_t) = clamp(epsilon_0 * (H_t / H_0), epsilon_min, epsilon_max)
   - High Certainty (H_t < 0.8 nats): Lowers threshold to epsilon_min (0.025).
     Protects fluent multi-step deduction from premature exit.
   - High Hesitation / Loop (H_t > 1.8 nats): Widens threshold to epsilon_max (0.090).
     Rapidly captures dynamical velocity stagnation and halts circular loops.

3. Pure Representation Steering (Zero-Scissors):
   P_perp = I - V_bias * V_bias^T
   h_clean = P_perp(h_last)
   If v_t^consensus < epsilon_t(H_t):
       h_clean = h_clean + lambda_exit * u_exit
================================================================================
"""

import math
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, StoppingCriteria


class ConsensusEntropyGovernor:
    """
    Production-grade Consensus Band + Shannon Entropy Coupled Governor.
    Dynamically adapts to any model depth (28 to 80+ layers).
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layer_band: Optional[List[int]] = None,
        base_threshold: float = 0.06,
        ref_entropy: float = 1.2,
        min_threshold: float = 0.025,
        max_threshold: float = 0.090,
        lambda_exit: float = 0.08,
        k_dim: int = 4
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = len(model.model.layers)
        self.d_model = model.config.hidden_size

        # If no band is passed, dynamically compute midpoint band
        if layer_band is None:
            mid = self.num_layers // 2
            # 3-layer consensus band around midpoint
            self.layer_band = [max(0, mid - 2), mid, min(self.num_layers - 1, mid + 2)]
        else:
            self.layer_band = layer_band

        self.base_threshold = base_threshold
        self.ref_entropy = ref_entropy
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.lambda_exit = lambda_exit
        self.k_dim = k_dim

        self.layers = [model.model.layers[idx] for idx in self.layer_band]
        self.hook_handles = []

        # Buffers
        self.trajectories: Dict[int, List[torch.Tensor]] = {idx: [] for idx in self.layer_band}
        self.last_velocities: Dict[int, float] = {idx: 1.0 for idx in self.layer_band}
        self.step_consensus_v: float = 1.0
        self.step_entropy: float = ref_entropy
        self.step_threshold: float = base_threshold
        self.consecutive_stagnations: int = 0
        self.tokens_generated: int = 0
        self.is_active: bool = False

        # Directional vectors
        self._init_vectors()

    def _init_vectors(self):
        """Pre-extract conclusion direction vector u_exit from embedding weights."""
        think_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        wait_ids = self.tokenizer.encode("Wait", add_special_tokens=False)
        embed_w = self.model.get_input_embeddings().weight.data

        w_think = embed_w[think_ids[0]].to(torch.float32)
        w_wait = embed_w[wait_ids[0]].to(torch.float32)
        target_device = embed_w.device
        dtype = embed_w.dtype

        self.u_exit = F.normalize(w_think - w_wait, dim=-1).to(target_device, dtype=dtype)

        bias_ids = self.tokenizer.encode("Let's think step by step", add_special_tokens=False)
        self.v_bias = F.normalize(embed_w[bias_ids].mean(dim=0).to(torch.float32), dim=-1).to(target_device, dtype=dtype)

    def reset(self):
        for idx in self.layer_band:
            self.trajectories[idx].clear()
            self.last_velocities[idx] = 1.0
        self.step_consensus_v = 1.0
        self.step_entropy = self.ref_entropy
        self.step_threshold = self.base_threshold
        self.consecutive_stagnations = 0
        self.tokens_generated = 0

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, outputs):
            if not self.is_active:
                return outputs

            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            # Skip prompt prefill
            if hidden.shape[1] > 1:
                return outputs

            curr_h = hidden[:, -1, :].clone()

            # 1. Project out prompt hesitation bias
            proj = (curr_h.float() @ self.v_bias.float().unsqueeze(-1)) * self.v_bias.float()
            h_clean = curr_h.float() - proj

            # 2. Track Layer Velocity
            traj = self.trajectories[layer_idx]
            if len(traj) > 0:
                prev_h = traj[-1]
                cos_sim = F.cosine_similarity(h_clean, prev_h, dim=-1).item()
                v_t = max(0.0, 1.0 - cos_sim)
                self.last_velocities[layer_idx] = v_t
            else:
                self.last_velocities[layer_idx] = 1.0

            traj.append(h_clean.clone())

            # 3. If Midpoint Layer and Consensus Settled -> Apply Voluntary Exit Nudge
            mid_idx = self.layer_band[len(self.layer_band) // 2]
            if layer_idx == mid_idx and self.step_consensus_v < self.step_threshold:
                h_clean = h_clean + self.lambda_exit * self.u_exit.float()

            h_clean = h_clean.to(hidden.dtype)
            if isinstance(outputs, tuple):
                return (torch.cat([hidden[:, :-1, :], h_clean.unsqueeze(1)], dim=1),) + outputs[1:]
            else:
                return torch.cat([hidden[:, :-1, :], h_clean.unsqueeze(1)], dim=1)

        return hook_fn

    def attach(self):
        self.reset()
        self.is_active = True
        for idx, layer in zip(self.layer_band, self.layers):
            self.hook_handles.append(layer.register_forward_hook(self._make_hook(idx)))

    def detach(self):
        self.is_active = False
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def update_logits(self, scores: Optional[torch.Tensor]):
        """
        Dynamically scale threshold epsilon_t using instantaneous Shannon Entropy.
        """
        self.tokens_generated += 1

        # 1. Compute Shannon Entropy H_t
        if scores is not None:
            if isinstance(scores, (list, tuple)) and len(scores) > 0:
                logits = scores[-1][0].float()
            elif isinstance(scores, torch.Tensor):
                logits = scores[-1, 0].float() if scores.ndim == 3 else (scores[0].float() if scores.ndim == 2 else scores.float())
            else:
                logits = None

            if logits is not None:
                probs = F.softmax(logits, dim=-1)
                log_p = torch.log(probs + 1e-12)
                self.step_entropy = -(probs * log_p).sum().item()
            else:
                self.step_entropy = self.ref_entropy
        else:
            self.step_entropy = self.ref_entropy

        # 2. Dynamic Threshold: epsilon_t = base * (H_t / H_0)
        ratio = self.step_entropy / max(0.01, self.ref_entropy)
        raw_threshold = self.base_threshold * ratio
        self.step_threshold = max(self.min_threshold, min(self.max_threshold, raw_threshold))

        # 3. Minimax Consensus Velocity
        self.step_consensus_v = max(self.last_velocities.values())
