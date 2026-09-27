"""
In-Flight Neural Steering Engine (Zero-Scissors Paradigm).
Applies dynamic orthogonal subspace projection P_perp and directional conclusion
steering inside Transformer forward hooks during autoregressive generation.
Operates with ZERO external stopping criteria (stopping_criteria=None).
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class InFlightSteeringHook:
    """
    In-flight Transformer forward hook that:
    1. Passively allows unconstrained exploration while velocity v_t >= epsilon.
    2. Once representation stabilizes (v_t < epsilon for k consecutive tokens),
       dynamically engages:
       - Orthogonal projection P_perp = I - v_bias @ v_bias.T (prompt bias deflation)
       - Directed conclusion alignment along u_exit = (w_think - w_wait) / ||...||
    3. Causes the model's internal logits to voluntarily emit </think> and conclude
       with ZERO external stopping criteria or text-based regex scissors.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        target_layer_idx: int = 14,
        velocity_threshold: float = 0.05,
        consensus_k: int = 3,
        steering_alpha: float = 0.08,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer_idx = target_layer_idx
        self.velocity_threshold = velocity_threshold
        self.consensus_k = consensus_k
        self.steering_alpha = steering_alpha

        # Locate decoder layer
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.target_layer = model.model.layers[target_layer_idx]
        elif hasattr(model, "layers"):
            self.target_layer = model.layers[target_layer_idx]
        else:
            raise ValueError("Unable to find decoder layers.")

        # Extract conclusion unembedding direction u_exit = w_think - w_wait
        self.u_exit: Optional[torch.Tensor] = None
        self._init_conclusion_direction()

        # State tracking
        self.v_bias: Optional[torch.Tensor] = None
        self.P_perp: Optional[torch.Tensor] = None
        self.trajectory: List[torch.Tensor] = []
        self.low_velocity_streak: int = 0
        self.steering_engaged: bool = False
        self.engagement_token_idx: Optional[int] = None
        self.hook_handle = None
        self.is_active = False

    def _init_conclusion_direction(self):
        """Computes normalized difference between </think> and 'Wait' in unembedding space."""
        try:
            lm_head = self.model.lm_head
            w_u = lm_head.weight.detach() # [vocab_size, hidden_dim]

            # Find token IDs
            think_id = self.tokenizer.encode("</think>", add_special_tokens=False)[-1]
            wait_id = self.tokenizer.encode("Wait", add_special_tokens=False)[0]

            w_think = w_u[think_id].float()
            w_wait = w_u[wait_id].float()

            diff = w_think - w_wait
            norm = torch.linalg.norm(diff)
            if norm > 1e-6:
                self.u_exit = (diff / norm).to(lm_head.weight.device)
            else:
                self.u_exit = None
        except Exception as e:
            print(f"[Warning] Could not initialize u_exit direction: {e}")
            self.u_exit = None

    def set_prompt_bias(self, prompt_hidden: torch.Tensor):
        """
        Extracts terminal prompt token representation as rank-1 bias vector
        and precomputes orthogonal projector P_perp = I - v @ v.T
        """
        vec = prompt_hidden.squeeze().float()
        norm = torch.linalg.norm(vec)
        if norm > 1e-8:
            v = (vec / norm).unsqueeze(1) # [D, 1]
            D = v.shape[0]
            I = torch.eye(D, device=v.device, dtype=torch.float32)
            self.P_perp = (I - torch.matmul(v, v.T)) # [D, D]
            self.v_bias = v
        else:
            self.P_perp = None
            self.v_bias = None

    def reset(self):
        """Resets trajectory state for a new generation run."""
        self.trajectory.clear()
        self.low_velocity_streak = 0
        self.steering_engaged = False
        self.engagement_token_idx = None

    def _hook_fn(self, module, inputs, outputs):
        if not self.is_active:
            return outputs

        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        # Only process decode tokens (seq_len == 1)
        if hidden.shape[1] > 1:
            return outputs

        current_h = hidden[:, -1, :].detach().float()
        step = len(self.trajectory) + 1

        # Track velocity against previous token
        if len(self.trajectory) > 0:
            prev_h = self.trajectory[-1]
            cos_sim = F.cosine_similarity(current_h, prev_h, dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)

            # Check stabilization
            if v_t < self.velocity_threshold:
                self.low_velocity_streak += 1
                if self.low_velocity_streak >= self.consensus_k:
                    if not self.steering_engaged:
                        self.steering_engaged = True
                        self.engagement_token_idx = step
            else:
                self.low_velocity_streak = 0

        self.trajectory.append(current_h)

        # Apply in-flight steering if engaged
        if self.steering_engaged:
            h_mod = hidden[:, -1, :].clone().float()
            
            # 1. Orthogonal Subspace Deflation
            if self.P_perp is not None:
                P = self.P_perp.to(h_mod.device)
                h_mod = torch.matmul(h_mod, P)

            # 2. Directed Conclusion Alignment
            if self.u_exit is not None and self.steering_alpha > 0:
                u = self.u_exit.to(h_mod.device).unsqueeze(0)
                norm_h = torch.linalg.norm(h_mod, dim=-1, keepdim=True)
                h_mod = h_mod + self.steering_alpha * norm_h * u

            # Inject modified hidden state back into residual stream
            hidden[:, -1, :] = h_mod.to(hidden.dtype)

        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden

    def attach(self):
        """Attaches hook to target layer."""
        self.reset()
        self.is_active = True
        self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

    def detach(self):
        """Detaches hook."""
        self.is_active = False
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
