"""
================================================================================
SELF-NORMALIZING Z-GOVERNOR (SubNeutralize v2.0)
Mathematical Formulation:
    Instead of a fixed dimension-dependent threshold epsilon, we dynamically track
    the streaming trajectory statistics:
        mu_t = E[v_1, ..., v_t]
        sigma_t = Std[v_1, ..., v_t]
        z_t = (v_t - mu_t) / (sigma_t + 1e-6)

    When z_t <= -1.5 (or -2.0), velocity has dropped significantly below the
    exploratory baseline, signaling mathematical deduction consensus without
    requiring manual per-architecture epsilon calibration.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfNormalizingZGovernor:
    def __init__(
        self,
        model: nn.Module,
        target_layer_idx: int,
        k_dim: int = 4,
        z_threshold: float = -1.5,
        min_deduction_tokens: int = 20,
        warmup_tokens: int = 15
    ):
        self.model = model
        self.target_layer = model.model.layers[target_layer_idx]
        self.d_model = model.config.hidden_size
        self.k_dim = k_dim
        self.z_threshold = z_threshold
        self.min_deduction_tokens = min_deduction_tokens
        self.warmup_tokens = warmup_tokens

        self.V_bias = None
        self.P_perp = None
        self.trajectory = []
        self.velocities = []
        self.active_neutralization = False
        self.tokens_generated = 0
        self.hook_handle = None

        # Welford's streaming variance algorithm
        self.count = 0
        self.mean_v = 0.0
        self.M2_v = 0.0

    def reset(self):
        self.V_bias = None
        self.P_perp = None
        self.trajectory.clear()
        self.velocities.clear()
        self.active_neutralization = False
        self.tokens_generated = 0
        self.count = 0
        self.mean_v = 0.0
        self.M2_v = 0.0

    def _update_stats(self, val: float):
        self.count += 1
        delta = val - self.mean_v
        self.mean_v += delta / self.count
        delta2 = val - self.mean_v
        self.M2_v += delta * delta2

    def get_std(self) -> float:
        if self.count < 2:
            return 1.0
        return math.sqrt(self.M2_v / (self.count - 1)) + 1e-6

    def extract_prompt_subspace(self, prompt_hidden_states: torch.Tensor):
        H = prompt_hidden_states.squeeze(0).float()
        H_centered = H - H.mean(dim=0, keepdim=True)
        try:
            _, _, V = torch.linalg.svd(H_centered, full_matrices=False)
            self.V_bias = V[:self.k_dim, :].T.to(prompt_hidden_states.device)
            I = torch.eye(self.d_model, device=prompt_hidden_states.device, dtype=torch.float32)
            self.P_perp = I - torch.matmul(self.V_bias, self.V_bias.T)
        except Exception:
            self.P_perp = torch.eye(self.d_model, device=prompt_hidden_states.device, dtype=torch.float32)

    def _hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            rest = outputs[1:]
        else:
            hidden_states = outputs
            rest = None

        if hidden_states.shape[1] > 1:
            self.extract_prompt_subspace(hidden_states.detach())
            return outputs

        self.tokens_generated += 1
        current_h = hidden_states[:, -1, :].detach().float()

        if len(self.trajectory) > 0:
            prev_h = self.trajectory[-1]
            cos_sim = F.cosine_similarity(current_h, prev_h, dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)
            self.velocities.append(v_t)

            # Warmup online statistics
            if self.tokens_generated <= self.warmup_tokens:
                self._update_stats(v_t)
            else:
                std_v = self.get_std()
                z_t = (v_t - self.mean_v) / std_v

                if self.tokens_generated >= self.min_deduction_tokens and z_t <= self.z_threshold:
                    self.active_neutralization = True
        else:
            self.velocities.append(1.0)

        self.trajectory.append(current_h)

        if self.active_neutralization and self.P_perp is not None:
            orig_dtype = hidden_states.dtype
            h_f = hidden_states.float()
            neutralized_h = torch.matmul(h_f, self.P_perp).to(orig_dtype)
            if rest is not None:
                return (neutralized_h,) + rest
            return neutralized_h

        return outputs

    def attach(self):
        self.reset()
        self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

    def detach(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
