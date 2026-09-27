"""
================================================================================
EXPERIMENT: PARAMETER SCALE VALIDATION (14 BILLION PARAMETERS)
MODEL: DeepSeek-R1-Distill-Qwen-14B (48 Layers, Hidden Size 5120)
HARDWARE: NVIDIA A100-SXM4 (40GB/80GB)
PURPOSE:
    Prove that SubNeutralize scales monotonically with parameter size (1.5B -> 7B -> 14B),
    confirming that the Prompt Bias Attractor (V_bias) intensifies in higher-capacity models.
================================================================================
"""

import gc
import json
import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ------------------------------------------------------------------------------
# 1. SUBSPACE NEUTRALIZER (d_model = 5120 for 14B)
# ------------------------------------------------------------------------------
class SubspaceNeutralizer:
    def __init__(self, d_model: int = 5120, k_dim: int = 4):
        self.d_model = d_model
        self.k_dim = k_dim
        self.V_bias = None
        self.P_perp = None

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

    def project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.P_perp is None:
            return hidden_state
        orig_dtype = hidden_state.dtype
        h_f = hidden_state.float()
        h_clean = torch.matmul(h_f, self.P_perp)
        return h_clean.to(orig_dtype)

# ------------------------------------------------------------------------------
# 2. DYNAMIC GOVERNOR (Layer 24 of 48 Layers)
# ------------------------------------------------------------------------------
class DynamicReasoningGovernor:
    def __init__(
        self, 
        model: nn.Module, 
        target_layer_idx: int = 24, 
        k_dim: int = 4, 
        velocity_threshold: float = 0.05, 
        min_deduction_tokens: int = 20
    ):
        self.model = model
        self.target_layer = model.model.layers[target_layer_idx]
        self.neutralizer = SubspaceNeutralizer(d_model=model.config.hidden_size, k_dim=k_dim)
        self.velocity_threshold = velocity_threshold
        self.min_deduction_tokens = min_deduction_tokens
        
        self.trajectory = []
        self.velocities = []
        self.active_neutralization = False
        self.tokens_generated = 0
        self.trigger_token_idx = None
        self.hook_handle = None

    def reset(self):
        self.trajectory.clear()
        self.velocities.clear()
        self.active_neutralization = False
        self.tokens_generated = 0
        self.trigger_token_idx = None

    def _hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            rest = outputs[1:]
        else:
            hidden_states = outputs
            rest = None

        if hidden_states.shape[1] > 1:
            self.neutralizer.extract_prompt_subspace(hidden_states.detach())
            return outputs

        self.tokens_generated += 1
        current_h = hidden_states[:, -1, :].detach().float()

        if len(self.trajectory) > 0:
            prev_h = self.trajectory[-1]
            cos_sim = F.cosine_similarity(current_h, prev_h, dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)
            self.velocities.append(v_t)

            if self.tokens_generated >= self.min_deduction_tokens and v_t < self.velocity_threshold:
                if not self.active_neutralization:
                    self.active_neutralization = True
                    self.trigger_token_idx = self.tokens_generated
        else:
            self.velocities.append(1.0)

        self.trajectory.append(current_h)

        if self.active_neutralization:
            neutralized_h = self.neutralizer.project(hidden_states)
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

# ------------------------------------------------------------------------------
# 3. EXTRACTION & STOPPING CRITERIA
# ------------------------------------------------------------------------------
def extract_answer(text: str):
    patterns = [
        r"\\boxed\{[^\d]*(-?\d+(?:,\d+)*(?:\.\d+)?)[^\d]*\}",
        r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"[Tt]he (?:final )?answer is:?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*$"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).replace(",", "").strip()
    return None

class QwenStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.complete_box_pattern = re.compile(r"\\boxed\{[^}]+\}")
        self.hash_pattern = re.compile(r"####\s*-?\d+")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_tokens = input_ids[0, self.prompt_len:]
        if len(gen_tokens) < 20:
            return False
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
        if "</think>" in text:
            if self.complete_box_pattern.search(text) or self.hash_pattern.search(text) or "<｜end of sentence｜>" in text:
                return True
        return False
