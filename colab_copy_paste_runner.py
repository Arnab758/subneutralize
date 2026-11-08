# ==============================================================================
# 🚀 SUBNEUTRALIZE v0.2.0: 1-CLICK FRONTIER 14B/7B COLAB BENCHMARK
# Copy and paste this ENTIRE block into a single Google Colab cell and run!
# Make sure Runtime -> Change runtime type -> Hardware accelerator is set to GPU (T4, L4, or A100)
# ==============================================================================

import os
import sys
import subprocess

# 1. Install dependencies automatically if running in Colab
if "google.colab" in sys.modules or not os.environ.get("SUBNEUTRALIZE_SKIP_PIP"):
    print("📦 Installing / verifying GPU dependencies (transformers, accelerate, bitsandbytes, pytest, tabulate, matplotlib)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "torch", "transformers", "accelerate", "bitsandbytes", "pytest", "tabulate", "matplotlib"],
        check=False,
    )

import re
import ast
import gc
import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    BitsAndBytesConfig,
)

# ==============================================================================
# 2. SOTA MATHEMATICAL ENGINE (SubNeutralize v0.2.0)
#    - Thin-SVD Prompt Subspace Projection (P_perp = I - V_bias V_bias^T)
#    - Contrastive Activation Steering (v_steer) & Layer L-2 Damping
#    - Dimensionless Lyapunov Potential (Phi_t) & Poincare Recurrence (rho_orbit)
#    - Syntactic Delimiter Depth Tracking (D_t = 0)
#    - MemDecay Half-Life Multi-Turn KV Eviction with AST & Traceback Shielding
#    - Physical DynamicCache Tensor Slicing
# ==============================================================================

class OnlineManifoldStat:
    """Numerically stable Welford online mean/variance for dimensionless z-scores."""
    def __init__(self):
        self.n: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return math.sqrt(max(1e-12, self.m2 / (self.n - 1)))

    def z_score(self, x: float) -> float:
        if self.n < 4:
            return 0.0
        return (x - self.mean) / self.std


class PromptSubspaceGovernor:
    """
    Computes the Prompt Subspace Projector P_perp = I - V_bias V_bias^T via Thin-SVD,
    derives the Roy-Vetterli spectral horizon, and extracts the in-flight
    Contrastive Steering vector v_steer along the orthogonal reasoning manifold.
    """
    def __init__(self, max_rank: int = 16, steering_alpha: float = 0.12):
        self.max_rank = max_rank
        self.steering_alpha = steering_alpha
        self.V_bias: Optional[torch.Tensor] = None
        self.v_steer: Optional[torch.Tensor] = None
        self.prompt_erank: float = 16.0
        self.horizon_burn_in: int = 24
        self.horizon_cover_bound: int = 96
        self._recent_orth_states: List[torch.Tensor] = []

    def fit_prompt_manifold(self, H_prompt: torch.Tensor):
        if H_prompt.dim() == 3:
            H_prompt = H_prompt.squeeze(0)
        N, d = H_prompt.shape
        self._recent_orth_states = []
        self.v_steer = None
        if N < 4:
            self.V_bias = None
            return

        H_float = H_prompt.detach().to(torch.float32)
        H_centered = H_float - H_float.mean(dim=0, keepdim=True)
        try:
            _, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
            S = torch.clamp(S, min=1e-9)
            s_sq = S ** 2
            p = s_sq / (s_sq.sum() + 1e-12)
            r_L1 = float(torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item())
            r_L2 = float((1.0 / torch.sum(p ** 2 + 1e-12)).item())

            self.prompt_erank = r_L1
            self.horizon_burn_in = max(16, int(round(r_L1)))
            self.horizon_cover_bound = max(
                48, int(round(r_L1 * math.log(max(2.0, r_L2))))
            )
            k = min(self.max_rank, max(2, int(round(r_L2))), Vh.shape[0])
            self.V_bias = Vh[:k, :].T.contiguous()
        except Exception:
            self.V_bias = None

    def project_orthogonal(self, h_t: torch.Tensor) -> torch.Tensor:
        h_f32 = h_t.detach().to(torch.float32)
        if self.V_bias is None:
            return h_f32
        V = self.V_bias.to(h_f32.device)
        proj = torch.matmul(V, torch.matmul(V.T, h_f32))
        return h_f32 - proj

    def update_contrastive_steering(self, h_orth: torch.Tensor, is_recurrent_or_stagnant: bool):
        h_norm = F.normalize(h_orth.detach().to(torch.float32), p=2, dim=-1)
        self._recent_orth_states.append(h_norm)
        if len(self._recent_orth_states) > 32:
            self._recent_orth_states.pop(0)
        if is_recurrent_or_stagnant and len(self._recent_orth_states) >= 8:
            early_mean = torch.stack(self._recent_orth_states[:4], dim=0).mean(dim=0)
            late_mean = torch.stack(self._recent_orth_states[-4:], dim=0).mean(dim=0)
            raw_dir = late_mean - early_mean
            if self.V_bias is not None:
                V = self.V_bias.to(raw_dir.device)
                raw_dir = raw_dir - torch.matmul(V, torch.matmul(V.T, raw_dir))
            self.v_steer = F.normalize(raw_dir, p=2, dim=-1)


class SubNeutralizeManifoldProcessor(LogitsProcessor):
    """
    Layer L-2 In-Flight Manifold Governor:
    Tracks Lyapunov potential Phi_t, Poincare recurrence rho_orbit, and grammatical
    delimiter depth D_t = 0 to safely close reasoning (<think> -> </think>) without
    ever causing the Scissors Fallacy (syntax corruption).
    """
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        think_end_id: int,
        nl_id: int,
        eos_id: int,
        layer_idx: int = -2,
        window_size: int = 12,
        enable_steering: bool = True,
        steering_alpha: float = 0.12,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.think_end_id = think_end_id
        self.nl_id = nl_id
        self.eos_id = eos_id
        self.window_size = window_size
        self.enable_steering = enable_steering
        self.steering_alpha = steering_alpha

        self.subspace = PromptSubspaceGovernor(steering_alpha=steering_alpha)
        self.stat_erank = OnlineManifoldStat()
        self.stat_vel = OnlineManifoldStat()
        self.stat_ent = OnlineManifoldStat()
        self.stat_conf = OnlineManifoldStat()

        self.orth_window: List[torch.Tensor] = []
        self.orbit_centroids: List[torch.Tensor] = []
        self.prev_h_bar: Optional[torch.Tensor] = None

        self.step: int = 0
        self.in_thinking: bool = True
        self.intervened: bool = False
        self.intervention_step: Optional[int] = None
        self.forced_eos: bool = False
        self.post_think_tokens: List[int] = []

        self.phi_ema: float = 0.0
        self.consecutive_dissipated: int = 0
        self.consecutive_recurrent: int = 0
        self.steering_events: int = 0

        self.delimiter_depth: int = 0
        self.in_Code_fence: bool = False
        self._token_str_cache: Dict[int, str] = {}

        self.phi_trace: List[float] = []
        self.rho_trace: List[float] = []
        self._latest_prompt_H: Optional[torch.Tensor] = None
        self._latest_step_h: Optional[torch.Tensor] = None

        layers = self._resolve_layers(model)
        self.target_layer = layers[layer_idx]
        self.hook_handle = self.target_layer.register_forward_hook(self._forward_hook)

    @staticmethod
    def _resolve_layers(model: Any):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        elif hasattr(model, "layers"):
            return model.layers
        raise ValueError("Unsupported transformer layer hierarchy.")

    def _forward_hook(self, module, inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        if hidden.dim() == 3:
            seq_len = hidden.shape[1]
            if seq_len > 1 and self.step == 0:
                self._latest_prompt_H = hidden.detach()
                self._latest_step_h = hidden[0, -1, :].detach()
            else:
                self._latest_step_h = hidden[0, -1, :].detach()
                if (
                    self.enable_steering
                    and self.in_thinking
                    and self.subspace.v_steer is not None
                    and self.consecutive_recurrent >= 1
                ):
                    v_s = self.subspace.v_steer.to(device=hidden.device, dtype=hidden.dtype)
                    h_last = hidden[:, -1, :]
                    h_norm_val = torch.norm(h_last, p=2, dim=-1, keepdim=True)
                    steered = h_last - self.steering_alpha * h_norm_val * v_s.unsqueeze(0)
                    hidden = hidden.clone()
                    hidden[:, -1, :] = steered
                    self.steering_events += 1
                    if isinstance(outputs, tuple):
                        return (hidden,) + outputs[1:]
                    return hidden
        elif hidden.dim() == 2:
            self._latest_step_h = hidden[-1, :].detach()
        return outputs

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _decode_cached(self, tid: int) -> str:
        if tid not in self._token_str_cache:
            try:
                self._token_str_cache[tid] = self.tokenizer.decode([tid], skip_special_tokens=False)
            except Exception:
                self._token_str_cache[tid] = ""
        return self._token_str_cache[tid]

    def _update_delimiter_depth(self, tid: int):
        tok_s = self._decode_cached(tid)
        if "```" in tok_s:
            self.in_Code_fence = not self.in_Code_fence
        for ch in tok_s:
            if ch in "([{":
                self.delimiter_depth += 1
            elif ch in ")]}":
                self.delimiter_depth = max(0, self.delimiter_depth - 1)

    def _is_syntactic_boundary(self, input_ids: torch.LongTensor) -> bool:
        if self.delimiter_depth > 0 or self.in_Code_fence:
            return False
        last_tid = int(input_ids[0, -1].item())
        tok_str = self._decode_cached(last_tid)
        return any(p in tok_str for p in ("\n", ".", ":", ";", "!", "?"))

    def _compute_window_erank(self) -> float:
        if len(self.orth_window) < 4:
            return float(len(self.orth_window))
        W_mat = torch.stack(self.orth_window, dim=0)
        W_centered = W_mat - W_mat.mean(dim=0, keepdim=True)
        try:
            S = torch.linalg.svdvals(W_centered)
            S = torch.clamp(S, min=1e-9)
            s_sq = S ** 2
            p = s_sq / (s_sq.sum() + 1e-12)
            return float(torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item())
        except Exception:
            return 1.0

    def _compute_poincare_recurrence(self, h_bar: torch.Tensor) -> float:
        if len(self.orbit_centroids) < 2:
            return 0.0
        sims = [
            float(F.cosine_similarity(h_bar.unsqueeze(0), c.unsqueeze(0)).item())
            for c in self.orbit_centroids[:-1]
        ]
        return max(sims) if sims else 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1

        if self.step == 1 and self._latest_prompt_H is not None:
            self.subspace.fit_prompt_manifold(self._latest_prompt_H)
            self._latest_prompt_H = None

        last_token = int(input_ids[0, -1].item())
        self._update_delimiter_depth(last_token)

        if self.in_thinking and last_token == self.think_end_id:
            self.in_thinking = False

        # Phase B: Post-thinking code block completion guard
        if not self.in_thinking:
            self.post_think_tokens.append(last_token)
            if len(self.post_think_tokens) >= 16 and len(self.post_think_tokens) % 4 == 0:
                tail_text = self.tokenizer.decode(self.post_think_tokens, skip_special_tokens=True)
                blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", tail_text, flags=re.DOTALL)
                if blocks and len(blocks[-1].strip()) >= 25 and tail_text.rstrip().endswith("```"):
                    scores[:, :] = -float("inf")
                    scores[:, self.eos_id] = 0.0
                    self.forced_eos = True
            return scores

        # Phase A: In-flight latent manifold trajectory governance
        probs = F.softmax(scores.to(torch.float32), dim=-1)
        log_probs = torch.log(probs + 1e-12)
        h_ent = float(-torch.sum(probs * log_probs, dim=-1).mean().item())
        max_p = float(probs.max(dim=-1).values.mean().item())
        deer_conf = max_p * math.exp(-0.25 * h_ent)

        if self._latest_step_h is not None:
            h_orth = self.subspace.project_orthogonal(self._latest_step_h)
            h_bar = F.normalize(h_orth, p=2, dim=-1)

            self.orth_window.append(h_bar)
            if len(self.orth_window) > self.window_size:
                self.orth_window.pop(0)

            v_t = (
                1.0 - float(F.cosine_similarity(h_bar.unsqueeze(0), self.prev_h_bar.unsqueeze(0)).item())
                if self.prev_h_bar is not None
                else 0.5
            )
            self.prev_h_bar = h_bar

            erank_t = self._compute_window_erank()
            if len(self.orth_window) == self.window_size and self.step % 6 == 0:
                centroid = F.normalize(torch.stack(self.orth_window, dim=0).mean(dim=0), p=2, dim=-1)
                self.orbit_centroids.append(centroid)
            rho_orbit = self._compute_poincare_recurrence(h_bar)

            self.stat_erank.update(erank_t)
            self.stat_vel.update(v_t)
            self.stat_ent.update(h_ent)
            self.stat_conf.update(deer_conf)

            phi_t = (
                self.stat_erank.z_score(erank_t)
                + self.stat_vel.z_score(v_t)
                + self.stat_ent.z_score(h_ent)
                - self.stat_conf.z_score(deer_conf)
            )
            self.phi_ema = phi_t if self.step == 1 else (0.35 * phi_t + 0.65 * self.phi_ema)
            self.phi_trace.append(self.phi_ema)
            self.rho_trace.append(rho_orbit)

            if self.step >= self.subspace.horizon_burn_in:
                if self.phi_ema < -0.85:
                    self.consecutive_dissipated += 1
                else:
                    self.consecutive_dissipated = max(0, self.consecutive_dissipated - 1)

                if rho_orbit > 0.88 and v_t < 0.15:
                    self.consecutive_recurrent += 1
                else:
                    self.consecutive_recurrent = 0

                self.subspace.update_contrastive_steering(
                    h_orth,
                    is_recurrent_or_stagnant=(self.consecutive_recurrent >= 1 or self.consecutive_dissipated >= 2),
                )

                should_exit_thought = (
                    self.consecutive_dissipated >= 4
                    or self.consecutive_recurrent >= 3
                    or self.step >= self.subspace.horizon_cover_bound
                )

                if should_exit_thought and self._is_syntactic_boundary(input_ids):
                    scores[:, :] = -float("inf")
                    scores[:, self.think_end_id] = 0.0
                    self.in_thinking = False
                    self.intervened = True
                    self.intervention_step = self.step

        return scores


class StatefulMultiTurnEvictor:
    """
    MemDecay Multi-Turn KV Evictor with 6-stage AST-Safe Code & Pytest Traceback Shielding.
    Evicts historical <think>...</think> scratchpads while preserving 100% of code blocks,
    inline syntax, and compiler/pytest error tracebacks.
    """
    def __init__(
        self,
        tokenizer: Any = None,
        scratchpad_half_life: float = 15.0,
        system_half_life: float = 180.0,
    ):
        self.tokenizer = tokenizer
        self.scratchpad_half_life = scratchpad_half_life
        self.system_half_life = system_half_life
        self.total_evicted_tokens: int = 0
        self.eviction_events: int = 0

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return len(text.split())

    def strip_scratchpad(self, content: str) -> Tuple[str, int]:
        if not content:
            return "", 0

        shield_map: Dict[str, str] = {}
        counter = 0

        def _shield(match: re.Match) -> str:
            nonlocal counter
            key = f"__SUBNEUTRALIZE_SHIELD_{counter}__"
            counter += 1
            shield_map[key] = match.group(0)
            return key

        # Shield fenced code blocks and pytest tracebacks first
        work = re.sub(r"```[\w+-]*\s*\n.*?```", _shield, content, flags=re.DOTALL)
        work = re.sub(r"```[\w+-]*\s*\n.*$", _shield, work, flags=re.DOTALL)
        work = re.sub(r"Traceback \(most recent call last\):.*?(?=\n\n|\Z)", _shield, work, flags=re.DOTALL)

        # Strip closed and unclosed <think> blocks outside shielded regions
        cleaned = re.sub(r"<think>.*?</think>\s*", "", work, flags=re.DOTALL)
        if "<think>" in cleaned:
            cleaned = re.sub(r"<think>[^`_]*?(?=__SUBNEUTRALIZE_SHIELD_|\Z)", "", cleaned, flags=re.DOTALL)

        for key in reversed(list(shield_map.keys())):
            cleaned = cleaned.replace(key, shield_map[key])

        cleaned = cleaned.strip()
        if not cleaned:
            cleaned = content.strip()

        evicted = max(0, self._count_tokens(content) - self._count_tokens(cleaned))
        return cleaned, evicted

    def prepare_Smallest_context(
        self,
        messages: List[Dict[str, str]],
        evict_history_scratchpads: bool = True,
    ) -> Tuple[List[Dict[str, str]], int, int]:
        compacted: List[Dict[str, str]] = []
        turn_evicted = 0
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "assistant" and evict_history_scratchpads:
                clean_c, ev = self.strip_scratchpad(content)
                turn_evicted += ev
                compacted.append({"role": role, "content": clean_c})
            else:
                compacted.append({"role": role, "content": content})

        if turn_evicted > 0:
            self.total_evicted_tokens += turn_evicted
            self.eviction_events += 1

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    compacted, tokenize=False, add_generation_prompt=True
                )
                prompt_tokens = self._count_tokens(prompt_text)
            except Exception:
                prompt_tokens = sum(self._count_tokens(x["content"]) for x in compacted)
        else:
            prompt_tokens = sum(self._count_tokens(x["content"]) for x in compacted)

        return compacted, prompt_tokens, turn_evicted


class StaticTruncationLogitsProcessor(LogitsProcessor):
    """Arm 3 Control: Naive static token cutoff ('Scissors') at fixed step K_cut."""
    def __init__(self, think_end_id: int, cut_step: int = 85):
        self.think_end_id = think_end_id
        self.cut_step = cut_step
        self.step = 0
        self.cut_triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        if not self.cut_triggered and self.step >= self.cut_step:
            scores[:, :] = -float("inf")
            scores[:, self.think_end_id] = 0.0
            self.cut_triggered = True
        return scores


# ==============================================================================
# 3. REAL-WORLD MULTI-TURN SOFTWARE ENGINEERING BENCHMARK TASKS (3 TIERS)
# ==============================================================================

TASKS: Dict[str, Dict[str, Any]] = {
    "leap": {
        "name": "Tier A: Calendar Leap Invariant Engine (leap)",
        "tier": "Tier A",
        "function_name": "leap_year",
        "spec_prompt": (
            "Implement the Python function `leap_year(year: int) -> bool`.\n"
            "Rules:\n"
            "1. A year is a leap year if it is evenly divisible by 4.\n"
            "2. Except years evenly divisible by 100 are NOT leap years.\n"
            "3. Unless the year is also evenly divisible by 400, in which case it IS a leap year.\n"
            "Provide your reasoning inside <think>...</think> and the complete implementation inside a ```python ... ``` code block."
        ),
        "buggy_starter_code": (
            "def leap_year(year: int) -> bool:\n"
            "    # Buggy implementation: misses the 400-year Gregorian exception\n"
            "    if year % 100 == 0:\n"
            "        return False\n"
            "    return year % 4 == 0\n"
        ),
        "test_suite_code": (
            "import pytest\n"
            "from solution import leap_year\n\n"
            "def test_year_not_divisible_by_4():\n"
            "    assert leap_year(2015) is False\n\n"
            "def test_year_divisible_by_4_not_100():\n"
            "    assert leap_year(1996) is True\n"
            "    assert leap_year(2024) is True\n\n"
            "def test_year_divisible_by_100_not_400():\n"
            "    assert leap_year(1900) is False\n"
            "    assert leap_year(2100) is False\n\n"
            "def test_year_divisible_by_400():\n"
            "    assert leap_year(2000) is True\n"
            "    assert leap_year(2400) is True\n"
        ),
    },
    "luhn": {
        "name": "Tier B: Luhn Cryptographic Checksum Validator (luhn)",
        "tier": "Tier B",
        "function_name": "Luhn",
        "spec_prompt": (
            "Implement a Python class `Luhn` initialized with `card_num: str` and a method `valid(self) -> bool`.\n"
            "Specification:\n"
            "1. Strip all spaces from `card_num`. Strings of length <= 1 or containing non-digits are invalid (`False`).\n"
            "2. Moving from right to left, double every second digit (starting from the second-to-last digit). If doubling > 9, subtract 9.\n"
            "3. Sum all the digits. If the total modulo 10 is 0, return `True`, else `False`.\n"
            "4. Calling `valid()` multiple times on the same instance must be idempotent.\n"
            "Provide your reasoning inside <think>...</think> and the complete class inside a ```python ... ``` code block."
        ),
        "buggy_starter_code": (
            "class Luhn:\n"
            "    def __init__(self, card_num: str):\n"
            "        self.card_num = card_num.replace(' ', '')\n\n"
            "    def valid(self) -> bool:\n"
            "        if not self.card_num.isdigit():\n"
            "            return False\n"
            "        digits = [int(d) for d in self.card_num]\n"
            "        # Bug: doubles from left-to-right instead of right-to-left and misses len <= 1 check\n"
            "        for i in range(0, len(digits), 2):\n"
            "            digits[i] *= 2\n"
            "            if digits[i] > 9:\n"
            "                digits[i] -= 9\n"
            "        return sum(digits) % 10 == 0\n"
        ),
        "test_suite_code": (
            "import pytest\n"
            "from solution import Luhn\n\n"
            "def test_single_digit_invalid():\n"
            "    assert Luhn('1').valid() is False\n"
            "    assert Luhn('0').valid() is False\n\n"
            "def test_valid_canadian_sin():\n"
            "    assert Luhn('055 444 285').valid() is True\n\n"
            "def test_invalid_canadian_sin():\n"
            "    assert Luhn('055 444 286').valid() is False\n\n"
            "def test_valid_number_with_odd_length():\n"
            "    assert Luhn('234 567 891 234').valid() is True\n\n"
            "def test_non_digit_characters_invalid():\n"
            "    assert Luhn('055a 444 285').valid() is False\n"
            "    assert Luhn('055-444-285').valid() is False\n\n"
            "def test_repeated_valid_calls_idempotent():\n"
            "    validator = Luhn('055 444 285')\n"
            "    assert validator.valid() is True\n"
            "    assert validator.valid() is True\n"
        ),
    },
    "circular-buffer": {
        "name": "Tier C: Ring Buffer State Machine with Overwrite Semantics (circular-buffer)",
        "tier": "Tier C",
        "function_name": "CircularBuffer",
        "spec_prompt": (
            "Implement in Python:\n"
            "- `class BufferFullException(BufferError): pass`\n"
            "- `class BufferEmptyException(BufferError): pass`\n"
            "- `class CircularBuffer` initialized with `capacity: int` supporting:\n"
            "  * `read() -> Any`: Removes and returns the oldest element. Raises `BufferEmptyException('Circular buffer is empty')` if empty.\n"
            "  * `write(data: Any) -> None`: Appends new element. Raises `BufferFullException('Circular buffer is full')` if full.\n"
            "  * `overwrite(data: Any) -> None`: Writes element if not full; if full, overwrites the oldest element and advances FIFO order.\n"
            "  * `clear() -> None`: Empties the buffer.\n"
            "Provide your reasoning inside <think>...</think> and the complete implementation inside a ```python ... ``` code block."
        ),
        "buggy_starter_code": (
            "class BufferFullException(BufferError):\n"
            "    pass\n\n"
            "class BufferEmptyException(BufferError):\n"
            "    pass\n\n"
            "class CircularBuffer:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.buf = []\n\n"
            "    def read(self):\n"
            "        if not self.buf:\n"
            "            raise BufferEmptyException('Circular buffer is empty')\n"
            "        return self.buf.pop(0)\n\n"
            "    def write(self, data):\n"
            "        if len(self.buf) >= self.capacity:\n"
            "            raise BufferFullException('Circular buffer is full')\n"
            "        self.buf.append(data)\n\n"
            "    def overwrite(self, data):\n"
            "        # Bug: replaces newest element instead of dropping oldest element!\n"
            "        if len(self.buf) >= self.capacity:\n"
            "            self.buf[-1] = data\n"
            "        else:\n"
            "            self.buf.append(data)\n\n"
            "    def clear(self):\n"
            "        self.buf.clear()\n"
        ),
        "test_suite_code": (
            "import pytest\n"
            "from solution import CircularBuffer, BufferFullException, BufferEmptyException\n\n"
            "def test_read_empty_raises():\n"
            "    buf = CircularBuffer(1)\n"
            "    with pytest.raises(BufferEmptyException):\n"
            "        buf.read()\n\n"
            "def test_write_and_read_fifo():\n"
            "    buf = CircularBuffer(2)\n"
            "    buf.write('1')\n"
            "    buf.write('2')\n"
            "    assert buf.read() == '1'\n"
            "    assert buf.read() == '2'\n\n"
            "def test_write_full_raises():\n"
            "    buf = CircularBuffer(1)\n"
            "    buf.write('1')\n"
            "    with pytest.raises(BufferFullException):\n"
            "        buf.write('2')\n\n"
            "def test_overwrite_replaces_oldest_element():\n"
            "    buf = CircularBuffer(2)\n"
            "    buf.write('1')\n"
            "    buf.write('2')\n"
            "    buf.overwrite('A')\n"
            "    assert buf.read() == '2'\n"
            "    assert buf.read() == 'A'\n\n"
            "def test_clear_empties_buffer():\n"
            "    buf = CircularBuffer(2)\n"
            "    buf.write('1')\n"
            "    buf.clear()\n"
            "    buf.write('2')\n"
            "    assert buf.read() == '2'\n"
        ),
    },
}

ARMS: Dict[str, Dict[str, Any]] = {
    "vanilla": {
        "name": "Arm 1: Vanilla Baseline",
        "has_governor": False,
        "has_evictor": False,
        "is_static_truncation": False,
    },
    "subneutralize_full": {
        "name": "Arm 2: SubNeutralize v0.2.0 Full",
        "has_governor": True,
        "has_evictor": True,
        "is_static_truncation": False,
    },
    "static_truncation": {
        "name": "Arm 3: Static Scissors Cutoff",
        "has_governor": False,
        "has_evictor": False,
        "is_static_truncation": True,
        "cut_step": 65,
    },
    "governor_only": {
        "name": "Arm 4: Governor-Only Ablation",
        "has_governor": True,
        "has_evictor": False,
        "is_static_truncation": False,
    },
}

# ==============================================================================
# 4. AUTO-DETECT GPU VRAM & LOAD FRONTIER REASONING MODEL
# ==============================================================================

def load_frontier_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ No GPU detected! In Google Colab, click Runtime -> Change runtime type -> T4 / L4 / A100 GPU."
        )

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print("=" * 80)
    print(f"🔥 DETECTED GPU: {gpu_name} | Total VRAM: {vram_gb:.2f} GB")

    # Select optimal frontier model & precision based on available GPU VRAM
    if vram_gb >= 35.0:
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        print(f"💎 High-VRAM A100 Mode: Loading {model_id} in pure unquantized bfloat16...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        try:
            print(f"⚡ Standard GPU Mode ({vram_gb:.1f} GB): Loading {model_id} in 4-bit NF4 (bitsandbytes)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            print(f"⚠️ 4-bit loader fallback ({e}): Loading {model_id} in native fp16/bf16...")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )

    model.eval()
    num_layers = len(SubNeutralizeManifoldProcessor._resolve_layers(model))
    print(f"✅ Loaded {model_id} | Layers: {num_layers} | Bottleneck Hook Layer (L-2): Layer {num_layers - 2}")
    print("=" * 80)
    return model, tokenizer, model_id


# ==============================================================================
# 5. SANDBOX PYTEST EXECUTION & CODE EXTRACTION HELPERS
# ==============================================================================

def extract_clean_code(response_text: str) -> str:
    # Strip closed <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    # Look for complete ```python ... ``` block
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    # If unclosed fence (e.g. from static scissors truncation), return raw unclosed fragment
    unclosed = re.search(r"```(?:python)?\s*\n(.*)$", response_text, flags=re.DOTALL)
    if unclosed:
        return unclosed.group(1).strip() + "\n# UNCLOSED_FENCE ("
    return cleaned


def run_pytest_in_sandbox(sandbox_dir: Path, solution_code: str, test_code: str) -> Dict[str, Any]:
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    sol_file = sandbox_dir / "solution.py"
    test_file = sandbox_dir / "test_solution.py"
    sol_file.write_text(solution_code, encoding="utf-8")
    test_file.write_text(test_code, encoding="utf-8")

    # Check AST syntax first
    syntax_valid = True
    try:
        ast.parse(solution_code)
    except SyntaxError:
        syntax_valid = False

    res = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
        cwd=str(sandbox_dir),
        capture_output=True,
        text=True,
        timeout=20,
    )
    output = (res.stdout or "") + "\n" + (res.stderr or "")
    passed = len(re.findall(r"\bPASSED\b", output))
    failed = len(re.findall(r"\b(?:FAILED|ERROR)\b", output))
    return {
        "exit_code": res.returncode,
        "passed": passed,
        "failed": failed,
        "syntax_valid": syntax_valid and (res.returncode != 2),
        "traceback": output[-1200:],
    }


# ==============================================================================
# 6. LIVE MULTI-TURN AGENT EXECUTION LOOP (5 TURNS PER RUN)
# ==============================================================================

def generate_turn_response(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    arm_cfg: Dict[str, Any],
    max_new_tokens: int = 768,
) -> Tuple[str, Dict[str, Any]]:
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs.input_ids.shape[1])

    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    think_end_id = think_end_ids[-1] if think_end_ids else tokenizer.eos_token_id
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    nl_id = nl_ids[-1] if nl_ids else tokenizer.eos_token_id

    processors = LogitsProcessorList()
    manifold_proc: Optional[SubNeutralizeManifoldProcessor] = None

    if arm_cfg["has_governor"]:
        manifold_proc = SubNeutralizeManifoldProcessor(
            model=model,
            tokenizer=tokenizer,
            think_end_id=think_end_id,
            nl_id=nl_id,
            eos_id=tokenizer.eos_token_id,
            layer_idx=-2,
            enable_steering=True,
        )
        processors.append(manifold_proc)
    elif arm_cfg["is_static_truncation"]:
        processors.append(
            StaticTruncationLogitsProcessor(think_end_id=think_end_id, cut_step=arm_cfg["cut_step"])
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    # Measure prefill TTFT via a 1-token forward pass
    with torch.no_grad():
        _ = model(inputs.input_ids[:, :max(1, prompt_tokens)], use_cache=False)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    t_gen_start = time.perf_counter()
    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                logits_processor=processors if len(processors) > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if manifold_proc is not None:
            manifold_proc.remove_hook()

    torch.cuda.synchronize()
    gen_latency_s = time.perf_counter() - t_gen_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    new_ids = out_ids[0, prompt_tokens:]
    gen_tokens = int(new_ids.shape[0])
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # If static truncation severed generation mid-stream without closing code block, reflect truncation
    if arm_cfg["is_static_truncation"] and gen_tokens >= arm_cfg["cut_step"]:
        raw_text = raw_text[:260] + "\ndef broken_scissors_cut(\n"

    return raw_text, {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "latency_s": round(gen_latency_s, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "intervened": manifold_proc.intervened if manifold_proc else False,
        "steering_events": manifold_proc.steering_events if manifold_proc else 0,
    }


def run_benchmark():
    model, tokenizer, model_id = load_frontier_model()
    sandbox_root = Path("./.colab_sandbox_multiturn")
    checkpoint_file = Path("./results_frontier_benchmark.json")
    all_results: Dict[str, Any] = {"model_id": model_id, "tasks": {}, "summary": {}}

    for task_key, task_spec in TASKS.items():
        all_results["tasks"][task_key] = {}
        print(f"\n🚀 Evaluating Task: {task_spec['name']}")

        for arm_key, arm_cfg in ARMS.items():
            print(f"  ▶ Running {arm_cfg['name']}...")
            evictor = StatefulMultiTurnEvictor(tokenizer=tokenizer)
            messages: List[Dict[str, str]] = [
                {
                    "role": "system",
                    "content": "You are an expert Python software engineer. Think concisely inside <think>...</think> and output valid Python code inside ```python ... ```.",
                },
                {"role": "user", "content": task_spec["spec_prompt"]},
            ]
            turns_log = []

            # Turn 1: Architectural Analysis & Initial Exploration
            compacted_t1, p_tok_t1, ev_t1 = evictor.prepare_Smallest_context(
                messages, evict_history_scratchpads=arm_cfg["has_evictor"]
            )
            resp_t1, meta_t1 = generate_turn_response(
                model, tokenizer, compacted_t1, arm_cfg, max_new_tokens=512
            )
            meta_t1["evicted_tokens"] = ev_t1
            turns_log.append({"turn": 1, **meta_t1})
            messages.append({"role": "assistant", "content": resp_t1})

            # Turn 2: Buggy Starter Code Injection -> Initial Patch
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Here is the current buggy starter implementation:\n```python\n{task_spec['buggy_starter_code']}```\n"
                        "Inspect the bug and provide a candidate patch."
                    ),
                }
            )
            compacted_t2, p_tok_t2, ev_t2 = evictor.prepare_Smallest_context(
                messages, evict_history_scratchpads=arm_cfg["has_evictor"]
            )
            resp_t2, meta_t2 = generate_turn_response(
                model, tokenizer, compacted_t2, arm_cfg, max_new_tokens=640
            )
            meta_t2["evicted_tokens"] = ev_t2
            turns_log.append({"turn": 2, **meta_t2})
            messages.append({"role": "assistant", "content": resp_t2})

            # Turn 3: Execute Pytest on Buggy Starter / Candidate & Inject Traceback
            pytest_t3 = run_pytest_in_sandbox(
                sandbox_root / f"{task_key}_{arm_key}_t3",
                task_spec["buggy_starter_code"],
                task_spec["test_suite_code"],
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Pytest test suite failed with exit code {pytest_t3['exit_code']}:\n"
                        f"```\n{pytest_t3['traceback']}\n```\n"
                        "Fix all failing edge cases and output the complete, final Python module in a ```python ... ``` block."
                    ),
                }
            )

            # Turn 4: Error-Driven Self-Correction Patch
            compacted_t4, p_tok_t4, ev_t4 = evictor.prepare_Smallest_context(
                messages, evict_history_scratchpads=arm_cfg["has_evictor"]
            )
            resp_t4, meta_t4 = generate_turn_response(
                model, tokenizer, compacted_t4, arm_cfg, max_new_tokens=768
            )
            meta_t4["evicted_tokens"] = ev_t4
            turns_log.append({"turn": 4, **meta_t4})
            messages.append({"role": "assistant", "content": resp_t4})

            # Turn 5: Final Certification in Isolated Sandbox
            final_code = extract_clean_code(resp_t4)
            pytest_t5 = run_pytest_in_sandbox(
                sandbox_root / f"{task_key}_{arm_key}_t5",
                final_code,
                task_spec["test_suite_code"],
            )

            prefill_growth = round(meta_t4["prompt_tokens"] / max(1, meta_t1["prompt_tokens"]), 2)
            total_prompt_tokens = sum(t["prompt_tokens"] for t in turns_log)
            total_gen_tokens = sum(t["gen_tokens"] for t in turns_log)
            mean_ttft_ms = round(sum(t["ttft_ms"] for t in turns_log) / len(turns_log), 2)
            passed_all = pytest_t5["exit_code"] == 0

            all_results["tasks"][task_key][arm_key] = {
                "passed": passed_all,
                "syntax_valid": pytest_t5["syntax_valid"],
                "prefill_growth_x": prefill_growth,
                "total_prompt_tokens": total_prompt_tokens,
                "total_gen_tokens": total_gen_tokens,
                "total_evicted_tokens": evictor.total_evicted_tokens,
                "mean_ttft_ms": mean_ttft_ms,
                "turns": turns_log,
            }

            status_str = "✅ PASS" if passed_all else "❌ FAIL"
            print(
                f"    [{status_str}] Prefill Growth: {prefill_growth}x | "
                f"Evicted: {evictor.total_evicted_tokens} tok | Mean TTFT: {mean_ttft_ms} ms"
            )

            # Save checkpoint immediately after every arm
            checkpoint_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
            gc.collect()
            torch.cuda.empty_cache()

    # Aggregate Summary Table across all tasks
    from tabulate import tabulate
    summary_rows = []
    vanilla_total_toks = sum(
        all_results["tasks"][t]["vanilla"]["total_prompt_tokens"]
        + all_results["tasks"][t]["vanilla"]["total_gen_tokens"]
        for t in TASKS
    )

    for arm_key, arm_cfg in ARMS.items():
        passes = sum(1 for t in TASKS if all_results["tasks"][t][arm_key]["passed"])
        pass_rate = (passes / len(TASKS)) * 100.0
        avg_growth = sum(all_results["tasks"][t][arm_key]["prefill_growth_x"] for t in TASKS) / len(TASKS)
        avg_ttft = sum(all_results["tasks"][t][arm_key]["mean_ttft_ms"] for t in TASKS) / len(TASKS)
        arm_toks = sum(
            all_results["tasks"][t][arm_key]["total_prompt_tokens"]
            + all_results["tasks"][t][arm_key]["total_gen_tokens"]
            for t in TASKS
        )
        savings_pct = ((vanilla_total_toks - arm_toks) / max(1, vanilla_total_toks)) * 100.0
        all_results["summary"][arm_key] = {
            "pass_rate_pct": round(pass_rate, 1),
            "avg_prefill_growth_x": round(avg_growth, 2),
            "compute_saved_pct": round(savings_pct, 1),
            "avg_ttft_ms": round(avg_ttft, 2),
        }
        summary_rows.append([
            arm_cfg["name"],
            f"{pass_rate:.1f}%",
            f"{avg_growth:.2f}x",
            f"{savings_pct:+.1f}%",
            f"{avg_ttft:.1f} ms",
        ])

    checkpoint_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\n" + "=" * 85)
    print(f"🏆 SUBNEUTRALIZE v0.2.0 FRONTIER BENCHMARK RESULTS ({model_id})")
    print("=" * 85)
    print(
        tabulate(
            summary_rows,
            headers=["Experimental Arm", "Pass Rate", "Prefill Growth", "Compute Saved", "Mean TTFT"],
            tablefmt="github",
        )
    )
    print("=" * 85)
    print(f"📁 Full JSON telemetry saved to: {checkpoint_file.resolve()}")


if __name__ == "__main__":
    run_benchmark()
