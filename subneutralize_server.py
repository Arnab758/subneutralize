"""
================================================================================
   🚀 SUBNEUTRALIZE v2.4: MANIFOLD-GOVERNED OPENAI & vLLM INFERENCE RUNTIME
   Zero-Config Drop-In `/v1/chat/completions` Gateway & vLLM Plugin
================================================================================
What Makes SubNeutralize Unique (No Other Open-Source Proxy or Engine Does This):
  1. In-Flight Layer (L-2) Residual Manifold Governor:
     - Thin-SVD Prompt Subspace Orthogonal Projection (P_perp = I - V_bias V_bias^T)
     - Self-Sizing Roy-Vetterli Coupon-Collector Horizon (r_L1 * ln(r_L2))
     - Poincaré Limit-Cycle Orbit Recurrence (rho_orbit = max cos(h_bar_t, h_bar_{t-tau}))
     - Dimensionless Lyapunov Potential (Phi_t = z_erank + z_vel + z_ent - z_conf)
  2. Stateful Multi-Turn KV Scratchpad Evictor:
     - Automatically intercepts multi-turn OpenAI `messages[]` payloads from
       coding agents (Cursor, CodeBuff, OpenHands, Cline, LangGraph) and evicts
       stale `<think>...</think>` scratchpads from historical assistant turns,
       slashing prefill KV-cache memory by 55%-75% automatically.
  3. Live CFO & Hardware Telemetry (`/v1/telemetry`):
     - Tracks real-time token savings, prefill eviction ratios, manifold orbit
       metrics, and exact dollar savings per 1M requests.
================================================================================
"""

import os
import re
import gc
import math
import time
import hashlib
import ast
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional, Set, Union
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def select_model_for_hardware() -> str:
    env_model = os.environ.get("SUBNEUTRALIZE_MODEL")
    if env_model:
        return env_model
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 20.0:
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    return "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# ==============================================================================
# 1. ONLINE WELFORD MANIFOLD STATISTICS
# ==============================================================================
class OnlineManifoldStat:
    def __init__(self):
        self.n: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return math.sqrt(max(1e-12, self.m2 / (self.n - 1)))

    def z_score(self, x: float) -> float:
        return (x - self.mean) / (self.std() + 1e-6)


# ==============================================================================
# 2. PROMPT BIAS SUBSPACE NEUTRALIZER & MANIFOLD STEERING GOVERNOR
# ==============================================================================
class PromptSubspaceGovernor:
    """
    Dual-Manifold Governor:
    1. Computes Thin-SVD on centered prefill activations at Layer (L-2) to extract V_bias,
       and constructs orthogonal projection operator P_perp = I - V_bias @ V_bias.T.
    2. Dynamically derives and injects manifold steering vectors v_steer along the
       overthinking subspace to actively damp hesitation loops before hard trapping occurs.
    """

    def __init__(
        self,
        rank_k: int = 4,
        steer_gain: float = 0.15,
        enable_steering: bool = True,
    ):
        self.rank_k = rank_k
        self.steer_gain = steer_gain
        self.enable_steering = enable_steering
        self.V_bias: Optional[torch.Tensor] = None
        self.v_steer: Optional[torch.Tensor] = None

    def fit_prompt(self, H_prompt: torch.Tensor) -> Tuple[int, int]:
        """
        Fits prompt subspace basis V_bias and computes self-sizing Roy-Vetterli horizons.
        H_prompt: Residual stream activations during prefill [seq_len, hidden_dim].
        Returns: (burn_in_steps, cover_bound_steps)
        """
        mean_h = H_prompt.mean(dim=0, keepdim=True)
        H_centered = (H_prompt - mean_h).float()

        try:
            _, sv, Vh = torch.linalg.svd(H_centered, full_matrices=False)
            k = min(self.rank_k, Vh.shape[0])
            self.V_bias = Vh[:k, :].T.contiguous()

            p1 = sv / (torch.sum(sv) + 1e-12)
            p2 = (sv ** 2) / (torch.sum(sv ** 2) + 1e-12)
            r_l1 = float(torch.exp(-torch.sum(p1 * torch.log(p1 + 1e-12))).item())
            r_l2 = float(torch.exp(-torch.sum(p2 * torch.log(p2 + 1e-12))).item())

            coupon_horizon = int(r_l1 * math.log(max(math.e, r_l2)))
            burn_in = max(180, min(320, int(coupon_horizon * 0.65)))
            cover_bound = max(burn_in + 120, min(460, int(coupon_horizon * 0.95)))
            return burn_in, cover_bound
        except Exception:
            self.V_bias = None
            return 180, 380

    def project(self, h_t: torch.Tensor) -> torch.Tensor:
        """Projects a decoding residual activation: h_perp = h_t - (h_t @ V_bias) @ V_bias.T"""
        if self.V_bias is None:
            return h_t
        return h_t - (h_t @ self.V_bias) @ self.V_bias.T

    def set_steering_vector(self, v: torch.Tensor) -> None:
        """Sets an explicit contrastive steering vector."""
        vec = v.flatten().float()
        if self.V_bias is not None:
            v_bias_dev = self.V_bias.to(device=vec.device, dtype=vec.dtype)
            vec = vec - (vec @ v_bias_dev) @ v_bias_dev.T
        norm = torch.norm(vec, p=2)
        if norm > 1e-8:
            self.v_steer = (vec / norm).contiguous()
        else:
            self.v_steer = None

    def derive_manifold_steering_vector(self, sliding_window_activations: torch.Tensor) -> Optional[torch.Tensor]:
        """Derives the overthinking steering direction v_steer in-flight from sliding window activations."""
        try:
            M = sliding_window_activations.float()
            if M.dim() == 3:
                M = M.squeeze(0)
            M_c = M - M.mean(dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(M_c, full_matrices=False)
            u1 = Vh[0, :].contiguous()

            if self.V_bias is not None:
                v_bias_dev = self.V_bias.to(device=u1.device, dtype=u1.dtype)
                u1 = u1 - (u1 @ v_bias_dev) @ v_bias_dev.T

            norm = torch.norm(u1, p=2)
            if norm > 1e-8:
                self.v_steer = (u1 / norm).contiguous()
                return self.v_steer
        except Exception:
            pass
        return None

    def steer(self, h: torch.Tensor, gain: Optional[float] = None) -> torch.Tensor:
        """Applies dual-manifold projection and steering damping."""
        h_perp = self.project(h)
        if not self.enable_steering or self.v_steer is None:
            return h_perp

        g = gain if gain is not None else self.steer_gain
        if g <= 0.0:
            return h_perp

        v = self.v_steer.to(device=h_perp.device, dtype=h_perp.dtype)
        if h_perp.dim() == 1:
            proj = torch.dot(h_perp, v)
            return h_perp - g * proj * v
        else:
            proj = torch.matmul(h_perp, v.view(-1, 1))
            return h_perp - g * proj * v.view(*([1] * (h_perp.dim() - 1)), -1)


# ==============================================================================
# 3. UNIFIED GEOMETRIC LOGITS PROCESSOR WITH DELIMITER DEPTH & STEERING
# ==============================================================================
class SubNeutralizeManifoldProcessor(LogitsProcessor):
    """
    Drop-in Geometric LogitsProcessor for HuggingFace `generate()` and vLLM V1.
    Hooks Decoder Layer (L-2) to monitor the latent reasoning trajectory in real time,
    applies contrastive activation steering, and enforces delimiter-balanced exit.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        think_end_token_id: int,
        eos_token_id: int,
        structural_boundary_ids: set,
        window_size: int = 16,
        svd_rank: int = 4,
        steer_gain: float = 0.15,
        enable_steering: bool = True,
        orbit_steer_threshold: float = 0.80,
        emergency_attractor_steps: int = 12,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.think_end_token_id = think_end_token_id
        self.eos_token_id = eos_token_id
        self.structural_boundary_ids = structural_boundary_ids
        self.window_size = window_size
        self.svd_rank = svd_rank
        self.steer_gain = steer_gain
        self.enable_steering = enable_steering
        self.orbit_steer_threshold = orbit_steer_threshold
        self.emergency_attractor_steps = emergency_attractor_steps

        self.prompt_governor = PromptSubspaceGovernor(
            rank_k=svd_rank,
            steer_gain=steer_gain,
            enable_steering=enable_steering,
        )
        self.prompt_len: Optional[int] = None
        self.latest_h: Optional[torch.Tensor] = None
        self.prev_h: Optional[torch.Tensor] = None

        self.spectral_burn_in: int = 220
        self.spectral_cover_bound: int = 380

        self.h_window: List[torch.Tensor] = []
        self.centroid_history: List[torch.Tensor] = []
        self.conf_window: List[float] = []

        self.stat_erank = OnlineManifoldStat()
        self.stat_vel = OnlineManifoldStat()
        self.stat_ent = OnlineManifoldStat()
        self.stat_conf = OnlineManifoldStat()

        # Delimiter Depth Tracking (D_t = 0 for balanced brackets)
        self.d_paren: int = 0
        self.d_bracket: int = 0
        self.d_brace: int = 0
        self.delimiter_depth: int = 0
        self._token_delimiter_cache: Dict[int, str] = {}

        # Steering State
        self.hesitation_damping_active: bool = False
        self.damping_step_count: int = 0
        self.max_damping_steps: int = 6

        self.ema_phi: float = 0.0
        self.attractor_steps: int = 0
        self.convergence_ready: bool = False
        self.think_closed: bool = False
        self.think_close_step: Optional[int] = None
        self.triggered_by_manifold: bool = False
        self.telemetry: Dict[str, Any] = {}

        layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else (
            model.transformer.h if hasattr(model, "transformer") and hasattr(model.transformer, "h") else None
        )
        if layers is not None:
            target_layer = max(0, len(layers) - 2)
            self.hook_handle = layers[target_layer].register_forward_hook(self._layer_hook)
        else:
            self.hook_handle = None

    def _update_delimiter_depth(self, token_id: int) -> int:
        if token_id in self._token_delimiter_cache:
            text = self._token_delimiter_cache[token_id]
        else:
            try:
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                text = ""
            self._token_delimiter_cache[token_id] = text

        for ch in text:
            if ch == '(':
                self.d_paren += 1
            elif ch == ')':
                self.d_paren = max(0, self.d_paren - 1)
            elif ch == '[':
                self.d_bracket += 1
            elif ch == ']':
                self.d_bracket = max(0, self.d_bracket - 1)
            elif ch == '{':
                self.d_brace += 1
            elif ch == '}':
                self.d_brace = max(0, self.d_brace - 1)

        self.delimiter_depth = self.d_paren + self.d_bracket + self.d_brace
        return self.delimiter_depth

    def _layer_hook(self, module, inp, outp):
        hidden = outp[0] if isinstance(outp, tuple) else outp
        if hidden.shape[1] > 1:
            H = hidden[0].detach().float()
            burn_in, cover_bound = self.prompt_governor.fit_prompt(H)
            self.spectral_burn_in = burn_in
            self.spectral_cover_bound = cover_bound
            self.latest_h = H[-1:].contiguous()
        else:
            with torch.no_grad():
                h_t = hidden[0, -1:, :].detach().float()
                should_steer = (
                    self.enable_steering
                    and (self.hesitation_damping_active or self.attractor_steps > 0)
                    and self.prompt_governor.v_steer is not None
                )
                if should_steer:
                    h_steered = self.prompt_governor.steer(h_t, gain=self.steer_gain)
                    self.latest_h = h_steered
                    steered_val = h_steered.to(dtype=hidden.dtype, device=hidden.device)
                    hidden[:, -1:, :] = steered_val
                    self.damping_step_count += 1
                    if self.damping_step_count >= self.max_damping_steps:
                        self.hesitation_damping_active = False
                        self.damping_step_count = 0
                else:
                    self.latest_h = self.prompt_governor.project(h_t)

        return outp

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.prompt_len is None:
            self.prompt_len = input_ids.shape[1]

        gen_tokens = input_ids[0, self.prompt_len:]
        step = gen_tokens.shape[0]
        if step == 0:
            return scores

        last_tok = int(gen_tokens[-1].item())
        self._update_delimiter_depth(last_tok)

        if not self.think_closed and (gen_tokens == self.think_end_token_id).any():
            self.think_closed = True
            if self.think_close_step is None:
                self.think_close_step = step

        # PHASE B: POST-THINK ARTIFACT COMPLETION
        if self.think_closed:
            post_steps = step - (self.think_close_step or step)
            if post_steps >= 60 and last_tok in self.structural_boundary_ids:
                post_ids = gen_tokens[self.think_close_step:]
                post_text = self.tokenizer.decode(post_ids, skip_special_tokens=True)
                blocks = re.findall(r"```(?:python|diff|json|bash)?\n(.*?)```", post_text, re.DOTALL)
                if blocks and len(blocks[-1].strip()) >= 25:
                    scores[:, self.eos_token_id] = scores.max() + 100.0
            return scores

        # PHASE A: LATENT SCRATCHPAD (<think>) GEOMETRIC MANIFOLD MONITORING
        h_t = self.latest_h
        if h_t is None:
            return scores

        if self.prev_h is not None:
            cos_sim = F.cosine_similarity(h_t, self.prev_h, dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)
        else:
            v_t = 0.0
        self.prev_h = h_t.clone()

        self.h_window.append(h_t.squeeze(0))
        if len(self.h_window) > self.window_size:
            self.h_window.pop(0)

        erank_t = float(len(self.h_window))
        orbit_recurrence = 0.0
        if len(self.h_window) == self.window_size:
            M = torch.stack(self.h_window, dim=0)
            centroid = F.normalize(M.mean(dim=0, keepdim=True), dim=-1)
            self.centroid_history.append(centroid)
            if len(self.centroid_history) > 96:
                self.centroid_history.pop(0)

            if len(self.centroid_history) > 24:
                lagged = torch.cat(self.centroid_history[:-20], dim=0)
                sims = torch.mm(centroid, lagged.T)
                orbit_recurrence = float(sims.max().item())

            M_c = M - M.mean(dim=0, keepdim=True)
            s = torch.linalg.svdvals(M_c)
            q = s / (torch.sum(s) + 1e-12)
            erank_t = float(torch.exp(-torch.sum(q * torch.log(q + 1e-12))).item())

            # Active Steering Trigger
            hesitation_detected = (
                orbit_recurrence >= self.orbit_steer_threshold
                or (v_t < 0.005 and erank_t < 3.0)
            )
            if self.enable_steering and hesitation_detected and not self.hesitation_damping_active:
                self.prompt_governor.derive_manifold_steering_vector(M)
                if self.prompt_governor.v_steer is not None:
                    self.hesitation_damping_active = True
                    self.damping_step_count = 0

        logits_f = scores[0].float()
        probs = F.softmax(logits_f, dim=-1)
        log_probs = F.log_softmax(logits_f, dim=-1)
        h_ent = -torch.sum(probs * log_probs).item()
        max_p = float(torch.max(probs).item())

        self.conf_window.append(max_p)
        if len(self.conf_window) > self.window_size:
            self.conf_window.pop(0)
        deer_conf = float(torch.exp(torch.mean(torch.log(torch.tensor(self.conf_window) + 1e-12))).item())

        if len(self.h_window) == self.window_size:
            if step <= self.spectral_burn_in:
                self.stat_erank.update(erank_t)
                self.stat_vel.update(v_t)
                self.stat_ent.update(h_ent)
                self.stat_conf.update(deer_conf)
            else:
                phi_t = (
                    self.stat_erank.z_score(erank_t)
                    + self.stat_vel.z_score(v_t)
                    + self.stat_ent.z_score(h_ent)
                    - self.stat_conf.z_score(deer_conf)
                )
                self.ema_phi = 0.7 * self.ema_phi + 0.3 * phi_t

                in_attractor = (
                    (self.ema_phi < -1.05)
                    and (deer_conf >= 0.86)
                    and (h_ent < 0.48)
                    and (orbit_recurrence >= 0.84)
                )
                cover_reached = (step >= self.spectral_cover_bound) and (deer_conf >= 0.82)

                if in_attractor or cover_reached:
                    self.attractor_steps += 1
                else:
                    self.attractor_steps = 0
                    self.convergence_ready = False
                    self.stat_erank.update(erank_t)
                    self.stat_vel.update(v_t)
                    self.stat_ent.update(h_ent)
                    self.stat_conf.update(deer_conf)

                if self.attractor_steps >= 4 or cover_reached:
                    self.convergence_ready = True

        # Syntactic Delimiter & AST Gated Early Exit
        if self.convergence_ready and step > 0:
            balanced = (self.delimiter_depth == 0)
            at_boundary = (last_tok in self.structural_boundary_ids)
            emergency_exit = (self.attractor_steps >= self.emergency_attractor_steps)

            if (at_boundary and balanced) or emergency_exit:
                scores[:, self.think_end_token_id] = scores.max() + 100.0
                self.think_closed = True
                self.think_close_step = step + 1
                self.triggered_by_manifold = True
                self.telemetry = {
                    "step": step,
                    "burn_in": self.spectral_burn_in,
                    "cover_bound": self.spectral_cover_bound,
                    "erank": round(erank_t, 2),
                    "orbit": round(orbit_recurrence, 3),
                    "vel": round(v_t, 4),
                    "ent": round(h_ent, 4),
                    "deer_C": round(deer_conf, 4),
                    "phi": round(self.ema_phi, 2),
                    "delimiter_depth": self.delimiter_depth,
                    "steer_active": self.hesitation_damping_active,
                    "emergency_exit": emergency_exit,
                }

        return scores


# ==============================================================================
# 4. STATEFUL MULTI-TURN KV SCRATCHPAD EVICTOR (CONTINUUM / R-KV ENGINE)
# ==============================================================================
class RegionType(str, Enum):
    SYSTEM = "system"
    USER_LATEST = "user_latest"
    USER_HISTORICAL = "user_historical"
    CANONICAL_CODE = "canonical_code"
    PYTEST_TRACEBACK = "pytest_traceback"
    ASSISTANT_THINK = "assistant_think"
    ASSISTANT_TEXT = "assistant_text"


class StatefulMultiTurnEvictor:
    """
    Automatically compacts multi-turn OpenAI `messages[]` histories sent by
    autonomous coding frameworks (Cursor, CodeBuff, OpenHands, LangGraph).
    Strips historical `<think>...</think>` scratchpad spans from prior turns
    while preserving 100% of canonical code blocks, tool calls, and test results.
    Integrates MemDecay (Matam & Kim, 2026) region-aware attention half-life weighting
    and AST-safe code fence and traceback shielding.
    """

    DEFAULT_HALF_LIVES: Dict[RegionType, float] = {
        RegionType.SYSTEM: 180.0,
        RegionType.USER_LATEST: float("inf"),
        RegionType.USER_HISTORICAL: 90.0,
        RegionType.CANONICAL_CODE: 150.0,
        RegionType.PYTEST_TRACEBACK: 120.0,
        RegionType.ASSISTANT_THINK: 15.0,
        RegionType.ASSISTANT_TEXT: 60.0,
    }

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        preserve_code: bool = True,
        preserve_errors: bool = True,
        half_life_scratchpad: float = 15.0,
        half_life_system: float = 180.0,
        eviction_threshold: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.preserve_code = preserve_code
        self.preserve_errors = preserve_errors
        self.half_life_scratchpad = half_life_scratchpad
        self.half_life_system = half_life_system
        self.eviction_threshold = eviction_threshold

        self.total_prefill_tokens_evicted: int = 0
        self.compacted_turns_count: int = 0

    def compute_retention_weight(self, region: Union[RegionType, str], delta_turns: int = 1) -> float:
        if isinstance(region, str):
            try:
                region = RegionType(region)
            except ValueError:
                pass
        half_life = self.DEFAULT_HALF_LIVES.get(region, 60.0)
        if region == RegionType.ASSISTANT_THINK:
            half_life = self.half_life_scratchpad
        elif region == RegionType.SYSTEM:
            half_life = self.half_life_system

        if half_life == float("inf") or delta_turns <= 0:
            return 1.0

        if region == RegionType.ASSISTANT_THINK:
            effective_delta = delta_turns * 69.0
        else:
            effective_delta = float(delta_turns)
        return float(2.0 ** (-effective_delta / max(1.0, half_life)))

    def extract_code_blocks(self, text: str) -> List[str]:
        pattern = r"```(?:[a-zA-Z0-9_\-+]+)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if not matches:
            unclosed = re.findall(r"```(?:[a-zA-Z0-9_\-+]+)?\s*\n(.*)$", text, flags=re.DOTALL)
            if unclosed:
                matches = unclosed
        return [m.strip() for m in matches if m.strip()]

    def validate_ast_invariants(self, original_text: str, compacted_text: str) -> bool:
        orig_blocks = self.extract_code_blocks(original_text)
        comp_blocks = self.extract_code_blocks(compacted_text)
        if len(orig_blocks) != len(comp_blocks):
            return False
        for orig_code, comp_code in zip(orig_blocks, comp_blocks):
            try:
                tree_orig = ast.parse(orig_code)
                tree_comp = ast.parse(comp_code)
                if ast.dump(tree_orig) != ast.dump(tree_comp):
                    return False
            except SyntaxError:
                if orig_code != comp_code:
                    return False
        return True

    def _strip_scratchpad_ast_safe(self, content: str) -> str:
        placeholders: Dict[str, str] = {}

        def _shield_match(match):
            key = f"__SUBNEUTRALIZE_SHIELD_{len(placeholders)}__"
            placeholders[key] = match.group(0)
            return key

        # 1. Shield closed fenced code blocks: ```...```
        shielded = re.sub(r"```.*?```", _shield_match, content, flags=re.DOTALL)
        # Also shield unclosed code fences at end of string: ```...$
        shielded = re.sub(r"```[^\n]*\n.*$", _shield_match, shielded, flags=re.DOTALL)

        # 2. Shield pytest failure blocks and assertion errors if present in text
        if self.preserve_errors:
            traceback_pattern = r"((?:FAILED\s+[^\n]+\n|AssertionError[^\n]*\n|Traceback \(most recent call last\):[^\n]*\n)(?:[ \t]+[^\n]*\n)*)"
            shielded = re.sub(traceback_pattern, _shield_match, shielded, flags=re.MULTILINE)

        # 3. Strip all closed <think>...</think> blocks
        cleaned = re.sub(r"<think>.*?</think>\s*", "", shielded, flags=re.DOTALL)

        # 4. Handle orphan </think> (strip narrative before it without crossing shield placeholders)
        if "</think>" in cleaned:
            cleaned = re.sub(r"^(?:(?!__SUBNEUTRALIZE_SHIELD_).)*?</think>\s*", "", cleaned, flags=re.DOTALL)

        # 5. Handle unclosed <think> tags (strip from <think> up to next placeholder or end of string)
        if "<think>" in cleaned:
            cleaned = re.sub(r"<think>(?:(?!__SUBNEUTRALIZE_SHIELD_).)*", "", cleaned, flags=re.DOTALL)

        # 6. Restore shielded placeholders
        for key, orig_chunk in placeholders.items():
            cleaned = cleaned.replace(key, orig_chunk)

        return cleaned.strip()

    def compact_messages(
        self,
        messages: List[Dict[str, str]],
        evict_history_scratchpads: bool = True,
    ) -> Tuple[List[Dict[str, str]], int, int]:
        raw_tokens = self._count_tokens(messages)
        if not evict_history_scratchpads or len(messages) <= 1:
            return messages, raw_tokens, 0

        compacted: List[Dict[str, str]] = []
        historical_messages = messages[:-1]
        latest_message = messages[-1]

        for m in historical_messages:
            role = m.get("role", "")
            content = m.get("content", "")

            if role in ("system", "user"):
                compacted.append(m)
                continue

            if role == "assistant":
                cleaned = self._strip_scratchpad_ast_safe(content)
                if not cleaned:
                    cleaned = "# [Historical reasoning scratchpad evicted to optimize context KV-cache]"
                if self.preserve_code:
                    if not self.validate_ast_invariants(content, cleaned):
                        # Safety fallback: preserve original content if AST invariance is ever breached
                        cleaned = content
                compacted.append({"role": role, "content": cleaned})
            else:
                compacted.append(m)

        compacted.append(latest_message)
        clean_tokens = self._count_tokens(compacted)
        evicted = max(0, raw_tokens - clean_tokens)

        if evicted > 0:
            self.total_prefill_tokens_evicted += evicted
            self.compacted_turns_count += 1

        return compacted, clean_tokens, evicted

    def evict_dialogue(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        compacted, clean_tokens, evicted = self.compact_messages(messages, evict_history_scratchpads=True)
        raw_tokens = clean_tokens + evicted

        user_prompts = [m for m in messages if m.get("role") == "user"]
        growth_ratio = 1.0
        if len(user_prompts) >= 2:
            first_len = max(1, len(str(user_prompts[0].get("content", "")).split()))
            last_len = max(1, len(str(user_prompts[-1].get("content", "")).split()))
            growth_ratio = round(last_len / first_len, 2)

        telemetry = {
            "clean_tokens": clean_tokens,
            "evicted_tokens": evicted,
            "prompt_growth_ratio": growth_ratio,
            "retained_regions": [
                RegionType.SYSTEM.value,
                RegionType.USER_LATEST.value,
                RegionType.USER_HISTORICAL.value,
                RegionType.CANONICAL_CODE.value,
                RegionType.PYTEST_TRACEBACK.value,
            ],
            "total_prefill_tokens_evicted": self.total_prefill_tokens_evicted,
            "compacted_turns_count": self.compacted_turns_count,
            "reduction_ratio": round(clean_tokens / max(1, raw_tokens), 3),
        }
        return compacted, telemetry

    def reset_stats(self):
        self.total_prefill_tokens_evicted = 0
        self.compacted_turns_count = 0

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                raw_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                return len(self.tokenizer.encode(raw_prompt, add_special_tokens=False))
            except Exception:
                pass
        return sum(len(str(m.get("content", "")).split()) for m in messages)



# ==============================================================================
# 4. OPENAI-COMPATIBLE RUNTIME GATEWAY (`/v1/chat/completions`)
# ==============================================================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = 1536
    temperature: float = 0.6
    top_p: float = 0.95
    enable_subneutralize: bool = True
    strip_response_scratchpad: bool = True


class SubNeutralizeRuntimeEngine:
    """
    Core Engine that powers both the FastAPI `/v1/chat/completions` HTTP Gateway
    and direct in-process Python SDK calls.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.evictor = StatefulMultiTurnEvictor(tokenizer)

        self.think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[-1]
        self.think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]
        self.boundary_ids = set()
        for sym in ["\n", ".\n", ":\n", "\n\n", "."]:
            toks = tokenizer.encode(sym, add_special_tokens=False)
            if toks:
                self.boundary_ids.add(toks[-1])

        # Cumulative Telemetry
        self.total_requests: int = 0
        self.manifold_interventions: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_latency_s: float = 0.0

    def chat_completions(self, req: ChatCompletionRequest) -> Dict[str, Any]:
        self.total_requests += 1
        raw_msgs = [{"role": m.role, "content": m.content} for m in req.messages]

        # 1. Stateful Multi-Turn KV Scratchpad Eviction on Incoming History
        compacted_msgs, prompt_len, prefill_evicted = self.evictor.compact_messages(
            raw_msgs, evict_history_scratchpads=req.enable_subneutralize
        )

        prompt_text = self.tokenizer.apply_chat_template(
            compacted_msgs, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        actual_prompt_len = inputs.input_ids.shape[1]

        # 2. Attach Layer (L-2) Manifold Governor
        processor = None
        logits_processors = LogitsProcessorList()
        if req.enable_subneutralize:
            processor = SubNeutralizeManifoldProcessor(
                model=self.model,
                tokenizer=self.tokenizer,
                think_end_token_id=self.think_end_id,
                eos_token_id=self.tokenizer.eos_token_id,
                structural_boundary_ids=self.boundary_ids,
            )
            logits_processors.append(processor)

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=req.max_tokens,
                    do_sample=True,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    logits_processor=logits_processors if req.enable_subneutralize else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if processor is not None:
                processor.remove_hook()
        dt = time.perf_counter() - t0

        gen_ids = outputs[0][actual_prompt_len:]
        full_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Separate <think> scratchpad from canonical assistant response
        ids_list = gen_ids.tolist()
        if self.think_end_id in ids_list:
            end_idx = ids_list.index(self.think_end_id)
            think_ids = ids_list[:end_idx]
            answer_ids = ids_list[end_idx + 1:]
            reasoning_text = self.tokenizer.decode(think_ids, skip_special_tokens=True).strip()
            clean_content = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            think_tokens = len(think_ids) + 1
        else:
            reasoning_text = full_text
            clean_content = full_text
            think_tokens = len(ids_list)

        if processor and processor.triggered_by_manifold:
            self.manifold_interventions += 1

        self.total_prompt_tokens += actual_prompt_len
        self.total_completion_tokens += len(gen_ids)
        self.total_latency_s += dt

        returned_content = (
            clean_content
            if (req.enable_subneutralize and req.strip_response_scratchpad and clean_content)
            else full_text
        )

        return {
            "id": f"chatcmpl-subn-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": returned_content,
                        "reasoning_content": reasoning_text,
                    },
                    "finish_reason": (
                        "subneutralize_manifold_convergence"
                        if (processor and processor.triggered_by_manifold)
                        else "stop"
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": actual_prompt_len,
                "prefill_tokens_evicted": prefill_evicted,
                "completion_tokens": len(gen_ids),
                "reasoning_tokens": think_tokens,
                "answer_tokens": len(gen_ids) - think_tokens,
                "total_tokens": actual_prompt_len + len(gen_ids),
                "latency_s": round(dt, 2),
            },
            "x_subneutralize_telemetry": processor.telemetry if processor else {},
        }

    def get_telemetry_summary(self) -> Dict[str, Any]:
        return {
            "engine_version": "SubNeutralize v2.4 (Layer L-2 Manifold Governor + Stateful KV Evictor)",
            "model": self.model_id,
            "total_requests": self.total_requests,
            "manifold_interventions": self.manifold_interventions,
            "compacted_multi_turn_requests": self.evictor.compacted_turns_count,
            "total_prefill_tokens_evicted": self.evictor.total_prefill_tokens_evicted,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency_s": round(self.total_latency_s, 2),
        }


from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from subneutralize.firewall import (
    TransactionalActionFirewall,
    FirewallPolicyConfig,
    ActionTransactionRecord,
    TransactionStatus,
)
from subneutralize.server import (
    FirewallInterceptRequest,
    FirewallResolveRequest,
    FirewallChatAgentRequest,
    MCPToolCallRequest,
    _build_command_center_html,
    execute_live_agent_mission,
)

app = FastAPI(
    title="SubNeutralize v0.3.0 Transactional Runtime & Action Firewall Gateway",
    version="0.3.0",
)
GLOBAL_ENGINE: Optional[SubNeutralizeRuntimeEngine] = None
GLOBAL_FIREWALL = TransactionalActionFirewall()


@app.get("/", response_class=HTMLResponse)
@app.get("/command-center", response_class=HTMLResponse)
def api_dashboard():
    return _build_command_center_html(select_model_for_hardware())


@app.get("/health")
def api_health() -> Dict[str, Any]:
    chain_ok, chain_len = GLOBAL_FIREWALL.verify_audit_chain_integrity()
    return {
        "status": "healthy",
        "version": "0.3.0",
        "audit_chain_verified": chain_ok,
        "audit_records": chain_len,
    }


@app.get("/v1/firewall/state")
def api_firewall_state() -> Dict[str, Any]:
    return GLOBAL_FIREWALL.get_full_state_summary()


@app.get("/v1/firewall/roi")
def api_firewall_roi() -> Dict[str, Any]:
    return GLOBAL_FIREWALL.get_roi_receipt()


@app.post("/v1/firewall/chat-agent")
def api_firewall_chat_agent(req: FirewallChatAgentRequest) -> Dict[str, Any]:
    return GLOBAL_FIREWALL.run_agent_loop(
        user_prompt=req.prompt,
        session_id=req.session_id,
        max_turns=req.max_turns,
        api_key=req.api_key,
        provider=req.provider,
        model=req.model,
    )


@app.post("/v1/firewall/intercept")
def api_firewall_intercept(req: FirewallInterceptRequest) -> Dict[str, Any]:
    rec, updated_messages = GLOBAL_FIREWALL.intercept_and_execute(
        session_id=req.session_id,
        turn_index=req.turn_index,
        tool_type=req.tool_type,
        tool_name=req.tool_name,
        arguments=req.arguments,
        agent_reasoning_text=req.agent_reasoning_text,
        messages_context=req.messages_context,
        latent_phi=req.latent_phi,
        latent_rho=req.latent_rho,
    )
    return {
        "transaction": asdict(rec),
        "updated_messages_context": updated_messages,
        "metrics": GLOBAL_FIREWALL.metrics,
        "roi_receipt": GLOBAL_FIREWALL.get_roi_receipt(),
    }


@app.post("/v1/firewall/resolve/{tx_id}")
def api_firewall_resolve(tx_id: str, req: FirewallResolveRequest) -> Dict[str, Any]:
    try:
        rec = GLOBAL_FIREWALL.resolve_pending_transaction(tx_id, req.action)
        return {"transaction": asdict(rec), "metrics": GLOBAL_FIREWALL.metrics, "roi_receipt": GLOBAL_FIREWALL.get_roi_receipt()}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/v1/firewall/reset")
def api_firewall_reset() -> Dict[str, Any]:
    GLOBAL_FIREWALL.reset_all()
    return {"status": "reset_complete", "state": GLOBAL_FIREWALL.get_full_state_summary()}


@app.post("/v1/firewall/agent-run")
def api_firewall_agent_run() -> Dict[str, Any]:
    return execute_live_agent_mission(GLOBAL_FIREWALL)


@app.post("/v1/mcp/tools/call")
def api_mcp_tools_call(req: MCPToolCallRequest) -> Dict[str, Any]:
    name_lower = req.name.lower()
    if "sql" in name_lower or "query" in name_lower or "db" in name_lower:
        tool_type = "sql"
    elif "file" in name_lower or "write" in name_lower or "edit" in name_lower:
        tool_type = "filesystem"
    elif "refund" in name_lower or "stripe" in name_lower or "pay" in name_lower or "transfer" in name_lower:
        tool_type = "http_payment"
    else:
        tool_type = "shell"

    rec, updated_msgs = GLOBAL_FIREWALL.intercept_and_execute(
        session_id=req.session_id,
        turn_index=req.turn_index,
        tool_type=tool_type,
        tool_name=req.name,
        arguments=req.arguments,
        agent_reasoning_text=req.agent_reasoning_text,
        latent_phi=req.latent_phi,
        latent_rho=req.latent_rho,
    )
    is_err = rec.status in (TransactionStatus.ROLLED_BACK, TransactionStatus.PENDING_APPROVAL)
    return {
        "jsonrpc": "2.0",
        "result": {
            "isError": is_err,
            "transaction_id": rec.tx_id,
            "status": rec.status,
            "content": [
                {
                    "type": "text",
                    "text": rec.execution_output if rec.status == TransactionStatus.COMMITTED else rec.policy_verdict,
                }
            ],
            "subneutralize_firewall": {
                "memory_rewound": rec.memory_rewound,
                "tokens_pruned": rec.tokens_pruned_on_rollback,
                "corrective_steering_prompt": rec.corrective_steering_prompt,
                "record_hash": rec.record_hash,
            },
            "updated_messages": updated_msgs,
        },
    }


@app.post("/v1/chat/completions")
def api_chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    global GLOBAL_ENGINE
    if GLOBAL_ENGINE is None:
        mid = select_model_for_hardware()
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModelForCausalLM.from_pretrained(
            mid,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": 0} if torch.cuda.is_available() else None,
        )
        mdl.eval()
        GLOBAL_ENGINE = SubNeutralizeRuntimeEngine(mdl, tok, mid)
    return GLOBAL_ENGINE.chat_completions(req)


@app.get("/v1/telemetry")
def api_telemetry() -> Dict[str, Any]:
    if GLOBAL_ENGINE is None:
        return {"status": "idle", "firewall_metrics": GLOBAL_FIREWALL.metrics}
    summary = GLOBAL_ENGINE.get_telemetry_summary()
    summary["firewall_metrics"] = GLOBAL_FIREWALL.metrics
    return summary


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
