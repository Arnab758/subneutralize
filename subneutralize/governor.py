"""
Scale-Free Dynamical Consensus Governor Core.

Provides dimension-invariant trajectory velocity, windowed attractor basin
detection, and token entropy monitoring for autoregressive reasoning models.
"""

import re
import math
import torch
import torch.nn.functional as F
from collections import deque
from typing import Optional, List, Dict, Any, Tuple, Union


class ScaleFreeDynamicalGovernor:
    """
    Dimension-Invariant, Scale-Free Inference Governor for Autoregressive Reasoning Models.
    
    Monitors layer hidden state dynamics in real time via forward hooks.
    Identifies when the reasoning trajectory has converged into a stable attractor basin
    (meaning the cognitive deduction is complete and subsequent tokens represent
    overthinking / looping), triggering a clean transition to answer emission.
    """
    def __init__(
        self,
        min_warmup_tokens: int = 50,
        consensus_ratio_threshold: float = 0.85,
        velocity_ceiling: float = 0.12,
        dispersion_ceiling: float = 0.08,
        ema_alpha: float = 0.12,
        entropy_threshold: float = 1.35,
        debounce_tokens: int = 3,
        window_size: int = 5,
    ):
        """
        Args:
            min_warmup_tokens: Minimum reasoning tokens before consensus can be declared.
            consensus_ratio_threshold: Threshold on v_t / EMA(v_t) indicating deceleration.
            velocity_ceiling: Absolute directional velocity ceiling (1 - cos_sim).
            dispersion_ceiling: Maximum centroid dispersion over the sliding window.
            ema_alpha: Smoothing factor for exponential moving average of velocity.
            entropy_threshold: Shannon entropy ceiling ensuring model is confident.
            debounce_tokens: Consecutive steps required in the attractor before stopping.
            window_size: Length of recent trajectory window used to verify attractor stability.
        """
        self.min_warmup_tokens = min_warmup_tokens
        self.consensus_ratio_threshold = consensus_ratio_threshold
        self.velocity_ceiling = velocity_ceiling
        self.dispersion_ceiling = dispersion_ceiling
        self.ema_alpha = ema_alpha
        self.entropy_threshold = entropy_threshold
        self.debounce_tokens = debounce_tokens
        self.window_size = window_size
        
        self.reset()

    def reset(self):
        """Resets tracking state for a new prompt."""
        self.step = 0
        self.last_hidden: Optional[torch.Tensor] = None
        self.ema_velocity: Optional[float] = None
        self.history: List[Dict[str, Any]] = []
        self.consensus_token: Optional[int] = None
        self.consecutive_stabilized = 0
        self._window: deque = deque(maxlen=self.window_size)

    def update(
        self,
        hidden_state: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> Union[bool, torch.Tensor]:
        """
        Step-wise update called on every emitted token.
        
        Args:
            hidden_state: Tensor of shape (B, D) or (B, 1, D) from target bottleneck layer.
            logits: Optional Tensor of shape (B, vocab_size) to compute token entropy.
            
        Returns:
            bool: True if dynamical consensus equilibrium is reached (single batch),
                  or a torch.BoolTensor of shape (B,) for batched generation.
        """
        self.step += 1
        
        # Normalize shape to (B, D)
        if hidden_state.ndim == 3:
            h_t = hidden_state[:, -1, :].detach().float()
        elif hidden_state.ndim == 2:
            h_t = hidden_state.detach().float()
        elif hidden_state.ndim == 1:
            h_t = hidden_state.unsqueeze(0).detach().float()
        else:
            h_t = hidden_state.view(hidden_state.shape[0], -1).detach().float()

        batch_size = h_t.shape[0]

        # For single-batch (common interactive and inference path)
        if batch_size == 1:
            return self._update_single(h_t[0:1], logits[0:1] if logits is not None else None)

        # For batched generation, evaluate per sample
        results = []
        for b in range(batch_size):
            h_b = h_t[b:b+1]
            l_b = logits[b:b+1] if logits is not None else None
            results.append(self._update_single(h_b, l_b))
        
        return torch.tensor(results, dtype=torch.bool, device=hidden_state.device)

    def _update_single(self, h_t: torch.Tensor, logits: Optional[torch.Tensor] = None) -> bool:
        """Internal step-wise update for a single stream (shape 1, D)."""
        if self.last_hidden is None:
            self.last_hidden = h_t
            self._window.append(h_t)
            return False

        # 1. Directional Cosine Velocity v_t in latent space: v_t = 1 - cos(h_t, h_{t-1})
        sim = F.cosine_similarity(h_t, self.last_hidden, dim=-1).clamp(-1.0, 1.0).item()
        v_t = max(0.0, 1.0 - sim)
        self.last_hidden = h_t
        self._window.append(h_t)

        # 2. Update Exponential Moving Average of Velocity (EMA)
        if self.ema_velocity is None:
            self.ema_velocity = v_t
        else:
            self.ema_velocity = self.ema_alpha * v_t + (1.0 - self.ema_alpha) * self.ema_velocity

        # 3. Compute Dimension-Invariant Consensus Ratio R_t
        r_t = v_t / max(self.ema_velocity, 1e-6)

        # 4. Compute Windowed Trajectory Dispersion (Attractor Radius)
        if len(self._window) >= 2:
            window_stack = torch.cat(list(self._window), dim=0) # (K, D)
            centroid = window_stack.mean(dim=0, keepdim=True)    # (1, D)
            sims_to_centroid = F.cosine_similarity(window_stack, centroid, dim=-1).clamp(-1.0, 1.0)
            dispersion = (1.0 - sims_to_centroid).mean().item()
        else:
            dispersion = 1.0

        # 5. Compute Instantaneous Shannon Entropy (if logits provided)
        entropy = 0.0
        if logits is not None:
            logits_vec = logits.view(-1).detach().float()
            probs = F.softmax(logits_vec, dim=-1)
            # Filter near-zero probabilities to prevent NaN
            nonzero_mask = probs > 1e-12
            probs_nz = probs[nonzero_mask]
            entropy = -(probs_nz * torch.log(probs_nz)).sum().item()

        self.history.append({
            "step": self.step,
            "velocity": v_t,
            "ema_velocity": self.ema_velocity,
            "ratio": r_t,
            "dispersion": dispersion,
            "entropy": entropy
        })

        # 6. Check Dynamical Equilibrium Condition
        if self.step >= self.min_warmup_tokens:
            # Trajectory is in an attractor if:
            # - Directional velocity is below the ceiling AND decelerating relative to EMA
            # OR window dispersion has contracted into an attractor basin
            is_velocity_stable = (v_t <= self.velocity_ceiling) and (r_t <= self.consensus_ratio_threshold)
            is_attractor_basin = (dispersion <= self.dispersion_ceiling) and (v_t <= self.velocity_ceiling * 1.5)

            is_candidate = is_velocity_stable or is_attractor_basin

            # Entropy safety: ensure model is not in high perplexity / confusion
            if logits is not None and entropy > self.entropy_threshold:
                is_candidate = False

            if is_candidate:
                self.consecutive_stabilized += 1
                if self.consecutive_stabilized >= self.debounce_tokens:
                    self.consensus_token = self.step
                    return True
            else:
                self.consecutive_stabilized = 0

        return False


def extract_reasoning_and_answer(raw_text: str) -> Tuple[str, str]:
    """
    Cleanly separates internal thinking (<think> ... </think>) from the final answer.
    Works universally across DeepSeek-R1, QwQ, and general reasoning formats.
    
    Returns:
        (reasoning_text, answer_text)
    """
    text = raw_text.strip()
    
    if "</think>" in text:
        parts = text.split("</think>", 1)
        reasoning = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return reasoning, answer
    elif "<think>" in text:
        reasoning = text.replace("<think>", "").strip()
        return reasoning, ""
    
    # No think tags present - whole output is the answer
    return "", text


def extract_clean_code(generated_text: str) -> str:
    """
    Universal multi-format code extractor.
    Handles all LLM emission styles across DeepSeek, Claude, GPT, and GLM:
    1. Embedded markdown code blocks (```python ... ```)
    2. Prompts continuing from or into markdown fences
    3. Raw Python functions/classes without markdown wrappers
    """
    text = generated_text.strip()
    
    # If the text has a </think> block, search after the think block first
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Format 1: Model emitted a standard full block somewhere in response
    block_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if block_match:
        return block_match.group(1).strip()

    # Format 2: Direct continuation from a prompt fence (e.g. ```python\n)
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
        
    if "```" in text:
        return text.split("```", 1)[0].strip()

    # Format 3: Raw code with def/class keywords
    lines = text.split("\n")
    code_lines = []
    inside_code = False
    for line in lines:
        if line.strip().startswith(("def ", "class ", "import ", "from ", "class:")):
            inside_code = True
        if inside_code:
            code_lines.append(line)
            
    if code_lines:
        return "\n".join(code_lines).strip()

    return text


def extract_clean_answer(generated_text: str) -> str:
    """
    Extracts the clean final answer for any task (math, general QA, or code).
    Strips internal thoughts and returns the final synthesized solution.
    """
    _, answer = extract_reasoning_and_answer(generated_text)
    if answer:
        return answer
    return generated_text.strip()
