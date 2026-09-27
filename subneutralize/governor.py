"""
Scale-Free Dynamical Consensus Governor Core.
"""

import re
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any


class ScaleFreeDynamicalGovernor:
    """
    Dimension-Invariant, Scale-Free Inference Governor for Autoregressive Reasoning Models.
    
    Monitors layer hidden state dynamics in real time via forward hooks.
    Detects the onset of the 'Overthinking Plateau' when the trajectory drops into a
    stable attractor basin, transitioning seamlessly to clean implementation emission.
    """
    def __init__(
        self,
        min_warmup_tokens: int = 60,
        consensus_ratio_threshold: float = 0.82,
        velocity_ceiling: float = 0.135,
        ema_alpha: float = 0.10,
        entropy_threshold: float = 1.20,
        debounce_tokens: int = 2,
    ):
        self.min_warmup_tokens = min_warmup_tokens
        self.consensus_ratio_threshold = consensus_ratio_threshold
        self.velocity_ceiling = velocity_ceiling
        self.ema_alpha = ema_alpha
        self.entropy_threshold = entropy_threshold
        self.debounce_tokens = debounce_tokens
        
        self.reset()

    def reset(self):
        """Resets dynamic tracking state for a new prompt."""
        self.step = 0
        self.last_hidden = None
        self.ema_velocity = None
        self.history: List[Dict[str, Any]] = []
        self.consensus_token: Optional[int] = None
        self.consecutive_stabilized = 0

    def update(self, hidden_state: torch.Tensor, logits: Optional[torch.Tensor] = None) -> bool:
        """
        Step-wise update called on every emitted token.
        
        Args:
            hidden_state: Tensor of shape (1, 1, d) or (1, d) from target bottleneck layer.
            logits: Optional Tensor of shape (1, vocab_size) to compute token entropy.
            
        Returns:
            bool: True if dynamical consensus equilibrium is reached, False otherwise.
        """
        self.step += 1
        h_t = hidden_state.view(1, -1).detach().float()
        
        if self.last_hidden is None:
            self.last_hidden = h_t
            return False

        # 1. Directional Cosine Velocity in latent space
        sim = F.cosine_similarity(h_t, self.last_hidden, dim=-1).clamp(-1.0, 1.0).item()
        v_t = 1.0 - sim
        self.last_hidden = h_t

        # 2. Update Exponential Moving Average of Velocity (EMA)
        if self.ema_velocity is None:
            self.ema_velocity = v_t
        else:
            self.ema_velocity = self.ema_alpha * v_t + (1.0 - self.ema_alpha) * self.ema_velocity

        # 3. Compute Dimension-Invariant Consensus Ratio R_t
        r_t = v_t / max(self.ema_velocity, 1e-6)

        # 4. Compute Instantaneous Shannon Entropy (if logits provided)
        entropy = 0.0
        if logits is not None:
            probs = F.softmax(logits.view(-1).detach().float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum().item()

        self.history.append({
            "step": self.step,
            "velocity": v_t,
            "ema_velocity": self.ema_velocity,
            "ratio": r_t,
            "entropy": entropy
        })

        # 5. Check Equilibrium Condition with Persistence Debounce
        if self.step >= self.min_warmup_tokens:
            is_candidate = (r_t < self.consensus_ratio_threshold) or (v_t < self.velocity_ceiling)
            
            # Entropy safety check: ensure model is not in high perplexity/confusion
            if logits is not None and entropy > self.entropy_threshold:
                is_candidate = False

            if is_candidate:
                self.consecutive_stabilized += 1
                if self.consecutive_stabilized >= self.debounce_tokens:
                    self.consensus_token = self.step
                    return True
            else:
                self.consecutive_stabilized = 0

        # Absolute Graceful Fallback: Runs unconstrained if consensus is not certified
        return False


def extract_clean_code(generated_text: str) -> str:
    """
    Universal multi-format code extractor.
    Handles all LLM emission styles across DeepSeek, Claude, GPT, and GLM:
    1. Embedded markdown code blocks (```python ... ```)
    2. Prompts continuing from or into markdown fences
    3. Raw Python functions/classes without markdown wrappers
    """
    text = generated_text.strip()
    
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
