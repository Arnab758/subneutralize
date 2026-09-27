"""
Attention-level Subspace Neutralization (SubNeutralize) hook
and inference stopping controller for reasoning models.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from transformers import StoppingCriteria


class SubspaceNeutralizationController(StoppingCriteria):
    """
    Monitors latent trajectory dynamics during autoregressive generation.
    When a candidate solution reaches representational stability, it suppresses
    bias re-activation to prevent pathological second-guessing.
    """

    def __init__(
        self,
        tokenizer,
        prompt_len: int,
        target_layer_idx: int = 16,
        velocity_threshold: float = 0.05,
        min_reasoning_tokens: int = 40,
        max_reflection_tokens: int = 150,
    ):
        """
        Args:
            tokenizer: HF PreTrainedTokenizer
            prompt_len: Length of input prompt tokens
            target_layer_idx: Deep decoder layer index to monitor
            velocity_threshold: Cosine velocity below which reasoning is deemed stable
            min_reasoning_tokens: Minimum reasoning tokens before early exit is considered
            max_reflection_tokens: Maximum tokens permitted after candidate solution is reached
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.target_layer_idx = target_layer_idx
        self.velocity_threshold = velocity_threshold
        self.min_reasoning_tokens = min_reasoning_tokens
        self.max_reflection_tokens = max_reflection_tokens

        self.candidate_found = False
        self.candidate_token_idx = None
        self.recent_hidden_states: List[torch.Tensor] = []
        self.tokens_since_candidate = 0
        self.stop_triggered = False

    def reset(self, prompt_len: int):
        self.prompt_len = prompt_len
        self.candidate_found = False
        self.candidate_token_idx = None
        self.recent_hidden_states = []
        self.tokens_since_candidate = 0
        self.stop_triggered = False

    def update_hidden_state(self, current_hidden: torch.Tensor):
        """
        Feeds the latest step's residual state [1, 1, hidden_dim]
        to evaluate trajectory velocity.
        """
        h = current_hidden.squeeze().detach()
        self.recent_hidden_states.append(h)
        
        # Keep sliding window of last 5 states
        if len(self.recent_hidden_states) > 5:
            self.recent_hidden_states.pop(0)

        if len(self.recent_hidden_states) >= 2:
            h_curr = self.recent_hidden_states[-1]
            h_prev = self.recent_hidden_states[-2]
            cos_sim = torch.cosine_similarity(h_curr.unsqueeze(0), h_prev.unsqueeze(0)).item()
            velocity = 1.0 - cos_sim

            total_generated = len(self.recent_hidden_states)
            if not self.candidate_found and total_generated > self.min_reasoning_tokens:
                # If velocity drops below threshold, the model has converged on an answer
                if velocity < self.velocity_threshold:
                    self.candidate_found = True
                    self.candidate_token_idx = total_generated

        if self.candidate_found:
            self.tokens_since_candidate += 1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Called by Hugging Face generate() at every token step."""
        current_seq_len = input_ids.shape[-1]
        reasoning_tokens = current_seq_len - self.prompt_len

        # If candidate was found and model enters redundant reflection exceeding budget
        if self.candidate_found and self.tokens_since_candidate >= self.max_reflection_tokens:
            self.stop_triggered = True
            return True

        # Check for standard final answer delimiter (e.g. </think> or \boxed{})
        recent_tokens = input_ids[0, -10:].tolist()
        decoded_recent = self.tokenizer.decode(recent_tokens, skip_special_tokens=False)
        if "</think>" in decoded_recent:
            self.stop_triggered = True
            return True

        return False


class SubspaceAttentionHook:
    """
    Applies orthogonal projection operator P_perp to attention query representations
    to prevent re-activating prompt-level bias directions.
    """

    def __init__(self, P_perp: torch.Tensor):
        """
        Args:
            P_perp: Precomputed orthogonal projection matrix [hidden_dim, hidden_dim]
        """
        self.P_perp = P_perp
        self.is_active = False

    def hook_fn(self, module: nn.Module, input_tensor: Tuple, output_tensor: torch.Tensor):
        if not self.is_active or self.P_perp is None:
            return output_tensor

        # For self-attention query projection: q = h @ W_q
        # output_tensor shape: [batch, seq_len, num_heads, head_dim] or [batch, seq_len, hidden_dim]
        device = output_tensor.device
        P = self.P_perp.to(device).to(output_tensor.dtype)

        with torch.no_grad():
            if output_tensor.ndim == 3:
                # [batch, seq, hidden] -> project
                neutralized = torch.matmul(output_tensor, P)
                return neutralized
            elif output_tensor.ndim == 4:
                # Flatten heads, project, restore
                b, s, h, d = output_tensor.shape
                flattened = output_tensor.reshape(b, s, h * d)
                if flattened.shape[-1] == P.shape[0]:
                    neutralized = torch.matmul(flattened, P).reshape(b, s, h, d)
                    return neutralized

        return output_tensor
