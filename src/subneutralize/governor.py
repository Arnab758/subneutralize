"""
Consensus Entropy Governor (SubNeutralize)
=========================================
Couples Riemannian trajectory velocity across a multi-layer cognitive 
bottleneck band with instantaneous Shannon entropy from the generation head.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, StoppingCriteria

from .subspace import BiasSubspaceAnalyzer
from .hooks import ResidualStreamHook


class SubNeutralizeStoppingCriteria(StoppingCriteria):
    """
    Standard Hugging Face StoppingCriteria wrapper that halts generation
    when the governor determines that reasoning has converged.
    """

    def __init__(self, governor: "ConsensusEntropyGovernor"):
        super().__init__()
        self.governor = governor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if scores is not None:
            self.governor.update_step(scores)
        return self.governor.should_halt()


class ConsensusEntropyGovernor:
    """
    Runtime inference governor for autoregressive reasoning models.
    
    1. Multi-Layer Consensus:
       v_t^consensus = max_{l in B} v_t^(l)
       Ensures generation continues if ANY monitored layer is actively computing.
       
    2. Dynamic Entropy Scaling:
       epsilon_t(H_t) = clamp(epsilon_0 * (H_t / H_0), epsilon_min, epsilon_max)
       Protects fluent generation when entropy is low; halts circular self-doubt when high.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[AutoTokenizer] = None,
        layer_band: Optional[List[int]] = None,
        base_threshold: float = 0.06,
        ref_entropy: float = 1.2,
        min_threshold: float = 0.025,
        max_threshold: float = 0.090,
        min_reasoning_tokens: int = 40,
        max_stagnant_tokens: int = 12,
        lambda_exit: float = 0.08
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.base_threshold = base_threshold
        self.ref_entropy = ref_entropy
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.min_reasoning_tokens = min_reasoning_tokens
        self.max_stagnant_tokens = max_stagnant_tokens
        self.lambda_exit = lambda_exit

        # Resolve model layers (supports LLaMA, Qwen, DeepSeek, Mistral)
        self.layers = self._extract_layers(model)
        self.num_layers = len(self.layers)

        # Dynamic midpoint consensus band
        if layer_band is None:
            mid = self.num_layers // 2
            self.layer_band = [max(0, mid - 2), mid, min(self.num_layers - 1, mid + 2)]
        else:
            self.layer_band = layer_band

        self.analyzer = BiasSubspaceAnalyzer()
        self.hooks: List[ResidualStreamHook] = []

        # Runtime state
        self.trajectories: Dict[int, List[torch.Tensor]] = {idx: [] for idx in self.layer_band}
        self.last_velocities: Dict[int, float] = {idx: 1.0 for idx in self.layer_band}
        self.step_consensus_v: float = 1.0
        self.step_entropy: float = ref_entropy
        self.step_threshold: float = base_threshold
        self.stagnant_count: int = 0
        self.tokens_generated: int = 0
        self.is_active: bool = False
        self.halt_requested: bool = False

        self._init_exit_vectors()

    def _extract_layers(self, model: nn.Module) -> List[nn.Module]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        elif hasattr(model, "layers"):
            return list(model.layers)
        else:
            raise ValueError("Unsupported model architecture. Could not locate decoder layers.")

    def _init_exit_vectors(self):
        """Initializes directional guidance vector u_exit from embeddings if available."""
        self.u_exit: Optional[torch.Tensor] = None
        if self.tokenizer is not None and hasattr(self.model, "get_input_embeddings"):
            try:
                think_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
                wait_ids = self.tokenizer.encode("Wait", add_special_tokens=False)
                if len(think_ids) > 0 and len(wait_ids) > 0:
                    embed_w = self.model.get_input_embeddings().weight.data
                    w_think = embed_w[think_ids[0]].to(torch.float32)
                    w_wait = embed_w[wait_ids[0]].to(torch.float32)
                    self.u_exit = F.normalize(w_think - w_wait, dim=-1).to(embed_w.device, dtype=embed_w.dtype)
            except Exception:
                self.u_exit = None

    def reset(self):
        """Resets tracking buffers for a new inference call."""
        for idx in self.layer_band:
            self.trajectories[idx].clear()
            self.last_velocities[idx] = 1.0
        self.step_consensus_v = 1.0
        self.step_entropy = self.ref_entropy
        self.step_threshold = self.base_threshold
        self.stagnant_count = 0
        self.tokens_generated = 0
        self.halt_requested = False

    def _hook_callback(self, layer_idx: int, module: nn.Module, inputs: Tuple, outputs: Union[Tuple, torch.Tensor]):
        if not self.is_active:
            return outputs

        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        # Skip prompt evaluation (prefill phase)
        if hidden.shape[1] > 1:
            return outputs

        curr_h = hidden[:, -1, :].clone()

        # 1. Project out prompt bias
        if self.analyzer.V_bias is not None:
            curr_h = self.analyzer.project_orthogonal(curr_h)

        # 2. Track layer velocity
        traj = self.trajectories[layer_idx]
        if len(traj) > 0:
            prev_h = traj[-1]
            cos_sim = F.cosine_similarity(curr_h.float(), prev_h.float(), dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)
            self.last_velocities[layer_idx] = v_t
        else:
            self.last_velocities[layer_idx] = 1.0

        traj.append(curr_h.clone())

        # 3. Soft exit steering if consensus reached
        mid_idx = self.layer_band[len(self.layer_band) // 2]
        if layer_idx == mid_idx and self.step_consensus_v < self.step_threshold and self.u_exit is not None:
            curr_h = curr_h + self.lambda_exit * self.u_exit.to(curr_h.device)

        curr_h = curr_h.to(hidden.dtype)
        if isinstance(outputs, tuple):
            return (torch.cat([hidden[:, :-1, :], curr_h.unsqueeze(1)], dim=1),) + outputs[1:]
        else:
            return torch.cat([hidden[:, :-1, :], curr_h.unsqueeze(1)], dim=1)

    def attach(self):
        """Attaches residual stream forward hooks."""
        self.reset()
        self.is_active = True
        for idx in self.layer_band:
            hook = ResidualStreamHook(self.layers[idx], idx, self._hook_callback)
            hook.register()
            self.hooks.append(hook)

    def detach(self):
        """Detaches all forward hooks safely."""
        self.is_active = False
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def update_step(self, scores: Optional[Union[torch.Tensor, List[torch.Tensor]]]):
        """Updates Shannon entropy and checks consensus criteria at generation step."""
        self.tokens_generated += 1

        # Calculate Shannon entropy H_t
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

        # Dynamic threshold scaling
        ratio = self.step_entropy / max(0.01, self.ref_entropy)
        raw_th = self.base_threshold * ratio
        self.step_threshold = max(self.min_threshold, min(self.max_threshold, raw_th))

        # Minimax consensus velocity
        self.step_consensus_v = max(self.last_velocities.values())

        # Check for sustained stagnation
        if self.tokens_generated >= self.min_reasoning_tokens:
            if self.step_consensus_v < self.step_threshold:
                self.stagnant_count += 1
                if self.stagnant_count >= self.max_stagnant_tokens:
                    self.halt_requested = True
            else:
                self.stagnant_count = max(0, self.stagnant_count - 1)

    def should_halt(self) -> bool:
        return self.halt_requested

    def as_stopping_criteria(self) -> SubNeutralizeStoppingCriteria:
        """Returns standard Hugging Face StoppingCriteria object."""
        return SubNeutralizeStoppingCriteria(self)
