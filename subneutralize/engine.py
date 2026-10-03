"""
High-Level Model Wrapper and Generation Engine for SubNeutralize.

Provides KV-cache-preserving in-flight transitions, chat-template handling,
and universal reasoning/code extraction.
"""

import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Tuple
from .governor import (
    ScaleFreeDynamicalGovernor,
    extract_clean_code,
    extract_clean_answer,
    extract_reasoning_and_answer,
)


@dataclass
class GovernedOutput:
    """Encapsulates results from SubNeutralize-governed generation."""
    text: str
    clean_code: str
    answer: str
    reasoning: str
    thinking_tokens: int
    code_tokens: int
    answer_tokens: int
    total_tokens: int
    wall_clock_seconds: float
    consensus_reached: bool
    consensus_step: Optional[int] = None
    history: Optional[List[Dict[str, Any]]] = None


def sample_next_token(logits: torch.Tensor, temperature: float = 0.6, top_p: float = 0.95) -> torch.Tensor:
    """Samples next token with temperature and nucleus (top-p) truncation."""
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / max(1e-5, temperature)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Truncate tokens whose cumulative probability exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least the first token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        scaled_logits = scaled_logits.masked_fill(indices_to_remove, -float("inf"))

    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class SubNeutralize:
    """
    SubNeutralize Runtime Inference Governor Wrapper.
    
    Attaches non-destructively to an autoregressive model's cognitive bottleneck layer,
    tracks latent Riemannian velocity, and transitions dynamically to code/answer emission
    upon reaching dynamical consensus equilibrium while preserving 100% of the KV cache.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        target_layer: Optional[int] = None,
        warmup_tokens: int = 50,
        consensus_threshold: float = 0.85,
        velocity_ceiling: float = 0.12,
        dispersion_ceiling: float = 0.08,
        debounce_tokens: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = torch.device("cpu")
        
        # 1. Resolve cognitive bottleneck layer (defaults to 50% depth)
        self.target_layer = target_layer
        self.layer_module = self._resolve_target_layer(target_layer) if model is not None else None
        
        # 2. Initialize Dynamical Governor
        self.governor = ScaleFreeDynamicalGovernor(
            min_warmup_tokens=warmup_tokens,
            consensus_ratio_threshold=consensus_threshold,
            velocity_ceiling=velocity_ceiling,
            dispersion_ceiling=dispersion_ceiling,
            debounce_tokens=debounce_tokens,
        )
        
        # Internal hook tracking
        self._current_hidden: Optional[torch.Tensor] = None
        self._hook_handle = None

    def _resolve_target_layer(self, target_layer: Optional[int]):
        """Detects model architecture and hooks into cognitive midpoint."""
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
            
        if layers is None:
            raise ValueError("Unable to automatically detect transformer layers in model. Please provide target_layer explicitly.")
            
        num_layers = len(layers)
        if target_layer is None:
            # Cognitive bottleneck band is universally situated at ~50% depth
            self.target_layer = num_layers // 2
        else:
            self.target_layer = target_layer
            
        return layers[self.target_layer]

    def attach(self):
        """Attaches forward hook to model's cognitive bottleneck layer."""
        if self._hook_handle is None and self.layer_module is not None:
            self._hook_handle = self.layer_module.register_forward_hook(self._hook_fn)

    def detach(self):
        """Detaches forward hook safely."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def as_stopping_criteria(self):
        """
        Returns a Hugging Face compatible StoppingCriteria callable.
        Allows drop-in integration with native model.generate(..., stopping_criteria=[...]).
        """
        self.attach()
        self.governor.reset()

        class _SubNeutralizeCriteria:
            def __init__(criteria_self, parent):
                criteria_self.parent = parent

            def __call__(criteria_self, input_ids: torch.LongTensor, scores: Optional[torch.FloatTensor] = None, **kwargs) -> Any:
                if criteria_self.parent._current_hidden is not None:
                    is_stop = criteria_self.parent.governor.update(
                        criteria_self.parent._current_hidden, 
                        logits=scores if scores is not None else None
                    )
                    if isinstance(is_stop, torch.Tensor):
                        return is_stop
                    if input_ids.shape[0] == 1:
                        return bool(is_stop)
                    return torch.tensor([bool(is_stop)] * input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                
                if input_ids.shape[0] == 1:
                    return False
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        return _SubNeutralizeCriteria(self)

    def _hook_fn(self, module, inp, outp):
        """Captures last token's residual hidden state non-destructively."""
        hidden = outp[0] if isinstance(outp, tuple) else outp
        # Extract the latest generated token's representation
        self._current_hidden = hidden[:, -1, :].detach()

    def _format_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """Applies model chat template if available to structure reasoning prompt."""
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None) is not None:
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                return formatted
            except Exception:
                pass
        if isinstance(prompt, list):
            parts = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in prompt]
            return "\n\n".join(parts) + "\n\nAssistant:"
        return prompt

    def _get_eos_ids(self) -> set:
        """Collects all valid EOS and stop token IDs for the tokenizer."""
        eos_ids = set()
        if self.tokenizer is not None and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                eos_ids.update(self.tokenizer.eos_token_id)
            else:
                eos_ids.add(self.tokenizer.eos_token_id)
        if self.tokenizer is not None and hasattr(self.tokenizer, "additional_special_tokens_ids"):
            eos_ids.update(self.tokenizer.additional_special_tokens_ids)
        return eos_ids

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 1500,
        answer_budget_tokens: int = 800,
        temperature: float = 0.6,
        top_p: float = 0.95,
        synthesis_prefix: Optional[str] = None,
    ) -> GovernedOutput:
        """
        Executes SubNeutralize governed generation with full KV-cache preservation.
        
        Args:
            prompt: User coding / reasoning / math prompt.
            max_new_tokens: Maximum total generation ceiling.
            answer_budget_tokens: Maximum tokens allocated for answer emission after consensus.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            synthesis_prefix: Optional prefix injected upon reaching equilibrium.
                              If None, standard newline / answer transition is used.
            
        Returns:
            GovernedOutput with full text, clean answer, code, token counts, and telemetry.
        """
        self.governor.reset()
        self._current_hidden = None
        self.attach()

        start_time = time.time()
        formatted_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        generated_ids = input_ids.clone()
        past_key_values = None
        eos_token_ids = self._get_eos_ids()

        consensus_reached = False
        think_tokens = 0
        answer_tokens = 0
        in_thinking_phase = True

        try:
            for step in range(max_new_tokens):
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = self.model(generated_ids, use_cache=True)
                    else:
                        outputs = self.model(generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)

                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

                next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)
                token_val = next_token.item()

                if token_val in eos_token_ids:
                    break

                emitted_chunk = self.tokenizer.decode(next_token[0], skip_special_tokens=False)

                if in_thinking_phase:
                    think_tokens += 1
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                    # Check for natural end of thinking </think>
                    if "</think>" in emitted_chunk:
                        in_thinking_phase = False
                        # Detach hook to avoid overhead during answer generation
                        self.detach()
                        continue

                    # Update dynamical governor
                    if self._current_hidden is not None:
                        is_equilibrium = self.governor.update(self._current_hidden, logits=logits)
                        if is_equilibrium:
                            consensus_reached = True
                            in_thinking_phase = False
                            self.detach()

                            # In-flight transition: Inject closing tag and optional prefix
                            trans_text = "\n</think>\n"
                            if synthesis_prefix:
                                trans_text += synthesis_prefix

                            trans_ids = self.tokenizer.encode(trans_text, add_special_tokens=False, return_tensors="pt").to(self.device)
                            # Single fast forward pass to register transition tokens in the KV cache
                            with torch.no_grad():
                                trans_outputs = self.model(trans_ids, past_key_values=past_key_values, use_cache=True)
                            past_key_values = trans_outputs.past_key_values
                            generated_ids = torch.cat([generated_ids, trans_ids], dim=-1)
                            continue
                else:
                    # Emitting final answer
                    answer_tokens += 1
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    if answer_tokens >= answer_budget_tokens:
                        break

        finally:
            self.detach()

        elapsed = time.time() - start_time
        total_tokens = think_tokens + answer_tokens
        full_text = self.tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)

        reasoning, answer = extract_reasoning_and_answer(full_text)
        clean_code = extract_clean_code(answer or full_text)
        clean_ans = extract_clean_answer(full_text)

        return GovernedOutput(
            text=full_text,
            clean_code=clean_code,
            answer=clean_ans,
            reasoning=reasoning,
            thinking_tokens=think_tokens,
            code_tokens=answer_tokens,
            answer_tokens=answer_tokens,
            total_tokens=total_tokens,
            wall_clock_seconds=elapsed,
            consensus_reached=consensus_reached,
            consensus_step=self.governor.consensus_token,
            history=self.governor.history,
        )


def govern(model: torch.nn.Module, tokenizer: Any, **kwargs) -> SubNeutralize:
    """Convenience factory function."""
    return SubNeutralize(model, tokenizer, **kwargs)
