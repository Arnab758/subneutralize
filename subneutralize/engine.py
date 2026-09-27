"""
High-Level Model Wrapper and Generation Engine for SubNeutralize.
"""

import time
import torch
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any
from .governor import ScaleFreeDynamicalGovernor, extract_clean_code


@dataclass
class GovernedOutput:
    """Encapsulates results from SubNeutralize-governed generation."""
    text: str
    clean_code: str
    thinking_tokens: int
    code_tokens: int
    total_tokens: int
    wall_clock_seconds: float
    consensus_reached: bool
    consensus_step: Optional[int] = None
    history: Optional[List[Dict[str, Any]]] = None


class SubNeutralize:
    """
    SubNeutralize Runtime Inference Governor Wrapper.
    
    Attaches non-destructively to an autoregressive model's cognitive bottleneck layer,
    tracks latent Riemannian velocity, and transitions dynamically to code/answer emission
    upon reaching dynamical consensus equilibrium.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        target_layer: Optional[int] = None,
        warmup_tokens: int = 60,
        consensus_threshold: float = 0.82,
        velocity_ceiling: float = 0.135,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # 1. Resolve cognitive bottleneck layer (defaults to 50% depth)
        self.target_layer = target_layer
        self.layer_module = self._resolve_target_layer(target_layer)
        
        # 2. Initialize Dynamical Governor
        self.governor = ScaleFreeDynamicalGovernor(
            min_warmup_tokens=warmup_tokens,
            consensus_ratio_threshold=consensus_threshold,
            velocity_ceiling=velocity_ceiling,
        )
        
        # Internal hook tracking
        self._current_hidden: Optional[torch.Tensor] = None
        self._hook_handle = None

    def _resolve_target_layer(self, target_layer: Optional[int]):
        """Detects model architecture and hooks into cognitive midpoint."""
        # Find layers container (Qwen, Llama, Mistral, DeepSeek)
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
            
        if layers is None:
            raise ValueError("Unable to automatically detect transformer layers in model. Please provide layer explicitly.")
            
        num_layers = len(layers)
        if target_layer is None:
            # Cognitive bottleneck band is universally situated at ~50% depth
            self.target_layer = num_layers // 2
        else:
            self.target_layer = target_layer
            
        return layers[self.target_layer]

    def attach(self):
        """Attaches forward hook to model's cognitive bottleneck layer."""
        if self._hook_handle is None:
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
            def __call__(criteria_self, input_ids: torch.LongTensor, scores: Optional[torch.FloatTensor] = None, **kwargs) -> bool:
                if criteria_self.parent._current_hidden is not None:
                    return criteria_self.parent.governor.update(
                        criteria_self.parent._current_hidden, 
                        logits=scores if scores is not None else None
                    )
                return False

        return _SubNeutralizeCriteria(self)

    def _hook_fn(self, module, inp, outp):
        """Captures last token's residual hidden state non-destructively."""
        hidden = outp[0] if isinstance(outp, tuple) else outp
        # Extract the latest generated token's representation
        self._current_hidden = hidden[:, -1, :].detach()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1500,
        code_budget_tokens: int = 600,
        temperature: float = 0.6,
        top_p: float = 0.95,
        synthesis_prefix: str = "\n```python\n",
    ) -> GovernedOutput:
        """
        Executes SubNeutralize governed generation.
        
        Args:
            prompt: User coding / reasoning prompt.
            max_new_tokens: Maximum total generation ceiling (identical for benchmark parity).
            code_budget_tokens: Maximum tokens allocated for clean code emission after consensus.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            synthesis_prefix: Transition prefix injected upon reaching equilibrium.
            
        Returns:
            GovernedOutput with generated text, clean code, token counts, and latency.
        """
        self.governor.reset()
        self._current_hidden = None
        
        # Attach hook to bottleneck layer
        self._hook_handle = self.layer_module.register_forward_hook(self._hook_fn)
        
        start_time = time.time()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        
        generated_ids = input_ids.clone()
        past_key_values = None
        
        consensus_reached = False
        think_tokens = 0
        
        # Stage 1: Autoregressive exploration monitored by Dynamical Governor
        try:
            for step in range(max_new_tokens):
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = self.model(generated_ids, use_cache=True)
                    else:
                        outputs = self.model(generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
                        
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                
                # Temperature & sampling
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                think_tokens += 1
                
                # Check for natural end of think </think>
                emitted_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                if "</think>" in emitted_text:
                    break
                    
                # Update governor with residual activation
                if self._current_hidden is not None:
                    is_equilibrium = self.governor.update(self._current_hidden, logits=logits)
                    if is_equilibrium:
                        consensus_reached = True
                        break
        finally:
            # Remove forward hook
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None

        # Stage 2: Synthesis & Code Emission
        full_text_stage1 = self.tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
        
        if consensus_reached:
            # Transition prompt into clean code generation mode
            synthesis_prompt = prompt + "\n" + full_text_stage1.strip() + "</think>" + synthesis_prefix
            syn_inputs = self.tokenizer(synthesis_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                code_gen = self.model.generate(
                    **syn_inputs,
                    max_new_tokens=code_budget_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                    use_cache=True
                )
                
            code_tokens = code_gen.shape[1] - syn_inputs["input_ids"].shape[1]
            raw_code_output = self.tokenizer.decode(code_gen[0, syn_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            final_text = synthesis_prefix + raw_code_output
        else:
            code_tokens = 0
            final_text = full_text_stage1
            
        elapsed = time.time() - start_time
        total_tokens = think_tokens + code_tokens
        clean_code = extract_clean_code(final_text)
        
        return GovernedOutput(
            text=final_text,
            clean_code=clean_code,
            thinking_tokens=think_tokens,
            code_tokens=code_tokens,
            total_tokens=total_tokens,
            wall_clock_seconds=elapsed,
            consensus_reached=consensus_reached,
            consensus_step=self.governor.consensus_token,
            history=self.governor.history
        )


def govern(model: torch.nn.Module, tokenizer: Any, **kwargs) -> SubNeutralize:
    """Convenience factory function."""
    return SubNeutralize(model, tokenizer, **kwargs)
