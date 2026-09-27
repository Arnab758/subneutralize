"""
PyTorch forward hook manager for extracting Transformer residual stream representations
and monitoring layer-wise hidden-state dynamics during autoregressive generation.
"""

from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn


class ResidualStreamProbe:
    """
    Attaches forward hooks to Transformer decoder layers to extract residual stream
    activations and attention dynamics without altering generation flow.
    """

    def __init__(self, model: nn.Module, target_layers: Optional[List[int]] = None):
        """
        Args:
            model: Hugging Face PreTrainedModel (e.g., Qwen2ForCausalLM, LlamaForCausalLM)
            target_layers: List of layer indices to probe. Defaults to all layers if None.
        """
        self.model = model
        self.layers = self._find_decoder_layers(model)
        self.num_layers = len(self.layers)

        if target_layers is None:
            self.target_layers = list(range(self.num_layers))
        else:
            self.target_layers = [idx for idx in target_layers if 0 <= idx < self.num_layers]

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.captured_hidden_states: Dict[int, List[torch.Tensor]] = {
            l: [] for l in self.target_layers
        }
        self.is_active = False

    def _find_decoder_layers(self, model: nn.Module) -> nn.ModuleList:
        """Finds the ModuleList of decoder layers across standard HF architectures."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        elif hasattr(model, "layers"):
            return model.layers
        else:
            raise ValueError(f"Could not locate decoder layers in model architecture: {type(model)}")

    def _create_hook(self, layer_idx: int) -> Callable:
        """Creates a forward hook to capture the output of a decoder layer."""
        def hook_fn(module, input_tensor, output_tensor):
            if not self.is_active:
                return
            # Decoder layer output is typically (hidden_states, ...) or just a tensor
            if isinstance(output_tensor, tuple):
                hidden = output_tensor[0]
            else:
                hidden = output_tensor

            # Store detached slice on CPU to avoid GPU VRAM exhaustion
            # Shape: [batch_size, seq_len, hidden_dim]
            with torch.no_grad():
                self.captured_hidden_states[layer_idx].append(hidden.detach().cpu())

        return hook_fn

    def attach(self):
        """Registers hooks to the specified decoder layers."""
        self.clear()
        self.hooks = []
        for l in self.target_layers:
            handle = self.layers[l].register_forward_hook(self._create_hook(l))
            self.hooks.append(handle)
        self.is_active = True

    def detach(self):
        """Removes all registered hooks."""
        self.is_active = False
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def clear(self):
        """Clears accumulated hidden states from memory."""
        self.captured_hidden_states = {l: [] for l in self.target_layers}

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach()

    def get_layer_trajectory(self, layer_idx: int) -> torch.Tensor:
        """
        Concatenates all captured steps for a specific layer into a single tensor.
        Returns:
            Tensor of shape [batch, total_tokens, hidden_dim]
        """
        if layer_idx not in self.captured_hidden_states:
            raise KeyError(f"Layer {layer_idx} was not probed.")
        if not self.captured_hidden_states[layer_idx]:
            raise ValueError(f"No hidden states captured for layer {layer_idx}.")

        return torch.cat(self.captured_hidden_states[layer_idx], dim=1)

    def get_prompt_representation(self, prompt_len: int, layer_idx: int) -> torch.Tensor:
        """
        Extracts the residual state at the terminal prompt token (x_N) for a layer.
        Returns:
            Tensor of shape [batch, hidden_dim]
        """
        trajectory = self.get_layer_trajectory(layer_idx)
        return trajectory[:, prompt_len - 1, :]
