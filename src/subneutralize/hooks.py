"""
PyTorch forward hooks for non-intrusive residual stream capture and projection.
"""

from typing import Callable, List, Optional
import torch
import torch.nn as nn


class ResidualStreamHook:
    """
    Manages lightweight forward hooks across targeted Transformer layers.
    Designed for zero allocation overhead during autoregressive decoding.
    """

    def __init__(self, layer: nn.Module, layer_idx: int, hook_fn: Callable):
        self.layer = layer
        self.layer_idx = layer_idx
        self.hook_fn = hook_fn
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def register(self):
        if self.handle is None:
            self.handle = self.layer.register_forward_hook(self._wrapper)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _wrapper(self, module, inputs, outputs):
        return self.hook_fn(self.layer_idx, module, inputs, outputs)
