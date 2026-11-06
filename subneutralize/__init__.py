"""
SubNeutralize: Parameter-Free Runtime Inference Governor for Reasoning Models.

Eliminates the 'Overthinking Crisis' in large autoregressive reasoning models
by tracking latent representation dynamics and transitioning immediately upon
reaching dynamical consensus equilibrium.
"""

from .governor import (
    ScaleFreeDynamicalGovernor,
    ManifoldDynamicalGovernor,
    extract_clean_code,
    extract_clean_answer,
    extract_reasoning_and_answer,
)
from .manifold import (
    PromptSubspaceGovernor,
    SubNeutralizeManifoldProcessor,
    OnlineManifoldStat,
)
from .evictor import StatefulMultiTurnEvictor
from .engine import SubNeutralize, GovernedOutput, govern
from .vllm_adapter import (
    VLLMSubNeutralizeLogitsProcessor,
    VLLMMultiTurnMiddleware,
    create_vllm_sampling_params,
)

# Legacy alias for backward compatibility
ConsensusEntropyGovernor = ScaleFreeDynamicalGovernor

__version__ = "0.2.0"
__author__ = "Arnab Dutta"

__all__ = [
    "SubNeutralize",
    "GovernedOutput",
    "ScaleFreeDynamicalGovernor",
    "ManifoldDynamicalGovernor",
    "PromptSubspaceGovernor",
    "SubNeutralizeManifoldProcessor",
    "OnlineManifoldStat",
    "StatefulMultiTurnEvictor",
    "ConsensusEntropyGovernor",
    "govern",
    "extract_clean_code",
    "extract_clean_answer",
    "extract_reasoning_and_answer",
    "VLLMSubNeutralizeLogitsProcessor",
    "VLLMMultiTurnMiddleware",
    "create_vllm_sampling_params",
    "__version__",
]
