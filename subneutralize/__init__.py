"""
SubNeutralize: Parameter-Free Runtime Inference Governor for Reasoning Models.

Eliminates the 'Overthinking Crisis' in large autoregressive reasoning models
by tracking latent representation dynamics and transitioning immediately upon
reaching dynamical consensus equilibrium.
"""

from .governor import ScaleFreeDynamicalGovernor, extract_clean_code
from .engine import SubNeutralize, govern

# Legacy alias for backward compatibility
ConsensusEntropyGovernor = ScaleFreeDynamicalGovernor

__version__ = "0.1.0"
__author__ = "Arnab Dutta"

__all__ = [
    "SubNeutralize",
    "ScaleFreeDynamicalGovernor",
    "ConsensusEntropyGovernor",
    "govern",
    "extract_clean_code",
    "__version__",
]
