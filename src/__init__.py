"""
Latent Reasoning Dynamics: Subspace Neutralization of Internal Bias
"""

from .consensus_entropy_governor import ConsensusEntropyGovernor
from .z_governor import SelfNormalizingZGovernor

__version__ = "0.2.0"
__all__ = ["ConsensusEntropyGovernor", "SelfNormalizingZGovernor"]
