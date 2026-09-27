"""
Latent Reasoning Dynamics: Subspace Neutralization of Internal Bias
"""

try:
    from .consensus_entropy_governor import ConsensusEntropyGovernor
except ImportError:
    ConsensusEntropyGovernor = None

try:
    from .z_governor import SelfNormalizingZGovernor
except ImportError:
    SelfNormalizingZGovernor = None

__version__ = "0.2.0"
__all__ = ["ConsensusEntropyGovernor", "SelfNormalizingZGovernor"]
