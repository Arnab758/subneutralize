"""
SubNeutralize: A Parameter-Free Runtime Governor for Autoregressive Reasoning Models
===================================================================================

Author: Arnab Dutta
DOI: 10.5281/zenodo.22941619
Repository: https://github.com/Arnab758/subneutralize
License: AGPLv3 (Commercial licenses available)
"""

__version__ = "1.0.0"
__author__ = "Arnab Dutta"

from .governor import ConsensusEntropyGovernor
from .subspace import BiasSubspaceAnalyzer
from .hooks import ResidualStreamHook

__all__ = [
    "ConsensusEntropyGovernor",
    "BiasSubspaceAnalyzer",
    "ResidualStreamHook",
    "__version__",
    "__author__",
]
