"""
SubNeutralize: Parameter-Free Runtime Inference Governor & Transactional Action Firewall.

Eliminates the 'Overthinking Crisis' in large autoregressive reasoning models
by tracking latent representation dynamics, evicting multi-turn KV scratchpads,
and enforcing ACID-style transactional staging (`BEGIN`, `STAGE`, `VALIDATE`, `COMMIT` / `ROLLBACK`)
with Zero-Poisoning Memory Rewind for autonomous tool calls.
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
from .evictor import StatefulMultiTurnEvictor, RegionType
from .engine import SubNeutralize, GovernedOutput, govern, PhysicalCacheSlicer
from .vllm_adapter import (
    VLLMSubNeutralizeLogitsProcessor,
    VLLMMultiTurnMiddleware,
    create_vllm_sampling_params,
)
from .firewall import (
    TransactionalActionFirewall,
    FirewallPolicyConfig,
    ActionTransactionRecord,
    TransactionStatus,
    ToolCategory,
    UniversalSQLSavepointAdapter,
    SQLiteSavepointAdapter,
    FilesystemShadowAdapter,
)

# Legacy alias for backward compatibility
ConsensusEntropyGovernor = ScaleFreeDynamicalGovernor

__version__ = "0.3.0"
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
    "RegionType",
    "PhysicalCacheSlicer",
    "ConsensusEntropyGovernor",
    "govern",
    "extract_clean_code",
    "extract_clean_answer",
    "extract_reasoning_and_answer",
    "VLLMSubNeutralizeLogitsProcessor",
    "VLLMMultiTurnMiddleware",
    "create_vllm_sampling_params",
    "TransactionalActionFirewall",
    "FirewallPolicyConfig",
    "ActionTransactionRecord",
    "TransactionStatus",
    "ToolCategory",
    "UniversalSQLSavepointAdapter",
    "SQLiteSavepointAdapter",
    "FilesystemShadowAdapter",
    "__version__",
]
