"""StegText v2 public API.

The package exposes four composable axes: backend, sampling policy, method and
payload coder.
"""

from .core.runner import StegoRunner
from .core.types import (
    DecisionPlan,
    MethodContext,
    PathRef,
    PromptState,
    TokenDistribution,
    VisibleTarget,
)

__all__ = [
    "DecisionPlan",
    "MethodContext",
    "PathRef",
    "PromptState",
    "StegoRunner",
    "TokenDistribution",
    "VisibleTarget",
]
