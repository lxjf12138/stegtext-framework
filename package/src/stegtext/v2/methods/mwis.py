from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.types import DecisionPlan, MethodContext, PathRef, PromptState, TokenDistribution, VisibleTarget
from .direct import DirectMethod, DirectState, exact_realization_plan


class MWISMethod(DirectMethod):
    """Prefix-independent support baseline.

    The implementation keeps the runner contract independent; candidate
    filtering is deliberately explicit and deterministic.
    """

    def plan(self, state: DirectState, distributions: list[TokenDistribution], ctx: MethodContext) -> DecisionPlan:
        ids, probs = distributions[0].compact()
        order = torch.argsort(probs, descending=True)
        kept: list[int] = []
        keys: list[bytes] = []
        for pos in order.tolist():
            token_id = int(ids[pos].item())
            key = ctx.codec.token_bytes(token_id)
            if key and not any(key.startswith(old) or old.startswith(key) for old in keys):
                kept.append(token_id)
                keys.append(key)
        selected = torch.as_tensor(kept, dtype=torch.long, device=probs.device)
        return exact_realization_plan(selected, distributions[0], ctx)
