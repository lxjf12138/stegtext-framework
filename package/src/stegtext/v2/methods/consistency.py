from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.errors import InvariantError
from ..core.types import DecisionPlan, MethodContext, PathRef, PromptState, TokenDistribution, VisibleTarget
from .direct import DirectMethod, exact_realization_plan, select_exact_member


@dataclass(frozen=True)
class ConsistencyState:
    path: PathRef
    resolved_bytes: bytes = b""
    terminal: bool = False
    tokens_used: int = 0


class ConsistencyMethod:
    """Canonical/Stepwise method with history-dependent candidate validation."""

    def __init__(self, oracle, *, initial_k: int = 32, max_k: int | None = None) -> None:
        if initial_k <= 0:
            raise ValueError("initial_k must be positive")
        self.oracle = oracle
        self.initial_k = int(initial_k)
        self.max_k = None if max_k is None else int(max_k)

    def start(self, prompt_state: PromptState) -> ConsistencyState:
        return ConsistencyState(prompt_state.path)

    def queries(self, state: ConsistencyState) -> list[PathRef]:
        return [state.path]

    def plan(self, state: ConsistencyState, distributions: list[TokenDistribution], ctx: MethodContext) -> DecisionPlan:
        dist = distributions[0]
        ids, _ = dist.compact()
        order = torch.argsort(dist.probs.index_select(0, ids), descending=True)
        ordered = ids.index_select(0, order)
        k = min(self.initial_k, int(ordered.numel()))
        valid_ids = None
        while k > 0:
            candidate_ids = ordered[:k]
            mask = self.oracle.valid_mask(torch.as_tensor(state.path.token_ids(), dtype=torch.long), candidate_ids)
            if bool(mask.any().item()):
                valid_ids = candidate_ids[mask]
                break
            if k == int(ordered.numel()) or (self.max_k is not None and k >= self.max_k):
                break
            k = min(int(ordered.numel()), max(k + 1, k * 2))
        if valid_ids is None or valid_ids.numel() == 0:
            raise InvariantError("Consistency found no canonical candidate")
        ctx.stats["stepwise_valid_support"] = int(valid_ids.numel())
        return exact_realization_plan(valid_ids, dist, ctx)

    def observe(self, state: ConsistencyState, plan: DecisionPlan, target: VisibleTarget, ctx: MethodContext) -> int:
        return DirectMethod.observe(self, state, plan, target, ctx)

    def advance(self, state: ConsistencyState, plan: DecisionPlan, action: int, rng, ctx: MethodContext) -> ConsistencyState:
        token_id = select_exact_member(plan, int(action), rng)
        return ConsistencyState(
            path=state.path.append(token_id),
            resolved_bytes=state.resolved_bytes + plan.opaque["keys"][int(action)],
            terminal=ctx.codec.is_eos(token_id),
            tokens_used=state.tokens_used + 1,
        )

    def done(self, state: ConsistencyState) -> bool:
        return state.terminal

    def output(self, state: ConsistencyState, ctx: MethodContext) -> bytes:
        data = state.resolved_bytes
        eos = getattr(ctx.codec, "eos_bytes", b"")
        while eos and data.endswith(eos):
            data = data[: -len(eos)]
        return data


# The paper and historical scripts use both names for the same canonicality
# protocol.  Keeping an alias avoids a second implementation and a second
# receiver state machine.
StepwiseMethod = ConsistencyMethod
