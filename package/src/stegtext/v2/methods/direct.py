from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.errors import InvariantError, ReceiverMismatchError
from ..core.types import DecisionPlan, MethodContext, PathRef, PromptState, TokenDistribution, VisibleTarget


@dataclass(frozen=True)
class DirectState:
    path: PathRef
    resolved_bytes: bytes = b""
    terminal: bool = False
    tokens_used: int = 0


def exact_realization_plan(
    ids: torch.Tensor,
    distribution: TokenDistribution,
    ctx: MethodContext,
) -> DecisionPlan:
    """Coalesce token ids that are indistinguishable to the receiver."""
    keys: list[bytes] = []
    members: list[list[int]] = []
    positions: dict[bytes, int] = {}
    for raw_id in ids.detach().cpu().tolist():
        token_id = int(raw_id)
        key = ctx.codec.token_bytes(token_id)
        if not key:
            raise InvariantError(
                f"positive-support token {token_id} has an empty visible realization"
            )
        position = positions.get(key)
        if position is None:
            position = len(keys)
            positions[key] = position
            keys.append(key)
            members.append([])
        members[position].append(token_id)
    masses = [
        distribution.probs.index_select(
            0,
            torch.as_tensor(group, dtype=torch.long, device=distribution.probs.device),
        ).sum()
        for group in members
    ]
    probs = torch.stack(masses)
    ctx.stats["exact_realization_groups"] = len(members)
    ctx.stats["exact_duplicate_tokens"] = sum(max(0, len(group) - 1) for group in members)
    return DecisionPlan(
        probs=probs / probs.sum(),
        opaque={"keys": keys, "members": members, "dist": distribution},
    )


def select_exact_member(plan: DecisionPlan, action: int, rng) -> int:
    members = plan.opaque["members"][int(action)]
    if len(members) == 1:
        return int(members[0])
    dist: TokenDistribution = plan.opaque["dist"]
    weights = [float(dist.probs[int(token_id)].item()) for token_id in members]
    total = sum(weights)
    if total <= 0.0:
        raise InvariantError("exact-realization action has zero member mass")
    value = float(rng.random()) * total
    cumulative = 0.0
    for token_id, weight in zip(members, weights):
        cumulative += weight
        if value < cumulative:
            return int(token_id)
    return int(members[-1])


class DirectMethod:
    def start(self, prompt_state: PromptState) -> DirectState:
        return DirectState(prompt_state.path)

    def queries(self, state: DirectState) -> list[PathRef]:
        return [state.path]

    def plan(self, state: DirectState, distributions: list[TokenDistribution], ctx: MethodContext) -> DecisionPlan:
        ids, _ = distributions[0].compact()
        return exact_realization_plan(ids, distributions[0], ctx)

    def observe(self, state: DirectState, plan: DecisionPlan, target: VisibleTarget, ctx: MethodContext) -> int:
        members = plan.opaque["members"]
        keys = plan.opaque["keys"]
        if hasattr(ctx.codec, "next_token"):
            try:
                token_id = int(ctx.codec.next_token(state.path.token_ids(), target.remaining))
            except (ValueError, IndexError) as exc:
                raise ReceiverMismatchError("direct receiver cannot tokenize the visible target") from exc
            for action, group in enumerate(members):
                if token_id in group and target.remaining.startswith(keys[action]):
                    target.consume(keys[action])
                    return action
        matches = [action for action, key in enumerate(keys) if target.remaining.startswith(key)]
        if len(matches) != 1:
            raise ReceiverMismatchError(f"direct receiver found {len(matches)} matching realizations")
        target.consume(keys[matches[0]])
        return matches[0]

    def advance(self, state: DirectState, plan: DecisionPlan, action: int, rng, ctx: MethodContext) -> DirectState:
        members = plan.opaque["members"]
        if int(action) < 0 or int(action) >= len(members):
            raise IndexError("direct action is outside the plan")
        token_id = select_exact_member(plan, int(action), rng)
        payload = plan.opaque["keys"][int(action)]
        return DirectState(
            path=state.path.append(token_id),
            resolved_bytes=state.resolved_bytes + payload,
            terminal=ctx.codec.is_eos(token_id),
            tokens_used=state.tokens_used + 1,
        )

    def done(self, state: DirectState) -> bool:
        return state.terminal

    def output(self, state: DirectState, ctx: MethodContext) -> bytes:
        data = state.resolved_bytes
        while data.endswith(getattr(ctx.codec, "eos_bytes", b"")) and getattr(ctx.codec, "eos_bytes", b""):
            data = data[: -len(ctx.codec.eos_bytes)]
        return data
