from __future__ import annotations

from collections import OrderedDict

import torch

from ..core.errors import InvariantError, ReceiverMismatchError
from ..core.types import DecisionPlan, MethodContext, TokenDistribution, VisibleTarget
from .direct import DirectMethod, DirectState


class SyncPoolMethod(DirectMethod):
    """SyncPool's coarse-grained pool selection and synchronized sampling."""

    def __init__(self):
        self._geometry = OrderedDict()
        self._geometry_codec = None

    def plan(self, state: DirectState, distributions: list[TokenDistribution], ctx: MethodContext) -> DecisionPlan:
        if len(distributions) != 1:
            raise InvariantError("SyncPool requires exactly one model distribution")
        ids, _ = distributions[0].compact()
        # Cache only support-dependent geometry, never probability masses.
        # Keep the codec alive and clear on replacement to prevent stale keys.
        if self._geometry_codec is not ctx.codec:
            self._geometry.clear()
            self._geometry_codec = ctx.codec
        device = distributions[0].probs.device
        cpu_ids = ids.detach().to(device="cpu", dtype=torch.long)
        signature = (device, cpu_ids.numpy().tobytes())
        geometry = self._geometry.get(signature)
        if geometry is None:
            entries = []
            for token_id in cpu_ids.tolist():
                key = ctx.codec.token_bytes(token_id)
                if not key:
                    raise InvariantError(
                        f"positive-support token {token_id} has an empty visible realization"
                    )
                entries.append((key, token_id))
            entries.sort(key=lambda item: (item[0], item[1]))
            groups: list[bytes] = []
            members: list[list[int]] = []
            for key, token_id in entries:
                if groups and key.startswith(groups[-1]):
                    members[-1].append(token_id)
                else:
                    groups.append(key)
                    members.append([token_id])
            indices = [torch.as_tensor(group, dtype=torch.long, device=device) for group in members]
            geometry = (groups, members, indices)
            self._geometry[signature] = geometry
            if len(self._geometry) > 8:
                self._geometry.popitem(last=False)
        else:
            self._geometry.move_to_end(signature)
        groups, members, indices = geometry
        masses = [
            distributions[0].probs.index_select(0, member_ids).sum()
            for member_ids in indices
        ]
        return DecisionPlan(
            torch.stack(masses),
            {"keys": groups, "members": members, "dist": distributions[0]},
        )

    def observe(self, state: DirectState, plan: DecisionPlan, target: VisibleTarget, ctx: MethodContext) -> int:
        matches = [i for i, key in enumerate(plan.opaque["keys"]) if target.remaining.startswith(key)]
        if len(matches) != 1:
            raise ReceiverMismatchError(f"SyncPool receiver found {len(matches)} matching groups")
        # The shortest key identifies the ambiguity pool, but it is not the
        # token emitted by the sender.  Consumption is deferred until advance,
        # after synchronized sampling identifies the actual pool member.
        return matches[0]

    def advance(self, state: DirectState, plan: DecisionPlan, action: int, rng, ctx: MethodContext) -> DirectState:
        if int(action) < 0 or int(action) >= len(plan.opaque["members"]):
            raise IndexError("SyncPool action is outside the plan")
        member_ids = plan.opaque["members"][int(action)]
        dist: TokenDistribution = plan.opaque["dist"]
        probs = dist.probs.index_select(
            0,
            torch.as_tensor(member_ids, dtype=torch.long, device=dist.probs.device),
        )
        u = float(rng.random()) * float(probs.sum().item())
        acc = 0.0
        chosen = member_ids[-1]
        for token_id, value in zip(member_ids, probs.tolist()):
            acc += float(value)
            if u < acc:
                chosen = int(token_id)
                break
        payload = ctx.codec.token_bytes(chosen)
        if ctx.mode == "decode":
            if ctx.target is None:
                raise ReceiverMismatchError("SyncPool decode state has no visible target")
            ctx.target.consume(payload)
        path = state.path.append(chosen)
        resolved = state.resolved_bytes + payload
        prompt_len = len(ctx.prompt.token_ids)
        latent = ctx.codec.tokens_to_bytes(path.token_ids()[prompt_len:])
        if latent != resolved:
            raise InvariantError(
                "SyncPool visible output diverged from the synchronized token path"
            )
        return DirectState(
            path,
            resolved,
            ctx.codec.is_eos(chosen),
            max(0, path.depth - ctx.prompt.path.depth),
        )
