#!/usr/bin/env python3
"""Shared belief-level planning and FullSync kernels for Round-4.1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from round3_common import REPO_ROOT
from round4_common import _linear_retained_assignment


ROUND4_1_DIR = REPO_ROOT / "engineering_survey" / "round4_1"


def xlogx(value: float) -> float:
    value = float(value)
    return 0.0 if value <= 0.0 else value * math.log2(value)


def entropy_from_aggregates(
    group_mass: float,
    exact_mass: float,
    strict_xlogx_sum: float,
) -> float:
    """Compute H(alpha, q_strict...) from unnormalized masses."""

    total = float(group_mass)
    if total <= 0.0:
        raise ValueError("posterior group mass must be positive")
    value = math.log2(total) - (
        xlogx(float(exact_mass)) + float(strict_xlogx_sum)
    ) / total
    return max(0.0, float(value))


@dataclass(frozen=True)
class BeliefCandidate:
    kind: str
    path: Any
    remaining_visible: bytes
    mass: float
    terminal: bool
    token_id: int | None = None
    retained_id: int | None = None

    def signature(self) -> tuple[Any, ...]:
        return (
            self.kind,
            self.path.token_ids(),
            self.remaining_visible,
            bool(self.terminal),
            self.token_id,
            self.retained_id,
        )


def _record_fresh_ids(record: dict[str, Any], index) -> list[int]:
    explicit = record.get("fresh_ids", ())
    if explicit:
        return [int(value) for value in explicit]
    return [
        int(token_id)
        for group in record.get("static", ())
        for token_id in index.lex_token_ids_cpu[group.lo : group.hi].tolist()
    ]


def materialize_candidates(state, plan, action: int, index, codec) -> list[BeliefCandidate]:
    """Materialize only the already-selected group for an advance/reference check."""

    record = plan.opaque["records"][int(action)]
    key = bytes(record["key"])
    dist = record.get("dist")
    fresh_ids = _record_fresh_ids(record, index)
    candidates: list[BeliefCandidate] = []
    if dist is not None and fresh_ids:
        positions = torch.as_tensor(
            fresh_ids, dtype=torch.long, device=dist.probs.device
        )
        weights = (
            dist.probs.index_select(0, positions)
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .tolist()
        )
        for token_id, probability in zip(fresh_ids, weights):
            payload = bytes(index.token_bytes[int(token_id)])
            mass = float(state.expansion_mass) * float(probability)
            if mass <= 0.0 or not payload.startswith(key):
                continue
            candidates.append(
                BeliefCandidate(
                    kind="fresh",
                    path=state.base_path.append(int(token_id)),
                    remaining_visible=payload,
                    mass=mass,
                    terminal=bool(codec.is_eos(int(token_id))),
                    token_id=int(token_id),
                )
            )
    for retained_id in record.get("retained_ids", ()):
        retained_id = int(retained_id)
        retained = state.retained[retained_id]
        mass = float(retained.mass)
        residual = bytes(retained.residual)
        if mass <= 0.0 or not residual.startswith(key):
            continue
        candidates.append(
            BeliefCandidate(
                kind="retained",
                path=retained.path,
                remaining_visible=residual,
                mass=mass,
                terminal=bool(retained.terminal),
                retained_id=retained_id,
            )
        )
    if not candidates:
        raise AssertionError("selected belief group has no positive-mass candidates")
    return candidates


def explicit_selected_metrics(state, plan, action: int, index, codec) -> dict[str, Any]:
    candidates = materialize_candidates(state, plan, action, index, codec)
    key = bytes(plan.opaque["records"][int(action)]["key"])
    total = sum(item.mass for item in candidates)
    exact = [item for item in candidates if item.remaining_visible == key]
    strict = [item for item in candidates if item.remaining_visible != key]
    exact_mass = sum(item.mass for item in exact)
    strict_xlogx_sum = sum(xlogx(item.mass) for item in strict)
    return {
        "group_mass": total,
        "exact_mass": exact_mass,
        "strict_mass": sum(item.mass for item in strict),
        "strict_xlogx_sum": strict_xlogx_sum,
        "exact_count": len(exact),
        "strict_count": len(strict),
        "las_posterior_entropy_bits": entropy_from_aggregates(
            total, exact_mass, strict_xlogx_sum
        ),
        "candidate_signatures": [item.signature() for item in candidates],
        "candidate_masses": [item.mass for item in candidates],
    }


def selected_cheap_metrics(plan, action: int) -> dict[str, Any]:
    record = plan.opaque["records"][int(action)]
    total = float(record["group_mass"])
    strict_mass = float(record["strict_mass"])
    return {
        "selected_group_key_hex": bytes(record["key"]).hex(),
        "selected_group_mass": total,
        "selected_group_exact_mass": float(record["exact_mass"]),
        "projected_strict_mass": strict_mass,
        "projected_strict_posterior_mass": strict_mass / total,
        "projected_next_frontier": int(record["strict_count"]),
        "exact_candidate_count": int(record["exact_count"]),
        "las_posterior_entropy_bits": float(
            record["las_posterior_entropy_bits"]
        ),
    }


def candidate_mass_map(state, plan, action: int, index, codec) -> dict[tuple[Any, ...], float]:
    result: dict[tuple[Any, ...], float] = {}
    for candidate in materialize_candidates(state, plan, action, index, codec):
        signature = candidate.signature()
        result[signature] = result.get(signature, 0.0) + float(candidate.mass)
    return result


def advance_belief_fullsync(state, plan, action: int, rng, ctx, index):
    """Collapse the selected belief group to one complete latent candidate."""

    from stegtext.v2.core.errors import InvariantError, ReceiverMismatchError
    from stegtext.v2.methods.las import LASState

    records = plan.opaque["records"]
    if int(action) < 0 or int(action) >= len(records):
        raise IndexError("BeliefFullSync action is outside the common plan")
    record = records[int(action)]
    key = bytes(record["key"])
    candidates = materialize_candidates(state, plan, int(action), index, ctx.codec)
    total = sum(candidate.mass for candidate in candidates)
    if total <= 0.0:
        raise InvariantError("BeliefFullSync selected group has zero mass")
    draw = float(rng.random()) * total
    cumulative = 0.0
    chosen_index = len(candidates) - 1
    for position, candidate in enumerate(candidates):
        cumulative += float(candidate.mass)
        if draw < cumulative:
            chosen_index = position
            break
    chosen = candidates[chosen_index]
    if not chosen.remaining_visible.startswith(key):
        raise InvariantError("BeliefFullSync candidate does not extend selected group")
    if ctx.mode == "decode":
        if ctx.target is None:
            raise ReceiverMismatchError("BeliefFullSync decode state has no target")
        # LAS-style observe already consumed the common group key.
        ctx.target.consume(chosen.remaining_visible[len(key) :])
    prompt_depth = len(getattr(getattr(ctx, "prompt", None), "token_ids", ()))
    next_state = LASState(
        base_path=chosen.path,
        expansion_mass=0.0 if chosen.terminal else 1.0,
        retained=(),
        resolved_bytes=state.resolved_bytes + chosen.remaining_visible,
        terminal=bool(chosen.terminal),
        tokens_used=max(0, chosen.path.depth - prompt_depth),
    )
    ctx.stats["belief_fullsync_candidate_index"] = chosen_index
    ctx.stats["belief_fullsync_candidate_kind"] = chosen.kind
    ctx.stats["belief_fullsync_candidate_signature"] = chosen.signature()
    ctx.stats["belief_fullsync_candidate_mass"] = float(chosen.mass)
    return next_state


def make_belief_planner(index):
    """Return the optimized common planner used by both LAS and FullSync."""

    from stegtext.v2.core.errors import InvariantError
    from stegtext.v2.core.types import DecisionPlan, TokenDistribution
    from stegtext.v2.methods.las import LASMethod

    class BeliefPlanner(LASMethod):
        def __init__(self) -> None:
            super().__init__(index, optimized=True)
            lex_group_ids = torch.empty(index.vocab_size, dtype=torch.long)
            for group_id, group in enumerate(index.base_groups):
                lex_group_ids[group.lo : group.hi] = group_id
            self.lex_group_ids = lex_group_ids

        def plan(self, state, distributions, ctx):
            if state.expansion_mass > 0.0:
                if len(distributions) != 1:
                    raise InvariantError(
                        "active belief fresh mass requires one distribution"
                    )
                dist: TokenDistribution | None = distributions[0]
            else:
                if distributions:
                    raise InvariantError(
                        "belief planner received a distribution with no fresh mass"
                    )
                dist = None

            residuals = [item.residual for item in state.retained if item.residual]
            base_masses = base_xlogx = base_counts = None
            fresh_lex_masses = None
            if dist is None:
                sparse_support = True
                dynamic = self._retained_groups(residuals)
            else:
                self._validate_visible_support(dist)
                ids, _ = dist.compact()
                sparse_support = int(ids.numel()) < self.index.vocab_size
                if sparse_support:
                    dynamic = self._support_groups(ids, residuals)
                else:
                    dynamic = self.index.dynamic_groups(residuals)
                    lex_ids = self.index.lex_token_ids_gpu.to(dist.probs.device)
                    fresh_lex_masses = (
                        dist.probs.index_select(0, lex_ids).to(torch.float64)
                        * float(state.expansion_mass)
                    )
                    group_ids = self.lex_group_ids.to(dist.probs.device)
                    base_masses = torch.zeros(
                        len(self.index.base_groups),
                        dtype=torch.float64,
                        device=dist.probs.device,
                    )
                    base_masses.scatter_add_(0, group_ids, fresh_lex_masses)
                    per_candidate_xlogx = torch.where(
                        fresh_lex_masses > 0,
                        fresh_lex_masses * torch.log2(fresh_lex_masses),
                        torch.zeros_like(fresh_lex_masses),
                    )
                    base_xlogx = torch.zeros_like(base_masses)
                    base_xlogx.scatter_add_(0, group_ids, per_candidate_xlogx)
                    base_counts = torch.zeros_like(base_masses, dtype=torch.long)
                    base_counts.scatter_add_(
                        0, group_ids, (fresh_lex_masses > 0).to(torch.long)
                    )

            assignments = _linear_retained_assignment(dynamic, state.retained)
            records: list[dict[str, Any]] = []
            masses: list[float] = []
            for (key, static_indices, fresh_ids), retained_ids in zip(
                dynamic, assignments
            ):
                static = [self.index.base_groups[position] for position in static_indices]
                fresh_mass = 0.0
                fresh_xlogx = 0.0
                fresh_count = 0
                fresh_exact_mass = 0.0
                fresh_exact_xlogx = 0.0
                fresh_exact_count = 0
                if dist is not None and fresh_ids:
                    positions = torch.as_tensor(
                        fresh_ids, dtype=torch.long, device=dist.probs.device
                    )
                    raw = (
                        dist.probs.index_select(0, positions).to(torch.float64)
                        * float(state.expansion_mass)
                    )
                    weights = raw.detach().cpu().tolist()
                    for token_id, mass in zip(fresh_ids, weights):
                        mass = float(mass)
                        if mass <= 0.0:
                            continue
                        fresh_mass += mass
                        fresh_xlogx += xlogx(mass)
                        fresh_count += 1
                        if bytes(self.index.token_bytes[int(token_id)]) == bytes(key):
                            fresh_exact_mass += mass
                            fresh_exact_xlogx += xlogx(mass)
                            fresh_exact_count += 1
                elif dist is not None and static_indices:
                    positions = torch.as_tensor(
                        static_indices, dtype=torch.long, device=base_masses.device
                    )
                    fresh_mass = float(base_masses.index_select(0, positions).sum().item())
                    fresh_xlogx = float(base_xlogx.index_select(0, positions).sum().item())
                    fresh_count = int(base_counts.index_select(0, positions).sum().item())
                    for static_index in static_indices:
                        group = self.index.base_groups[int(static_index)]
                        if bytes(group.key) != bytes(key):
                            continue
                        exact_positions = list(group.exact_positions)
                        if exact_positions:
                            exact_tensor = torch.as_tensor(
                                exact_positions,
                                dtype=torch.long,
                                device=fresh_lex_masses.device,
                            )
                            values = fresh_lex_masses.index_select(0, exact_tensor)
                            fresh_exact_mass += float(values.sum().item())
                            fresh_exact_xlogx += float(
                                torch.where(
                                    values > 0,
                                    values * torch.log2(values),
                                    torch.zeros_like(values),
                                ).sum().item()
                            )
                            fresh_exact_count += int((values > 0).sum().item())

                retained_mass = 0.0
                retained_xlogx = 0.0
                retained_count = 0
                retained_exact_mass = 0.0
                retained_exact_xlogx = 0.0
                retained_exact_count = 0
                for retained_id in retained_ids:
                    retained = state.retained[int(retained_id)]
                    mass = float(retained.mass)
                    if mass <= 0.0:
                        continue
                    retained_mass += mass
                    retained_xlogx += xlogx(mass)
                    retained_count += 1
                    if bytes(retained.residual) == bytes(key):
                        retained_exact_mass += mass
                        retained_exact_xlogx += xlogx(mass)
                        retained_exact_count += 1

                group_mass = fresh_mass + retained_mass
                exact_mass = fresh_exact_mass + retained_exact_mass
                exact_xlogx = fresh_exact_xlogx + retained_exact_xlogx
                all_xlogx = fresh_xlogx + retained_xlogx
                exact_count = fresh_exact_count + retained_exact_count
                total_count = fresh_count + retained_count
                strict_mass = max(0.0, group_mass - exact_mass)
                strict_xlogx = all_xlogx - exact_xlogx
                strict_count = total_count - exact_count
                if group_mass <= 0.0:
                    continue
                if exact_count <= 0:
                    raise InvariantError(
                        f"belief group {bytes(key)!r} has no exact representative"
                    )
                record = {
                    "key": bytes(key),
                    "static": static,
                    "fresh_ids": [int(value) for value in fresh_ids],
                    "retained_ids": [int(value) for value in retained_ids],
                    "dist": dist,
                    "group_mass": group_mass,
                    "exact_mass": exact_mass,
                    "strict_mass": strict_mass,
                    "strict_xlogx_sum": strict_xlogx,
                    "exact_count": exact_count,
                    "strict_count": strict_count,
                    "candidate_count": total_count,
                    "las_posterior_entropy_bits": entropy_from_aggregates(
                        group_mass, exact_mass, strict_xlogx
                    ),
                }
                records.append(record)
                masses.append(group_mass)

            if not masses or sum(masses) <= 0.0:
                raise InvariantError("belief planner produced no positive-mass groups")
            dtype = dist.probs.dtype if dist is not None else torch.float64
            device = dist.probs.device if dist is not None else torch.device("cpu")
            probs = torch.as_tensor(masses, dtype=dtype, device=device)
            probs = probs / probs.sum()
            ctx.stats["belief_plan_calls"] = int(
                ctx.stats.get("belief_plan_calls", 0)
            ) + 1
            ctx.stats["las_group_count"] = len(records)
            ctx.stats["las_retained_frontier"] = len(state.retained)
            ctx.stats["las_retained_merge"] = "sorted_linear"
            ctx.stats["belief_aggregate_metrics"] = True
            ctx.stats["las_static_vocab"] = not sparse_support
            return DecisionPlan(probs=probs, opaque={"records": records, "masses": masses})

        def advance_fullsync(self, state, plan, action, rng, ctx):
            return advance_belief_fullsync(
                state, plan, int(action), rng, ctx, self.index
            )

    return BeliefPlanner()

