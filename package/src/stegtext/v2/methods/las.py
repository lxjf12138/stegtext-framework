from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..core.errors import InvariantError, ReceiverMismatchError
from ..core.types import DecisionPlan, MethodContext, PathRef, PromptState, TokenDistribution, VisibleTarget
from ..model.vocab_index import StaticGroup, TokenByteIndex


@dataclass(frozen=True)
class RetainedPath:
    path: PathRef
    residual: bytes
    mass: float
    terminal: bool = False


@dataclass(frozen=True)
class LASState:
    base_path: PathRef
    expansion_mass: float = 1.0
    retained: tuple[RetainedPath, ...] = ()
    resolved_bytes: bytes = b""
    terminal: bool = False
    tokens_used: int = 0


def _choice(items: list[tuple[Any, float]], rng: Any) -> Any:
    if not items:
        raise InvariantError("cannot sample an empty exact LAS set")
    total = sum(max(0.0, float(w)) for _, w in items)
    if total <= 0:
        raise InvariantError("exact LAS set has zero probability mass")
    u = float(rng.random()) * total
    acc = 0.0
    for value, weight in items:
        acc += max(0.0, float(weight))
        if u < acc:
            return value
    return items[-1][0]


class LASMethod:
    """Look-ahead synchronization with dense fresh and sparse retained mass."""

    def __init__(self, index: TokenByteIndex, *, optimized: bool = True) -> None:
        self.index = index
        self.optimized = bool(optimized)

    def start(self, prompt_state: PromptState) -> LASState:
        return LASState(base_path=prompt_state.path)

    def queries(self, state: LASState) -> list[PathRef]:
        if state.terminal or state.expansion_mass <= 0.0:
            return []
        return [state.base_path]

    def _lex_cdf(self, dist: TokenDistribution) -> torch.Tensor:
        """Return a single GPU prefix sum in cached lexical vocabulary order."""
        ids = self.index.lex_token_ids_gpu.to(device=dist.probs.device)
        # Float64 keeps subtraction between nearby interval endpoints stable
        # without changing or repairing the sampler's support.
        p_lex = dist.probs.index_select(0, ids).to(dtype=torch.float64)
        cdf = torch.cumsum(p_lex, dim=0)
        return torch.cat((torch.zeros(1, dtype=cdf.dtype, device=cdf.device), cdf))

    def _static_mass(
        self,
        dist: TokenDistribution,
        groups: list[StaticGroup],
        cdf: torch.Tensor | None = None,
    ) -> float:
        if not groups:
            return 0.0
        if dist.probs.device.type == "cpu":
            # Keep the small reference path bit-exact with NaiveLAS: both
            # accumulate the same lexical token order in Python float64.
            return sum(
                float(dist.probs[int(token_id)].item())
                for group in groups
                for token_id in self.index.lex_token_ids_cpu[group.lo : group.hi]
            )
        if cdf is None:
            cdf = self._lex_cdf(dist)
        lo = torch.as_tensor([group.lo for group in groups], dtype=torch.long, device=cdf.device)
        hi = torch.as_tensor([group.hi for group in groups], dtype=torch.long, device=cdf.device)
        intervals = cdf.index_select(0, hi) - cdf.index_select(0, lo)
        if bool((intervals < 0).any().item()):
            raise InvariantError("LAS lexical CDF produced a negative prefix interval mass")
        return float(intervals.sum().item())

    def _validate_visible_support(self, dist: TokenDistribution) -> None:
        empty_ids = self.index.empty_token_ids_cpu
        if int(empty_ids.size) == 0:
            return
        ids = torch.as_tensor(empty_ids, dtype=torch.long, device=dist.probs.device)
        positive = ids[dist.probs.index_select(0, ids) > 0]
        if int(positive.numel()) > 0:
            sample = positive[:8].detach().cpu().tolist()
            raise InvariantError(
                "LAS sampler support contains tokens with empty visible realizations: "
                f"{sample}"
            )

    def plan(self, state: LASState, distributions: list[TokenDistribution], ctx: MethodContext) -> DecisionPlan:
        if state.expansion_mass > 0.0:
            if len(distributions) != 1:
                raise InvariantError("active LAS fresh mass requires exactly one model distribution")
            dist: TokenDistribution | None = distributions[0]
        else:
            if distributions:
                raise InvariantError("LAS received a model distribution with no fresh mass to expand")
            dist = None
        residuals = [r.residual for r in state.retained if r.residual]
        if dist is None:
            sparse_support = True
            dynamic = self._retained_groups(residuals)
            cdf = None
        else:
            self._validate_visible_support(dist)
            ids, _ = dist.compact()
            sparse_support = int(ids.numel()) < self.index.vocab_size
            if sparse_support:
                dynamic = self._support_groups(ids, residuals)
                cdf = None
            else:
                dynamic = [(key, static_indices, []) for key, static_indices, _ in self.index.dynamic_groups(residuals)]
                cdf = self._lex_cdf(dist)
        records: list[dict[str, Any]] = []
        masses: list[float] = []
        for key, static_indices, fresh_ids in dynamic:
            retained_ids = [i for i, r in enumerate(state.retained) if r.residual.startswith(key)]
            static = [self.index.base_groups[i] for i in static_indices]
            if dist is None:
                fresh_mass = 0.0
            elif fresh_ids:
                fresh_mass = float(dist.probs.index_select(0, torch.as_tensor(fresh_ids, device=dist.probs.device)).sum().item())
            else:
                fresh_mass = self._static_mass(dist, static, cdf)
            retained_mass = sum(float(state.retained[i].mass) for i in retained_ids)
            masses.append(state.expansion_mass * fresh_mass + retained_mass)
            records.append({"key": key, "static": static, "fresh_ids": fresh_ids, "retained_ids": retained_ids, "dist": dist})
        if not masses or sum(masses) <= 0:
            raise InvariantError("LAS produced no positive-mass action group")
        dtype = dist.probs.dtype if dist is not None else torch.float64
        device = dist.probs.device if dist is not None else torch.device("cpu")
        probs = torch.as_tensor(masses, dtype=dtype, device=device)
        probs = probs / probs.sum()
        ctx.stats["las_group_count"] = len(records)
        ctx.stats["las_retained_frontier"] = len(state.retained)
        ctx.stats["las_retained_frontier_max"] = max(
            int(ctx.stats.get("las_retained_frontier_max", 0)), len(state.retained)
        )
        ctx.stats["las_static_vocab"] = not sparse_support
        ctx.stats["las_fresh_branch_active"] = dist is not None
        return DecisionPlan(probs=probs, opaque={"records": records, "masses": masses})

    @staticmethod
    def _retained_groups(residuals: list[bytes]) -> list[tuple[bytes, list[int], list[int]]]:
        out: list[tuple[bytes, list[int], list[int]]] = []
        for key in sorted(set(bytes(value) for value in residuals if value)):
            if out and key.startswith(out[-1][0]):
                continue
            out.append((key, [], []))
        return out

    def _support_groups(self, ids: torch.Tensor, residuals: list[bytes]):
        entries: list[tuple[bytes, int | None]] = [
            (self.index.token_bytes[int(token_id)], int(token_id)) for token_id in ids.tolist()
            if self.index.token_bytes[int(token_id)]
        ]
        entries.extend((bytes(value), None) for value in residuals if value)
        entries.sort(key=lambda item: (item[0], -1 if item[1] is None else item[1]))
        out: list[tuple[bytes, list[int], list[int]]] = []
        for key, token_id in entries:
            if out and key.startswith(out[-1][0]):
                if token_id is not None:
                    out[-1][2].append(token_id)
                continue
            out.append((key, [], [] if token_id is None else [token_id]))
        return out

    def observe(self, state: LASState, plan: DecisionPlan, target: VisibleTarget, ctx: MethodContext) -> int:
        matches = [i for i, record in enumerate(plan.opaque["records"]) if target.remaining.startswith(record["key"])]
        if len(matches) != 1:
            raise ReceiverMismatchError(f"LAS receiver found {len(matches)} matching groups")
        target.consume(plan.opaque["records"][matches[0]]["key"])
        return matches[0]

    def advance(self, state: LASState, plan: DecisionPlan, action: int, rng: Any, ctx: MethodContext) -> LASState:
        records = plan.opaque["records"]
        if action < 0 or action >= len(records):
            raise IndexError("LAS action is outside the plan")
        record = records[int(action)]
        key = record["key"]
        static_count = sum(int(group.hi - group.lo) for group in record.get("static", ()))
        fresh_count = len(record.get("fresh_ids", ())) or static_count
        ctx.stats["las_selected_group_size"] = fresh_count + len(record.get("retained_ids", ()))
        group_mass = float(plan.opaque["masses"][int(action)])
        if group_mass <= 0:
            raise InvariantError("selected LAS group has zero mass")
        dist: TokenDistribution | None = record["dist"]
        static: list[StaticGroup] = record["static"]
        fresh_ids: list[int] = record.get("fresh_ids", [])
        exact: list[tuple[tuple[str, Any], float]] = []
        strict: list[tuple[PathRef, bytes, float, bool]] = []
        candidate_id_lists = [fresh_ids] if fresh_ids else [
            self.index.lex_token_ids_cpu[static_group.lo : static_group.hi].tolist()
            for static_group in static
        ]
        candidate_ids = [int(token_id) for ids in candidate_id_lists for token_id in ids]
        if dist is None:
            if candidate_ids:
                raise InvariantError("LAS plan has fresh candidates without a model distribution")
            candidate_probs: list[float] = []
        elif dist.probs.device.type == "cpu":
            # Preserve the reference accumulation values exactly on CPU.
            candidate_probs = [float(dist.probs[token_id].item()) for token_id in candidate_ids]
        else:
            id_tensor = torch.as_tensor(candidate_ids, dtype=torch.long, device=dist.probs.device)
            candidate_probs = (
                dist.probs.index_select(0, id_tensor)
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            )
        for token_id, probability in zip(candidate_ids, candidate_probs):
            payload = self.index.token_bytes[token_id]
            mass = state.expansion_mass * float(probability)
            if payload == key and mass > 0:
                exact.append((("fresh", (token_id,)), mass))
            elif payload.startswith(key) and len(payload) > len(key) and mass > 0:
                strict.append(
                    (
                        state.base_path.append(token_id),
                        payload[len(key):],
                        mass,
                        self._is_eos(ctx, token_id),
                    )
                )
        for retained_id in record["retained_ids"]:
            retained = state.retained[int(retained_id)]
            mass = float(retained.mass)
            if retained.residual == key and mass > 0:
                exact.append((("retained", retained), mass))
            elif retained.residual.startswith(key) and len(retained.residual) > len(key) and mass > 0:
                strict.append((retained.path, retained.residual[len(key):], mass, retained.terminal))
        if not exact:
            raise InvariantError(f"LAS group {key!r} has no exact representative")
        selected_kind, selected = _choice(exact, rng)
        if selected_kind == "fresh":
            token_id = int(selected[0])
            base_path = state.base_path.append(token_id)
            selected_terminal = self._is_eos(ctx, token_id)
        else:
            retained: RetainedPath = selected
            base_path = retained.path
            selected_terminal = retained.terminal
        if selected_terminal and strict:
            raise InvariantError(
                "LAS terminal realization has a strict extension; tokenizer EOS contract is invalid"
            )
        next_retained: list[RetainedPath] = []
        for path, residual, mass, terminal in strict:
            next_retained.append(
                RetainedPath(path=path, residual=residual, mass=float(mass / group_mass), terminal=terminal)
            )
        exact_mass = sum(weight for _, weight in exact)
        next_expansion_mass = 0.0 if selected_terminal else float(exact_mass / group_mass)
        # Some lightweight parity/unit-test contexts do not carry a prompt
        # object.  Production ``MethodContext`` always does, but treating a
        # missing prompt as depth zero keeps the state transition total and
        # does not affect the normal runner path.
        prompt_depth = len(getattr(getattr(ctx, "prompt", None), "token_ids", ()))
        return LASState(
            base_path=base_path,
            expansion_mass=next_expansion_mass,
            retained=tuple(next_retained),
            resolved_bytes=state.resolved_bytes + key,
            terminal=selected_terminal,
            tokens_used=max(0, base_path.depth - prompt_depth),
        )

    @staticmethod
    def _is_eos(ctx: MethodContext, token_id: int) -> bool:
        return bool(ctx.codec.is_eos(int(token_id)))

    def done(self, state: LASState) -> bool:
        return state.terminal

    def output(self, state: LASState, ctx: MethodContext) -> bytes:
        data = state.resolved_bytes
        path_ids = state.base_path.token_ids()
        prompt_ids = tuple(int(value) for value in ctx.prompt.token_ids)
        if path_ids[: len(prompt_ids)] != prompt_ids:
            raise InvariantError("LAS active path does not descend from the rendered prompt")
        latent = ctx.codec.tokens_to_bytes(path_ids[len(prompt_ids) :])
        if latent != data:
            raise InvariantError("LAS visible output diverged from its synchronized token path")
        eos = getattr(ctx.codec, "eos_bytes", b"")
        while eos and data.endswith(eos):
            data = data[: -len(eos)]
        return data


class NaiveLASMethod(LASMethod):
    """Small, intentionally direct LAS reference implementation.

    Unlike :class:`LASMethod`, this version materializes one lightweight
    candidate record per positive-support token on every plan.  It is kept as
    a correctness oracle for small vocabularies; it must not be used for a
    full-vocabulary experiment.
    """

    def __init__(self, index: TokenByteIndex) -> None:
        super().__init__(index, optimized=False)

    def plan(
        self,
        state: LASState,
        distributions: list[TokenDistribution],
        ctx: MethodContext,
    ) -> DecisionPlan:
        if state.expansion_mass > 0.0:
            if len(distributions) != 1:
                raise InvariantError("active naive LAS fresh mass requires exactly one model distribution")
            dist: TokenDistribution | None = distributions[0]
            self._validate_visible_support(dist)
            ids, _ = dist.compact()
        else:
            if distributions:
                raise InvariantError("naive LAS received a distribution with no fresh mass")
            dist = None
            ids = torch.empty(0, dtype=torch.long)
        entries: list[tuple[bytes, str, int]] = []
        for token_id in ids.detach().cpu().tolist():
            payload = self.index.token_bytes[int(token_id)]
            if payload:
                entries.append((payload, "fresh", int(token_id)))
        for retained_id, retained in enumerate(state.retained):
            if retained.residual:
                entries.append((bytes(retained.residual), "retained", int(retained_id)))
        entries.sort(key=lambda item: (item[0], item[1], item[2]))

        # The first string in each prefix chain is the group key.  Every
        # later candidate extending it belongs to that same receiver action.
        records: list[dict[str, Any]] = []
        for payload, kind, value in entries:
            if records and payload.startswith(records[-1]["key"]):
                if kind == "fresh":
                    records[-1]["fresh_ids"].append(value)
                else:
                    records[-1]["retained_ids"].append(value)
                continue
            records.append(
                {
                    "key": payload,
                    "static": [],
                    "fresh_ids": [value] if kind == "fresh" else [],
                    "retained_ids": [value] if kind == "retained" else [],
                    "dist": dist,
                }
            )

        masses: list[float] = []
        for record in records:
            fresh_mass = (
                sum(float(dist.probs[int(token_id)].item()) for token_id in record["fresh_ids"])
                if dist is not None
                else 0.0
            )
            retained_mass = sum(
                float(state.retained[int(retained_id)].mass)
                for retained_id in record["retained_ids"]
            )
            masses.append(state.expansion_mass * fresh_mass + retained_mass)
        if not masses or sum(masses) <= 0.0:
            raise InvariantError("naive LAS produced no positive-mass action group")
        dtype = dist.probs.dtype if dist is not None else torch.float64
        device = dist.probs.device if dist is not None else torch.device("cpu")
        probs = torch.as_tensor(masses, dtype=dtype, device=device)
        probs = probs / probs.sum()
        ctx.stats["las_group_count"] = len(records)
        ctx.stats["las_retained_frontier"] = len(state.retained)
        ctx.stats["las_reference_mode"] = True
        return DecisionPlan(probs=probs, opaque={"records": records, "masses": masses})

    def advance(
        self,
        state: LASState,
        plan: DecisionPlan,
        action: int,
        rng: Any,
        ctx: MethodContext,
    ) -> LASState:
        records = plan.opaque["records"]
        if action < 0 or action >= len(records):
            raise IndexError("naive LAS action is outside the plan")
        record = records[int(action)]
        key = record["key"]
        ctx.stats["las_selected_group_size"] = len(record.get("fresh_ids", ())) + len(record.get("retained_ids", ()))
        group_mass = float(plan.opaque["masses"][int(action)])
        if group_mass <= 0.0:
            raise InvariantError("selected naive LAS group has zero mass")
        dist: TokenDistribution | None = record["dist"]

        exact: list[tuple[tuple[str, Any], float]] = []
        strict: list[tuple[PathRef, bytes, float, bool]] = []
        for token_id in record["fresh_ids"]:
            if dist is None:
                raise InvariantError("naive LAS plan has fresh candidates without a distribution")
            payload = self.index.token_bytes[int(token_id)]
            mass = state.expansion_mass * float(dist.probs[int(token_id)].item())
            if payload == key and mass > 0.0:
                exact.append((("fresh", (int(token_id),)), mass))
            elif payload.startswith(key) and len(payload) > len(key) and mass > 0.0:
                strict.append(
                    (
                        state.base_path.append(int(token_id)),
                        payload[len(key) :],
                        mass,
                        self._is_eos(ctx, int(token_id)),
                    )
                )
        for retained_id in record["retained_ids"]:
            retained = state.retained[int(retained_id)]
            mass = float(retained.mass)
            if retained.residual == key and mass > 0.0:
                exact.append((("retained", retained), mass))
            elif retained.residual.startswith(key) and len(retained.residual) > len(key) and mass > 0.0:
                strict.append((retained.path, retained.residual[len(key) :], mass, retained.terminal))
        if not exact:
            raise InvariantError(f"naive LAS group {key!r} has no exact representative")

        selected_kind, selected = _choice(exact, rng)
        if selected_kind == "fresh":
            token_id = int(selected[0])
            base_path = state.base_path.append(token_id)
            selected_terminal = self._is_eos(ctx, token_id)
        else:
            retained = selected
            base_path = retained.path
            selected_terminal = retained.terminal
        if selected_terminal and strict:
            raise InvariantError(
                "naive LAS terminal realization has a strict extension; tokenizer EOS contract is invalid"
            )
        exact_mass = sum(weight for _, weight in exact)
        prompt_depth = len(getattr(getattr(ctx, "prompt", None), "token_ids", ()))
        return LASState(
            base_path=base_path,
            expansion_mass=0.0 if selected_terminal else float(exact_mass / group_mass),
            retained=tuple(
                RetainedPath(path=path, residual=residual, mass=float(mass / group_mass), terminal=terminal)
                for path, residual, mass, terminal in strict
            ),
            resolved_bytes=state.resolved_bytes + key,
            terminal=selected_terminal,
            tokens_used=max(0, base_path.depth - prompt_depth),
        )
