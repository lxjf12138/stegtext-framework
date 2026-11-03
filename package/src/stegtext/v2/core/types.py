from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import torch

from .errors import ProbabilityError, ReceiverMismatchError


@dataclass(frozen=True)
class PathRef:
    """Persistent token path; fresh vocabulary children share their parent."""

    parent: Optional["PathRef"]
    token_id: Optional[int]
    depth: int

    @classmethod
    def root(cls) -> "PathRef":
        return cls(parent=None, token_id=None, depth=0)

    def append(self, token_id: int) -> "PathRef":
        return PathRef(self, int(token_id), self.depth + 1)

    def token_ids(self) -> tuple[int, ...]:
        out: list[int] = []
        cur: Optional[PathRef] = self
        while cur is not None and cur.token_id is not None:
            out.append(int(cur.token_id))
            cur = cur.parent
        out.reverse()
        return tuple(out)


@dataclass(frozen=True)
class PromptState:
    path: PathRef
    token_ids: tuple[int, ...]
    prompt_bytes: bytes
    prompt_text: str = ""


@dataclass(frozen=True)
class TokenDistribution:
    """A normalized model law. ``probs`` is always dense over vocabulary ids."""

    probs: torch.Tensor
    support_ids: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        p = self.probs
        if not isinstance(p, torch.Tensor) or p.ndim != 1:
            raise ProbabilityError("TokenDistribution.probs must be a rank-1 tensor")
        if not torch.is_floating_point(p):
            raise ProbabilityError("TokenDistribution.probs must be floating point")
        if not bool(torch.isfinite(p).all().item()):
            raise ProbabilityError("TokenDistribution contains NaN or infinity")
        if bool((p < 0).any().item()):
            raise ProbabilityError("TokenDistribution contains a negative probability")
        total = float(p.sum().item())
        if total <= 0.0:
            raise ProbabilityError("TokenDistribution has zero total mass")
        if self.support_ids is not None:
            ids = self.support_ids
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise ProbabilityError("support_ids must be a rank-1 integer tensor")
            if bool((ids < 0).any().item()) or bool((ids >= p.numel()).any().item()):
                raise ProbabilityError("support_ids contains an out-of-range token id")

    def compact(self) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.support_ids
        if ids is None:
            ids = torch.nonzero(self.probs > 0, as_tuple=False).flatten()
        else:
            ids = ids.to(device=self.probs.device, dtype=torch.long)
            ids = ids[self.probs.index_select(0, ids) > 0]
        if ids.numel() == 0:
            raise ProbabilityError("TokenDistribution has no positive-support token")
        vals = self.probs.index_select(0, ids)
        vals = vals / vals.sum()
        return ids, vals


@dataclass
class DecisionPlan:
    probs: torch.Tensor
    opaque: Any

    def __post_init__(self) -> None:
        if self.probs.ndim != 1 or self.probs.numel() == 0:
            raise ProbabilityError("DecisionPlan.probs must be non-empty and rank-1")
        if not bool(torch.isfinite(self.probs).all().item()):
            raise ProbabilityError("DecisionPlan contains NaN or infinity")
        if bool((self.probs < 0).any().item()):
            raise ProbabilityError("DecisionPlan contains a negative probability")
        total = float(self.probs.sum().item())
        if total <= 0:
            raise ProbabilityError("DecisionPlan has zero total mass")
        self.probs = self.probs / total


@dataclass
class VisibleTarget:
    data: bytes
    cursor: int = 0

    @property
    def remaining(self) -> bytes:
        return self.data[self.cursor :]

    def consume(self, prefix: bytes) -> None:
        if not self.remaining.startswith(prefix):
            raise ReceiverMismatchError(
                f"visible target does not start with action bytes {prefix!r}"
            )
        self.cursor += len(prefix)

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.data)


@dataclass
class MethodContext:
    codec: Any
    prompt: PromptState
    mode: str
    target: Optional[VisibleTarget] = None
    step: int = 0
    max_tokens: Optional[int] = None
    stats: dict[str, Any] = field(default_factory=dict)


class Method(Protocol):
    def start(self, prompt_state: PromptState) -> object: ...
    def queries(self, state: object) -> list[PathRef]: ...
    def plan(
        self,
        state: object,
        distributions: list[TokenDistribution],
        ctx: MethodContext,
    ) -> DecisionPlan: ...
    def observe(
        self,
        state: object,
        plan: DecisionPlan,
        target: VisibleTarget,
        ctx: MethodContext,
    ) -> int: ...
    def advance(
        self,
        state: object,
        plan: DecisionPlan,
        action: int,
        rng: Any,
        ctx: MethodContext,
    ) -> object: ...
    def done(self, state: object) -> bool: ...
    def output(self, state: object, ctx: MethodContext) -> bytes: ...


class Backend(Protocol):
    codec: Any

    def render_prompt(self, prompt: str) -> PromptState: ...
    def next_logits(self, path: PathRef) -> torch.Tensor: ...
