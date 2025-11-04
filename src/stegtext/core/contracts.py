from __future__ import annotations
from typing import Dict, Any, List, Callable, Protocol, Sequence, Tuple, Optional
import abc
import torch
from .data import Candidate, GroupingResult

class SupportsRandom(Protocol):
    def random(self) -> float: ...

class Plan:
    def __init__(self, groups: GroupingResult, group_probs: torch.Tensor, meta: Dict[str, Any]):
        self.groups = groups
        self.group_probs = group_probs
        self.meta = meta

class Disambiguator(abc.ABC):
    def init(self) -> None: ...
    def reset(self) -> None: ...

    @abc.abstractmethod
    def plan(self, candidates: List[Candidate]) -> Plan: ...

    @abc.abstractmethod
    def advance(
        self,
        plan: Plan,
        chosen_group_idx: int,
        rng: SupportsRandom,
        source_generate: Callable[[torch.LongTensor, Optional[int]], List[Candidate]],
    ) -> Tuple[List[Candidate], Candidate] | Candidate: ...

class Source(abc.ABC):
    """Unified source contract used by the engine."""

    @abc.abstractmethod
    def init(self, prompt: str, *, rng=None) -> List[Candidate]:
        """Return first-step candidates for the supplied prompt."""

    @abc.abstractmethod
    def generate(
        self,
        prefix_tokens: torch.LongTensor,
        *,
        prompt_len: Optional[int] = None,
        rng=None,
    ) -> List[Candidate]:
        """Extend the supplied prefix and return continuation candidates."""

    def decode(
        self,
        candidate: Candidate,
        *,
        from_payload: bool = False,
        strip_eos: bool = False,
    ) -> str:
        raise NotImplementedError

    def reset_sentence(self) -> None: ...

    def estimate_tokens(self, text: str) -> int:
        return len(text)


class AfterPickResult(Protocol):
    bits: str
    group_idx: int


class BaseCodingStrategy(abc.ABC):
    def init(self, rng: SupportsRandom | None = None) -> None: ...

    @abc.abstractmethod
    def encode(
        self,
        group_probs: Sequence[float] | torch.Tensor,
        bit_stream: str,
        *,
        rng: SupportsRandom,
    ) -> Tuple[str, int]: ...

    @abc.abstractmethod
    def decode(
        self,
        group_probs: Sequence[float] | torch.Tensor,
        chosen_group_idx: int,
        *,
        rng: SupportsRandom,
    ) -> str: ...
