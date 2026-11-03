from __future__ import annotations

from typing import Any, Protocol, Sequence, Tuple

import torch


class CoderSession(Protocol):
    def encode(
        self, probs: Sequence[float] | torch.Tensor, bit_stream: str, *, rng: Any
    ) -> Tuple[str, int]: ...

    def decode(
        self, probs: Sequence[float] | torch.Tensor, action: int, *, rng: Any
    ) -> str: ...


class PayloadCoder(Protocol):
    def start(self, mode: str, *, rng: Any = None) -> CoderSession: ...
