from __future__ import annotations

from typing import Sequence

import torch

from ...core.contracts import BaseCodingStrategy, SupportsRandom
from ...core.prob import sanitize1d


class RandomSampler(BaseCodingStrategy):
    """Coder that ignores the bitstream and samples group indices via RNG."""

    def init(self, rng: SupportsRandom | None = None) -> None:  # pragma: no cover - stateless
        # Stateless: nothing to reset, but keep signature for consistency.
        return None

    def _sample_index(self, probs: torch.Tensor, rng: SupportsRandom) -> int:
        weights = torch.clamp(probs.detach().to(dtype=torch.float64, device="cpu"), min=0.0)
        total = float(weights.sum().item())
        if total <= 0.0 or weights.numel() == 0:
            return 0
        cumulative = torch.cumsum(weights, dim=0)
        threshold = float(rng.random()) * total
        idx = int(torch.searchsorted(cumulative, torch.tensor(threshold, dtype=torch.float64)).item())
        if idx >= int(weights.numel()):
            idx = int(weights.numel()) - 1
        return idx

    def encode(
        self,
        group_probs: Sequence[float] | torch.Tensor,
        bit_stream: str,
        *,
        rng: SupportsRandom,
    ) -> tuple[str, int]:
        probs = sanitize1d(group_probs)
        idx = self._sample_index(probs, rng)
        return "", idx

    def decode(
        self,
        group_probs: Sequence[float] | torch.Tensor,
        chosen_group_idx: int,
        *,
        rng: SupportsRandom,
    ) -> str:
        # Decode pathway piggybacks on deterministic selection coming from the engine;
        # no bits were emitted during encode, so nothing to recover.
        return ""
