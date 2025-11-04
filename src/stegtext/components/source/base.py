from __future__ import annotations

from typing import Protocol, List, Optional

import torch

from ...core.data import Candidate

# Unique sentinel appended to EOS token byte streams. Make it a long, unlikely
# ASCII sequence so natural model output will not collide with it.
EOS_STEGA = b"<eos_stega>"


class Source(Protocol):
    """Common interface implemented by all text sources."""

    def init(self, prompt: str, *, rng=None) -> List[Candidate]:
        """Return first-step candidates for the given prompt."""

    def generate(
        self,
        prefix_tokens: torch.LongTensor,
        *,
        prompt_len: Optional[int] = None,
        rng=None,
    ) -> List[Candidate]:
        """Extend the supplied prefix and return candidate continuations."""

    def decode(
        self,
        candidate: Candidate,
        *,
        from_payload: bool = False,
        strip_eos: bool = False,
    ) -> str:
        """Return a human-readable view of a candidate using the source vocab."""

    def reset_cache(self, *, full: bool = False) -> None:
        """Clear any cached model state prior to a new encode/decode run."""
