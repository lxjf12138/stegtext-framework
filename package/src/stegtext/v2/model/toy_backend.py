from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ..core.types import PathRef, PromptState
from .canonicality import ExactCanonicalityOracle
from .tokenizer_codec import EOS_STEGA
from .vocab_index import TokenByteIndex


@dataclass(frozen=True)
class ToyBackendConfig:
    token_bytes: Mapping[int, bytes]
    transitions: Mapping[tuple[int, ...], Sequence[tuple[int, float]]]
    prompt_tokens: Mapping[str, Sequence[int]] | None = None
    eos_ids: Sequence[int] = ()


class ToyCodec:
    def __init__(self, token_bytes: Mapping[int, bytes], eos_ids: Sequence[int]) -> None:
        self._bytes = dict((int(k), bytes(v)) for k, v in token_bytes.items())
        self.eos_ids = frozenset(int(x) for x in eos_ids)
        self.eos_bytes = EOS_STEGA
        self.vocab_size = max(self._bytes.keys(), default=-1) + 1

    def token_bytes(self, token_id: int) -> bytes:
        return self._bytes[int(token_id)]

    def tokens_to_bytes(self, token_ids) -> bytes:
        return b"".join(self.token_bytes(i) for i in token_ids)

    def bytes_to_text(self, data: bytes, *, strip_eos: bool = False) -> str:
        if strip_eos:
            while data.endswith(EOS_STEGA):
                data = data[:-len(EOS_STEGA)]
        return data.decode("utf-8", errors="strict")

    def is_eos(self, token_id: int) -> bool:
        return int(token_id) in self.eos_ids

    def next_token(self, history, visible: bytes) -> int:
        if visible.startswith(EOS_STEGA) and self.eos_ids:
            return min(self.eos_ids)
        matches = [i for i, value in self._bytes.items() if value and visible.startswith(value)]
        if not matches:
            raise ValueError("ToyCodec cannot tokenize visible target")
        # Deterministic toy canonicalization: prefer the longest realization,
        # then the lowest id.  Tests with prefix ambiguity can provide a
        # custom oracle/codec when another contract is intended.
        return min(matches, key=lambda i: (-len(self._bytes[i]), i))


class ToyBackend:
    """Deterministic backend used for exhaustive method tests."""

    def __init__(self, config: ToyBackendConfig) -> None:
        self.config = config
        self.codec = ToyCodec(config.token_bytes, config.eos_ids)
        self.vocab_size = max(self.codec.vocab_size, max((k for s in config.transitions.values() for k, _ in s), default=-1) + 1)
        table = list(self.codec._bytes.items())
        table.extend((i, b"") for i in range(self.vocab_size) if i not in self.codec._bytes)
        self.codec._bytes.update(table)
        self.index = TokenByteIndex([self.codec._bytes.get(i, b"") for i in range(self.vocab_size)])
        self.oracle = ExactCanonicalityOracle()

    def render_prompt(self, prompt: str) -> PromptState:
        ids = tuple(int(x) for x in (self.config.prompt_tokens or {}).get(prompt, ()))
        path = PathRef.root()
        for token_id in ids:
            path = path.append(token_id)
        return PromptState(path=path, token_ids=ids, prompt_bytes=self.codec.tokens_to_bytes(ids), prompt_text=prompt)

    def next_logits(self, path: PathRef) -> torch.Tensor:
        specs = self.config.transitions.get(path.token_ids())
        if specs is None:
            raise RuntimeError(f"ToyBackend has no transition for path {path.token_ids()}")
        probs = torch.zeros(self.vocab_size, dtype=torch.float64)
        for token_id, value in specs:
            probs[int(token_id)] = float(value)
        if float(probs.sum().item()) <= 0:
            raise RuntimeError(f"ToyBackend has no transition for path {path.token_ids()}")
        # Preserve the Toy transition support exactly.  Turning zero mass into
        # a tiny positive value would make ``full_softmax`` expose invalid
        # actions and hide support/parity bugs.
        logits = torch.full_like(probs, -torch.inf)
        positive = probs > 0
        logits[positive] = torch.log(probs[positive])
        return logits
