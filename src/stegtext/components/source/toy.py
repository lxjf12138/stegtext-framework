from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .base import Candidate, Source
from .vocab import TokenByteVocab


CandidateSpec = Tuple[int, float]
TransitionMap = Dict[Tuple[int, ...], Sequence[CandidateSpec]]


@dataclass
class ToySourceConfig:
    """Configuration driving a deterministic toy source.

    Parameters
    ----------
    vocab: TokenByteVocab
        Token to byte mapping shared by the toy source.
    transitions: Dict[Tuple[int, ...], Sequence[Tuple[int, float]]]
        Mapping from a prefix token tuple to candidate specifications.
    end_token_ids: Sequence[int]
        Token ids considered terminal; all terminal candidates will be
        emitted with ``is_eos=True``.
    prompt_to_tokens: Optional[Callable[[str], Sequence[int]]]
        Optional hook converting human-readable prompts into the token space.
        When omitted, the source falls back to a simple byte-by-byte encoder
        that requires single-byte tokens to be present in ``vocab``.
    """

    vocab: TokenByteVocab
    transitions: TransitionMap
    end_token_ids: Sequence[int] = ()
    prompt_to_tokens: Optional[Callable[[str], Sequence[int]]] = None


class ToySource(Source):
    """Minimal in-memory source useful for tests and demos."""

    def __init__(self, config: ToySourceConfig) -> None:
        self._config = config
        self._vocab = config.vocab
        self._end_token_ids = {int(t) for t in config.end_token_ids}
        self._transitions: Dict[Tuple[int, ...], List[CandidateSpec]] = {
            tuple(int(tok) for tok in key): [(int(tid), float(prob)) for tid, prob in values]
            for key, values in config.transitions.items()
        }
        self._prompt_encoder: Callable[[str], Sequence[int]] = (
            config.prompt_to_tokens if config.prompt_to_tokens is not None else self._default_prompt_encoder
        )
        self._byte_lookup = self._build_reverse_vocab()

    # ------------------------------------------------------------------
    # Source interface
    # ------------------------------------------------------------------
    def init(self, prompt: str, *, rng=None) -> List[Candidate]:
        prefix_tokens = self._ensure_tensor(self._prompt_encoder(prompt))
        prompt_len = int(prefix_tokens.numel())
        return self._candidates_for_prefix(prefix_tokens, prompt_len)

    def generate(
        self,
        prefix_tokens: torch.LongTensor,
        *,
        prompt_len: int | None = None,
        rng=None,
    ) -> List[Candidate]:
        tokens = self._ensure_tensor(prefix_tokens)
        if prompt_len is None:
            prompt_len = int(tokens.numel())
        return self._candidates_for_prefix(tokens, prompt_len)

    def decode(
        self,
        candidate: Candidate,
        *,
        from_payload: bool = False,
        strip_eos: bool = False,
    ) -> str:
        if from_payload:
            payload = candidate.vb[candidate.prompt_byte_len :]
        else:
            payload = candidate.vb
        return self._vocab.bytes_to_text(payload, strip_eos=strip_eos)

    def reset_cache(self, *, full: bool = False) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_tensor(self, tokens: Iterable[int]) -> torch.LongTensor:
        if isinstance(tokens, torch.Tensor):
            return tokens.detach().clone().to(dtype=torch.long, device="cpu")
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()  # type: ignore[assignment]
        return torch.tensor([int(t) for t in tokens], dtype=torch.long)

    def _candidates_for_prefix(self, prefix_tokens: torch.LongTensor, prompt_len: int) -> List[Candidate]:
        prefix_list = prefix_tokens.tolist()
        if prefix_list and prefix_list[-1] in self._end_token_ids:
            return []

        key = tuple(prefix_list)
        specs = self._transitions.get(key)
        if specs is None and prefix_list:
            specs = self._transitions.get(tuple(prefix_list[-1:]))
        if specs is None:
            specs = self._transitions.get((), [])

        prefix_bytes = self._vocab.tokens_to_bytes(prefix_list)
        prompt_bytes = self._vocab.tokens_to_bytes(prefix_list[:prompt_len])
        prompt_byte_len = len(prompt_bytes)
        out: List[Candidate] = []
        for token_id, prob in specs:
            cand_tokens = torch.cat(
                [prefix_tokens, torch.tensor([int(token_id)], dtype=torch.long)], dim=0
            )
            vb = prefix_bytes + self._vocab.token_bytes(int(token_id))
            cand_prompt_len = prompt_len
            out.append(
                Candidate(
                    p=float(prob),
                    tokens=cand_tokens,
                    vb=vb,
                    is_eos=token_id in self._end_token_ids,
                    prompt_len=cand_prompt_len,
                    prompt_byte_len=prompt_byte_len,
                )
            )
        return out

    def _build_reverse_vocab(self) -> Dict[bytes, int]:
        lookup: Dict[bytes, int] = {}
        for token_id, payload in self._vocab.mapping.items():
            if len(payload) == 1 and payload not in lookup:
                lookup[payload] = int(token_id)
        return lookup

    def _default_prompt_encoder(self, prompt: str) -> Sequence[int]:
        tokens: List[int] = []
        for byte in prompt.encode("utf-8"):
            key = bytes([byte])
            if key not in self._byte_lookup:
                raise ValueError(
                    f"byte {byte} not representable in toy vocab; consider providing prompt_to_tokens"
                )
            tokens.append(self._byte_lookup[key])
        return tokens
