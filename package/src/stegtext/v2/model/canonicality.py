from __future__ import annotations

from typing import Any, Protocol

import torch


class CanonicalityOracle(Protocol):
    def valid_mask(self, history: torch.Tensor, candidate_ids: torch.Tensor) -> torch.BoolTensor: ...


class ExactCanonicalityOracle:
    """Reference oracle for ToyBackend: every supplied candidate is canonical."""

    def valid_mask(self, history: torch.Tensor, candidate_ids: torch.Tensor) -> torch.BoolTensor:
        return torch.ones(candidate_ids.numel(), dtype=torch.bool, device=candidate_ids.device)


class TokenizerCanonicalityOracle:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def _check(self, history: list[int], token_id: int) -> bool:
        ids = history + [int(token_id)]
        text = self.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors=None)
        retok = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if hasattr(retok, "tolist"):
            retok = retok.tolist()
        if retok and isinstance(retok[0], list):
            retok = retok[0]
        return tuple(int(x) for x in retok) == tuple(ids)

    def valid_mask(self, history: torch.Tensor, candidate_ids: torch.Tensor) -> torch.BoolTensor:
        h = [int(x) for x in history.detach().cpu().flatten().tolist()]
        ids = candidate_ids.detach().cpu().flatten().tolist()
        out = [self._check(h, int(i)) for i in ids]
        return torch.tensor(out, dtype=torch.bool, device=candidate_ids.device)
