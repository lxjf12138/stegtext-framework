from __future__ import annotations

from typing import Protocol

import torch

from ..core.errors import ProbabilityError
from ..core.types import TokenDistribution


class SamplingPolicy(Protocol):
    def __call__(self, logits: torch.Tensor) -> TokenDistribution: ...


def _softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError("model logits must be rank-1")
    if bool(torch.isnan(logits).any().item()) or bool(torch.isposinf(logits).any().item()):
        raise ProbabilityError("model logits contain NaN or positive infinity")
    if bool(torch.isneginf(logits).all().item()):
        raise ProbabilityError("all model logits are masked")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Model heads are often bf16.  Keeping the probabilities in bf16 loses
    # enough mass across a 150k-token vocabulary to make the sum exceed one
    # after lexical aggregation.  Promote before normalization so the public
    # TokenDistribution is a genuine categorical law.
    work_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    p = torch.softmax(logits.to(dtype=work_dtype) / temperature, dim=0)
    p = p / p.sum()
    if not torch.isfinite(p).all() or float(p.sum().item()) <= 0:
        raise ProbabilityError("softmax produced an invalid distribution")
    return p


class FullSoftmax:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def __call__(self, logits: torch.Tensor) -> TokenDistribution:
        return TokenDistribution(_softmax(logits, self.temperature))


class Temperature(FullSoftmax):
    pass


class TopK:
    def __init__(self, k: int, temperature: float = 1.0) -> None:
        if int(k) <= 0:
            raise ValueError("top-k must be positive")
        self.k = int(k)
        self.temperature = float(temperature)

    def __call__(self, logits: torch.Tensor) -> TokenDistribution:
        p = _softmax(logits, self.temperature)
        k = min(self.k, p.numel())
        _, ids = torch.topk(p, k=k)
        out = torch.zeros_like(p)
        out[ids] = p[ids]
        out /= out.sum()
        return TokenDistribution(out, ids)


class TopP:
    def __init__(self, p: float, temperature: float = 1.0) -> None:
        if not 0.0 < float(p) <= 1.0:
            raise ValueError("top-p must be in (0, 1]")
        self.p = float(p)
        self.temperature = float(temperature)

    def __call__(self, logits: torch.Tensor) -> TokenDistribution:
        probs = _softmax(logits, self.temperature)
        if self.p >= 1.0:
            return TokenDistribution(probs)
        vals, ids = torch.sort(probs, descending=True)
        keep = torch.cumsum(vals, 0) <= self.p
        keep[0] = True
        ids = ids[keep]
        out = torch.zeros_like(probs)
        out[ids] = probs[ids]
        out /= out.sum()
        return TokenDistribution(out, ids)


class MinP:
    def __init__(self, min_p: float, temperature: float = 1.0) -> None:
        if not 0.0 < float(min_p) <= 1.0:
            raise ValueError("min-p must be in (0, 1]")
        self.min_p = float(min_p)
        self.temperature = float(temperature)

    def __call__(self, logits: torch.Tensor) -> TokenDistribution:
        probs = _softmax(logits, self.temperature)
        threshold = float(probs.max().item()) * self.min_p
        ids = torch.nonzero(probs >= threshold, as_tuple=False).flatten()
        out = torch.zeros_like(probs)
        out[ids] = probs[ids]
        out /= out.sum()
        return TokenDistribution(out, ids)
