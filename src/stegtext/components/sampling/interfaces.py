from __future__ import annotations
from typing import Protocol, Optional
import torch

class Sampler(Protocol):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Return filtered/proportioned probabilities (sum to 1)."""

class Reweighter(Protocol):
    def __call__(self, probs: torch.Tensor) -> torch.Tensor:
        """Return reweighted probabilities (sum to 1)."""

def compose_sampler(*samplers: Sampler):
    def _fn(logits: torch.Tensor)->torch.Tensor:
        probs = logits
        for s in samplers:
            probs = s(probs)  # type: ignore
        return probs
    return _fn

# Example implementations referencing existing helpers
from .samplers import softmax, apply_top_k, apply_top_p, apply_typical, apply_min_p

class TemperatureSampler:
    def __init__(self, temperature: float = 1.0): self.t = max(1e-6, float(temperature))
    def __call__(self, logits: torch.Tensor)->torch.Tensor:
        return softmax(logits, self.t)

class TopKSampler:
    def __init__(self, k: int): self.k = int(k)
    def __call__(self, probs: torch.Tensor)->torch.Tensor:
        return apply_top_k(probs, self.k)

class TopPSampler:
    def __init__(self, p: float): self.p = float(p)
    def __call__(self, probs: torch.Tensor)->torch.Tensor:
        return apply_top_p(probs, self.p)

class MinPSampler:
    def __init__(self, p: float): self.p = float(p)
    def __call__(self, probs: torch.Tensor)->torch.Tensor:
        return apply_min_p(probs, self.p)

class TypicalSampler:
    def __init__(self, p: float): self.p = float(p)
    def __call__(self, probs: torch.Tensor)->torch.Tensor:
        return apply_typical(probs, self.p)
