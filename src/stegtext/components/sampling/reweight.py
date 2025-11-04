from __future__ import annotations
from typing import Optional, Callable
import torch

class ExternalReweighter:
    """Proof-of-concept: combine current distribution with an external model's
    distribution to shape KL in a controlled way (e.g., product-of-experts).
    The `external_logits_fn` returns logits for the *same* vocabulary.
    """
    def __init__(self, external_logits_fn: Callable[[torch.Tensor], torch.Tensor], alpha: float = 0.5):
        self.fn = external_logits_fn
        self.alpha = float(alpha)

    def __call__(self, probs: torch.Tensor)->torch.Tensor:
        # Convert probs back to logits (safe)
        logits = torch.log(probs + 1e-12)
        ext_logits = self.fn(logits)
        # Combine via convex combination then renormalize
        mix = (1.0 - self.alpha) * logits + self.alpha * ext_logits
        out = torch.softmax(mix, dim=-1)
        return out
