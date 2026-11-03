"""A finite token source, shared by every planner, without changing its kernel.

At N generated source tokens the next distribution is a deterministic EOS.
This extra EOS is an internal end-of-stream delimiter: it adds no visible
bytes, consumes no source-token budget, and is never reported as natural EOS.
Counting each path separately matters when LAS retains different tokenizations.
"""
from __future__ import annotations

import torch

from stegtext.v2.core.types import TokenDistribution


class TokenBudgetSource:
    def __init__(self, source, max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        self.source = source
        self.max_new_tokens = max_new_tokens
        self.prompt_ids = None

    def __getattr__(self, name):
        return getattr(self.source, name)

    def bind(self, prompt):
        result = self.source.bind(prompt)
        self.prompt_ids = tuple(result.token_ids)
        return result

    def count(self, path):
        ids = tuple(path.token_ids())
        if self.prompt_ids is None or ids[:len(self.prompt_ids)] != self.prompt_ids:
            raise ValueError("Token budget requires a path under the bound prompt")
        return len(ids)-len(self.prompt_ids)

    def distribution(self, path):
        count = self.count(path)
        if count > self.max_new_tokens:
            raise ValueError("A terminated token-budget path cannot be expanded")
        if count == self.max_new_tokens:
            # The existing codecs already reserve EOS bytes outside text. Its
            # deterministic emission flushes retained residuals through the
            # unchanged planner instead of clipping a group or sampling a
            # representative prematurely.
            probs = torch.zeros(self.backend.codec.vocab_size, dtype=torch.float64)
            probs[min(self.eos_ids)] = 1.
            return TokenDistribution(probs)
        return self.source.distribution(path)

    def status(self, state):
        path = state.base_path if hasattr(state, 'base_path') else state.path
        count = self.count(path)
        forced = bool(state.terminal and count == self.max_new_tokens+1)
        return dict(max_new_tokens=self.max_new_tokens, source_terminal=bool(state.terminal),
                    token_limit_reached=forced,
                    representative_generated_tokens=count-int(forced),
                    unit='source_generated_tokens', natural_eos_counts=True,
                    internal_boundary_eos_counts=False)
