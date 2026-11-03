from __future__ import annotations

from typing import Protocol

import torch

from ..core.types import PathRef, PromptState


class Backend(Protocol):
    codec: object

    def render_prompt(self, prompt: str) -> PromptState: ...
    def next_logits(self, path: PathRef) -> torch.Tensor: ...
