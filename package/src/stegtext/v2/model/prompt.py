from __future__ import annotations

from typing import Any, Mapping, Protocol


class PromptPolicy(Protocol):
    def render(self, tokenizer: Any, prompt: str) -> Any: ...


class DefaultPromptPolicy:
    """Use a tokenizer chat template when available, otherwise plain text."""

    def __init__(self, *, template_options: Mapping[str, Any] | None = None) -> None:
        self.template_options = dict(template_options or {})

    def render(self, tokenizer: Any, prompt: str):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(prompt)}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    **self.template_options,
                )
            except (TypeError, ValueError):
                pass
        return tokenizer(str(prompt), add_special_tokens=True, return_tensors="pt")


class PlainCompletionPromptPolicy:
    """Explicit plain autoregressive continuation; never applies a chat template."""

    def __init__(self, *, add_special_tokens: bool = True) -> None:
        self.add_special_tokens = bool(add_special_tokens)

    def render(self, tokenizer: Any, prompt: str):
        return tokenizer(str(prompt), add_special_tokens=self.add_special_tokens, return_tensors="pt")


class ChatPromptPolicy:
    """Explicit chat rendering; failure is fatal rather than silently falling back."""

    def __init__(self, *, template_options: Mapping[str, Any] | None = None) -> None:
        self.template_options = dict(template_options or {})

    def render(self, tokenizer: Any, prompt: str):
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("tokenizer has no chat template; use PlainCompletionPromptPolicy")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt)}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **self.template_options,
        )
