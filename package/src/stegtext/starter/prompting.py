from __future__ import annotations
from stegtext.v2.model.prompt import PlainCompletionPromptPolicy, ChatPromptPolicy


def make_prompt_policy(mode: str, *, add_special_tokens: bool = True, template_options=None):
    mode=str(mode).strip().lower()
    if mode == "plain_completion":
        return PlainCompletionPromptPolicy(add_special_tokens=add_special_tokens)
    if mode == "chat":
        return ChatPromptPolicy(template_options=template_options)
    raise ValueError("prompt mode must be explicit: plain_completion or chat")
