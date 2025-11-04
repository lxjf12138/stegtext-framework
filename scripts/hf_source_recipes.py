#!/usr/bin/env python3
"""Example recipes for instantiating `HFSource` with chat vs completion models.

This module shows how to build `HFSource` objects for:
  * Qwen2.5 chat-style models (using their chat template)
  * Qwen2.5 completion-style models (plain prompt completion)
  * Meta-Llama-3 completion-style models (instruction tuned, BOS-required)

The examples avoid JSON-based configuration so you can inspect the Python-level
construction directly and adapt them for your own projects.

Usage (illustrative):

    from scripts.hf_source_recipes import make_qwen25_source

    source = make_qwen25_source("~/path/Qwen2.5-3B", device="cuda:0", chat=True)
    candidates = source.init("你好，请用一句话介绍自己。")

Each helper returns a fully initialised `HFSource` placed on the requested
`device`.  They do not cache the models, so make sure to reuse the returned
`HFSource` instead of calling the helper repeatedly in a tight loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stegtext.components.source.hf import HFSource, HFSourceConfig

__all__ = [
    "make_qwen25_source",
    "make_qwen3_source",
    "make_deepseek_source",
    "make_glm4_source",
    "make_llama3_8b_source",
]


_QWEN25_END_TOKENS = (
    "<|im_end|>",
    "<|endoftext|>",
)

_QWEN3_END_TOKENS = (
    "<|im_end|>",
    "<|endoftext|>",
)

_DEEPSEEK_END_TOKENS = (
    "<｜end▁of▁sentence｜>",
)

_GLM4_END_TOKENS = (
    "<|endoftext|>",
    "<eop>",
    "<|user|>",
)

_LLAMA3_END_TOKENS = (
    "<|eot_id|>",
    "<|end_of_text|>",
)


def _expand(path: str) -> str:
    return str(Path(path).expanduser())


def _resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype alias '{name}'")
    return mapping[key]


def _load_model_and_tokenizer(
    model_path: str,
    *,
    device: str,
    dtype: Optional[str],
    trust_remote_code: bool,
    tokenizer_kwargs: Optional[dict] = None,
    model_kwargs: Optional[dict] = None,
):
    tok_kwargs = dict(tokenizer_kwargs or {})
    tokenizer = AutoTokenizer.from_pretrained(
        _expand(model_path),
        trust_remote_code=trust_remote_code,
        **tok_kwargs,
    )

    mdl_kwargs = dict(model_kwargs or {})
    torch_dtype = _resolve_dtype(dtype or mdl_kwargs.pop("torch_dtype", None))
    if torch_dtype is not None:
        mdl_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        _expand(model_path),
        trust_remote_code=trust_remote_code,
        **mdl_kwargs,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def _token_id(tokenizer, token: Optional[object]) -> Optional[int]:
    if token is None:
        return None
    if isinstance(token, int):
        return token if token >= 0 else None
    if not isinstance(token, str) or not token:
        return None
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        tid = None
    if tid is None or int(tid) < 0:
        return None
    return int(tid)


def _convert_stop_tokens(tokenizer, tokens: Sequence[object]) -> list[int]:
    if not tokens:
        raise ValueError("end_tokens must be provided")
    ids: list[int] = []
    seen: set[int] = set()
    for tok in tokens:
        tid = _token_id(tokenizer, tok)
        if tid is None:
            raise ValueError(f"Unable to resolve stop token {tok!r}")
        if tid in seen:
            continue
        seen.add(tid)
        ids.append(tid)
    return ids


def _build_chat_cfg(
    tokenizer,
    model,
    *,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    system_prompt: Optional[str],
    end_tokens: Sequence[object],
    think_token: Optional[str] = None,
    think_max_steps: int = 4096,
) -> HFSourceConfig:
    cfg = HFSourceConfig(
        tokenizer=tokenizer,
        model=model,
        end_token_ids=_convert_stop_tokens(tokenizer, end_tokens),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        prompt_template=None,
        system_prompt=system_prompt,
    )
    if think_token:
        cfg.think_end_token_id = _token_id(tokenizer, think_token)
        cfg.think_max_steps = think_max_steps
    return cfg


def _build_completion_cfg(
    tokenizer,
    model,
    *,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    template: str,
    system_prompt: Optional[str] = None,
    end_tokens: Sequence[object],
    think_token: Optional[str] = None,
    think_max_steps: int = 4096,
) -> HFSourceConfig:
    cfg = HFSourceConfig(
        tokenizer=tokenizer,
        model=model,
        end_token_ids=_convert_stop_tokens(tokenizer, end_tokens),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        prompt_template=template,
        system_prompt=system_prompt,
    )
    if think_token:
        cfg.think_end_token_id = _token_id(tokenizer, think_token)
        cfg.think_max_steps = think_max_steps
    return cfg


# ---------------------------------------------------------------------------
# Qwen 2.5 (chat or completion)
# ---------------------------------------------------------------------------


def make_qwen25_source(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: Optional[str] = "float16",
    chat: bool = True,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    system_prompt: Optional[str] = None,
) -> HFSource:
    tokenizer, model = _load_model_and_tokenizer(
        model_path,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
        tokenizer_kwargs={"use_fast": False},
    )

    if chat:
        cfg = _build_chat_cfg(
            tokenizer,
            model,
            temperature=temperature,
            top_k=None,
            top_p=top_p,
            system_prompt=system_prompt,
            end_tokens=_QWEN25_END_TOKENS,
        )
    else:
        bos = tokenizer.bos_token or ""
        template = f"{bos}{{prompt}}"
        cfg = _build_completion_cfg(
            tokenizer,
            model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            template=template,
            end_tokens=_QWEN25_END_TOKENS,
        )
    return HFSource(cfg)


# ---------------------------------------------------------------------------
# Qwen 3 (supports thinking)
# ---------------------------------------------------------------------------


def make_qwen3_source(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: Optional[str] = "float16",
    chat: bool = True,
    enable_thinking: bool = True,
    think_token: str = "</think>",
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    system_prompt: Optional[str] = None,
) -> HFSource:
    tokenizer, model = _load_model_and_tokenizer(
        model_path,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
        tokenizer_kwargs={"use_fast": False},
    )

    think = think_token if enable_thinking else None

    if chat:
        cfg = _build_chat_cfg(
            tokenizer,
            model,
            temperature=temperature,
            top_k=None,
            top_p=top_p,
            system_prompt=system_prompt,
            end_tokens=_QWEN3_END_TOKENS,
            think_token=think,
            think_max_steps=4096,
        )
    else:
        bos = tokenizer.bos_token or ""
        template = f"{bos}{{prompt}}"
        cfg = _build_completion_cfg(
            tokenizer,
            model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            template=template,
            end_tokens=_QWEN3_END_TOKENS,
            think_token=think,
            think_max_steps=4096,
        )
    return HFSource(cfg)


# ---------------------------------------------------------------------------
# DeepSeek (R1 distillation, thinking capable)
# ---------------------------------------------------------------------------


def make_deepseek_source(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: Optional[str] = "float16",
    chat: bool = True,
    enable_thinking: bool = True,
    think_token: str = "</think>",
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    system_prompt: Optional[str] = None,
) -> HFSource:
    tokenizer, model = _load_model_and_tokenizer(
        model_path,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
        tokenizer_kwargs={"use_fast": False},
    )

    think = think_token if enable_thinking else None

    if chat:
        cfg = _build_chat_cfg(
            tokenizer,
            model,
            temperature=temperature,
            top_k=None,
            top_p=top_p,
            system_prompt=system_prompt,
            end_tokens=_DEEPSEEK_END_TOKENS,
            think_token=think,
            think_max_steps=4096,
        )
    else:
        bos = tokenizer.bos_token or ""
        template = f"{bos}{{prompt}}"
        cfg = _build_completion_cfg(
            tokenizer,
            model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            template=template,
            end_tokens=_DEEPSEEK_END_TOKENS,
            think_token=think,
            think_max_steps=4096,
        )
    return HFSource(cfg)


# ---------------------------------------------------------------------------
# GLM-4 chat models
# ---------------------------------------------------------------------------


def make_glm4_source(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: Optional[str] = "float16",
    chat: bool = True,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.9,
    system_prompt: Optional[str] = None,
) -> HFSource:
    tokenizer, model = _load_model_and_tokenizer(
        model_path,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
        tokenizer_kwargs={"use_fast": False},
    )

    if chat:
        cfg = _build_chat_cfg(
            tokenizer,
            model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            system_prompt=system_prompt,
            end_tokens=_GLM4_END_TOKENS,
        )
    else:
        bos = tokenizer.bos_token or ""
        template = f"{bos}{{prompt}}"
        cfg = _build_completion_cfg(
            tokenizer,
            model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            template=template,
            end_tokens=_GLM4_END_TOKENS,
        )
    return HFSource(cfg)


# ---------------------------------------------------------------------------
# Meta-Llama-3 8B Instruct
# ---------------------------------------------------------------------------


def make_llama3_8b_source(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: Optional[str] = "bfloat16",
    chat: bool = False,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.9,
    system_prompt: Optional[str] = None,
) -> HFSource:
    tokenizer, model = _load_model_and_tokenizer(
        model_path,
        device=device,
        dtype=dtype,
        trust_remote_code=False,
    )

    if chat:
        bos = tokenizer.bos_token or "<|begin_of_text|>"
        eos = tokenizer.eos_token or "<|eot_id|>"
        template_parts = [
            bos,
            "<|start_header_id|>system<|end_header_id|>\n",
            system_prompt or "You are a helpful assistant.",
            eos,
            "<|start_header_id|>user<|end_header_id|>\n",
            "{prompt}",
            eos,
            "<|start_header_id|>assistant<|end_header_id|>\n",
        ]
        template = "".join(template_parts)
        cfg = _build_completion_cfg(
            tokenizer,
            model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            template=template,
            system_prompt=None,
            end_tokens=_LLAMA3_END_TOKENS,
        )
    else:
        bos = tokenizer.bos_token or "<|begin_of_text|>"
        eos = tokenizer.eos_token or "<|eot_id|>"
        template = f"{bos}{{prompt}}{eos}"
        cfg = _build_completion_cfg(
            tokenizer,
            model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            template=template,
            system_prompt=system_prompt,
            end_tokens=_LLAMA3_END_TOKENS,
        )
    return HFSource(cfg)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Instantiate HFSource recipes")
    parser.add_argument(
        "model",
        choices=["qwen25", "qwen3", "deepseek", "glm4", "llama3"],
        help="which recipe to instantiate",
    )
    parser.add_argument("path", help="local path or HF hub id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()

    if args.model == "qwen25":
        src = make_qwen25_source(args.path, device=args.device, dtype=args.dtype)
    elif args.model == "qwen3":
        src = make_qwen3_source(args.path, device=args.device, dtype=args.dtype)
    elif args.model == "deepseek":
        src = make_deepseek_source(args.path, device=args.device, dtype=args.dtype)
    elif args.model == "glm4":
        src = make_glm4_source(args.path, device=args.device, dtype=args.dtype)
    else:
        src = make_llama3_8b_source(args.path, device=args.device, dtype=args.dtype)

    print(f"HFSource ready: {src}")
