from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from .base import Candidate, Source
from .vocab import TokenByteVocab


@dataclass
class HFSourceConfig:
    """Configuration wrapper for HuggingFace-backed causal sources."""

    tokenizer: object
    model: torch.nn.Module
    end_token_ids: Sequence[object]
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = None
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    think_end_token_id: Optional[int] = None
    think_max_steps: int = 12800
    # Optional: when provided, the assistant response will be prefixed with
    # this string as part of the prompt (so generation continues after it).
    assistant_prefix: Optional[str] = None


@dataclass
class _CacheState:
    tokens: torch.LongTensor
    attention_mask: torch.Tensor
    past_key_values: Any
    last_logits: torch.Tensor
    vb: bytes
    prompt_byte_len: int


class HFSource(Source):
    """Source emitting candidates from a supplied HuggingFace causal model."""

    def __init__(self, config: HFSourceConfig) -> None:
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")
        if config.top_k is not None and config.top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if config.top_p is not None and not (0.0 < config.top_p <= 1.0):
            raise ValueError("top_p must lie in the interval (0, 1]")

        self._config = config
        self._tokenizer = config.tokenizer
        self._model = config.model.eval()

        # Infer the primary execution device from the model parameters.
        try:
            first_param = next(self._model.parameters())
            self._model_device = first_param.device
        except StopIteration:  # pragma: no cover - unlikely for standard HF models
            self._model_device = torch.device("cpu")

        eos_candidates = self._normalize_end_token_ids(config.end_token_ids)
        self._end_token_ids = set()
        filtered_end_ids: list[int] = []
        if (
            config.think_end_token_id is not None
            and config.think_end_token_id in eos_candidates
        ):
            warnings.warn(
                "think_end_token_id overlaps end_token_ids; removed from EOS mapping to keep think phase distinguishable",
                RuntimeWarning,
            )
        for tid in eos_candidates:
            if tid == config.think_end_token_id:
                continue
            if tid not in self._end_token_ids:
                self._end_token_ids.add(tid)
                filtered_end_ids.append(tid)

        self._vocab = TokenByteVocab.from_tokenizer(
            self._tokenizer,
            end_token_ids=filtered_end_ids,
        )
        self._ensure_special_token_bytes()
        self._init_cache: dict[str, torch.LongTensor] = {}
        self._cache_state: Optional[_CacheState] = None
        self._last_prompt: Optional[str] = None

    # ------------------------------------------------------------------
    # Source interface
    # ------------------------------------------------------------------
    def init(self, prompt: str, *, rng=None) -> list[Candidate]:
        self.reset_cache()
        self._last_prompt = prompt
        cached = self._init_cache.get(prompt)
        if cached is None:
            prefix_tokens = self._encode_prompt(prompt).to(self._model_device)
            prefix_tokens = self._prefill_think(prefix_tokens)
            self._init_cache[prompt] = prefix_tokens.detach().cpu()
        else:
            prefix_tokens = cached.to(self._model_device)
        prompt_len = int(prefix_tokens.numel())
        return self._build_candidates(prefix_tokens, prompt_len)

    def generate(
        self,
        prefix_tokens: torch.LongTensor,
        *,
        prompt_len: int | None = None,
        rng=None,
    ) -> list[Candidate]:
        if not isinstance(prefix_tokens, torch.Tensor):
            raise TypeError("prefix_tokens must be a torch.Tensor")
        prefix_tokens = prefix_tokens.detach()
        if prefix_tokens.device != self._model_device:
            prefix_tokens = prefix_tokens.to(self._model_device)
        if prefix_tokens.dtype != torch.long:
            prefix_tokens = prefix_tokens.to(torch.long)
        if prompt_len is None:
            prompt_len = int(prefix_tokens.numel())
        return self._build_candidates(prefix_tokens, prompt_len=prompt_len)

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

    # ------------------------------------------------------------------
    # Perplexity utility
    # ------------------------------------------------------------------
    def ppl_full(
        self,
        prompt: str,
        visible_text: str,
        *,
        system_prompt: Optional[str] = None,
        template_override: Optional[str] = None,
        assistant_only: bool = False,
    ) -> tuple[float, int]:
        """Compute PPL over the full sequence: prompt + visible_text.

        The rendering matches this source's configuration:
          - chat: use tokenizer.apply_chat_template with assistant content=visible_text
          - completion: format the configured template and append visible_text

        Returns (ppl, token_count).
        """
        # Render full text using the same chat/completion logic as encoding.
        template = self._config.prompt_template
        sys_prompt = system_prompt if system_prompt is not None else self._config.system_prompt

        # Heuristic to decide chat vs completion consistent with _encode_prompt.
        use_chat = False
        if template is None and hasattr(self._tokenizer, "apply_chat_template"):
            use_chat = True
        elif template and template.strip().lower() == "chat" and hasattr(
            self._tokenizer, "apply_chat_template"
        ):
            use_chat = True

        if use_chat:
            messages_user = []
            if sys_prompt:
                messages_user.append({"role": "system", "content": sys_prompt})
            messages_user.append({"role": "user", "content": prompt})
            # Full conversation with assistant content
            messages_full = list(messages_user)
            messages_full.append({"role": "assistant", "content": visible_text})

            rendered_user = self._tokenizer.apply_chat_template(
                messages_user,
                tokenize=False,
                add_generation_prompt=False,
            )
            rendered_full = self._tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            base = template_override if template_override is not None else (
                template if template not in (None, "chat") else "{prompt}"
            )
            rendered_user = base.format(prompt=prompt)
            rendered_full = rendered_user + visible_text

        # Tokenize user-only and full sequences
        tok_full = self._tokenizer(
            rendered_full,
            add_special_tokens=False,
            return_tensors="pt",
        )
        if not hasattr(tok_full, "input_ids"):
            raise TypeError("tokenizer must return input_ids for PPL computation")
        ids = tok_full.input_ids.to(self._model_device)
        attn = None
        if hasattr(tok_full, "attention_mask") and tok_full.attention_mask is not None:
            attn = tok_full.attention_mask.to(self._model_device)

        if assistant_only:
            tok_user = self._tokenizer(
                rendered_user,
                add_special_tokens=False,
                return_tensors="pt",
            )
            pre_len = int(tok_user.input_ids.numel()) if hasattr(tok_user, "input_ids") else 0
            labels = ids.clone()
            # Mask out everything before assistant span
            labels[:, :pre_len] = -100
        else:
            labels = ids

        # Teacher-forced LM loss over the whole sequence.
        with torch.no_grad():
            outputs = self._model(input_ids=ids, attention_mask=attn, labels=labels)
            loss = outputs.loss.detach().to(torch.float32)
        ppl = float(torch.exp(loss).item())
        n_tokens = int(ids.numel())
        return ppl, n_tokens

    def ppl_from_ids(
        self,
        ids: torch.LongTensor,
        *,
        mask_prefix_len: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[float, int]:
        """Compute PPL directly from precomputed token ids.

        - ids: 1-D tensor of token ids representing the full sequence used during encoding
        - mask_prefix_len: number of leading tokens to mask out from loss (labels=-100),
          e.g., to exclude prompt/user tokens and only score assistant payload.
        - attention_mask: optional attention mask; if None, will use ones.
        Returns (ppl, token_count).
        """
        if ids.ndim != 1:
            ids = ids.view(-1)
        ids = ids.to(dtype=torch.long, device=self._model_device)
        input_ids = ids.unsqueeze(0)
        if attention_mask is None:
            attn = torch.ones_like(input_ids, dtype=torch.long, device=self._model_device)
        else:
            attn = attention_mask.to(self._model_device)
        labels = input_ids.clone()
        if mask_prefix_len > 0:
            m = int(mask_prefix_len)
            if m > 0 and m <= labels.size(1):
                labels[:, :m] = -100
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = outputs.loss.detach().to(torch.float32)
        ppl = float(torch.exp(loss).item())
        return ppl, int(input_ids.numel())

    # ------------------------------------------------------------------
    # Prompt & prefilling helpers
    # ------------------------------------------------------------------
    def _encode_prompt(self, prompt: str) -> torch.LongTensor:
        template = self._config.prompt_template
        system_prompt = self._config.system_prompt
        assistant_prefix = self._config.assistant_prefix or ""

        use_chat = False
        if template is None and hasattr(self._tokenizer, "apply_chat_template"):
            use_chat = True
        elif template and template.strip().lower() == "chat" and hasattr(
            self._tokenizer, "apply_chat_template"
        ):
            use_chat = True

        if use_chat:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if assistant_prefix:
                # Append assistant prefix into the assistant turn before generation
                rendered = rendered + assistant_prefix
        else:
            template_str = template if template not in (None, "chat") else "{prompt}"
            rendered = template_str.format(prompt=prompt)
            if assistant_prefix:
                rendered = rendered + assistant_prefix
        tokenized = self._tokenizer(
            rendered,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None,
        )

        if isinstance(tokenized, dict):
            input_ids = tokenized.get("input_ids")
        elif hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids
        else:
            input_ids = tokenized

        if isinstance(input_ids, (list, tuple)) and input_ids and isinstance(input_ids[0], (list, tuple)):
            input_ids = input_ids[0]

        if not isinstance(input_ids, (list, tuple)):
            raise TypeError("tokenizer must return a sequence of token ids")

        return torch.tensor([int(t) for t in input_ids], dtype=torch.long)

    def _prefill_think(self, tokens: torch.LongTensor) -> torch.LongTensor:
        target = self._config.think_end_token_id
        if target is None or tokens.numel() == 0:
            return tokens

        prompt_len = int(tokens.numel())
        current = tokens.detach().to(self._model_device)
        for _ in range(self._config.think_max_steps):
            if current[-1].item() == target:
                break
            logits = self._next_token_logits(current, prompt_len)
            if logits is None:
                break
            next_id = int(torch.argmax(logits).item())
            current = torch.cat([
                current,
                torch.tensor([next_id], dtype=torch.long, device=self._model_device),
            ])
            prompt_len = int(current.numel())
            if next_id == target:
                break
        if self._cache_state is not None:
            # Any cached logits correspond to the state *before* we appended the
            # latest token, so drop them to avoid serving stale predictions.
            self._cache_state = None
        return current

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------
    def _build_candidates(self, prefix_tokens: torch.LongTensor, prompt_len: int) -> list[Candidate]:
        if prefix_tokens.ndim != 1:
            raise ValueError("prefix_tokens must be a 1-D tensor")
        if prefix_tokens.numel() == 0:
            raise ValueError("prefix_tokens must contain at least one token")

        logits = self._next_token_logits(prefix_tokens, prompt_len)
        if logits is None:
            return []

        logits = logits / self._config.temperature
        logits = self._apply_sampling_filters(logits)

        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        sorted_probs = sorted_probs / sorted_probs.sum()

        positive_mask = sorted_probs > 0
        if not positive_mask.any():
            return []

        selected_probs = sorted_probs[positive_mask]
        selected_indices = sorted_indices[positive_mask]
        count = int(selected_probs.shape[0])

        cache = self._cache_state
        if cache is not None and torch.equal(cache.tokens, prefix_tokens):
            prefix_bytes = cache.vb
            prompt_byte_len = cache.prompt_byte_len
        else:
            prefix_tokens_cpu = prefix_tokens.detach().cpu()
            prefix_raw_bytes = self._vocab.tokens_to_bytes(prefix_tokens_cpu)
            prompt_prefix_bytes = self._vocab.tokens_to_bytes(prefix_tokens_cpu[:prompt_len])
            prefix_bytes = prefix_raw_bytes
            prompt_byte_len = len(prompt_prefix_bytes)

        candidate_tokens = torch.empty(
            (count, prefix_tokens.size(0) + 1),
            dtype=torch.long,
            device=self._model_device,
        )
        if prefix_tokens.numel() > 0:
            candidate_tokens[:, :-1] = prefix_tokens
        candidate_tokens[:, -1] = selected_indices.to(self._model_device)

        selected_indices_cpu = selected_indices.detach().cpu()
        candidates: list[Candidate] = []
        for idx in range(count):
            prob = float(round(float(selected_probs[idx].item()), 12))
            token_id = int(selected_indices_cpu[idx].item())
            vb = prefix_bytes + self._vocab.token_bytes(token_id)
            cand_tokens = candidate_tokens[idx].clone()

            candidates.append(
                Candidate(
                    p=prob,
                    tokens=cand_tokens,
                    vb=vb,
                    is_eos=token_id in self._end_token_ids,
                    prompt_len=prompt_len,
                    prompt_byte_len=prompt_byte_len,
                )
            )

        return candidates

    # ------------------------------------------------------------------
    # Model utilities & sampling helpers
    # ------------------------------------------------------------------
    def _next_token_logits(self, tokens: torch.LongTensor, prompt_len: int) -> Optional[torch.Tensor]:
        if tokens.numel() == 0:
            return None
        tokens_device = tokens.to(dtype=torch.long, device=self._model_device)
        return self._forward_with_cache(tokens_device, prompt_len)

    def _forward_with_cache(self, tokens: torch.LongTensor, prompt_len: int) -> torch.Tensor:
        device = self._model_device
        seq_len = int(tokens.size(0))

        cache = self._cache_state
        if cache is not None:
            cached_len = int(cache.tokens.size(0))
            if seq_len == cached_len and torch.equal(tokens, cache.tokens):
                return cache.last_logits.clone()
            if (
                seq_len > cached_len
                and torch.equal(tokens[:cached_len], cache.tokens)
                and cache.past_key_values is not None
            ):
                suffix = tokens[cached_len:]
                suffix_len = int(suffix.size(0))
                if suffix_len > 0:
                    input_ids = suffix.unsqueeze(0).to(device)
                    attn_suffix = torch.ones(
                        (1, suffix_len),
                        device=device,
                        dtype=cache.attention_mask.dtype,
                    )
                    attention_mask = torch.cat([cache.attention_mask, attn_suffix], dim=-1)
                    position_ids = torch.arange(
                        cached_len,
                        seq_len,
                        device=device,
                        dtype=torch.long,
                    ).unsqueeze(0)
                    with torch.no_grad():
                        outputs = self._model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=cache.past_key_values,
                            use_cache=True,
                        )
                    logits = outputs.logits[0, -1, :].detach().to(torch.float32)
                    past = getattr(outputs, "past_key_values", None)
                    if past is not None:
                        vb = cache.vb
                        for token_id in suffix.tolist():
                            vb += self._vocab.token_bytes(int(token_id))
                        new_tokens = torch.cat([cache.tokens, suffix.clone()], dim=0)
                        self._cache_state = _CacheState(
                            tokens=new_tokens,
                            attention_mask=attention_mask.detach(),
                            past_key_values=past,
                            last_logits=logits,
                            vb=vb,
                            prompt_byte_len=cache.prompt_byte_len,
                        )
                    else:
                        self._cache_state = None
                    return logits

        input_ids = tokens.unsqueeze(0).to(device)
        attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
        logits = outputs.logits[0, -1, :].detach().to(torch.float32)
        past = getattr(outputs, "past_key_values", None)
        prefix_tokens_cpu = tokens.detach().cpu()
        prefix_bytes = self._vocab.tokens_to_bytes(prefix_tokens_cpu)
        prompt_prefix_bytes = self._vocab.tokens_to_bytes(prefix_tokens_cpu[:prompt_len])
        prompt_byte_len = len(prompt_prefix_bytes)
        if past is not None:
            self._cache_state = _CacheState(
                tokens=tokens.clone(),
                attention_mask=attention_mask.detach(),
                past_key_values=past,
                last_logits=logits,
                vb=prefix_bytes,
                prompt_byte_len=prompt_byte_len,
            )
        else:
            self._cache_state = None
        return logits

    def _apply_sampling_filters(self, logits: torch.Tensor) -> torch.Tensor:
        filtered = logits.clone()

        top_k = self._config.top_k
        if top_k is not None and top_k < filtered.size(-1):
            kth_values, kth_indices = torch.topk(filtered, k=top_k)
            mask = torch.full_like(filtered, float("-inf"))
            mask.scatter_(0, kth_indices, kth_values)
            filtered = mask

        top_p = self._config.top_p
        if top_p is not None and top_p < 1.0:
            filtered = self._apply_top_p(filtered, top_p)

        return filtered

    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to the logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # Shift cumulative distribution to ensure at least one token remains.
        cutoff = cumulative > top_p
        cutoff[..., 0] = False
        sorted_logits[cutoff] = float("-inf")

        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(0, sorted_indices, sorted_logits)
        return filtered


    def _normalize_end_token_ids(self, raw_ids: Sequence[object]) -> list[int]:
        if raw_ids is None:
            raise ValueError("HFSourceConfig.end_token_ids must be provided")
        if len(raw_ids) == 0:
            raise ValueError("HFSourceConfig.end_token_ids must be non-empty")

        resolved: list[int] = []
        seen: set[int] = set()
        for item in raw_ids:
            token_id: Optional[int]
            if isinstance(item, int):
                token_id = int(item)
            elif isinstance(item, torch.Tensor):
                token_id = int(item.item())
            else:
                try:
                    token_id = self._tokenizer.convert_tokens_to_ids(item)  # type: ignore[arg-type]
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Unable to convert end token {item!r} to id") from exc
            if token_id is None or token_id < 0:
                raise ValueError(f"Invalid end token id derived from {item!r}")
            if token_id in seen:
                continue
            seen.add(token_id)
            resolved.append(token_id)
        return resolved

    def _ensure_special_token_bytes(self) -> None:
        replacements: list[tuple[int, bytes]] = []

        # Preserve explicit <think> tokens if present in the vocab.
        try:
            think_start = self._tokenizer.convert_tokens_to_ids("<think>")
            if think_start is not None and int(think_start) >= 0:
                replacements.append((int(think_start), b"<think>"))
        except Exception:
            pass

        if self._config.think_end_token_id is not None:
            replacements.append((int(self._config.think_end_token_id), b"</think>"))

        for token_id, literal in replacements:
            self._vocab.set_token_bytes(token_id, literal)

    def reset_cache(self, *, full: bool = False) -> None:
        self._cache_state = None
        if full:
            self._init_cache.clear()
            self._last_prompt = None
