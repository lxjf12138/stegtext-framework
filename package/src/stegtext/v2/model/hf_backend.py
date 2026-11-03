from __future__ import annotations

from collections import OrderedDict
import random
from typing import Any, Optional

import torch

from ..core.types import PathRef, PromptState
from .device import resolve_physical_device
from .canonicality import TokenizerCanonicalityOracle
from .prompt import DefaultPromptPolicy, PromptPolicy
from .tokenizer_codec import TokenizerCodec
from .vocab_index import TokenByteIndex


def _eos_ids(tokenizer: Any, model: Any) -> list[int]:
    values: list[int] = []
    for owner in (getattr(model, "generation_config", None), getattr(model, "config", None), tokenizer):
        value = getattr(owner, "eos_token_id", None)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(int(x) for x in value)
        else:
            values.append(int(value))
    return sorted(set(values))


class HFBackend:
    """Generic Hugging Face causal-LM backend; sampling lives elsewhere."""

    def __init__(self, tokenizer: Any, model: Any, *, device: str, eos_ids: list[int], cache_size: int = 2, physical_gpu: int | None = None, prompt_policy: PromptPolicy | None = None, model_id: str | None = None) -> None:
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = torch.device(device)
        self.prompt_policy = prompt_policy or DefaultPromptPolicy()
        self.codec = TokenizerCodec(tokenizer, eos_ids)
        self.codec_index = TokenByteIndex(self.codec.token_bytes_table, device=self.device)
        self.oracle = TokenizerCanonicalityOracle(tokenizer)
        self.model_id = model_id
        self.device_metadata = self._device_metadata(physical_gpu)
        self._logit_cache: OrderedDict[tuple[int, ...], torch.Tensor] = OrderedDict()
        self._active_kv_key: tuple[int, ...] | None = None
        self._active_kv: Any = None
        self.cache_size = max(0, int(cache_size))
        self.stats = {"full_forwards": 0, "incremental_forwards": 0, "tokens_recomputed": 0, "kv_cache_hits": 0, "kv_cache_misses": 0}

    def _device_metadata(self, physical_gpu: int | None) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "physical_gpu": physical_gpu,
            "logical_device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "model_id": self.model_id,
            "eos_token_ids": sorted(self.codec.eos_ids),
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(self.device)
            metadata["gpu_name"] = properties.name
            metadata["gpu_total_memory_bytes"] = int(properties.total_memory)
            metadata["gpu_uuid"] = str(getattr(properties, "uuid", "unknown"))
        else:
            metadata["gpu_name"] = "cpu"
            metadata["gpu_total_memory_bytes"] = 0
            metadata["gpu_uuid"] = None
        return metadata

    def reset_stats(self) -> None:
        for key in self.stats:
            self.stats[key] = 0

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | None = None,
        physical_gpu: int = 1,
        dtype: str | torch.dtype = "bfloat16",
        trust_remote_code: bool = False,
        device_map: str | dict[str, Any] | None = None,
        cache_size: int = 2,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        prompt_policy: PromptPolicy | None = None,
    ) -> "HFBackend":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("HFBackend requires transformers") from exc
        target = device or resolve_physical_device(physical_gpu)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, **(tokenizer_kwargs or {}))
        kwargs = dict(model_kwargs or {})
        kwargs.setdefault("trust_remote_code", trust_remote_code)
        if dtype != "auto":
            kwargs.setdefault("torch_dtype", getattr(torch, str(dtype).split(".")[-1], dtype))
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device_map is None:
            model.to(target)
        return cls(
            tokenizer,
            model,
            device=target,
            eos_ids=_eos_ids(tokenizer, model),
            cache_size=cache_size,
            physical_gpu=physical_gpu,
            prompt_policy=prompt_policy,
            model_id=str(model_id),
        )

    def render_prompt(self, prompt: str) -> PromptState:
        encoded = self.prompt_policy.render(self.tokenizer, str(prompt))
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        elif isinstance(encoded, dict) and "input_ids" in encoded:
            encoded = encoded["input_ids"]
        if not hasattr(encoded, "tolist"):
            raise TypeError("chat template did not return token ids")
        values = encoded.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        ids = tuple(int(x) for x in values)
        self.stats["prompt_tokens"] = len(ids)
        path = PathRef.root()
        for token_id in ids:
            path = path.append(token_id)
        return PromptState(path=path, token_ids=ids, prompt_bytes=self.codec.tokens_to_bytes(ids), prompt_text=str(prompt))

    def next_logits(self, path: PathRef) -> torch.Tensor:
        key = path.token_ids()
        if self.cache_size and key in self._logit_cache:
            self.stats["kv_cache_hits"] += 1
            logits = self._logit_cache.pop(key)
            self._logit_cache[key] = logits
            return logits
        self.stats["kv_cache_misses"] += 1
        ids = torch.tensor([list(path.token_ids())], dtype=torch.long, device=self.device)
        incremental = (
            self._active_kv is not None
            and self._active_kv_key is not None
            and len(key) == len(self._active_kv_key) + 1
            and key[:-1] == self._active_kv_key
        )
        with torch.no_grad():
            if incremental:
                outputs = self.model(
                    input_ids=ids[:, -1:],
                    past_key_values=self._active_kv,
                    use_cache=True,
                )
            else:
                outputs = self.model(input_ids=ids, use_cache=True)
        if incremental:
            self.stats["incremental_forwards"] += 1
            self.stats["tokens_recomputed"] += 1
        else:
            self.stats["full_forwards"] += 1
            self.stats["tokens_recomputed"] += int(ids.shape[1])
        past = getattr(outputs, "past_key_values", None)
        self._active_kv_key = key
        self._active_kv = past
        logits = outputs.logits[0, -1].detach()
        # Some checkpoints pad the output head beyond tokenizer.get_vocab().
        # Those ids have no canonical byte realization and are part of the
        # explicit public validity mask, rather than an implicit LAS drop.
        if logits.numel() > self.codec.vocab_size:
            logits = logits.clone()
            logits[self.codec.vocab_size :] = -torch.inf
        if self.cache_size:
            self._logit_cache[key] = logits
            while len(self._logit_cache) > self.cache_size:
                self._logit_cache.popitem(last=False)
        return logits

    def validate_tokenizer_contract(self) -> None:
        samples = ("hello world", "你好，世界", "boundary\ncheck", "é byte")
        sequences = [self.tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"] for text in samples]
        normalized = []
        for ids in sequences:
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            normalized.append(tuple(int(x) for x in ids))
        # Include special-token and random boundary-heavy sequences.  The
        # random generator is local and deterministic so validation is
        # reproducible without changing the experiment RNG stream.
        special_ids = getattr(self.tokenizer, "all_special_ids", ())
        normalized.extend((int(token_id),) for token_id in special_ids if not self.codec.is_eos(int(token_id)))
        generator = random.Random(0x5EED)
        for _ in range(16):
            length = generator.randint(1, 8)
            normalized.append(tuple(generator.randrange(self.codec.vocab_size) for _ in range(length)))
        self.codec.validate(normalized)
