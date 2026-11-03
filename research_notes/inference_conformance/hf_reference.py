"""Explicit Transformers 4.57.6 reference adapter, outside the frozen package.

Scope: batch size 1, unpadded plain completion, Qwen3/Llama, DynamicCache,
single-sequence greedy or sampling. A non-prefix query replays from the SAME
prompt prefill and then one token at a time. This fixes the numerical evaluation
schedule; a full prefill of a longer history is not assumed bit-equivalent.
"""
from __future__ import annotations

import copy

import torch
import transformers
from transformers import LogitsProcessorList
from transformers.cache_utils import DynamicCache


class HFReference:
    def __init__(self, backend, config=None, *, validity_mask=None):
        if transformers.__version__ != "4.57.6":
            raise RuntimeError("Re-audit private generation helpers for this Transformers version")
        self.backend = backend
        self.model = backend.model
        self.device = backend.device
        self.config = copy.deepcopy(config or self.model.generation_config)
        if self.config.num_beams != 1 or self.config.num_return_sequences != 1:
            raise ValueError("Only single-sequence greedy/sampling is audited")
        if self.config.use_cache is not True or self.config.cache_implementation not in (None, "dynamic"):
            raise ValueError("Only default DynamicCache is audited")
        if self.config.token_healing or self.config.constraints or self.config.force_words_ids:
            raise ValueError("Unsupported generation mode")
        self.validity_mask = validity_mask
        self.prompt = None
        self.key = None
        self.cache = None
        self.logits = None
        self.forwards = 0
        self.replays = 0

    def bind_prompt(self, token_ids):
        self.prompt = tuple(int(t) for t in token_ids)
        if not self.prompt:
            raise ValueError("Explicit nonempty tokenized prompt required")
        self.key = self.cache = self.logits = None
        self.model._prepare_special_tokens(self.config, kwargs_has_attention_mask=True, device=self.device)
        extra = LogitsProcessorList()
        if self.validity_mask is not None:
            # HF user processors run after penalties, before sampling warpers.
            mask = self.validity_mask
            class Validity:
                def __call__(self, input_ids, scores):
                    return scores.masked_fill(~mask, -torch.inf)
            extra.append(Validity())
        self.processors = self.model._get_logits_processor(
            self.config, input_ids_seq_length=len(self.prompt),
            encoder_input_ids=None, prefix_allowed_tokens_fn=None,
            logits_processor=extra, device=self.device, model_kwargs={})

    @torch.inference_mode()
    def next_logits(self, token_ids):
        key = tuple(int(t) for t in token_ids)
        if self.prompt is None or key[:len(self.prompt)] != self.prompt:
            raise ValueError("History must extend the bound prompt")
        if key == self.key:
            return self.logits
        if self.key is None or len(key) < len(self.key) or key[:len(self.key)] != self.key:
            self.key = None
            self.cache = DynamicCache(config=self.model.config)
            self.replays += 1
        if self.key is None:
            self._forward(self.prompt, 0)
        while len(self.key) < len(key):
            self._forward(key[:len(self.key)+1], len(self.key))
        return self.logits

    def _forward(self, key, start):
        ids = torch.tensor([key], dtype=torch.long, device=self.device)
        positions = torch.arange(start, len(key), device=self.device)
        inputs = self.model.prepare_inputs_for_generation(
            ids, past_key_values=self.cache, attention_mask=torch.ones_like(ids),
            cache_position=positions, use_cache=True, logits_to_keep=1)
        outputs = self.model(**inputs, return_dict=True)
        self.cache = outputs.past_key_values
        self.key = tuple(key)
        self.logits = outputs.logits[0, -1].detach().float()
        self.forwards += 1

    def scores(self, token_ids, logits=None):
        if logits is None:
            logits = self.next_logits(token_ids)
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        return self.processors(ids, logits.float().clone().unsqueeze(0))[0]

    def probabilities(self, token_ids, logits=None):
        scores = self.scores(token_ids, logits)
        if not self.config.do_sample:
            probs = torch.zeros_like(scores)
            probs[scores.argmax()] = 1
            return probs
        return torch.softmax(scores, dim=-1)
