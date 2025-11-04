"""Per-sentence encode/decode engine coordinating source, disambiguator and coder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
from stegtext.components.source.base import EOS_STEGA 
from .contracts import Disambiguator, Source
from .data import Candidate
from .rng import CSPRNG, derive_rng

_DEFAULT_PADDING_BITS = 1024


@dataclass
class EncodeOutput:
    text: str
    tokens: int
    emitted_bits: str
    stop_reason: str
    finished: bool


@dataclass
class DecodeOutput:
    bits: str
    steps: int
    stop_reason: str


@dataclass
class _RunResult:
    text: str
    tokens: int
    bits: str
    steps: int
    finished: bool
    stop_reason: str


class StegoEngine:
    """Single-sentence driver. Call `encode` or `decode` per sentence."""

    def __init__(
        self,
        source: Source,
        disambiguator: Disambiguator,
        coder,
        *,
        hooks: Optional[Sequence[Any]] = None,
    ) -> None:
        self.source = source
        self.dis = disambiguator
        self.coder = coder
        self.hooks: List[Any] = list(hooks or [])

    # ------------------------------------------------------------------
    def encode(
        self,
        prompt: str,
        bit_stream: str,
        *,
        rng: CSPRNG,
        max_steps: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> EncodeOutput:
        padding = _derive_padding(bit_stream, _DEFAULT_PADDING_BITS)
        result = self._run(
            mode="encode",
            prompt=prompt,
            rng=rng,
            bit_stream=bit_stream + padding,
            target_text=None,
            max_steps=max_steps,
            max_tokens=max_tokens,
        )
        return EncodeOutput(
            text=result.text,
            tokens=result.tokens,
            emitted_bits=result.bits,
            stop_reason=result.stop_reason,
            finished=result.finished,
        )

    # ------------------------------------------------------------------
    def decode(
        self,
        prompt: str,
        visible_text: str,
        *,
        rng: CSPRNG,
        max_steps: Optional[int] = None,
    ) -> DecodeOutput:
        result = self._run(
            mode="decode",
            prompt=prompt,
            rng=rng,
            bit_stream="",
            target_text=visible_text,
            max_steps=max_steps,
            max_tokens=None,
        )
        return DecodeOutput(bits=result.bits, steps=result.steps, stop_reason=result.stop_reason)

    def _run(
        self,
        *,
        mode: str,
        prompt: str,
        rng: CSPRNG,
        bit_stream: str,
        target_text: Optional[str],
        max_steps: Optional[int],
        max_tokens: Optional[int],
    ) -> _RunResult:
        sentence_rng = derive_rng(rng, "sentence")
        init_rng = derive_rng(sentence_rng, "source_init")
        self.source.reset_cache(full=True)
        candidates = self.source.init(prompt, rng=init_rng)
        if not candidates:
            raise RuntimeError("source.init produced no candidates")

        runner = _SentenceRunner(
            mode=mode,
            source=self.source,
            dis=self.dis,
            coder=self.coder,
            hooks=self.hooks,
            sentence_rng=sentence_rng,
            candidates=candidates,
            target_text=target_text,
            bit_stream=bit_stream,
            max_steps=max_steps,
            max_tokens=max_tokens,
        )
        return runner.run()


class _SentenceRunner:
    def __init__(
        self,
        *,
        mode: str,
        source: Source,
        dis: Disambiguator,
        coder,
        hooks: Sequence[Any],
        sentence_rng: CSPRNG,
        candidates: List[Candidate],
        target_text: Optional[str],
        bit_stream: str,
        max_steps: Optional[int],
        max_tokens: Optional[int],
    ) -> None:
        self.mode = mode
        self.source = source
        self.dis = dis
        self.coder = coder
        self.hooks = list(hooks)
        self.sentence_rng = sentence_rng
        self.current_candidates = candidates
        first_candidate = self.current_candidates[0] if self.current_candidates else None
        prompt_byte_len = int(getattr(first_candidate, "prompt_byte_len", 0)) if first_candidate else 0
        self.prefix_bytes = b""
        if first_candidate is not None and prompt_byte_len > 0:
            self.prefix_bytes = first_candidate.vb[:prompt_byte_len]
        if mode == "decode":
            if target_text is None:
                raise RuntimeError("decode mode requires visible text")
             # local import to avoid circular deps

            payload_bytes = target_text.encode("utf-8")
            self.target_bytes = self.prefix_bytes + payload_bytes + EOS_STEGA
        else:
            self.target_bytes = None
        self.bit_stream = bit_stream
        self.max_steps = max_steps
        # Optional hard cap measured in payload tokens (tokens after the prompt).
        # If provided, this takes precedence over step-count differences across
        # disambiguators so that all methods stop after producing the same
        # number of payload tokens.
        self.max_tokens = max_tokens

        self.bits: List[str] = []
        self.step = 0
        self.current_prefix_len = 0
        self.finished = False
        self.stop_reason = "ongoing"
        self.final_candidate: Optional[Candidate] = None

        self._notify("on_sentence_start")

    # ------------------------------------------------------------------
    def run(self) -> _RunResult:
        while self.current_candidates:
            if self.max_steps is not None and self.step >= self.max_steps:
                self.stop_reason = "max_steps"
                break

            if all(c.is_eos for c in self.current_candidates):
                if len(self.current_candidates) != 1:
                    raise RuntimeError("multiple EOS candidates present; disambiguator must collapse")
                self.final_candidate = self.current_candidates[0]
                self.finished = True
                self.stop_reason = "eos"
                break

            self.step += 1

            # expose pre-plan candidates to hooks for diagnostics/metrics (e.g., KL before/after disambiguation)
            self._notify("on_candidates", step=self.step, candidates=self.current_candidates)

            plan = self.dis.plan(self.current_candidates)
            groups = plan.groups.groups
            self._notify(
                "on_plan",
                step=self.step,
                groups=len(groups),
                group_keys=[g.key for g in groups],
                group_probs=plan.group_probs.tolist() if isinstance(plan.group_probs, torch.Tensor) else plan.group_probs,
            )

            step_rng = derive_rng(self.sentence_rng, f"step:{self.step}")
            if self.mode == "encode":
                emitted, chosen_idx = self._coder_encode(plan.group_probs, step_rng)
            else:
                chosen_idx = self._choose_group_for_decode(groups)
                emitted = self.coder.decode(plan.group_probs, chosen_idx, rng=step_rng)
            self.bits.append(emitted)

            advance_rng = derive_rng(self.sentence_rng, f"advance:{self.step}")

            def _generate(prefix: torch.LongTensor, prompt_len: Optional[int] = None) -> List[Candidate]:
                return self.source.generate(prefix, prompt_len=prompt_len, rng=advance_rng)

            outcome = self.dis.advance(plan, chosen_idx, advance_rng, _generate)
            if isinstance(outcome, Candidate):
                selected = outcome
                next_candidates: List[Candidate] = []
            elif isinstance(outcome, list):
                selected = getattr(self.dis, "last_selected", None)
                if selected is None:
                    raise RuntimeError("disambiguator did not expose last_selected candidate")
                next_candidates = outcome
            else:
                raise TypeError("disambiguator advance must return Candidate or List[Candidate]")

            self.final_candidate = selected
            self.current_prefix_len = len(selected.vb)
            # Optional: stop when we have emitted the desired number of payload
            # tokens regardless of how many planning steps were required. This
            # makes "budget" semantics comparable across different
            # disambiguators (e.g., lookahead vs baseline/syncpool) where one
            # planning step may or may not advance the visible/token prefix.
            if self.max_tokens is not None:
                total_tokens = int(selected.tokens.numel())
                prompt_len = int(getattr(selected, "prompt_len", 0))
                payload_tokens = max(0, total_tokens - prompt_len)
                if payload_tokens >= int(self.max_tokens):
                    self.stop_reason = "max_tokens"
                    break

            self._notify(
                "on_gen_prefix",
                step=self.step,
                prefix_ids=selected.tokens.detach().cpu().tolist(),
            )
            self._notify(
                "on_step",
                step=self.step,
                chosen_group=chosen_idx,
                chosen_key=groups[chosen_idx].key,
                emitted_bits=emitted,
            )

            if selected.is_eos:
                self.finished = True
                self.stop_reason = "eos"
                break

            self.current_candidates = next_candidates
            if not self.current_candidates:
                self.stop_reason = "stalled"
                break

        if self.final_candidate is None:
            raise RuntimeError("no final candidate produced")

        full_text = self.source.decode(self.final_candidate, from_payload=False)
        payload_text = self.source.decode(self.final_candidate, from_payload=True, strip_eos=True)

        prompt_byte_len = getattr(self.final_candidate, "prompt_byte_len", 0)
        total_tokens = int(self.final_candidate.tokens.numel())
        prompt_len = int(getattr(self.final_candidate, "prompt_len", 0))
        payload_tokens = max(0, total_tokens - prompt_len)
        bits_joined = "".join(self.bits)
        self._notify("on_sentence_end", text=full_text, steps=self.step, bits_real=len(bits_joined))
        visible = payload_text
        return _RunResult(
            text=visible,
            tokens=payload_tokens,
            bits=bits_joined,
            steps=self.step,
            finished=self.finished,
            stop_reason=self.stop_reason,
        )

    # ------------------------------------------------------
    def _coder_encode(self, group_probs, rng: CSPRNG) -> Tuple[str, int]:
        emitted, idx = self.coder.encode(group_probs, self.bit_stream, rng=rng)
        consume = len(emitted)
        if consume > 0:
            self.bit_stream = self.bit_stream[consume:]
        return emitted, idx

    def _choose_group_for_decode(self, groups) -> int:
        if self.target_bytes is None:
            raise RuntimeError("decode mode requires visible bytes")
        matches: List[int] = []
        for idx, group in enumerate(groups):
            key = group.key
            if len(key) <= self.current_prefix_len:
                continue
            if self.target_bytes.startswith(key):
                matches.append(idx)
        if not matches:
            print("[DEBUG] decode no match: target=", self.target_bytes, "prefix_len=", self.current_prefix_len)
            raise RuntimeError("no group matches target bytes")
        if len(matches) > 1:
            raise RuntimeError("multiple groups match target bytes")
        return matches[0]

    def _notify(self, name: str, **payload: Any) -> None:
        for hook in self.hooks:
            fn = getattr(hook, name, None)
            if callable(fn):
                try:
                    fn(mode=self.mode, **payload)
                except Exception:
                    continue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_padding(bit_stream: str, length: int) -> str:
    seed = bit_stream.encode("ascii", errors="ignore")
    if not seed:
        seed = b"\x00"
    counter = 0
    out = []
    while len("".join(out)) < length:
        counter_bytes = counter.to_bytes(4, "big")
        digest = hashlib.sha256(seed + counter_bytes).digest()
        out.append("".join(f"{byte:08b}" for byte in digest))
        counter += 1
    return ("".join(out))[:length]
