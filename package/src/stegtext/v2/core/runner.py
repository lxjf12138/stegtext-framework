from __future__ import annotations

import hashlib
import resource
import time
from dataclasses import dataclass
from typing import Any, Optional

from .errors import ReceiverMismatchError
from .types import MethodContext, PromptState, VisibleTarget


@dataclass(frozen=True)
class EncodeResult:
    text: str
    emitted_bits: str
    tokens: int
    steps: int
    finished: bool
    stop_reason: str
    stats: dict[str, Any]


@dataclass(frozen=True)
class DecodeResult:
    bits: str
    text: str
    steps: int
    finished: bool
    stop_reason: str
    stats: dict[str, Any]


def _padding(bits: str, length: int = 2048) -> str:
    seed = bits.encode("ascii") or b"\x00"
    out = bytearray()
    counter = 0
    while len(out) * 8 < length:
        out.extend(hashlib.sha256(seed + counter.to_bytes(8, "big")).digest())
        counter += 1
    return "".join(f"{x:08b}" for x in out)[:length]


class StegoRunner:
    """Method-agnostic v2 orchestration loop."""

    def __init__(self, backend, sampler, method, coder) -> None:
        self.backend = backend
        self.sampler = sampler
        self.method = method
        self.coder = coder

    def encode(
        self,
        prompt: str,
        payload_bits: str,
        *,
        rng: Any,
        max_steps: int | None = None,
        max_tokens: int | None = None,
        collect_trace: bool = False,
    ) -> EncodeResult:
        reset = getattr(self.backend, "reset_stats", None)
        if callable(reset):
            reset()
        self._reset_peak_memory()
        prompt_state = self.backend.render_prompt(prompt)
        state = self.method.start(prompt_state)
        coder = self._coder_session("encode", rng)
        ctx = MethodContext(codec=self.backend.codec, prompt=prompt_state, mode="encode", max_tokens=max_tokens)
        stream = str(payload_bits) + _padding(str(payload_bits))
        state, steps, finished, reason = self._loop(
            state,
            ctx,
            coder,
            rng,
            stream,
            None,
            max_steps,
            collect_trace=collect_trace,
        )
        emitted_bits = str(ctx.stats.pop("_emitted_bits", ""))
        # _loop stores emitted bits in stats to keep the state transition code
        # identical for encode and decode.
        text = self.backend.codec.bytes_to_text(self.method.output(state, ctx), strip_eos=True)
        payload = str(payload_bits)
        ctx.stats["payload_bits"] = len(payload)
        ctx.stats["decoded_bits"] = len(emitted_bits)
        compare_len = max(len(payload), len(emitted_bits))
        errors = sum(a != b for a, b in zip(payload, emitted_bits)) + abs(len(payload) - len(emitted_bits))
        ctx.stats["ber"] = float(errors / compare_len) if compare_len else 0.0
        ctx.stats["exact_message"] = emitted_bits.startswith(payload)
        tokens = max(int(getattr(state, "tokens_used", 0)), 1)
        ctx.stats["bpt"] = float(len(emitted_bits) / tokens)
        ctx.stats["bps"] = float(len(emitted_bits) / max(float(ctx.stats.get("total_ms", 0.0)) / 1000.0, 1e-12))
        self._record_terminal_metadata(state, ctx)
        self._merge_backend_stats(ctx)
        return EncodeResult(text, emitted_bits, int(getattr(state, "tokens_used", 0)), steps, finished, reason, dict(ctx.stats))

    def decode(
        self,
        prompt: str,
        visible_text: str,
        *,
        rng: Any,
        max_steps: int | None = None,
        max_tokens: int | None = None,
        collect_trace: bool = False,
    ) -> DecodeResult:
        reset = getattr(self.backend, "reset_stats", None)
        if callable(reset):
            reset()
        self._reset_peak_memory()
        prompt_state = self.backend.render_prompt(prompt)
        state = self.method.start(prompt_state)
        coder = self._coder_session("decode", rng)
        eos = getattr(self.backend.codec, "eos_bytes", b"")
        text_to_bytes = getattr(self.backend.codec, "text_to_bytes", None)
        visible_bytes = (
            text_to_bytes(str(visible_text))
            if callable(text_to_bytes)
            else str(visible_text).encode("utf-8")
        )
        target = VisibleTarget(visible_bytes + eos)
        ctx = MethodContext(codec=self.backend.codec, prompt=prompt_state, mode="decode", target=target, max_tokens=max_tokens)
        state, steps, finished, reason = self._loop(
            state,
            ctx,
            coder,
            rng,
            "",
            target,
            max_steps,
            collect_trace=collect_trace,
        )
        text = self.backend.codec.bytes_to_text(self.method.output(state, ctx), strip_eos=True)
        bits = str(ctx.stats.pop("_emitted_bits", ""))
        ctx.stats["payload_bits"] = None
        ctx.stats["decoded_bits"] = len(bits)
        ctx.stats["ber"] = None
        ctx.stats["exact_message"] = None
        tokens = max(int(getattr(state, "tokens_used", 0)), 1)
        ctx.stats["bpt"] = float(len(bits) / tokens)
        ctx.stats["bps"] = float(len(bits) / max(float(ctx.stats.get("total_ms", 0.0)) / 1000.0, 1e-12))
        self._record_terminal_metadata(state, ctx)
        self._merge_backend_stats(ctx)
        return DecodeResult(bits, text, steps, finished, reason, dict(ctx.stats))

    def _merge_backend_stats(self, ctx: MethodContext) -> None:
        counters = getattr(self.backend, "stats", None)
        if isinstance(counters, dict):
            for key, value in counters.items():
                ctx.stats[f"model_{key}"] = value
        metadata = getattr(self.backend, "device_metadata", None)
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                ctx.stats[f"model_{key}"] = value

    @staticmethod
    def _reset_peak_memory() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            # Memory accounting is observational; it must not change method
            # semantics when CUDA is unavailable or being torn down.
            return

    def _coder_session(self, mode: str, rng: Any):
        factory = self.coder() if isinstance(self.coder, type) else self.coder
        if not hasattr(factory, "start"):
            raise TypeError("v2 coder must expose start(mode, rng=...) -> CoderSession")
        return factory.start(mode, rng=rng)

    def _loop(
        self,
        state,
        ctx: MethodContext,
        coder,
        master_rng: Any,
        bit_stream: str,
        target: Optional[VisibleTarget],
        max_steps: int | None,
        *,
        collect_trace: bool,
    ):
        started = time.perf_counter()
        step = 0
        finished = False
        reason = "ongoing"
        all_bits: list[str] = []
        trace: list[dict[str, Any]] = []
        while not self.method.done(state):
            if max_steps is not None and step >= int(max_steps):
                reason = "max_steps"
                break
            if ctx.max_tokens is not None and int(getattr(state, "tokens_used", 0)) >= int(ctx.max_tokens):
                reason = "max_tokens"
                break
            ctx.step = step + 1
            query_started = time.perf_counter()
            paths = self.method.queries(state)
            distributions = [self.sampler(self.backend.next_logits(path)) for path in paths]
            ctx.stats["model_queries"] = ctx.stats.get("model_queries", 0) + len(paths)
            ctx.stats["model_forward_ms"] = ctx.stats.get("model_forward_ms", 0.0) + (time.perf_counter() - query_started) * 1000.0
            plan_started = time.perf_counter()
            plan = self.method.plan(state, distributions, ctx)
            ctx.stats["grouping_ms"] = ctx.stats.get("grouping_ms", 0.0) + (time.perf_counter() - plan_started) * 1000.0
            step_rng = _derive(master_rng, f"step:{ctx.step}:coder")
            coder_started = time.perf_counter()
            if ctx.mode == "encode":
                bits, action = coder.encode(plan.probs, bit_stream, rng=step_rng)
                bit_stream = bit_stream[len(bits) :]
            else:
                action = self.method.observe(state, plan, target, ctx)
                bits = coder.decode(plan.probs, action, rng=step_rng)
            ctx.stats["coder_ms"] = ctx.stats.get("coder_ms", 0.0) + (time.perf_counter() - coder_started) * 1000.0
            all_bits.append(str(bits))
            before = self.method.output(state, ctx) if collect_trace else b""
            state = self.method.advance(state, plan, int(action), _derive(master_rng, f"step:{ctx.step}:advance"), ctx)
            if collect_trace:
                trace.append(
                    self._trace_step(
                        ctx.step,
                        distributions,
                        plan,
                        int(action),
                        state,
                        before,
                        str(bits),
                        ctx,
                    )
                )
            step += 1
            if ctx.max_tokens is not None and int(getattr(state, "tokens_used", 0)) >= int(ctx.max_tokens):
                reason = "max_tokens"
                break
        if self.method.done(state):
            if target is not None and not target.done:
                raise ReceiverMismatchError(
                    "visible target contains trailing bytes after the method reached EOS"
                )
            finished, reason = True, "eos"
        ctx.stats["_emitted_bits"] = "".join(all_bits)
        if collect_trace:
            ctx.stats["trace"] = trace
        ctx.stats["steps"] = step
        if target is not None:
            ctx.stats["target_consumed"] = target.cursor
            ctx.stats["target_remaining"] = len(target.remaining)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        tokens = max(int(getattr(state, "tokens_used", 0)), 1)
        ctx.stats["total_ms"] = elapsed_ms
        ctx.stats["model_queries_per_token"] = float(ctx.stats.get("model_queries", 0)) / tokens
        try:
            import torch

            ctx.stats["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        except (ImportError, RuntimeError):
            ctx.stats["peak_gpu_memory_bytes"] = 0
        # Linux reports ru_maxrss in KiB; normalize to bytes for portability of
        # the experiment schema on this platform.
        ctx.stats["peak_cpu_memory_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        return state, step, finished, reason

    def _trace_step(
        self,
        step: int,
        distributions,
        plan,
        action: int,
        state,
        before: bytes,
        emitted_bits: str,
        ctx: MethodContext,
    ) -> dict[str, Any]:
        support_size = 0
        eos_probability = 0.0
        eos_rank = None
        if distributions:
            dist = distributions[0]
            probs = dist.probs
            support_ids, _ = dist.compact()
            support_size = int(support_ids.numel())
            eos_ids = sorted(
                int(token_id)
                for token_id in getattr(ctx.codec, "eos_ids", ())
                if 0 <= int(token_id) < int(probs.numel())
            )
            eos_probability = sum(float(probs[token_id].item()) for token_id in eos_ids)
            positive_eos = [token_id for token_id in eos_ids if float(probs[token_id].item()) > 0.0]
            if positive_eos:
                best = max(float(probs[token_id].item()) for token_id in positive_eos)
                eos_rank = 1 + int((probs > best).sum().item())
        after = self.method.output(state, ctx)
        delta = after[len(before) :] if after.startswith(before) else after
        try:
            resolved_text = ctx.codec.bytes_to_text(after, strip_eos=True)
            valid_utf8 = True
        except Exception:
            resolved_text = None
            valid_utf8 = False
        path = getattr(state, "path", getattr(state, "base_path", None))
        token_ids = path.token_ids() if path is not None and hasattr(path, "token_ids") else ()
        return {
            "step": int(step),
            "action": int(action),
            "action_probability": float(plan.probs[int(action)].item()),
            "action_count": int(plan.probs.numel()),
            "support_size": support_size,
            "eos_probability": float(eos_probability),
            "eos_rank": eos_rank,
            "selected_bytes_hex": delta.hex(),
            "resolved_bytes_hex": after.hex(),
            "resolved_text": resolved_text,
            "valid_utf8": valid_utf8,
            "emitted_bits": emitted_bits,
            "terminal": bool(self.method.done(state)),
            "active_token_id": int(token_ids[-1]) if token_ids else None,
        }

    @staticmethod
    def _record_terminal_metadata(state, ctx: MethodContext) -> None:
        path = getattr(state, "path", getattr(state, "base_path", None))
        token_ids = path.token_ids() if path is not None and hasattr(path, "token_ids") else ()
        terminal_token_id = int(token_ids[-1]) if token_ids and bool(getattr(state, "terminal", False)) else None
        ctx.stats["terminal_token_id"] = terminal_token_id
        ctx.stats["eos_token_ids"] = sorted(int(x) for x in getattr(ctx.codec, "eos_ids", ()))
        prompt_length = len(ctx.prompt.token_ids)
        generated_ids = tuple(int(x) for x in token_ids[prompt_length:])
        ctx.stats["latent_token_ids"] = list(generated_ids)
        try:
            latent_bytes = ctx.codec.tokens_to_bytes(generated_ids)
            ctx.stats["latent_text"] = ctx.codec.bytes_to_text(latent_bytes, strip_eos=True)
            ctx.stats["latent_text_error"] = None
        except Exception as exc:
            # Latent-path rendering is diagnostic only.  Preserve the exact
            # exception instead of changing or repairing method output.
            ctx.stats["latent_text"] = None
            ctx.stats["latent_text_error"] = f"{type(exc).__name__}: {exc}"


def _derive(rng: Any, scope: str):
    from .rng import derive_rng

    return derive_rng(rng, scope)
