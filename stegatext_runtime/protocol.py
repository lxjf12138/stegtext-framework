"""LAS/FullSync/Adaptive transport sharing Source with ordinary inference.

Returns the recovered/consumed bit prefix. No claim of finite-message framing.
The receiver reads public text, source contract, protocol and a shared key only.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import torch

from perf_research_selective import make_selective_planner
from round4_2_common import CounterRNG
from stegtext.starter.adaptive import select_kernel
from stegtext.v2.coders.discop import _DisCopSession
from stegtext.v2.core.errors import TokenizerContractError
from stegtext.v2.core.types import MethodContext, VisibleTarget

from .discop import CODER_ID, DisCopSession


def _state_digest(state):
    value = {
        "base": state.base_path.token_ids(), "resolved": state.resolved_bytes.hex(),
        "mass": state.expansion_mass, "terminal": state.terminal,
        "retained": [(r.path.token_ids(), r.residual.hex(), r.mass, r.terminal) for r in state.retained],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _text(codec, state):
    wire = state.resolved_bytes
    if state.terminal:
        if not wire.endswith(codec.eos_bytes):
            raise ValueError("Missing EOS realization")
        wire = wire[:-len(codec.eos_bytes)]
    return codec.bytes_to_text(wire)


def encode(source, prompt, stream, *, key, policy="las", tau=None, max_decisions=64):
    if not stream or any(bit not in "01" for bit in stream):
        raise ValueError("stream must contain supplied 0/1 bits")
    if not source.config.do_sample:
        raise ValueError("Checkpoint is greedy and has zero payload capacity; select explicit stego profile")
    if policy not in ("las", "fullsync", "adaptive"):
        raise ValueError("Unknown policy")
    threshold = 0. if policy == "las" else math.inf if policy == "fullsync" else float(tau)
    if math.isnan(threshold) or threshold < 0 or max_decisions <= 0:
        raise ValueError("Invalid threshold or decision budget")
    packet = {"version": 2, "coder": CODER_ID, "source": source.manifest(), "prompt": prompt,
        "policy": policy, "tau": "infinity" if math.isinf(threshold) else threshold,
        "stopping": {"kind": "decision_budget_then_utf8_boundary", "max_decisions": max_decisions,
                     "max_utf8_completion_decisions": 32}}
    result = _run(source, packet, key=key, stream=stream)
    packet.update(text=result["text"], natural_eos=result["natural_eos"])
    return packet, result


def decode(source, packet, *, key):
    if packet["version"] not in (1, 2) or source.manifest() != packet["source"]:
        raise ValueError("Public source contract does not match local source")
    if packet["version"] == 2 and packet.get("coder") != CODER_ID:
        raise ValueError("Unsupported public coder contract")
    return _run(source, packet, key=key, stream=None)


@torch.inference_mode()
def _run(source, packet, *, key, stream):
    if len(key) < 16:
        raise ValueError("Supply at least 16 bytes of shared key")
    sending = stream is not None
    backend = source.backend
    codec = backend.codec
    prompt = source.bind(packet["prompt"])
    planner = make_selective_planner(backend.codec_index)
    state = planner.start(prompt)
    wire = None if sending else codec.text_to_bytes(packet["text"]) + (codec.eos_bytes if packet["natural_eos"] else b"")
    target = None if sending else VisibleTarget(wire)
    ctx = MethodContext(codec, prompt, "encode" if sending else "decode", target=target)
    # Version 1 retains its historical literal-direction decoding convention.
    coder = DisCopSession() if packet["version"] == 2 else _DisCopSession()
    tau = math.inf if packet["tau"] == "infinity" else float(packet["tau"])
    limit = int(packet["stopping"]["max_decisions"])
    cap = limit + int(packet["stopping"]["max_utf8_completion_decisions"])
    emitted = []
    offset = 0
    trace = []
    gates = Counter()
    reason = "decision_cap"
    for step in range(1, cap+1):
        if state.terminal or (not sending and target.done):
            reason = "eos" if state.terminal else "packet_boundary"
            break
        distributions = [source.distribution(state.base_path)] if state.expansion_mass > 0 else []
        plan = planner.plan(state, distributions, ctx)
        if sending:
            out, action = coder.encode(plan.probs, stream[offset:], rng=CounterRNG(key, "coder", step))
            if offset+len(out) > len(stream) or stream[offset:offset+len(out)] != out:
                raise ValueError("Insufficient supplied bits; implicit zero padding is disabled")
        else:
            action = planner.observe(state, plan, target, ctx)
            out = coder.decode(plan.probs, action, rng=CounterRNG(key, "coder", step))
        record = plan.opaque["records"][action]
        gate = select_kernel(record["las_posterior_entropy_bits"], tau)
        state = getattr(planner, "advance" if gate == "las" else "advance_fullsync")(
            state, plan, action, CounterRNG(key, "advance", step), ctx)
        mass = state.expansion_mass+math.fsum(state.retained.masses)
        if not ((state.terminal and mass == 0) or abs(mass-1) < 2e-6):
            raise ValueError("Invalid posterior mass")
        emitted.append(out)
        offset += len(out)
        gates[gate] += 1
        trace.append({"step": step, "action": int(action), "kernel": gate,
            "retained": len(state.retained), "bits": offset,
            "u_post_bits": record["las_posterior_entropy_bits"]})
        if sending and (state.terminal or step >= limit):
            try:
                _text(codec, state)
            except TokenizerContractError:
                continue
            reason = "eos" if state.terminal else "decision_budget_utf8_boundary"
            break
    if not sending and not target.done:
        raise ValueError("Receiver did not consume the complete public packet")
    text = _text(codec, state)
    from .native_tokens import record_state
    return {"text": text, "bits": "".join(emitted), "bits_count": offset,
        "native_token_sequence": record_state(source, prompt, state, natural_eos=state.terminal),
        "steps": len(trace), "natural_eos": state.terminal, "stop_reason": reason,
        "receiver_consumed": target.done if target is not None else None,
        "state_sha256": _state_digest(state), "trace": trace,
        "kernel_counts": dict(gates), "source": source.manifest()}
