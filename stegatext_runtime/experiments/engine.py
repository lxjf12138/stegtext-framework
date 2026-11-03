"""One configurable planner loop; original mathematical kernels are reused."""
from __future__ import annotations

import hashlib
import inspect
import math
import os
from pathlib import Path
import time
import types

import torch

from .config import ROOT
from . import transport
from stegatext_runtime.discop import DisCopSession
from research_notes.full_validation import suite, systems
from research_notes.problem_resolution import bounded_systems
from research_notes.problem_resolution.paper_adapters import ShapeSafeStepwise, SingletonSafeLiao
from research_notes.problem_resolution.text_rules import complete_sentences, sentence_end
from round4_2_common import CounterRNG
from stegtext.starter.adaptive import select_kernel
from stegtext.v2.core.types import MethodContext, VisibleTarget
from stegtext.v2.core.errors import TokenizerContractError


def method(source, name):
    if name in ('basic_bytes_v1', 'mwis_bytes_v1', 'stepwise_chat_v1'):
        from .baseline_adapters import BytePrefixBaseline, ChatStepwise
        return ChatStepwise(source) if name == 'stepwise_chat_v1' else BytePrefixBaseline(name.split('_')[0])
    return ShapeSafeStepwise(source) if name == "stepwise_reviewed" else suite.method(source, name)


def coder(name, mode, options=None):
    if name in ('imec_window_exact', 'imec_prefix_exact'):
        from stegatext_runtime.imec import IMECSession
        return IMECSession(mode=mode, drain_all=name == 'imec_prefix_exact', **(options or {}))
    if name == "discop_copy": return DisCopSession()
    if name == "liao_stability_reviewed": return SingletonSafeLiao()
    return suite.coder("discop" if name == "discop_legacy" else name, mode)


def contract(variant, stopping):
    if "mode" in stopping:
        from .stopping import contract as v2_contract
        return v2_contract(variant, stopping)
    kind = variant["kind"]
    if kind == "upstream":
        return dict(kind="upstream-native-v1", system=variant["system"],
                    stopping="Original full_validation system protocol; see native_parameters", unit="native_protocol",
                    native_parameters={"sparsamp": "minimum 48 tokens, block 16", "anstega": "64 tokens, P32, t16, arbitrary initial state",
                        "anstega_normalized": "64 tokens, P32, t16, known leading 1", "imec_official": "64 message bits, block 4",
                        "meteor_borrowed": "64 message bits, P32", "rrc_verified": "64 message bits, verify, max 256 tokens",
                        "rrc_paper_stop": "64 message bits, paper stop, max 256 tokens", "arimec": "16 bits, max 64 tokens",
                        "arimec_merge": "16 bits, max 64 tokens"}[variant["system"]], query_watchdog=512)
    if kind == "carrier":
        return dict(kind="bounded-config-v1", system=variant["system"], unit="generated_tokens",
                    max_steps=stopping["max_steps"], utf8_extension=32, natural_eos=True,
                    seed_schedule="101 + experiment seed", terminal_state="public for ANS", scope="one carrier, confirmed prefix")
    return dict(kind="ordinary-v1" if kind == "ordinary" else "planner-v1",
                unit="generated_tokens" if kind == "ordinary" else "protocol_decisions", **stopping,
                utf8_completion_applies=kind == "planner", natural_eos=True,
                sentence_rule="substantive_pysbd_en_cjk_zh_v2" if "sentences" in stopping else None)


def goal(text, stop, language, bits):
    if bits < stop.get("min_bits", 0): return False
    if "sentences" in stop:
        return len(complete_sentences(text, language)) >= stop["sentences"] and sentence_end(text, language)
    if "paragraphs" in stop:
        return suite.shape(text)["paragraphs"] >= stop["paragraphs"] and sentence_end(text, language)
    return False


@torch.inference_mode()
def planner(source, packet, *, key, stream=None):
    if "mode" in packet["stopping"]:
        from .engine_v2 import planner as planner_v2
        return planner_v2(source, packet, key=key, stream=stream)
    sending = stream is not None
    if not source.config.do_sample: raise ValueError("Planner requires sampling, not a zero-capacity greedy source")
    variant, stop = packet["variant"], packet["stopping"]
    codec = source.backend.codec
    prompt = source.bind(packet["prompt"])
    m = method(source, variant["method"])
    state = m.start(prompt)
    target = None if sending else VisibleTarget(codec.text_to_bytes(packet["text"]) + (codec.eos_bytes if packet["natural_eos"] else b""))
    ctx = MethodContext(codec, prompt, "encode" if sending else "decode", target=target)
    c = coder(variant["coder"], ctx.mode)
    selective = variant["method"] in ("las", "fullsync", "adaptive")
    tau = (0. if variant["method"] == "las" else math.inf if variant["method"] == "fullsync" else variant.get("tau"))
    pieces, trace = [], []
    offset, reason = 0, "budget_utf8_boundary"
    for step in range(1, stop["max_steps"]+stop["utf8_extension"]+1):
        if state.terminal or (target is not None and target.done):
            reason = "eos" if state.terminal else "packet_boundary"
            break
        ctx.step = step
        paths = ([state.base_path] if state.expansion_mass > 0 else []) if selective else m.queries(state)
        distributions = [source.distribution(path) for path in paths]
        plan = m.plan(state, distributions, ctx)
        coverage = 1.
        if isinstance(plan.opaque, dict) and "members" in plan.opaque and distributions:
            retained = sorted({int(t) for group in plan.opaque["members"] for t in group})
            coverage = float(distributions[0].probs[retained].sum())
        if sending:
            out, action = c.encode(plan.probs.cpu(), stream[offset:], rng=CounterRNG(key, "coder", step))
            if out != stream[offset:offset+len(out)]: raise ValueError("Encoder did not consume the supplied prefix")
        else:
            action = m.observe(state, plan, target, ctx)
            out = c.decode(plan.probs.cpu(), action, rng=CounterRNG(key, "coder", step))
        kernel = None
        advance = m.advance
        if selective:
            kernel = select_kernel(plan.opaque["records"][action]["las_posterior_entropy_bits"], tau)
            advance = m.advance if kernel == "las" else m.advance_fullsync
        state = advance(state, plan, action, CounterRNG(key, "advance", step), ctx)
        if selective:
            mass = state.expansion_mass + math.fsum(state.retained.masses)
            if not ((state.terminal and mass == 0) or abs(mass-1) < 2e-6):
                raise ValueError("Invalid posterior mass")
        offset += len(out); pieces.append(out)
        trace.append(dict(step=step, action=int(action), bits=offset, kernel=kernel, coverage=coverage,
                          retained=len(getattr(state, "retained", ())), resolved_bytes=len(state.resolved_bytes)))
        if sending:
            try:
                text = suite.rendered(codec, state)
            except TokenizerContractError:
                continue
            if state.terminal or goal(text, stop, packet["language"], offset) or step >= stop["max_steps"]:
                reason = "eos" if state.terminal else "text_boundary" if goal(text, stop, packet["language"], offset) else "budget_utf8_boundary"
                break
    if target is not None and not target.done: raise ValueError("Receiver left unconsumed public bytes")
    text = suite.rendered(codec, state)
    from stegatext_runtime.native_tokens import record_state
    return dict(text=text, bits="".join(pieces), bits_count=offset, natural_eos=state.terminal,
                native_token_sequence=record_state(source, prompt, state, natural_eos=state.terminal),
                state_sha256=suite.digest(state), trace=trace, steps=len(trace), stop_reason=reason,
                receiver_consumed=None if sending else target.done,
                mean_coverage=sum(r["coverage"] for r in trace)/len(trace) if trace else 1.,
                min_coverage=min((r["coverage"] for r in trace), default=1.))


def key_for(config, seed):
    spec = config["randomness"]
    if spec["mode"] == "fixture":
        return hashlib.sha256(f"stegatext-experiment-fixture-key:v1:{seed}".encode()).digest()
    key = bytes.fromhex(Path(spec["file"]).read_text().strip())
    if hashlib.sha256(key).hexdigest() != spec["sha256"]: raise ValueError("Key file changed after resolving configuration")
    # Independent experiment seeds are separate shared-randomness domains.
    import hmac
    return hmac.new(key, b"stegatext-experiment-seed:v1:"+seed.to_bytes(8, "big"), hashlib.sha256).digest()


def message_for(config, seed, *, context=None):
    spec = config["message"]
    from .randomness import independent_cases
    if independent_cases(config):
        if not context:
            raise ValueError('Independent payload needs the complete case context')
        if spec.get('mode') != 'fixture' or spec.get('pattern') != 'random':
            raise ValueError('Independent payload requires a uniform random message fixture')
        import hmac
        payload_key = hmac.new(key_for(config, seed),
            b'stegatext-experimental-payload-v1:'+context.encode(), hashlib.sha256).digest()
        block = hashlib.shake_256(payload_key).digest((spec['bits']+7)//8)
        return ''.join(f'{b:08b}' for b in block)[:spec['bits']]
    if spec["mode"] == "file":
        bits = Path(spec["file"]).read_text().strip()
        if hashlib.sha256(bits.encode()).hexdigest() != spec["sha256"]: raise ValueError("Message file changed after resolving configuration")
        return bits
    n = spec["bits"]
    if spec["pattern"] != "random": return ("0" if spec["pattern"] == "zeros" else "1")*n
    block = hashlib.shake_256(f"stegatext-experiment-message:v1:{seed}".encode()).digest((n+7)//8)
    return "".join(f"{b:08b}" for b in block)[:n]


def clone(function, **bindings):
    function = inspect.unwrap(function)
    return types.FunctionType(function.__code__, dict(function.__globals__, **bindings), function.__name__, function.__defaults__, function.__closure__)


def send(source, case, config):
    if config["schema_version"] == 2:
        from .engine_v2 import send as send_v2
        return send_v2(source, case, config)
    variant = case["variant"]
    base = dict(version="unified-experiment-v1", source=source.manifest(), prompt=case["prompt"]["text"],
                language=case["prompt"]["language"], seed=case["seed"], variant=variant,
                stopping=config["stopping"], engine_contract=contract(variant, config["stopping"]))
    if variant["kind"] == "ordinary":
        from .randomness import generate
        return None, generate(source, case, config, max_new_tokens=config["stopping"]["max_steps"])
    key, stream = key_for(config, case["seed"]), message_for(config, case["seed"])
    if variant["kind"] == "planner":
        result = planner(source, base, key=key, stream=stream)
        return dict(base, text=result["text"], natural_eos=result["natural_eos"]), result
    if variant["kind"] == "upstream":
        original = clone(systems.send, bits_for=lambda _: stream)
        packet, result = original(source, dict(system=variant["system"], prompt=base["prompt"], seed=case["seed"]), key)
        if variant["system"] == "anstega_normalized":
            if not result["bits"].startswith("1"): raise ValueError("Missing known ANS initial bit")
            result.update(bits=result["bits"][1:], bits_count=result["bits_count"]-1)
        public = dict(base, system=packet["system"], auxiliary=packet["auxiliary"], text=packet["text"], natural_eos=packet["natural_eos"])
        return transport.attach(source, public, result), result
    bound = lambda src, prompt: bounded_systems.BoundedSource(src, prompt, token_budget=config["stopping"]["max_steps"])
    original = clone(bounded_systems.send_carrier, BoundedSource=bound)
    packet, result = original(source, variant["system"], base["prompt"], stream, key, case["seed"])
    packet["auxiliary"]["token_budget"] = config["stopping"]["max_steps"]
    public = dict(base, system=packet["system"], auxiliary=packet["auxiliary"], text=packet["text"],
                  natural_eos=packet["natural_eos"], confirmed_bits=packet["confirmed_bits"], stop_reason=packet["stop_reason"])
    return transport.attach(source, public, result), dict(result, text=packet["text"], natural_eos=packet["natural_eos"])


def receive(source, packet, config):
    if packet["version"] == "unified-experiment-v2":
        from .engine_v2 import receive as receive_v2
        return receive_v2(source, packet, config)
    if "mode" in packet["stopping"]:
        raise ValueError("v2 stopping requires a v2 packet version")
    if packet["version"] != "unified-experiment-v1" or packet["source"] != source.manifest():
        raise ValueError("Source or packet version differs from the receiver contract")
    if packet["engine_contract"] != contract(packet["variant"], packet["stopping"]): raise ValueError("Engine contract mismatch")
    key = key_for(config, packet["seed"])
    kind = packet["variant"]["kind"]
    if kind == "planner": return planner(source, packet, key=key)
    if kind == "upstream":
        result = systems.receive(source, packet, transport.text_tokens(source, packet), key)
        if packet["system"] == "anstega_normalized":
            if not result["bits"].startswith("1"): raise ValueError("Missing known ANS initial bit")
            result.update(bits=result["bits"][1:], bits_count=result["bits_count"]-1)
        return result
    if kind == "carrier": return clone(bounded_systems.receive_carrier, text_tokens=transport.text_tokens)(source, packet, key)
    raise ValueError("Ordinary text has no secret receiver")
