"""Real-model validation harness. Frozen package/upstream code is never edited.

Sender and receiver are separate CLI invocations. Receiver reads public packets
and test-fixture keys only; the summary, not the receiver, opens sender traces.
This is an experimental adapter, not a new production protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import torch
import pysbd

from stegatext_runtime import Source
from stegatext_runtime.protocol import _state_digest
from perf_research_selective import make_selective_planner
from round4_2_common import CounterRNG
from stegtext.starter.adaptive import select_kernel
from stegtext.v2.core.types import DecisionPlan, MethodContext, VisibleTarget
from stegtext.v2.methods.registry import method_factories
from stegtext.v2.methods.direct import DirectMethod, exact_realization_plan
from stegtext.v2.methods.syncpool import SyncPoolMethod
from stegtext.v2.coders.registry import CODER_TYPES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs/full_validation_v1"
PROMPT = ("A local newspaper reports on a community library. Its article describes "
          "the opening, the books and the services available to residents.\n\n"
          "The new community library opened on Monday morning")
EOS_PROMPT = ("A short story for children, with a clear ending.\n\n"
              "Mia found a lost puppy in the park. She read the phone number on its collar "
              "and called its owner. Ten minutes later, the owner arrived and thanked her. "
              "The puppy wagged its tail as they walked home together.\n\nThe End.")
ZH_PROMPT = "社区新闻：新图书馆正式开放\n\n周一上午，新建的社区图书馆迎来了第一批读者。馆内设有儿童阅读区、自习室和电子阅览区。工作人员介绍"
METHODS = ["direct", "basic_upstream", "mwis_upstream", "stepwise_upstream",
           "syncpool_upstream", "consistency_local", "mwis_greedy_local", "syncpool",
           "las_local", "naive_las", "las", "fullsync", "adaptive"]
CODERS = ["discop", "imec", "range_ac", "huffman",
          "liao_differential", "liao_binary", "liao_stability", "discop_upstream"]
SCENARIOS = {
    "single": dict(prompt=PROMPT, cap=128, sentences=1),
    "standalone_single": dict(prompt="A local newspaper reports on a community library, its opening, books and services for residents.\n\nLocal News: Community Library Opens\n\n", cap=128, sentences=1),
    "multi": dict(prompt=PROMPT, cap=160, sentences=3),
    "multi_checked": dict(prompt=PROMPT, cap=192, sentences=3, sentence_rule="pysbd_with_continuation_sentinel"),
    "paragraphs": dict(prompt=PROMPT, cap=192, paragraphs=2),
    "budget": dict(prompt=PROMPT, cap=24),
    "long": dict(prompt=PROMPT, cap=512),
    "eos": dict(prompt=EOS_PROMPT, cap=256),
    "chinese": dict(prompt=ZH_PROMPT, cap=96, sentences=2),
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def log(**value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def fixture_key(model, case):
    return hashlib.sha256(f"full-validation-v1:{model}:{case}".encode()).digest()


def bits_for(key):
    return "".join(f"{b:08b}" for b in hashlib.shake_256(b"payload:" + key).digest(16384))


def shape(text):
    return {"sentences": len(re.findall(r'[.!?。！？][\"”’\']?(?=\s|$)', text)),
            "paragraphs": len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]),
            "characters": len(text), "replacement_characters": text.count("\ufffd")}


def goal(text, spec):
    d = shape(text)
    if spec.get("sentence_rule")=="pysbd_with_continuation_sentinel":
        # Appending a continuation reveals whether the final dot only closes
        # an abbreviation (e.g. Mr.) rather than a sentence. The sentinel is
        # used only by the stopping rule and is never sent to the LM.
        units=pysbd.Segmenter(language="en",clean=False).segment(text+" zxqcontinuation")
        completed=[u for u in units if "zxqcontinuation" not in u and re.search(r'[.!?。！？][\"”’\']?\s*$',u)]
        return len(completed)>=spec["sentences"] and bool(re.search(r'[.!?。！？][\"”’\']?\s*$',text))
    return (("sentences" in spec and d["sentences"] >= spec["sentences"] and
             bool(re.search(r'[.!?。！？][\"”’\']?\s*$', text))) or
            ("paragraphs" in spec and d["paragraphs"] >= spec["paragraphs"] and
             bool(re.search(r'[.!?。！？][\"”’\']?\s*$', text))))


def rendered(codec, state):
    data = state.resolved_bytes
    if state.terminal:
        if not data.endswith(codec.eos_bytes):
            raise ValueError("Terminal state lacks EOS bytes")
        data = data[:-len(codec.eos_bytes)]
    return codec.bytes_to_text(data)


def digest(state):
    if hasattr(state, "retained") and not isinstance(state.retained, tuple):
        return _state_digest(state)
    fields = {"terminal": state.terminal, "resolved": state.resolved_bytes.hex()}
    for name in ("path", "base_path"):
        if hasattr(state, name):
            fields[name] = getattr(state, name).token_ids()
    if hasattr(state, "retained"):
        fields["mass"] = state.expansion_mass
        fields["retained"] = [(r.path.token_ids(), r.residual.hex(), r.mass, r.terminal) for r in state.retained]
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def load_pools():
    sys.path.insert(0, str(ROOT / "external_baselines/Consistency/src"))
    from tokenization_consistency.pools import build_pool
    return build_pool


class PoolMethod(DirectMethod):
    """Unmodified upstream pool construction, explicit byte transport adapter."""
    def __init__(self, source, strategy):
        self.source, self.strategy = source, strategy
        self.build_pool = load_pools()
        self.grouped = strategy == "syncpool"

    def plan(self, state, distributions, ctx):
        dist = distributions[0]
        positive = int((dist.probs > 0).sum())
        pool = self.build_pool(self.strategy, self.source.tokenizer, list(state.path.token_ids()),
                               dist.probs, top_k=positive)
        # Expose zero-mass fallback as a failure, never silently invent a source.
        if not len(pool) or float(pool.probs.sum()) <= 0:
            raise ValueError("Upstream pool has no positive retained probability")
        ctx.stats["coverage"] = pool.coverage
        if pool.groups is None:
            return exact_realization_plan(pool.token_ids, dist, ctx)
        members = [g.tolist() for g in pool.groups]
        # In the upstream string view EOS decodes to ''. An EOS-connected
        # single pool is identifiable without inspecting a visible prefix.
        keys = [min((ctx.codec.token_bytes(t) for t in g), key=len) for g in members]
        if len(members) == 1:
            keys = [b""]
        return DecisionPlan(pool.normalized(), dict(keys=keys, members=members, dist=dist))

    def observe(self, state, plan, target, ctx):
        if self.grouped:
            return SyncPoolMethod.observe(self, state, plan, target, ctx)
        return super().observe(state, plan, target, ctx)

    def advance(self, state, plan, action, rng, ctx):
        if self.grouped:
            return SyncPoolMethod.advance(self, state, plan, action, rng, ctx)
        return super().advance(state, plan, action, rng, ctx)


class Huffman:
    """Explicit textbook reconstruction; dyadic output generally changes law."""
    def codes(self, probs):
        queue = [(float(p), i, {i: ""}) for i, p in enumerate(probs.tolist()) if p > 0]
        heapq.heapify(queue)
        serial = len(probs)
        while len(queue) > 1:
            a, b = heapq.heappop(queue), heapq.heappop(queue)
            codes = {i: "0" + c for i, c in a[2].items()}
            codes.update({i: "1" + c for i, c in b[2].items()})
            heapq.heappush(queue, (a[0]+b[0], serial, codes))
            serial += 1
        return queue[0][2]

    def encode(self, probs, stream, *, rng):
        for action, code in self.codes(probs).items():
            if stream.startswith(code):
                return code, action
        raise ValueError("Insufficient Huffman bits")

    def decode(self, probs, action, *, rng):
        return self.codes(probs)[action]


def coder(name, mode):
    if name in CODER_TYPES:
        return CODER_TYPES[name]().start(mode)
    if name == "huffman":
        return Huffman()
    from research_notes.full_validation.upstream import upstream_coder
    return upstream_coder(name, mode)


def method(source, name):
    if name in ("las", "fullsync", "adaptive"):
        return make_selective_planner(source.backend.codec_index)
    if name.endswith("_upstream"):
        strategy = {"stepwise": "consistency"}.get(name[:-9], name[:-9])
        return PoolMethod(source, strategy)
    local = {"consistency_local": "consistency", "mwis_greedy_local": "mwis", "las_local": "las"}.get(name, name)
    return method_factories(source.backend)[local]()


@torch.inference_mode()
def run(source, packet, *, key, stream=None):
    sending = stream is not None
    codec = source.backend.codec
    prompt = source.bind(packet["prompt"])
    m = method(source, packet["method"])
    state = m.start(prompt)
    target = None if sending else VisibleTarget(codec.text_to_bytes(packet["text"]) +
                                              (codec.eos_bytes if packet["natural_eos"] else b""))
    ctx = MethodContext(codec, prompt, "encode" if sending else "decode", target=target)
    c = coder(packet["coder"], ctx.mode)
    spec = packet["scenario_spec"]
    trace, pieces, coverages = [], [], []
    offset = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    fwd_start = source.reference.forwards
    stop = "cap"
    cap = spec["cap"]
    for step in range(1, cap+33):
        if state.terminal or (target is not None and target.done):
            stop = "eos" if state.terminal else "packet_boundary"
            break
        ctx.step = step
        paths = ([state.base_path] if state.expansion_mass > 0 else []) if packet["method"] in ("las", "fullsync", "adaptive") else m.queries(state)
        ds = [source.distribution(p) for p in paths]
        plan = m.plan(state, ds, ctx)
        opaque = plan.opaque
        if isinstance(opaque, dict) and "members" in opaque and ds:
            support = sorted({int(t) for g in opaque["members"] for t in g})
            coverage = float(ds[0].probs[support].sum())
        else:
            coverage = 1.0
        coverages.append(coverage)
        if sending:
            out, action = c.encode(plan.probs.cpu(), stream[offset:], rng=CounterRNG(key, "coder", step))
            if out != stream[offset:offset+len(out)]:
                raise ValueError("Coder output differs from supplied payload prefix")
        else:
            action = m.observe(state, plan, target, ctx)
            out = c.decode(plan.probs.cpu(), action, rng=CounterRNG(key, "coder", step))
        advance = m.advance
        gate = None
        if packet["method"] in ("las", "fullsync", "adaptive"):
            tau = {"las": 0.0, "fullsync": math.inf, "adaptive": .5}[packet["method"]]
            gate = select_kernel(opaque["records"][action]["las_posterior_entropy_bits"], tau)
            advance = m.advance if gate == "las" else m.advance_fullsync
        state = advance(state, plan, action, CounterRNG(key, "advance", step), ctx)
        pieces.append(out)
        offset += len(out)
        trace.append(dict(step=step, action=int(action), bits=offset, kernel=gate,
                          coverage=coverage, retained=len(getattr(state, "retained", ())),
                          resolved_bytes=len(state.resolved_bytes)))
        if sending:
            try:
                text = rendered(codec, state)
            except Exception:
                if step <= cap+31:
                    continue
                raise
            enough = offset >= spec.get("min_bits", 0)
            if state.terminal or (enough and goal(text, spec)) or step >= cap:
                stop = "eos" if state.terminal else "text_boundary" if goal(text, spec) and enough else "budget_utf8_boundary"
                break
    if target is not None and not target.done:
        raise ValueError(f"Unconsumed public bytes: {len(target.remaining)}")
    text = rendered(codec, state)
    return dict(text=text, bits="".join(pieces), bits_count=offset, steps=len(trace),
                natural_eos=state.terminal, state_sha256=digest(state),
                receiver_consumed=target.done if target is not None else None,
                stop_reason=stop, shape=shape(text), scenario_goal=goal(text, spec),
                mean_coverage=sum(coverages)/len(coverages) if coverages else 1.0,
                min_coverage=min(coverages, default=1.0), trace=trace,
                wall_seconds=time.monotonic()-started,
                model_forwards=source.reference.forwards-fwd_start,
                peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30, pid=os.getpid())


def cases():
    for m in METHODS:
        for scene in ("single", "standalone_single", "multi", "multi_checked", "paragraphs", "budget"):
            yield dict(case=f"method__{m}__{scene}", method=m, coder="discop", scene=scene)
    for c in CODERS:
        if c != "discop":
            for scene in ("single", "standalone_single", "multi", "multi_checked", "budget"):
                yield dict(case=f"coder__{c}__{scene}", method="syncpool", coder=c, scene=scene)
    for m in ("syncpool", "las", "fullsync", "adaptive"):
        for scene in ("long", "eos", "chinese"):
            yield dict(case=f"extended__{m}__{scene}", method=m, coder="discop", scene=scene)


def stage(model, which, pattern=""):
    source = Source.load(model, "stego")
    from research_notes.full_validation.cache import accelerate
    accelerate(source)
    snapshots=[]
    def capture(ids,probabilities):
        if len(snapshots)>=32: return
        p=probabilities.detach().double().cpu()
        support=torch.nonzero(p>0).flatten()
        snapshots.append({"history_length":len(ids),"token_ids":support.tolist(),"probabilities":p[support].tolist()})
    source.query_log=capture
    directory = OUT/model
    records = []
    for item in cases():
        name = item["case"]
        if pattern and pattern not in name:
            continue
        destination = directory/("private" if which == "sender" else "receiver")/(name+".json")
        if destination.exists():
            records.append(json.loads(destination.read_text()))
            continue
        key = fixture_key(model, name)
        try:
            if which == "sender":
                spec = SCENARIOS[item["scene"]]
                packet = {**item, "version": "experimental-full-validation-v1", "source": source.manifest(),
                          "prompt": spec["prompt"], "scenario_spec": {k: v for k, v in spec.items() if k != "prompt"}}
                result = run(source, packet, key=key, stream=bits_for(key))
                packet.update(text=result["text"], natural_eos=result["natural_eos"])
                write(directory/"public"/(name+".json"), packet)
            else:
                public = directory/"public"/(name+".json")
                if not public.exists():
                    continue
                packet = json.loads(public.read_text())
                if packet["source"] != source.manifest():
                    raise ValueError("Source contract mismatch")
                result = run(source, packet, key=key)
            result = {**item, "status": "ok", **result}
        except Exception as exc:
            result = {**item, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc(), "pid": os.getpid()}
        write(destination, result)
        records.append(result)
        log(stage=which, model=model, **{k:v for k,v in result.items() if k in ("case", "status", "error", "steps", "bits_count", "wall_seconds", "natural_eos", "scenario_goal")})
    write(HERE/f"{model}_{which}_index.json", {"model":model,"stage":which,
        "kv_snapshot_restorations":source.reference.snapshot_restorations,
        "kv_snapshot_oracle_exact_checks":source.reference.snapshot_oracle_checks,
        "records":[{k:r[k] for k in ("case","status","error") if k in r} for r in records]})
    if snapshots:
        snapshot_path=OUT/model/f"{which}_probability_snapshots.json"
        if snapshot_path.exists(): snapshot_path=OUT/model/f"{which}_probability_snapshots_{os.getpid()}.json"
        write(snapshot_path,snapshots)


def summary():
    rows=[]
    for model in ("qwen", "llama"):
        for item in cases():
            name=item["case"]
            paths=[OUT/model/p/(name+".json") for p in ("private","receiver")]
            s,r=[json.loads(p.read_text()) if p.exists() else {} for p in paths]
            ok=s.get("status")==r.get("status")=="ok" and all(s[k]==r[k] for k in ("bits","text","state_sha256","trace")) and r["receiver_consumed"] and s["pid"]!=r["pid"]
            row={"model":model,**item,"roundtrip":bool(ok),"sender_status":s.get("status","missing"),"receiver_status":r.get("status","missing"),
                 **{k:s[k] for k in ("steps","bits_count","shape","natural_eos","scenario_goal","stop_reason","mean_coverage","min_coverage","peak_cuda_gib","wall_seconds") if k in s}}
            if not ok:
                row["error"]=r.get("error",s.get("error","bit/text/state mismatch or missing result"))
            rows.append(row)
    write(HERE/"results.json",{"rows":rows,"attempts":len(rows),"exact_roundtrips":sum(r["roundtrip"] for r in rows)})
    log(attempts=len(rows),exact_roundtrips=sum(r["roundtrip"] for r in rows))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=("qwen","llama"))
    p.add_argument("--stage",choices=("sender","receiver","summary"),required=True)
    p.add_argument("--filter",default="")
    a=p.parse_args()
    if a.stage=="summary": summary()
    else: stage(a.model,a.stage,a.filter)
