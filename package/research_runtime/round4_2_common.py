"""Frozen Round-4.1 adapters and fair experiment utilities for Round-4.2."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch

from round3_common import REPO_ROOT, ROUND3_DIR, DEFAULT_V2_REPO, MODEL_SPECS, add_v2_to_path, canonical_hash, source_manifest, sha256_file, model_file_manifest, load_prompts

OUT = REPO_ROOT / "engineering_survey" / "round4_2"
GRID = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 0)
MILESTONES = (64, 128, 256, 512)
DECISIONS = (16, 32, 64)
BASELINE = ("round4_1_common.py", "round4_common.py", "round3_sequential.py")


def label(k):
    return "full_softmax" if int(k) == 0 else f"topk{int(k)}"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze(output=OUT):
    add_v2_to_path()
    reference = json.loads((ROUND3_DIR / "v2_version_freeze.json").read_text())
    actual = canonical_hash(source_manifest(DEFAULT_V2_REPO / "src/stegtext/v2"))
    if actual != reference["v2_source_tree_hash_sha256"]:
        raise RuntimeError("external v2 source changed after Round-3 freeze")
    current = {name: sha256_file(REPO_ROOT / "scripts" / name) for name in BASELINE}
    dest = output / "config.json"
    if dest.exists():
        prior = json.loads(dest.read_text())
        if prior["baseline_scripts_sha256"] != current:
            raise RuntimeError("Round-4.1 baseline source changed during k-sweep")
        for model,spec in MODEL_SPECS.items():
            if prior["models"][model]["revision_files"] != model_file_manifest(spec["path"]):
                raise RuntimeError(f"model/tokenizer configuration changed: {model}")
        return prior
    config = {
        "round": "4.2", "temperature": 1.0, "k_grid": list(GRID),
        "full_softmax_encoding": 0, "visible_byte_milestones": list(MILESTONES),
        "decision_diagnostics": list(DECISIONS), "initial_paired_cases": 12,
        "expanded_paired_cases": 32, "bootstrap_replicates": 20000,
        "entropy_tolerance_float64": 1e-10,
        "visible_progress": "unescaped_receiver_data_bytes_excluding_prompt_and_eos",
        "baseline_scripts_sha256": current, "frozen_v2_tree_sha256": actual,
        "frozen_v2_parent_commit": reference["target"]["base_commit"],
        "randomness": "HMAC(case_key, domain, decision_index, draw_index); no k or method",
        "case_key_inputs": ["model", "prompt_id", "base_seed", "case_index"],
        "sampler": "frozen v2 TopK/FullSoftmax; bf16 head promoted to fp32 in sequential; float64 in structural audit",
        "phase_ab_state_semantics": "single_history conditional laws at saved natural or snapshot base histories; no retained prior",
        "models": {m: {"path": str(s["path"]), "revision_files": model_file_manifest(s["path"])} for m,s in MODEL_SPECS.items()},
    }
    write_json(dest, config)
    return config


class CounterRNG:
    """Each decision gets a fresh domain-separated counter stream."""
    def __init__(self, key: bytes, domain: str, decision: int):
        self.key = hmac.new(key, domain.encode() + b":" + struct.pack(">Q", decision), hashlib.sha256).digest()
        self.counter = 0
    def random(self):
        self.counter += 1
        value = hmac.new(self.key, struct.pack(">Q", self.counter), hashlib.sha256).digest()
        return (int.from_bytes(value[:8], "big") & ((1 << 53)-1)) / float(1 << 53)


def case_specs(model, count):
    prompts = load_prompts()
    strata = sorted({(p["language"], p["domain"]) for p in prompts})
    grouped = {s: [p for p in prompts if (p["language"], p["domain"]) == s] for s in strata}
    specs = []
    for i in range(count):
        s = strata[i % len(strata)]
        within = i // len(strata)
        p = grouped[s][within % len(grouped[s])]
        seed = 42000 + within // len(grouped[s])
        key = hashlib.sha256(f"round4.2:{model}:{p['prompt_id']}:{seed}:{i}".encode()).digest()
        rng = CounterRNG(key, "payload", 0)
        payload = "".join("1" if rng.random() >= 0.5 else "0" for _ in range(65536))
        specs.append({**p, "model": model, "case_index": i, "base_seed": seed,
                      "case_id": key.hex()[:24], "case_key_hex": key.hex(),
                      "payload": payload, "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                      "prompt_sha256": hashlib.sha256(p["prompt"].encode()).hexdigest()})
    return specs


def sampler(k):
    from stegtext.v2.sampling.policies import TopK, FullSoftmax
    return TopK(int(k), 1.0) if k else FullSoftmax(1.0)


def entropy(values):
    p = np.asarray(values, dtype=np.float64)
    p = p[p > 0]
    return float(-np.dot(p, np.log2(p)))


def visible_count(resolved: bytes, eos: bytes):
    data = resolved[:-len(eos)] if eos and resolved.endswith(eos) else resolved
    # All data token realizations and group boundaries are escaped byte pairs.
    if len(data) % 2 or any(data[i] != 1 for i in range(0, len(data), 2)):
        raise AssertionError("visible progress ended inside an escaped byte")
    return len(data) // 2


def make_codec(model):
    from transformers import AutoTokenizer
    from stegtext.v2.model.tokenizer_codec import TokenizerCodec
    from stegtext.v2.model.vocab_index import TokenByteIndex
    path = MODEL_SPECS[model]["path"]
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=MODEL_SPECS[model]["trust_remote_code"])
    config = json.loads((path / "config.json").read_text())
    gen = json.loads((path / "generation_config.json").read_text())
    eos = set()
    for source in (config, gen, {"eos_token_id": tokenizer.eos_token_id}):
        v = source.get("eos_token_id")
        if v is not None:
            eos.update(v if isinstance(v, list) else [v])
    codec = TokenizerCodec(tokenizer, sorted(eos))
    return tokenizer, codec, TokenByteIndex(codec.token_bytes_table, device="cpu")


def masked_logits(raw, tokenizer, codec, dtype=torch.float64):
    values = raw[:codec.vocab_size].to(dtype=dtype).clone()
    valid = torch.zeros(codec.vocab_size, dtype=torch.bool, device=values.device)
    valid[list(tokenizer.get_vocab().values())] = True
    valid &= torch.tensor([bool(x) for x in codec.token_bytes_table], device=values.device)
    values[~valid] = -torch.inf
    return values, int((~valid).sum()), max(0, int(raw.numel())-codec.vocab_size)


def plan_accounting(state, plan, dist, index):
    """Independent candidate-mass entropy audit; creates no PathRefs."""
    p = None if dist is None else dist.probs.detach().to("cpu", torch.float64).numpy()
    groups = []
    for i, record in enumerate(plan.opaque["records"]):
        ids = record["fresh_ids"]
        if not ids:
            ids = [int(t) for g in record["static"] for t in index.lex_token_ids_cpu[g.lo:g.hi]]
        fresh = np.array([], dtype=np.float64) if p is None else p[ids] * float(state.expansion_mass)
        exact_mask = np.array([index.token_bytes[t] == record["key"] for t in ids], dtype=bool)
        retained = [state.retained[j] for j in record["retained_ids"]]
        exact = np.concatenate((fresh[exact_mask], [r.mass for r in retained if r.residual == record["key"]]))
        strict = np.concatenate((fresh[~exact_mask], [r.mass for r in retained if r.residual != record["key"]]))
        exact = exact[exact > 0]; strict = strict[strict > 0]
        mass = float(exact.sum() + strict.sum())
        a = float(exact.sum() / mass)
        internal = entropy(np.concatenate((exact, strict)) / mass)
        u = entropy(np.concatenate(([a], strict / mass)))
        loss = a * entropy(exact / exact.sum()) if a else 0.0
        groups.append({"group_index": i, "mass": mass, "internal": internal,
                       "u": u, "loss": loss, "alpha": a,
                       "exact_count": len(exact), "strict_count": len(strict),
                       "strict_mass": float(strict.sum() / mass),
                       "identity_error": abs(internal-u-loss),
                       "cheap_u_error": abs(u-record["las_posterior_entropy_bits"])})
    weights = np.array([g["mass"] for g in groups]); weights /= weights.sum()
    hg = entropy(weights)
    hc = sum(w*g["internal"] for w,g in zip(weights, groups))
    eu = sum(w*g["u"] for w,g in zip(weights, groups))
    el = sum(w*g["loss"] for w,g in zip(weights, groups))
    hprior = entropy([state.expansion_mass] + [r.mass for r in state.retained])
    hx = hprior + (float(state.expansion_mass)*entropy(p) if p is not None else 0.0)
    return {"candidate_entropy_bits": hx, "group_entropy_bits": hg,
            "conditional_group_entropy_bits": hc, "expected_u_bits": eu,
            "expected_exact_loss_bits": el, "entropy_identity_error": abs(hx-hg-hc),
            "las_identity_error": abs(hc-eu-el),
            "max_group_las_identity_error": max(g["identity_error"] for g in groups),
            "largest_group_mass": float(weights.max()),
            "top1_group_mass": float(weights.max()), "top5_group_mass": float(np.sort(weights)[-5:].sum()),
            "group_count": len(groups)}, groups


class ProductionTimers:
    """Wrap unchanged functions for CPU phase timing only."""
    def __init__(self):
        import round4_1_common as base
        self.base = base
        self.merge = base._linear_retained_assignment
        self.materialize = base.materialize_candidates
        self.reset()
        def merge(*a, **kw):
            t=time.perf_counter(); value=self.merge(*a, **kw)
            self.retained_merge_ms += (time.perf_counter()-t)*1000
            return value
        def materialize(*a, **kw):
            t=time.perf_counter(); value=self.materialize(*a, **kw)
            self.candidate_materialization_ms += (time.perf_counter()-t)*1000
            return value
        base._linear_retained_assignment=merge
        base.materialize_candidates=materialize
    def reset(self):
        self.retained_merge_ms=0.0; self.candidate_materialization_ms=0.0
    def close(self):
        self.base._linear_retained_assignment=self.merge
        self.base.materialize_candidates=self.materialize
