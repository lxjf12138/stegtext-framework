"""Strict TOML configuration, inheritance, and deterministic case expansion."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import re
import tomllib

from stegatext_runtime.models import MODEL_IDS, prompt_mode


ROOT = Path(__file__).resolve().parents[2]
SAMPLING = {"do_sample", "temperature", "top_k", "top_p", "min_p", "typical_p", "repetition_penalty"}
METHODS = {"direct", "basic_upstream", "mwis_upstream", "stepwise_upstream", "stepwise_reviewed",
           "syncpool_upstream", "consistency_local", "mwis_greedy_local", "syncpool", "las_local",
           "naive_las", "las", "fullsync", "adaptive", "basic_bytes_v1", "mwis_bytes_v1", "stepwise_chat_v1"}
CODERS = {"discop_copy", "discop_legacy", "discop_upstream", "imec", "imec_window_exact", "imec_prefix_exact", "range_ac", "huffman",
          "liao_differential", "liao_binary", "liao_stability", "liao_stability_reviewed"}
SYSTEMS = {"sparsamp", "anstega", "anstega_normalized", "imec_official", "meteor_borrowed",
           "rrc_verified", "rrc_paper_stop", "arimec", "arimec_merge"}
CARRIERS = {"anstega_reviewed", "anstega_keyed", "meteor_borrowed"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def checked_keys(value, allowed, context):
    extra = set(value)-set(allowed)
    if extra:
        raise ValueError(f"Unknown {context} fields: {sorted(extra)}")


def merge(base, patch):
    result = copy.deepcopy(base)
    for key, value in patch.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def read_tree(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError("Configuration inheritance cycle")
    data = tomllib.loads(path.read_text())
    for prompt in data.get("inputs", {}).get("prompts", []):
        if "file" in prompt:
            prompt["file"] = str((path.parent/prompt["file"]).resolve())
    for section in ("message", "randomness"):
        if "file" in data.get(section, {}):
            data[section]["file"] = str((path.parent/data[section]["file"]).resolve())
    parent = data.pop("extends", None)
    return merge(read_tree(path.parent/parent, stack+(path,)), data) if parent else data


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def resolve(path):
    data = read_tree(path)
    checked_keys(data, {"schema_version", "name", "source", "inputs", "message", "randomness", "stopping", "variants", "execution"}, "root")
    if data.get("schema_version") not in (1, 2):
        raise ValueError("schema_version must be 1 or 2")
    v2 = data["schema_version"] == 2
    if 'execution' in data:
        checked_keys(data['execution'], {'order', 'order_seed'}, 'execution')
        if data['execution'].get('order') != 'hash_interleaved_v1':
            raise ValueError('Unknown execution order')
        seed = data['execution'].get('order_seed')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('execution.order_seed must be uint32')
    if not re.fullmatch(r"[A-Za-z0-9_-]+", data.get("name", "")):
        raise ValueError("name must contain letters, digits, underscore or dash")
    source = data["source"]
    checked_keys(source, {"models", "profile", "sampling", "kv_cache", "prompt_mode", "languages", "sampling_contract"}, "source")
    if not source.get("models") or not set(source["models"]) <= set(MODEL_IDS) or len(set(source["models"])) != len(source["models"]):
        raise ValueError(f"source.models must list distinct values from {sorted(MODEL_IDS)}")
    requested_mode = source.get("prompt_mode", "model_default")
    if requested_mode not in ("model_default", "chat", "plain_completion"):
        raise ValueError("Unsupported source.prompt_mode")
    if requested_mode != "model_default" and any(prompt_mode(m) != requested_mode for m in source["models"]):
        raise ValueError("source.prompt_mode disagrees with the mandatory model protocol")
    if "languages" in source:
        languages = source["languages"]
        if not isinstance(languages, dict) or set(languages) != set(source["models"]) or not set(languages.values()) <= {"en", "zh"}:
            raise ValueError("source.languages must assign en/zh to every selected model")
    source.setdefault("profile", "stego")
    source.setdefault("sampling", {})
    source.setdefault("kv_cache", "canonical")
    if source["profile"] not in ("stego", "checkpoint") or source["kv_cache"] not in ("canonical", "cpu_snapshots"):
        raise ValueError("Unsupported source profile or cache schedule")
    checked_keys(source["sampling"], SAMPLING, "sampling")
    for key, value in source["sampling"].items():
        if key == "do_sample":
            if type(value) is not bool: raise ValueError("do_sample must be bool")
        elif key == "top_k":
            if type(value) is not int or value < 0: raise ValueError("top_k must be nonnegative integer")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not float("-inf") < value < float("inf"):
            raise ValueError(f"{key} must be finite")
        elif key in ("top_p", "typical_p") and not 0 < value <= 1:
            raise ValueError(f"{key} must be in (0,1]")
        elif key == "min_p" and not 0 <= value <= 1:
            raise ValueError("min_p must be in [0,1]")
        elif key in ("temperature", "repetition_penalty") and value <= 0:
            raise ValueError(f"{key} must be positive")
    if source.get('sampling_contract') not in (None, 'controlled_topk_T1_v1'):
        raise ValueError('Unknown sampling contract')
    if source.get('sampling_contract'):
        expected = dict(temperature=1.0, top_p=1.0, do_sample=True, repetition_penalty=1.0)
        if any(source['sampling'].get(k) != v for k,v in expected.items()) or source['sampling'].get('top_k') not in (0,16,32,128,256,512,1024,2048):
            raise ValueError('Controlled T1 contract requires explicit sampler settings')
        if message_value := data.get('message'):
            if message_value.get('delivery', 'prefix_stream') != 'prefix_stream':
                raise ValueError('Controlled T1 contract forbids padded finite frames')
    inputs = data["inputs"]
    checked_keys(inputs, {"prompts", "seeds", "ordinary_seed_policy"}, "inputs")
    from .randomness import POLICIES
    if inputs.get('ordinary_seed_policy', 'shared_replicate_v1') not in POLICIES:
        raise ValueError('Unknown ordinary_seed_policy')
    if not inputs.get("prompts") or not inputs.get("seeds"):
        raise ValueError("prompts and seeds must be nonempty")
    if len(set(inputs["seeds"])) != len(inputs["seeds"]) or any(type(s) is not int or not 0 <= s < 2**32 for s in inputs["seeds"]):
        raise ValueError("seeds must be distinct uint32 integers")
    prompt_ids = set()
    for p in inputs["prompts"]:
        checked_keys(p, {"id", "file", "text", "language", "assistant_prefix"}, "prompt")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", p.get("id", "")) or p["id"] in prompt_ids:
            raise ValueError("Prompt IDs must be distinct safe names")
        prompt_ids.add(p["id"])
        if ("file" in p) == ("text" in p):
            raise ValueError("Each prompt requires exactly one of file/text")
        if "file" in p: p["text"] = Path(p.pop("file")).read_text()
        if not p["text"]: raise ValueError("Empty prompt")
        p.setdefault("language", "en")
        if p["language"] not in ("en", "zh"): raise ValueError("language must be en or zh")
        p["sha256"] = hashlib.sha256(p["text"].encode()).hexdigest()
        if 'assistant_prefix' in p:
            if not v2 or any(prompt_mode(m) != 'chat' for m in source['models']):
                raise ValueError('Assistant prefill requires schema v2 and Instruct chat models')
            if not isinstance(p['assistant_prefix'], str) or not p['assistant_prefix']:
                raise ValueError('assistant_prefix must be a nonempty string')
            p['sha256'] = fingerprint(dict(user=p['text'], assistant_prefix=p['assistant_prefix']))
    for model, language in source.get("languages", {}).items():
        if not any(p["language"] == language for p in inputs["prompts"]):
            raise ValueError(f"No {language} prompt for {model}")
    message = data.setdefault("message", {"mode": "fixture", "bits": 131072, "pattern": "random"})
    checked_keys(message, {"mode", "bits", "pattern", "file"} | ({"delivery"} if v2 else set()), "message")
    if v2:
        message.setdefault("delivery", "prefix_stream")
        if message["delivery"] not in ("prefix_stream", "finite_frame"):
            raise ValueError("message.delivery must be prefix_stream or finite_frame")
    if message.get("mode") == "file":
        if set(message)-{"mode", "file", "delivery"}: raise ValueError("file message accepts only mode/file/delivery")
        bits = Path(message["file"]).read_text().strip()
        if not bits or set(bits)-{"0", "1"}: raise ValueError("Message file must contain binary digits")
        message.update(bits=len(bits), sha256=hashlib.sha256(bits.encode()).hexdigest())
    elif message.get("mode") == "fixture":
        if "file" in message: raise ValueError("fixture message has no file")
        message.setdefault("bits", 131072); message.setdefault("pattern", "random")
        positive_integer(message["bits"], "message.bits")
        if message["pattern"] not in ("random", "zeros", "ones"): raise ValueError("Invalid fixture pattern")
    else:
        raise ValueError("message.mode must be fixture or file")
    randomness = data.setdefault("randomness", {"mode": "fixture"})
    checked_keys(randomness, {"mode", "file", "stream_policy", "experiment_id"}, "randomness")
    base_randomness = {k:v for k,v in randomness.items() if k not in ('stream_policy', 'experiment_id')}
    if base_randomness == {"mode": "fixture"}: pass
    elif set(base_randomness) == {"mode", "file"} and randomness["mode"] == "key_file":
        key = bytes.fromhex(Path(randomness["file"]).read_text().strip())
        if len(key) < 16: raise ValueError("Shared key needs at least 16 bytes")
        randomness["sha256"] = hashlib.sha256(key).hexdigest()
    else: raise ValueError("randomness must be fixture or key_file with file")
    if 'stream_policy' in randomness or 'experiment_id' in randomness:
        from .randomness import CASE_POLICY, INDEPENDENT
        if randomness.get('stream_policy') != CASE_POLICY or not v2 or randomness['mode'] != 'key_file':
            raise ValueError('independent_cases_v1 requires schema v2 and a key_file experiment root')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', randomness.get('experiment_id', '')):
            raise ValueError('Independent experiment_id must be a safe nonempty name')
        if inputs.get('ordinary_seed_policy') != INDEPENDENT:
            raise ValueError('Independent cases require independent ordinary sampling')
        if message['mode'] != 'fixture' or message['pattern'] != 'random' or message['delivery'] != 'prefix_stream':
            raise ValueError('Independent cases require an unpadded random fixture message')
    stopping = data.setdefault("stopping", {"max_steps": 64, "utf8_extension": 32})
    if v2:
        from .stopping import normalize
        stopping = data["stopping"] = normalize(stopping)
    else:
        checked_keys(stopping, {"max_steps", "utf8_extension", "sentences", "paragraphs", "min_bits"}, "stopping")
        stopping.setdefault("utf8_extension", 32)
        positive_integer(stopping["max_steps"], "stopping.max_steps")
        if type(stopping["utf8_extension"]) is not int or stopping["utf8_extension"] < 0:
            raise ValueError("utf8_extension must be nonnegative integer")
        if "sentences" in stopping and "paragraphs" in stopping: raise ValueError("Choose sentences or paragraphs")
        for key in ("sentences", "paragraphs", "min_bits"):
            if key in stopping: positive_integer(stopping[key], "stopping."+key)
    ids = set()
    if not data.get("variants"): raise ValueError("At least one variant is required")
    for variant in data["variants"]:
        checked_keys(variant, {"id", "kind", "method", "coder", "coder_parameters", "tau", "system", "implementation"}, "variant")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", variant.get("id", "")) or variant["id"] in ids:
            raise ValueError("Variant IDs must be distinct safe names")
        ids.add(variant["id"])
        kind = variant["kind"]
        if v2 and kind not in ("ordinary", "planner"):
            raise ValueError("v2 stopping is implemented for ordinary/planner only; native protocols retain v1")
        if v2 and message["delivery"] == "finite_frame" and kind == "planner" and variant.get("coder") != "discop_copy":
            raise ValueError("v2 finite_frame currently requires discop_copy")
        if kind == "planner":
            if variant.get("method") not in METHODS or variant.get("coder") not in CODERS: raise ValueError("Unknown method/coder")
            if variant.get('method') in ('basic_bytes_v1', 'mwis_bytes_v1', 'stepwise_chat_v1') and not v2:
                raise ValueError('Versioned baseline channel adapters require schema v2')
            if 'implementation' in variant and (not v2 or variant.get('method') != 'syncpool' or variant['implementation'] != 'shared_planner_v1'):
                raise ValueError('implementation is only supported for v2 SyncPool shared_planner_v1')
            if variant['coder'] == 'imec_prefix_exact' and not v2:
                raise ValueError('Terminal prefix delivery requires schema v2')
            if variant['coder'] == 'imec_prefix_exact':
                # Resolve new experiment defaults into manifests/public packets.
                # Low-level defaults stay unchanged for legacy packet replay.
                parameters = variant.setdefault('coder_parameters', dict(block_size=6, window_size=3))
                parameters.setdefault('coupling_strategy', 'greedy')
                parameters.setdefault('schedule', 'head_first')
            if "coder_parameters" in variant:
                if variant['coder'] not in ('imec_window_exact', 'imec_prefix_exact') or not v2:
                    raise ValueError('Explicit coder_parameters require v2 imec_window_exact')
                checked_keys(variant['coder_parameters'], {'block_size','window_size','coupling_strategy','schedule'} if variant['coder'] == 'imec_prefix_exact' else {'block_size','window_size'}, 'coder_parameters')
                for k in ('block_size','window_size'):
                    positive_integer(variant['coder_parameters'].get(k), k)
                if variant['coder_parameters'].get('coupling_strategy', 'greedy') not in ('greedy','prefix') or variant['coder_parameters'].get('schedule','max_entropy') not in ('max_entropy','head_first'):
                    raise ValueError('Invalid iMEC coupling/scheduling strategy')
                if variant['coder_parameters']['block_size'] > 12:
                    raise ValueError('iMEC diagnostic supports block_size <=12')
            if "system" in variant: raise ValueError("planner has no system")
            if variant["method"] == "adaptive":
                tau = variant.get("tau")
                if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0 <= tau < float("inf"):
                    raise ValueError("Adaptive requires an explicit finite nonnegative tau")
            elif "tau" in variant: raise ValueError("tau only applies to adaptive")
        elif kind in ("upstream", "carrier"):
            if set(variant)-{"id", "kind", "system"}: raise ValueError("System variants accept only id/kind/system")
            if variant.get("system") not in (SYSTEMS if kind == "upstream" else CARRIERS): raise ValueError("Unknown system")
            if kind == "carrier" and (stopping["max_steps"] > 128 or stopping["utf8_extension"] != 32):
                raise ValueError("Reviewed carrier wrapper supports max_steps <=128 and utf8_extension=32")
        elif kind != "ordinary" or set(variant) != {"id", "kind"}:
            raise ValueError("Invalid ordinary variant or unknown kind")
        if kind != "ordinary" and (source["sampling"].get("do_sample") is False or (source["profile"] == "checkpoint" and "qwen" in source["models"] and source["sampling"].get("do_sample") is not True)):
            raise ValueError("This steganographic experiment requires a sampling source")
        if not v2 and kind != "planner" and set(stopping)-{"max_steps", "utf8_extension"}:
            raise ValueError("Sentence/paragraph/min_bits stopping currently requires planner-only variants")
        if v2 and kind == "ordinary" and stopping['mode'] != 'sentence_boundary' and any(k in stopping for k in ("sentences", "paragraphs", "min_bits")):
            raise ValueError("Text/bit goals currently require planner-only bounded variants")
    lock = json.loads((ROOT/"environment/models.lock.json").read_text())
    data["model_revisions"] = {m: next(s for s in lock["models"] if s["repo_id"] == MODEL_IDS[m]) for m in source["models"]}
    return data


def cases(config):
    for model, prompt, seed, variant in itertools.product(config["source"]["models"], config["inputs"]["prompts"], config["inputs"]["seeds"], config["variants"]):
        if model in config["source"].get("languages", {}) and prompt["language"] != config["source"]["languages"][model]:
            continue
        item = dict(model=model, prompt=prompt, seed=seed, variant=variant)
        item["id"] = f"{model}__{prompt['id']}__{seed}__{variant['id']}"
        yield item
