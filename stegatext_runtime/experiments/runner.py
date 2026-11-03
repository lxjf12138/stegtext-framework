"""Isolated workers, immutable attempts, provenance, resume and comparison."""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid

from .config import ROOT, canonical, cases, fingerprint, resolve


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, record):
    from .recording import canonical_digest
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(record)
    value["record_sha256"] = canonical_digest(record)
    temporary = path.with_suffix(path.suffix+".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read(path):
    from .recording import canonical_digest, ends_with_object
    with path.open("rb") as handle:
        if not ends_with_object(handle):
            raise ValueError(f"Incomplete record (missing final object): {path}")
        handle.seek(0)
        value = json.load(handle)
    digest = value.pop("record_sha256")
    if canonical_digest(value) != digest: raise ValueError(f"Corrupted record: {path}")
    return value


def environment():
    import torch
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).strip()
    model_inputs = {}
    for spec in json.loads((ROOT/"environment/models.lock.json").read_text())["models"]:
        folder = ROOT/spec["local_path"]
        model_inputs[spec["repo_id"]] = {
            "configuration_and_tokenizer": {p.name: file_hash(p) for p in sorted(folder.iterdir())
                                            if p.is_file() and p.suffix == ".json"},
            "weights_stat": {p.name: dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                             for p in sorted(folder.glob("*.safetensors"))},
        }
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), gpu=gpu,
                packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}, model_inputs=model_inputs,
                cuda_build=torch.version.cuda, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                env={name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS")},
                lock_sha256=file_hash(ROOT/"uv.lock"), project_sha256=file_hash(ROOT/"pyproject.toml"), models_lock_sha256=file_hash(ROOT/"environment/models.lock.json"),
                runtime=dict(device="cuda:0", dtype="bfloat16", attention="sdpa", quantization="none"))


def code_state():
    paths = list((ROOT/"stegatext_runtime").rglob("*.py"))
    paths += [ROOT/p for p in ("research_notes/inference_conformance/hf_reference.py", "research_notes/full_validation/suite.py",
              "research_notes/full_validation/upstream.py", "research_notes/full_validation/systems.py", "research_notes/full_validation/cache.py",
              "research_notes/problem_resolution/paper_adapters.py", "research_notes/problem_resolution/bounded_systems.py", "research_notes/problem_resolution/text_rules.py")]
    paths += [p for p in (ROOT/"package").rglob("*.py") if "__pycache__" not in p.parts]
    paths += [p for p in (ROOT/"external_baselines").rglob("*") if p.is_file() and p.suffix in (".py", ".pyx") and ".git" not in p.parts]
    paths += list((ROOT/"runs/full_validation_v1/build/discop").glob("stega_cy*.so"))
    paths += [ROOT/"package/PACKAGE_SHA256SUMS.txt", ROOT/"BASELINE_SOURCES.json"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def planning(config):
    from .engine import contract
    from .randomness import policy
    expanded = list(cases(config))
    return dict(name=config["name"], configuration_sha256=fingerprint(config), cases=len(expanded),
                models=config["source"]["models"], source=config["source"],
                variants=[dict(**v, contract=contract(v, config["stopping"])) for v in config["variants"]],
                execution=config.get('execution', {'order':'config_product_v1'}),
                ordinary_seed_policy=policy(config),
                stego_stream_policy=config.get('randomness', {}).get('stream_policy', 'legacy_paired_v2'),
                case_ids=[c["id"] for c in expanded],
                fixture_pairing="v1: message/key by seed; v2: message by seed, key additionally domain-separated by source/prompt/seed nonce, shared across variants.",
                independence="Separate sender and receiver processes; common locked environment.")


def load_source(config, model):
    from stegatext_runtime import Source
    from research_notes.inference_conformance.hf_reference import HFReference
    source = Source.load(model, config["source"]["profile"])
    overrides = config["source"]["sampling"]
    for name, value in overrides.items(): setattr(source.config, name, value)
    source.config.validate()
    source.overrides.update(overrides)
    source.reference = HFReference(source.backend, source.config, validity_mask=source.validity)
    if config["source"]["kv_cache"] == "cpu_snapshots":
        from research_notes.full_validation.cache import accelerate
        accelerate(source)
    if config['source'].get('sampling_contract'):
        source.expected_sampling = dict(overrides)
        source.assert_sampling_contract()
    return source


def conditions(case, config, execution, source):
    from .engine import contract
    from .randomness import descriptor, policy, INDEPENDENT, protocol_descriptor
    message = dict(config["message"])
    randomness = dict(config["randomness"])
    message.pop("file", None); randomness.pop("file", None)
    result = dict(environment_sha256=fingerprint(execution["environment"]), code_sha256=fingerprint(execution["code"]),
                source=source, prompt_sha256=case["prompt"]["sha256"], seed=case["seed"],
                message=message, randomness=randomness, kv_cache=config["source"]["kv_cache"],
                execution_order=config.get('execution', {'order':'config_product_v1'}),
                stopping=contract(case["variant"], config["stopping"]))
    if case['variant']['kind'] == 'ordinary' and policy(config) == INDEPENDENT:
        result['sampling_randomness'] = descriptor(config, case, source)
    elif (stream := protocol_descriptor(config, case, source)) is not None:
        result['protocol_randomness'] = stream
    return result


def ordered_cases(config):
    rows = list(cases(config))
    execution = config.get('execution')
    if execution:
        seed = execution['order_seed']
        rows.sort(key=lambda row: hashlib.sha256(f"case-order-v1:{seed}:{row['id']}".encode()).digest())
    return rows


def base_record(case, role, execution):
    return dict(case_id=case["id"], stage=role, execution_sha256=fingerprint(execution), pid=os.getpid())


def cached(path, expected):
    if not path.exists(): return False
    record = read(path)
    for name, value in expected.items():
        if name != "pid" and record.get(name) != value: raise ValueError(f"Resume contract mismatch in {path}: {name}")
    return record.get("status") == "ok"


def common_evaluation_spec():
    from stegatext_runtime.models import MODEL_IDS
    lock = json.loads((ROOT/'environment/models.lock.json').read_text())
    return next(v for v in lock['models'] if v['repo_id'] == MODEL_IDS['qwen_instruct'])


def worker(directory, model, role):
    import numpy as np
    import torch
    from .engine import receive, send
    manifest = read(directory/"manifest.json")
    config, execution = manifest["configuration"], manifest["execution"]
    if environment() != execution["environment"] or code_state() != execution["code"]:
        raise ValueError("Worker environment/code differs from resolved execution")
    source = None
    evaluation_tokenizer = None
    for case in ordered_cases(config):
        if case["model"] != model or (role == "receiver" and case["variant"]["kind"] == "ordinary"): continue
        path = directory/role/(case["id"]+".json")
        base = base_record(case, role, execution)
        public_path = directory/"public"/(case["id"]+".json")
        if role == "receiver" and public_path.exists(): base["input_sha256"] = file_hash(public_path)
        if cached(path, base):
            if role == "sender" and case["variant"]["kind"] != "ordinary":
                saved = read(path)
                if not public_path.exists() or file_hash(public_path) != saved["public_sha256"]:
                    raise ValueError("Cached sender packet is missing or changed")
                read(public_path)
            continue
        if role == "receiver" and not public_path.exists():
            write(path, dict(base, status="unavailable", error="Sender did not produce a public packet"))
            continue
        if source is None:
            source = load_source(config, model)
            from stegatext_runtime.prompts import configure_prompt
            configure_prompt(source, case['prompt'].get('assistant_prefix'))
            if config['source']['kv_cache'] == 'cpu_snapshots' and any(v.get('coder') in ('imec_window_exact','imec_prefix_exact') for v in config['variants']):
                # Complete the canonical oracle preflight outside measured cases.
                source.bind(case['prompt']['text'])
                ref = source.reference
                p = source.probabilities(source.prompt.token_ids)
                a,b = p.topk(2).indices.cpu().tolist()
                for i in range(20):
                    source.probabilities(source.prompt.token_ids+(a if i%2 == 0 else b,))
                assert ref.snapshot_oracle_checks == 16
        torch.manual_seed(case["seed"]); np.random.seed(case["seed"]); random.seed(case["seed"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, forwards = time.monotonic(), source.reference.forwards
        snapshot_restores = getattr(source.reference, 'snapshot_restorations', 0)
        physical_calls = [0]
        def count_forward(_module, _inputs):
            physical_calls[0] += 1
        forward_hook = source.model.register_forward_pre_hook(count_forward)
        try:
            with torch.inference_mode():
                if role == "sender":
                    packet, result = send(source, case, config)
                    if packet is not None:
                        if packet["text"] is None and not (packet["version"] == "unified-experiment-v2" and "visible_bytes_hex" in packet):
                            raise ValueError("Upstream output is not a valid UTF-8 text carrier")
                        packet.update(case_id=case["id"], execution_sha256=fingerprint(execution))
                        write(public_path, packet)
                else:
                    packet = read(public_path)
                    if packet["case_id"] != case["id"] or packet["execution_sha256"] != fingerprint(execution): raise ValueError("Public packet execution mismatch")
                    if packet["variant"] != case["variant"] or packet["stopping"] != config["stopping"] or packet["seed"] != case["seed"] or packet["prompt"] != case["prompt"]["text"]:
                        raise ValueError("Public packet differs from resolved experiment")
                    if packet.get('assistant_prefix') != case['prompt'].get('assistant_prefix'):
                        raise ValueError('Public assistant prefill differs from resolved experiment')
                    result = receive(source, packet, config)
            record = dict(base, status="ok", result=result,
                          conditions=conditions(case, config, execution, source.manifest()))
        except Exception as exc:
            record = dict(base, status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                          conditions=conditions(case, config, execution, source.manifest()))
        finally:
            forward_hook.remove()
        torch.cuda.synchronize()
        record["performance"] = dict(wall_seconds=time.monotonic()-start, model_forwards=source.reference.forwards-forwards,
                                     actual_model_forward_calls=physical_calls[0],
                                     oracle_model_forward_calls=physical_calls[0]-(source.reference.forwards-forwards),
                                     peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
                                     snapshot_restorations=getattr(source.reference, 'snapshot_restorations', 0)-snapshot_restores,
                                     cuda_synchronized_timing=True,
                                     snapshot_oracle_checks=getattr(source.reference, "snapshot_oracle_checks", 0))
        if record.get("status") == "ok":
            result = record.get("result", {})
            native_tokens = result.get("native_output_token_count", result.get("generated_token_count"))
            visible_bytes = result.get("total_visible_bytes")
            decisions = result.get("protocol_decision_count", result.get("steps"))
            wall = record["performance"]["wall_seconds"]
            record["performance"].update(
                output_native_tokens=native_tokens,
                output_visible_bytes=visible_bytes,
                forwards_per_output_token=(
                    record["performance"]["actual_model_forward_calls"] / native_tokens
                    if native_tokens else None),
                milliseconds_per_output_token=(1000.0 * wall / native_tokens if native_tokens else None),
                milliseconds_per_protocol_decision=(1000.0 * wall / decisions if decisions else None),
                planner_seconds=result.get("planning_seconds"),
                coder_seconds=result.get("coder_seconds"),
                planner_milliseconds_per_decision=(
                    1000.0 * result["planning_seconds"] / decisions
                    if decisions and result.get("planning_seconds") is not None else None),
                coder_milliseconds_per_decision=(
                    1000.0 * result["coder_seconds"] / decisions
                    if decisions and result.get("coder_seconds") is not None else None),
            )
        if record['status'] == 'ok' and config['stopping'].get('mode') in ('sentence_boundary', 'token_budget'):
            text = record['result']['text']
            evaluation_spec = source.spec
            if config['source'].get('sampling_contract') == 'controlled_topk_T1_v1':
                # One public tokenizer for BOTH models; post-timing metric only.
                from transformers import AutoTokenizer
                evaluation_spec = common_evaluation_spec()
                if evaluation_tokenizer is None:
                    evaluation_tokenizer = AutoTokenizer.from_pretrained(str(ROOT/evaluation_spec['local_path']),local_files_only=True)
            tok = evaluation_tokenizer if evaluation_tokenizer is not None else source.tokenizer
            record['result']['evaluation_token_count'] = len(tok.encode(text,add_special_tokens=False)) if text is not None else None
            record['result']['evaluation_tokenizer'] = evaluation_spec['repo_id']+'@'+evaluation_spec['tokenizer_revision']
        if role == "sender" and public_path.exists(): record["public_sha256"] = file_hash(public_path)
        write(path, record)
        print(canonical(dict(case=case["id"], stage=role, status=record["status"], error=record.get("error"))), flush=True)


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(set(a)|set(b)):
            rows += differences(a.get(key), b.get(key), f"{prefix}.{key}" if prefix else key)
        return rows
    return [] if a == b else [dict(field=prefix, before=a, after=b)]


def summarize(directory):
    manifest = read(directory/"manifest.json")
    rows = []
    for case in cases(manifest["configuration"]):
        s = read(directory/"sender"/(case["id"]+".json"))
        rpath = directory/"receiver"/(case["id"]+".json")
        r = read(rpath) if rpath.exists() else {}
        ordinary = case["variant"]["kind"] == "ordinary"
        exact = None if ordinary else s["status"] == r.get("status") == "ok" and s["pid"] != r["pid"] and s["result"]["bits"] == r["result"]["bits"]
        if exact and case["variant"]["kind"] == "planner":
            exact = all(s["result"][key] == r["result"][key] for key in ("text", "trace", "state_sha256")) and r["result"]["receiver_consumed"]
            if 'coder_state' in s['result']:
                exact = exact and s['result']['coder_state'] == r['result'].get('coder_state')
            if 'native_token_sequence' in s['result']:
                exact = exact and s['result']['native_token_sequence'] == r['result'].get('native_token_sequence')
        mismatch = None
        if exact is False and s["status"] == r.get("status") == "ok":
            left, right = s["result"]["bits"], r["result"]["bits"]
            first = next((i for i, (x, y) in enumerate(zip(left, right)) if x != y), min(len(left), len(right)))
            mismatch = f"Bit/text/state mismatch: sender {len(left)} bits, receiver {len(right)} bits; first bit difference or end at {first}"
        row = dict(case_id=case["id"], model=case["model"], prompt_id=case["prompt"]["id"], seed=case["seed"],
                   variant=case["variant"], sender_status=s["status"], receiver_status=r.get("status"), exact=exact,
                   conditions=s.get("conditions"), error=s.get("error") or r.get("error") or mismatch,
                   bits=s.get("result", {}).get("bits_count"), performance=s.get("performance"), receiver_performance=r.get("performance"))
        if "result" in s:
            from research_notes.problem_resolution.prompt_ablation import metrics
            text = s["result"].get("text", "")
            row["quality_screen"] = metrics(text, case["prompt"]["language"]) if text is not None else None
            row["quality_scope"] = "Shape/repetition only; drift regex has known false negatives. No human equivalence claim."
            result = s["result"]
            if 'sampling_randomness' in result:
                row['sampling_randomness'] = result['sampling_randomness']
            for key in ('protocol_randomness', 'message_randomness'):
                if key in result:
                    row[key] = result[key]
            row["output"] = dict(steps=result.get("steps"), tokens=len(result.get("token_ids", result.get("tokens", []))) if ("token_ids" in result or "tokens" in result) else None,
                                 text_utf8_bytes=len(text.encode()) if text is not None else None, natural_eos=result.get("natural_eos"))
            native = result.get('native_token_sequence')
            if native is not None:
                row['output'].update(tokens=native['source_token_count'],
                    body_tokens=native['body_token_count'], native_sequence_status=native['status'],
                    token_count_kind='recorded_runtime_token_ids')
            if 'evaluation_token_count' in result:
                row['output']['evaluation_tokens'] = result['evaluation_token_count']
        public_path = directory/"public"/(case["id"]+".json")
        if public_path.exists():
            packet = read(public_path)
            aux = packet.get("auxiliary", {})
            row["side_information"] = dict(ans_final_state_bits=aux.get("final_state_bits", 0),
                                           known_initial_bits=len(aux.get("public_initial_prefix", "")) or aux.get("public_initial_prefix_bits", 0),
                                           eos_identity_bits=packet.get("text_transport", {}).get("eos_index_bits", 0),
                                           public_packet_file_bytes=public_path.stat().st_size, auxiliary=aux)
        result = s.get("result", {})
        row.update(terminal_reason=result.get("terminal_reason", result.get("stop_reason", "sender_error")),
                   censored=result.get("censored"), framing_status=result.get("framing_status"),
                   watchdog=result.get("watchdog"))
        if 'token_budget' in result:
            row['token_budget'] = result['token_budget']
        if 'sentence_goal_reached' in result:
            row['sentence'] = {k:result[k] for k in ('sentence_goal_reached','first_sentence_utf8_bytes',
                                                      'overshoot_utf8_bytes','completed_sentences','last_action_preserved')}
        if not ordinary:
            from .delivery import measure
            from .engine import message_for
            from .randomness import case_nonce
            side = row.get("side_information", {})
            auxiliary = side.get("ans_final_state_bits", 0)+side.get("eos_identity_bits", 0)
            payload_context = case_nonce(manifest['configuration'], case, s.get('conditions', {}).get('source', {}))
            row.update(measure(message_for(manifest["configuration"], case["seed"], context=payload_context), result.get("bits", ""),
                               r.get("result", {}).get("bits") if r.get("status") == "ok" else None,
                               delivery=manifest["configuration"]["message"].get("delivery", "prefix_stream"),
                               replay_exact=exact, auxiliary_bits=auxiliary))
            if exact and (not row["sender_prefix_valid"] or result.get("bits_count") != len(result.get("bits", ""))):
                row["exact"] = False
                row["full_message_delivered"] = False
                row["error"] = "Sender bits/count do not match requested wire prefix"
            row.update(online_committed_bits=result.get('online_committed_bits', result.get('bits_count', 0)),
                       terminal_extra_prefix_bits=result.get('terminal_extra_prefix_bits', 0),
                       total_confirmed_bits=row['strict_verified_payload_bits'],
                       sender_target_prefix_exact=row['sender_prefix_valid'], receiver_prefix_exact=bool(exact),
                       prefix_audit=result.get('prefix_audit'), message_input=result.get('message_input'))
            seconds = (row.get("performance") or {}).get("wall_seconds", 0)
            receiver_seconds = (row.get("receiver_performance") or {}).get("wall_seconds", 0)
            row["confirmed_bits_per_second"] = row["confirmed_prefix_bits"]/seconds if seconds else None
            row['strict_verified_bits_per_second'] = row['strict_verified_payload_bits']/seconds if seconds else None
            row["end_to_end_confirmed_bits_per_second"] = row["confirmed_prefix_bits"]/(seconds+receiver_seconds) if seconds+receiver_seconds else None
        rows.append(row)
    from .delivery import aggregate
    write(directory/"results.json", dict(rows=rows, cases=len(rows), exact=sum(r["exact"] is True for r in rows),
                                        stego_cases=sum(r["exact"] is not None for r in rows), aggregates=aggregate(rows)))
    return rows


def run(config_path, output_root, resume=None):
    config = resolve(config_path)
    execution = dict(configuration_sha256=fingerprint(config), environment=environment(), code=code_state())
    lock_path = ROOT/".cache/experiment_gpu0.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError("Another unified experiment owns GPU0") from exc
        if resume:
            directory = Path(resume).resolve()
            manifest = read(directory/"manifest.json")
            if manifest["configuration"] != config or manifest["execution"] != execution:
                raise ValueError("Resume requires identical resolved configuration, environment and code")
        else:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            directory = Path(output_root).resolve()/config["name"]/(stamp+"_"+fingerprint(config)[:10]+"_"+uuid.uuid4().hex[:6])
            directory.mkdir(parents=True, exist_ok=False)
            write(directory/"manifest.json", dict(schema_version=1, configuration=config, execution=execution, plan=planning(config)))
        print(canonical(dict(run_directory=str(directory))), flush=True)
        for model in config["source"]["models"]:
            for role in ("sender", "receiver"):
                command = [sys.executable, "-B", "-m", "stegatext_runtime", "experiment", "_worker",
                           "--run-dir", str(directory), "--model", model, "--stage", role]
                log_path = directory/"logs"/(model+"_"+role+".log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a") as log:
                    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f"{model} {role} worker failed; inspect {log_path}")
                print(canonical(dict(model=model, stage=role, status="finished")), flush=True)
        rows = summarize(directory)
        print(canonical(dict(run_directory=str(directory), cases=len(rows), exact=sum(r["exact"] is True for r in rows),
                             errors=sum(r["sender_status"] != "ok" or r["exact"] is False for r in rows))), flush=True)
        return directory, rows


def compare(directories):
    loaded = [(Path(d).resolve(), read(Path(d)/"results.json")) for d in directories]
    rows = []
    if len(loaded) == 1:
        _, results = loaded[0]
        groups = {}
        for row in results["rows"]: groups.setdefault((row["model"], row["prompt_id"], row["seed"]), []).append(row)
        pairs = [(group[0], row) for group in groups.values() for row in group[1:]]
    elif len(loaded) == 2:
        lookup = {r["case_id"]: r for r in loaded[0][1]["rows"]}
        pairs = [(lookup[r["case_id"]], r) for r in loaded[1][1]["rows"] if r["case_id"] in lookup]
        missing = sorted(set(lookup) ^ {r["case_id"] for r in loaded[1][1]["rows"]})
        if missing: rows.append(dict(unmatched_case_ids=missing))
    else: raise ValueError("compare accepts one run (variants) or two runs (matched case IDs)")
    for a, b in pairs:
        delta = differences(a.get("conditions"), b.get("conditions"))
        rows.append(dict(before=a["case_id"], after=b["case_id"], condition_differences=delta,
                         variant_differences=differences(a["variant"], b["variant"]),
                         same_recorded_conditions=not delta and a.get("conditions") is not None,
                         exact_before=a["exact"], exact_after=b["exact"]))
    return dict(comparisons=rows, scope="Recorded controls only; native protocol/token/decision units remain explicit. No security or quality equivalence claim.")
