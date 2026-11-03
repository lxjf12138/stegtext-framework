"""Small, externally bounded k16/Full timing pilot on existing review prompts.

No runtime/coder changes. Independent workers reuse the frozen execution path;
alarms record resource failures without resampling or altering normal stopping.
"""
from __future__ import annotations

import argparse
import atexit
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from stegatext_runtime.experiments.config import ROOT, cases, fingerprint, read_tree, resolve
from stegatext_runtime.experiments.runner import (
    code_state, environment, file_hash, planning, read, summarize, worker, write,
)
from tools.controlled_comparison.reproduce import write_config

OUT = ROOT / "reports/review_budget_pilot_20261023"
SELF = Path(__file__).resolve()
LIMITS = dict(cpu_threads=1, process_tree_rss_gib=12, minimum_available_gib=4,
              maximum_swap_growth_gib=.5, artifact_gib=2, single_record_mib=128,
              minimum_disk_free_gib=20, total_wall_seconds=1200,
              per_case_stage_seconds=180, case_hard_grace_seconds=15)


def prepare():
    if (OUT / "DESIGN.json").exists():
        raise RuntimeError("Pilot design exists; do not replace or resample")
    from tools.review_long_sweep_20261022 import variants
    prior = ROOT / "reports/review_long_sweep_20261022"
    documents = read(prior / "DATASET_MANIFEST.json")["documents"]
    audit = read(prior / "PROMPT_AUDIT.json")
    selected = [next(d for d in documents if d["language"] == language)
                for language in ("en", "zh")]
    write(OUT / "DATASET_MANIFEST.json", dict(
        documents=selected, selection="First existing document per language in the frozen development manifest; no output-based selection",
        source_manifest_sha256=file_hash(prior / "DATASET_MANIFEST.json")))
    write(OUT / "PROMPT_AUDIT.json", dict(rows=[
        r for r in audit["rows"] if r["prompt_id"] in {d["id"] for d in selected}]))
    key = OUT / "experiment_root.key"
    key.parent.mkdir(parents=True, exist_ok=True)
    with key.open("x") as handle:
        handle.write(os.urandom(32).hex()+"\n")
    key.chmod(0o600)
    jobs = []
    for k in (16, 0):
        for model, language in (("llama_instruct", "en"), ("qwen_instruct", "zh")):
            base_path = prior / "configs" / f"review_long_{model}_k32_s430001.toml"
            config = copy.deepcopy(read_tree(base_path))
            doc = next(d for d in selected if d["language"] == language)
            name = f"budget_{model}_k{k}"
            config["name"] = name
            config["source"].update(models=[model], languages={model: language})
            config["source"]["sampling"]["top_k"] = k
            config["inputs"] = dict(ordinary_seed_policy="independent_conditions_v1", seeds=[460001],
                prompts=[dict(id=doc["id"], language=language, text=doc["user_message"],
                              assistant_prefix=doc["prefix"])])
            config["variants"] = [v for v in variants()
                                  if k != 0 or v.get("method") != "stepwise_chat_v1"]
            config["randomness"] = dict(mode="key_file", file=str(key),
                stream_policy="independent_cases_v1", experiment_id="review-budget-pilot-20261023")
            config["execution"] = dict(order="hash_interleaved_v1", order_seed=2026102303)
            path = OUT / "configs" / (name+".toml")
            write_config(path, config)
            resolved = resolve(path)
            count = len(list(cases(resolved)))
            assert count == (11 if k == 0 else 12)
            jobs.append(dict(name=name, model=model, language=language, top_k=k,
                config=str(path.relative_to(ROOT)), config_sha256=file_hash(path), cases=count,
                prompt_id=doc["id"]))
    write(OUT / "DESIGN.json", dict(
        frozen_before_generation=True, created_unix=time.time(), jobs=jobs,
        cases=sum(j["cases"] for j in jobs), stego_cases=42,
        documents_per_language=1, repeats_per_document=1, seed=460001,
        limits=LIMITS, coder="imec_prefix_exact/greedy/head_first/6/3",
        stopping="Natural EOS or common 256 source-token cap; no minimum length or retries",
        full="top_k=0, top_p=1; existing byte-validity mask disclosed; no Stepwise Full",
        scope="Development timing and recovery pilot, not a PPL equivalence or capacity validation",
        timeout_policy="Per-stage external alarm; retain error/unavailable records; do not substitute or retry",
        sampling="Independent message/protocol/Ordinary streams; same seed is not a common token trajectory",
        script_sha256=file_hash(SELF), runtime_code_sha256=fingerprint(code_state()),
        dataset_manifest_sha256=file_hash(OUT / "DATASET_MANIFEST.json"),
        prompt_audit_sha256=file_hash(OUT / "PROMPT_AUDIT.json")))
    print(json.dumps(dict(prepared=True, cases=46, jobs=4, limits=LIMITS)), flush=True)


def bounded_worker(directory, model, role):
    import torch
    import stegatext_runtime.experiments.engine as engine
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original = engine.send if role == "sender" else engine.receive

    def expired(_signum, _frame):
        raise TimeoutError(f"Pilot resource limit: {LIMITS['per_case_stage_seconds']} seconds per case stage")

    def bounded(source, payload, config):
        identifier = payload["id"] if role == "sender" else payload["case_id"]
        active = dict(kind="case", status="running", job=directory.name, model=model,
                      top_k=config["source"]["sampling"]["top_k"], stage=role,
                      case_id=identifier, started_unix=time.time(), model_queries=0,
                      max_generated_history_tokens=0)
        write(OUT / "ACTIVE.json", active)
        last_write = [time.monotonic()]
        previous_hook = source.query_log

        def monitor(ids, probabilities):
            active["model_queries"] += 1
            prompt_len = len(source.prompt.token_ids) if source.prompt is not None else len(ids)
            active["max_generated_history_tokens"] = max(active["max_generated_history_tokens"], len(ids)-prompt_len)
            if time.monotonic()-last_write[0] >= 2:
                write(OUT / "ACTIVE.json", active)
                last_write[0] = time.monotonic()
            if previous_hook is not None:
                previous_hook(ids, probabilities)

        source.query_log = monitor
        old_handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["per_case_stage_seconds"])
        try:
            result = original(source, payload, config)
            active["outcome"] = "returned"
            return result
        except Exception as exc:
            active["outcome"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            source.query_log = previous_hook
            active.update(status="finished", finished_unix=time.time())
            write(OUT / "ACTIVE.json", active)
            write(directory / "resource_cases" / (role+"_"+identifier+".json"), active)

    if role == "sender":
        engine.send = bounded
    else:
        engine.receive = bounded
    worker(directory, model, role)


def execute():
    from tools.expanded_records_20261021 import archive
    design = read(OUT / "DESIGN.json")
    assert file_hash(SELF) == design["script_sha256"]
    assert fingerprint(code_state()) == design["runtime_code_sha256"]
    common_execution = dict(environment=environment(), code=code_state())
    completed = []
    for job in design["jobs"]:
        config_path = ROOT / job["config"]
        assert file_hash(config_path) == job["config_sha256"]
        config = resolve(config_path)
        directory = OUT / "runs" / job["name"]
        directory.mkdir(parents=True, exist_ok=False)
        execution = dict(common_execution, configuration_sha256=fingerprint(config))
        write(directory / "manifest.json", dict(schema_version=1, configuration=config,
              execution=execution, plan=planning(config), pilot_design_sha256=file_hash(OUT / "DESIGN.json")))
        with (ROOT / ".cache/experiment_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for role in ("sender", "receiver"):
                write(OUT / "ACTIVE.json", dict(kind="worker", status="starting", job=job["name"],
                      stage=role, started_unix=time.time()))
                with (OUT / (job["name"]+"_"+role+".log")).open("x") as log:
                    subprocess.run([sys.executable, "-B", str(SELF), "worker", "--directory", str(directory),
                                    "--model", job["model"], "--role", role],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = summarize(directory)
        write(OUT / "ACTIVE.json", dict(kind="scoring", status="running", job=job["name"], started_unix=time.time()))
        with (OUT / (job["name"]+"_score.log")).open("x") as log:
            subprocess.run([sys.executable, "-B", str(SELF), "score", "--directory", str(directory)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        completed.append(dict(job, directory=str(directory),
            successful_senders=sum(r["sender_status"] == "ok" for r in rows),
            independent_exact=sum(r["exact"] is True for r in rows)))
        write(OUT / "PROGRESS.json", dict(completed_jobs=completed,
              completed_cases=sum(j["cases"] for j in completed), planned_cases=46))
        archive(directory)
        print(json.dumps(completed[-1]), flush=True)
    write(OUT / "COMPLETE.json", dict(completed_jobs=4, attempts=46, finished_unix=time.time()))


def score(directory):
    import torch
    from stegatext_runtime.experiments.native_ppl import evaluate
    torch.set_num_threads(1)
    evaluate([directory], directory / "ppl")


def guard():
    import psutil
    design = read(OUT / "DESIGN.json")
    assert design["limits"] == LIMITS
    assert file_hash(SELF) == design["script_sha256"]
    started = time.monotonic()
    initial_swap = psutil.swap_memory().used
    lock = (OUT / "guard.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status="running", started_unix=time.time(), peak_rss_gib=0,
                 maximum_artifact_mib=0, maximum_swap_growth_gib=0, limits=LIMITS)
    with (OUT / "driver.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-B", str(SELF), "execute"], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state["driver_pid"] = proc.pid

        def cleanup():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                except ProcessLookupError:
                    pass

        atexit.register(cleanup)
        with (OUT / "resources.jsonl").open("x") as history:
            while proc.poll() is None:
                rss = 0
                try:
                    family = [psutil.Process(os.getpid())]+psutil.Process(os.getpid()).children(recursive=True)
                except psutil.Error:
                    family = []
                for process in family:
                    try:
                        rss += process.memory_info().rss
                    except psutil.Error:
                        pass
                sizes, largest_record = [], 0
                for path in OUT.rglob("*"):
                    if path.is_file():
                        try:
                            size = path.stat().st_size
                        except FileNotFoundError:
                            continue
                        sizes.append(size)
                        if path.parent.name in ("sender", "receiver") and path.suffix == ".json":
                            largest_record = max(largest_record, size)
                swap = max(0, psutil.swap_memory().used-initial_swap)/2**30
                available = psutil.virtual_memory().available/2**30
                elapsed = time.monotonic()-started
                active = read(OUT / "ACTIVE.json") if (OUT / "ACTIVE.json").exists() else {}
                state.update(elapsed_seconds=elapsed, peak_rss_gib=max(state["peak_rss_gib"], rss/2**30),
                    maximum_artifact_mib=max(state["maximum_artifact_mib"], sum(sizes)/2**20),
                    maximum_swap_growth_gib=max(state["maximum_swap_growth_gib"], swap))
                live = dict(unix=time.time(), elapsed_seconds=elapsed, rss_gib=rss/2**30,
                    available_gib=available, swap_growth_gib=swap, artifact_mib=sum(sizes)/2**20, active=active)
                history.write(json.dumps(live)+"\n")
                history.flush()
                write(OUT / "LIVE.json", live)
                write(OUT / "RESOURCES.json", state)
                reasons = []
                if rss/2**30 > LIMITS["process_tree_rss_gib"]: reasons.append("rss")
                if available < LIMITS["minimum_available_gib"]: reasons.append("available_memory")
                if swap > LIMITS["maximum_swap_growth_gib"]: reasons.append("swap")
                if sum(sizes)/2**30 > LIMITS["artifact_gib"]: reasons.append("artifact_size")
                if largest_record/2**20 > LIMITS["single_record_mib"]: reasons.append("record_size")
                if shutil.disk_usage(OUT).free/2**30 < LIMITS["minimum_disk_free_gib"]: reasons.append("disk")
                if elapsed > LIMITS["total_wall_seconds"]: reasons.append("total_time")
                if active.get("kind") == "case" and active.get("status") == "running" and time.time()-active["started_unix"] > LIMITS["per_case_stage_seconds"]+LIMITS["case_hard_grace_seconds"]:
                    reasons.append("case_hard_time")
                if (OUT / "STOP_REQUESTED.json").exists(): reasons.append("explicit_stop")
                if reasons:
                    state.update(status="resource_stop", reasons=reasons)
                    write(OUT / "STOP_STATE.json", dict(reasons=reasons, live=live))
                    cleanup()
                    break
                time.sleep(5)
        code = proc.wait()
    state.update(returncode=code, elapsed_seconds=time.monotonic()-started, finished_unix=time.time())
    if state["status"] == "running":
        state["status"] = "finished" if code == 0 else "failed"
    write(OUT / "RESOURCES.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "guard", "execute", "worker", "score"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--role", choices=("sender", "receiver"))
    args = parser.parse_args()
    if args.phase == "worker": bounded_worker(args.directory, args.model, args.role)
    elif args.phase == "score": score(args.directory)
    else: {"prepare": prepare, "execute": execute, "guard": guard}[args.phase]()
