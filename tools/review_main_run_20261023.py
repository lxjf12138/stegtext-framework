"""Serial main-table driver. Preparing/preflighting never loads a model.

Run explicitly with ``run`` after preflight. Resume preserves completed errors
as well as successes. Resources and elapsed time are observed without automatic
time/resource stops; an explicit user stop remains supported.
"""
from __future__ import annotations

import argparse
import atexit
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from stegatext_runtime.experiments.config import ROOT, cases, fingerprint, read_tree, resolve
from stegatext_runtime.experiments import runner
from tools.controlled_comparison.reproduce import write_config
from tools.expanded_records_20261021 import archive, raw_digest, read_record

SELF = Path(__file__).resolve()
PREP = ROOT / "research_notes/baseline_experiment_design_20261022/main_table_uniform_preparation_20261023"
DEFAULT_OUT = ROOT / "reports/review_main_batched_20261024"
SMOKE_OUT = ROOT / "reports/review_main_runner_smoke_20261024"
BATCH_SIZE = 10
# A zero interval disables setitimer in the existing worker/scorer. No wall
# budget or memory/disk threshold terminates a main-table attempt.
LIMITS = dict(cpu_threads=1, total_wall_seconds=0, truncated_case_seconds=0,
    full_case_seconds=0, score_case_seconds=0, automatic_resource_stops=False)
THREAD_ENV = {name:"1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
THREAD_ENV.update(TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")


def code_inputs():
    paths = [SELF, ROOT/"tools/review_main_report_20261023.py", ROOT/"tools/review_budget_pilot_20261023.py",
        ROOT/"tools/expanded_records_20261021.py", ROOT/"tools/expanded_native_ppl_20261021.py",
        ROOT/"tools/controlled_comparison/reproduce.py", ROOT/"environment/activate.sh"]
    return {str(p.relative_to(ROOT)):runner.file_hash(p) for p in paths}


def config_signature(config):
    value = copy.deepcopy(config)
    value.pop("name")
    value["inputs"].pop("prompts")
    return fingerprint(value)


def verify_slice(parent, shard):
    """Changing batch boundaries must preserve cases and all random domains."""
    from stegatext_runtime.experiments.randomness import case_nonce, descriptor
    assert config_signature(parent)==config_signature(shard), "Non-slicing parameters changed"
    original={c["id"]:c for c in cases(parent)}
    # The actual source manifest is identical because the source and prompt
    # configuration are equal. The arbitrary digest checks the derivations
    # without loading model weights.
    source={"sha256":fingerprint(parent["source"])}
    for case in cases(shard):
        baseline=original[case["id"]]
        assert baseline==case, "Sliced case differs from parent"
        assert case_nonce(parent,baseline,source)==case_nonce(shard,case,source)
        if case["variant"]["kind"]=="ordinary":
            assert descriptor(parent,baseline,source)==descriptor(shard,case,source)


def verify_matched_inputs(configs):
    """Every condition for a model must use identical prompts and repeats."""
    reference={}
    for config in configs:
        for model in config["source"]["models"]:
            value=config["inputs"]
            if model in reference:
                assert value==reference[model], "All k/methods must use identical prompts and repeat coordinates"
            else:
                reference[model]=value


def shard_slices(prompt_count, top_k):
    """Small GPU shards never cross a scientific ten-document batch boundary."""
    width=1 if top_k==0 else 4
    for batch_index, batch_start in enumerate(range(0,prompt_count,BATCH_SIZE)):
        batch_stop=min(batch_start+BATCH_SIZE,prompt_count)
        for part_index,start in enumerate(range(batch_start,batch_stop,width)):
            yield dict(batch_index=batch_index,part_index=part_index,
                comparison_block=(start-batch_start)//4,
                prompt_start=start,prompt_stop=min(start+width,batch_stop))


def batch_plan(shards):
    result=[]
    for index in sorted({j["batch_index"] for j in shards}):
        jobs=[j for j in shards if j["batch_index"]==index]
        result.append(dict(batch_index=index,prompt_start=min(j["prompt_start"] for j in jobs),
            prompt_stop=max(j["prompt_stop"] for j in jobs),
            cases=sum(j["cases"] for j in jobs),stego_cases=sum(j["stego_cases"] for j in jobs),
            shard_names=[j["name"] for j in jobs]))
    return result


def prepare(out, *, smoke=False):
    if (out/"DESIGN.json").exists():
        design=preflight(out)
        assert design["mode"]==("smoke" if smoke else "main"), "Output mode differs"
        return design
    prepared = json.loads((PREP/"PREPARATION.json").read_text())
    for name, digest in prepared["file_sha256"].items():
        assert runner.file_hash(PREP/name) == digest, name
    documents = json.loads((PREP/"DATASET_MANIFEST.json").read_text())
    audits = json.loads((PREP/"PROMPT_AUDIT.json").read_text())
    jobs = prepared["jobs"]
    if not smoke:
        verify_matched_inputs([resolve(ROOT/j["config"]) for j in jobs])
    if smoke:
        jobs = [next(j for j in jobs if j["model"]=="qwen_instruct" and j["top_k"]==128)]
        dev = json.loads((ROOT/"reports/review_dataset_quality_20261023/DATASET_MANIFEST.json").read_text())
        doc = next(d for d in dev["documents"] if d["corpus"]=="douban_40k")
        documents = dict(documents=[doc], scope="Development-only plumbing smoke; never main-table data")
        audits = dict(rows=[dict(r,model="qwen_instruct") for r in json.loads((ROOT/"reports/review_dataset_quality_20261023/PROMPT_AUDIT.json").read_text())["rows"] if r["prompt_id"]==doc["id"]])
    out.mkdir(parents=True, exist_ok=True)
    runner.write(out/"DATASET_MANIFEST.json", documents)
    runner.write(out/"PROMPT_AUDIT.json", audits)
    shards, semantic_cases = [], set()
    for job in jobs:
        parent = ROOT/job["config"]
        assert runner.file_hash(parent)==job["config_sha256"]
        base = read_tree(parent)
        if smoke:
            base["inputs"]["prompts"] = [dict(id=doc["id"], language="zh", text=doc["user_message"], assistant_prefix=doc["prefix"])]
            base["variants"] = [v for v in base["variants"] if v["id"] in ("ordinary", "las_imec", "adaptive_1000_imec")]
            base["stopping"]["max_new_tokens"] = 16
            base["randomness"]["experiment_id"] = "review-main-driver-smoke-20261023"
        for placement in shard_slices(len(base["inputs"]["prompts"]),job["top_k"]):
            start,stop=placement["prompt_start"],placement["prompt_stop"]
            config = copy.deepcopy(base)
            config["name"] = ("smoke" if smoke else base["name"])+f"_b{placement['batch_index']+1:02d}_p{start:02d}"
            config["inputs"]["prompts"] = base["inputs"]["prompts"][start:stop]
            path = out/"configs"/(config["name"]+".toml")
            write_config(path, config)
            resolved = resolve(path)
            expanded = list(cases(resolved))
            if not smoke:
                verify_slice(resolve(parent),resolved)
            for c in expanded:
                identity = (job["top_k"], c["id"])
                assert identity not in semantic_cases
                semantic_cases.add(identity)
            shards.append(dict(name=config["name"], config=str(path.relative_to(ROOT)),
                config_sha256=runner.file_hash(path), parent_config=job["config"],
                **placement,prompt_ids=[p["id"] for p in config["inputs"]["prompts"]],
                model=job["model"], top_k=job["top_k"], cases=len(expanded),
                case_ids=[c["id"] for c in expanded],
                stego_cases=sum(c["variant"]["kind"]!="ordinary" for c in expanded)))
    if not smoke:
        expected={(j["top_k"],c["id"]) for j in jobs for c in cases(resolve(ROOT/j["config"]))}
        assert semantic_cases==expected and len(expected)==prepared["generated_cases"]
        assert sum(j["stego_cases"] for j in shards)==prepared["independent_receivers"]
    # Alternate languages and k at each document chunk, so partial accounting
    # does not silently privilege one model's whole matrix.
    shards.sort(key=lambda j:(j["batch_index"], j["comparison_block"], [16,32,128,256,0].index(j["top_k"]), j["model"],j["part_index"]))
    limits = dict(LIMITS)
    runner.write(out/"DESIGN.json", dict(mode="smoke" if smoke else "main", frozen_before_generation=True,
        shards=shards, cases=sum(j["cases"] for j in shards), stego_cases=sum(j["stego_cases"] for j in shards),
        batch_size=BATCH_SIZE,batches=batch_plan(shards),
        matched_inputs_across_k=not smoke,source_preparation=str(PREP.relative_to(ROOT)),
        limits=limits, source_preparation_sha256=runner.file_hash(PREP/"PREPARATION.json"),
        runtime_code=runner.code_state(), driver_code=code_inputs(),
        input_sha256={name:runner.file_hash(out/name) for name in ("DATASET_MANIFEST.json", "PROMPT_AUDIT.json")},
        resume_policy="Keep every completed success/error; resume only missing records; record cumulative wall time without a time limit",
        resource_policy="Monitoring only; no automatic wall-time, case-time, RSS, swap, log or disk stop; explicit user stop remains supported",
        shard_order="Complete every condition on documents 1-10, then 11-20, 21-30, 31-32; within each batch use common four-document blocks, k and model; semantic random streams unchanged",
        scoring="Recorded native IDs; enriched existing scorer; raw full-vocabulary PPL and processed-source entropy",
        status="Prepared, no model generation launched"))
    return preflight(out)


def preflight(out):
    design = runner.read(out/"DESIGN.json")
    assert design["runtime_code"]==runner.code_state(), "Runtime code changed"
    assert design["driver_code"]==code_inputs(), "Driver/scoring code changed"
    assert design["source_preparation_sha256"]==runner.file_hash(PREP/"PREPARATION.json")
    for name, digest in design["input_sha256"].items():
        assert runner.file_hash(out/name)==digest, name
    semantic=set()
    assert design["batches"]==batch_plan(design["shards"])
    assert [j["batch_index"] for j in design["shards"]]==sorted(j["batch_index"] for j in design["shards"])
    if design["mode"]=="main":
        parents=sorted({j["parent_config"] for j in design["shards"]})
        verify_matched_inputs([resolve(ROOT/p) for p in parents])
    for job in design["shards"]:
        path=ROOT/job["config"]
        assert runner.file_hash(path)==job["config_sha256"]
        expanded=list(cases(resolve(path)))
        prompts=resolve(path)["inputs"]["prompts"]
        assert [p["id"] for p in prompts]==job["prompt_ids"]
        if design["mode"]=="main":
            parent=resolve(ROOT/job["parent_config"])
            assert prompts==parent["inputs"]["prompts"][job["prompt_start"]:job["prompt_stop"]]
            assert job["prompt_start"]//BATCH_SIZE==job["batch_index"]==(job["prompt_stop"]-1)//BATCH_SIZE
            verify_slice(parent,resolve(path))
        assert [c["id"] for c in expanded]==job["case_ids"]
        for c in expanded:
            key=(job["top_k"],c["id"])
            assert key not in semantic
            semantic.add(key)
    assert len(semantic)==design["cases"]
    print(json.dumps(dict(status="preflight_passed_no_model_loaded", mode=design["mode"],
        shards=len(design["shards"]), cases=design["cases"], independent_receivers=design["stego_cases"])), flush=True)
    return design


def checkpoint_batches(out, design):
    for batch in design["batches"]:
        markers=[out/"runs"/name/"COMPLETE.json" for name in batch["shard_names"]]
        if not all(p.exists() for p in markers):
            break
        path=out/"batches"/f"batch_{batch['batch_index']+1:02d}"/"COMPLETE.json"
        hashes={str(p.relative_to(out)):runner.file_hash(p) for p in markers}
        if path.exists():
            assert runner.read(path)["shard_completion_sha256"]==hashes
        else:
            runner.write(path,dict(batch_index=batch["batch_index"],cases=batch["cases"],
                stego_cases=batch["stego_cases"],finished_unix=time.time(),
                shard_completion_sha256=hashes,scope="All planned observations processed; failures retained"))


def missing_cases(config, directory, role, execution):
    """Do not turn a recorded failure into an unreported retry on resume."""
    pending=[]
    for case in runner.ordered_cases(config):
        if role=="receiver" and case["variant"]["kind"]=="ordinary":
            continue
        path=directory/role/(case["id"]+".json")
        if path.exists() or path.with_suffix(".json.gz").exists():
            record=read_record(path)
            assert record["case_id"]==case["id"] and record["stage"]==role
            assert record["execution_sha256"]==fingerprint(execution)
            if role=="receiver" and record.get("input_sha256"):
                assert record["input_sha256"]==runner.file_hash(directory/"public"/(case["id"]+".json"))
        else:
            pending.append(case)
    return pending


def work(out, directory, model, role):
    import tools.review_budget_pilot_20261023 as bounded
    manifest=runner.read(directory/"manifest.json")
    config=manifest["configuration"]
    pending=missing_cases(config,directory,role,manifest["execution"])
    # Only the local worker's traversal is restricted. Core files stay intact.
    runner.ordered_cases=lambda _:pending
    bounded.OUT=out
    limits=runner.read(out/"DESIGN.json")["limits"]
    bounded.LIMITS=dict(limits, per_case_stage_seconds=limits["full_case_seconds"] if config["source"]["sampling"]["top_k"]==0 else limits["truncated_case_seconds"])
    bounded.bounded_worker(directory,model,role)


def score(out, directory):
    import torch
    from stegatext_runtime.experiments.native_ppl import METRIC
    from stegatext_runtime.prompts import configure_prompt
    from tools.expanded_native_ppl_20261021 import enriched_score
    torch.set_num_threads(1)
    output=directory/"ppl"
    output.mkdir(exist_ok=True)
    config=runner.read(directory/"manifest.json")["configuration"]
    previous=runner.read(output/"scores.json")["rows"] if (output/"scores.json").exists() else []
    saved={r["case_id"]:r for r in previous}
    runner.write(output/"contract.json", dict(metric=METRIC, retokenization=False,
        scorer="tools.expanded_native_ppl_20261021.enriched_score", eos="body primary, natural EOS inclusive secondary; no synthetic boundary",
        conditioning="actual saved prompt and preceding native output IDs", logits="raw full vocabulary"))
    source=None
    seconds=runner.read(out/"DESIGN.json")["limits"]["score_case_seconds"]
    with (ROOT/".cache/experiment_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in runner.read(directory/"results.json")["rows"]:
            path=directory/"sender"/(row["case_id"]+".json")
            digest=raw_digest(path)
            if row["case_id"] in saved:
                assert saved[row["case_id"]]["sender_record_sha256"]==digest
                continue
            sender=read_record(path)
            item=dict(case_id=row["case_id"], sender_record_sha256=digest, model=row["model"],
                variant=row["variant"], seed=row["seed"], prompt_id=row["prompt_id"], metric=METRIC)
            active=dict(kind="case", stage="score", status="running", job=directory.name,
                case_id=row["case_id"], started_unix=time.time(), top_k=config["source"]["sampling"]["top_k"])
            runner.write(out/"ACTIVE.json",active)
            score_started=time.monotonic()
            def expired(_signal, _frame):
                raise TimeoutError(f"PPL case exceeded {seconds} seconds")
            old=signal.signal(signal.SIGALRM,expired)
            signal.setitimer(signal.ITIMER_REAL,seconds)
            try:
                seq=sender.get("result",{}).get("native_token_sequence")
                if sender["status"]!="ok" or not seq or seq.get("status")!="verified":
                    item.update(evaluator_status="unavailable_no_verified_native_sequence", nll_total=None, evaluator_tokens=0, ppl=None)
                else:
                    if source is None:
                        source=runner.load_source(config,row["model"])
                    prompt=next(p for p in config["inputs"]["prompts"] if p["id"]==row["prompt_id"])
                    configure_prompt(source,prompt.get("assistant_prefix"))
                    item.update(enriched_score(source,seq), native_sequence_sha256=fingerprint(seq))
            except Exception as exc:
                item.update(evaluator_status="evaluation_error", error=f"{type(exc).__name__}: {exc}", nll_total=None, evaluator_tokens=0, ppl=None)
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                signal.signal(signal.SIGALRM,old)
                active.update(status="finished", finished_unix=time.time())
                runner.write(out/"ACTIVE.json",active)
                item["scorer_wall_seconds"]=time.monotonic()-score_started
            previous.append(item)
            runner.write(output/"scores.json",dict(metric=METRIC,rows=previous))
            print(json.dumps(dict(scored=len(previous),case_id=row["case_id"],status=item["evaluator_status"])),flush=True)


def execute(out):
    design=preflight(out)
    execution=dict(environment=runner.environment(),code=runner.code_state())
    checkpoint_batches(out,design)
    for job in design["shards"]:
        directory=out/"runs"/job["name"]
        if (directory/"COMPLETE.json").exists():
            marker=runner.read(directory/"COMPLETE.json")
            for name,digest in marker["sha256"].items():
                assert runner.file_hash(directory/name)==digest
            continue
        directory.mkdir(parents=True,exist_ok=True)
        config=resolve(ROOT/job["config"])
        bound=dict(execution,configuration_sha256=fingerprint(config))
        if (directory/"manifest.json").exists():
            manifest=runner.read(directory/"manifest.json")
            assert manifest["execution"]==bound and manifest["configuration"]==config
        else:
            runner.write(directory/"manifest.json",dict(configuration=config,execution=bound,
                plan=runner.planning(config),main_design_sha256=runner.file_hash(out/"DESIGN.json")))
        state=runner.read(directory/"STAGES.json") if (directory/"STAGES.json").exists() else {}
        for role in ("sender","receiver"):
            if state.get(role):
                continue
            stage_started=time.monotonic()
            with (ROOT/".cache/experiment_gpu0.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with (directory/(role+".log")).open("a") as log:
                    subprocess.run([sys.executable,"-B","-m","tools.review_main_run_20261023","worker",
                        "--output",str(out),"--directory",str(directory),"--model",job["model"],"--role",role],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert not missing_cases(config,directory,role,bound)
            state[role]=True
            state[role+"_wall_seconds"]=time.monotonic()-stage_started
            runner.write(directory/"STAGES.json",state)
        if not state.get("summary"):
            runner.summarize(directory)
            state["summary"]=True
            runner.write(directory/"STAGES.json",state)
        if not state.get("score"):
            stage_started=time.monotonic()
            with (directory/"score.log").open("a") as log:
                subprocess.run([sys.executable,"-B","-m","tools.review_main_run_20261023","score",
                    "--output",str(out),"--directory",str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert {r["case_id"] for r in runner.read(directory/"ppl/scores.json")["rows"]}==set(job["case_ids"])
            state["score"]=True
            state["score_wall_seconds"]=time.monotonic()-stage_started
            runner.write(directory/"STAGES.json",state)
        from tools.review_main_report_20261023 import collect
        runner.write(directory/"metrics.json",dict(rows=collect(out,directory,job)))
        archive(directory)
        runner.write(directory/"COMPLETE.json",dict(cases=job["cases"],sha256={
            name:runner.file_hash(directory/name) for name in ("manifest.json","results.json","ppl/scores.json","TRACE_ARCHIVE.json","metrics.json")}))
        checkpoint_batches(out,design)
        from tools.review_main_report_20261023 import analyze
        analyze(out)
    runner.write(out/"COMPLETE.json",dict(planned_cases=design["cases"],finished_unix=time.time()))
    checkpoint_batches(out,design)
    from tools.review_main_report_20261023 import analyze
    analyze(out)


def requested_stop_reasons(out):
    return ["explicit_stop"] if (out/"STOP_REQUESTED.json").exists() else []


def guard(out):
    import psutil
    design=preflight(out)
    with (out/"guard.lock").open("a") as guard_lock:
        fcntl.flock(guard_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempts=out/"attempts"
        attempts.mkdir(exist_ok=True)
        records=sorted(attempts.glob("*/RESOURCES.json"))
        prior=sum(runner.read(p)["elapsed_seconds"] for p in records)
        if (out/"STOP_REQUESTED.json").exists():
            raise RuntimeError("STOP_REQUESTED.json is present")
        attempt=attempts/f"{len(records):03d}"
        attempt.mkdir(exist_ok=False)
        started=time.monotonic()
        initial_swap=psutil.swap_memory().used
        state=dict(status="running",elapsed_seconds=0.,previous_execution_seconds=prior,
            started_unix=time.time(),peak_rss_gib=0.,maximum_artifact_mib=0.,maximum_swap_growth_gib=0.)
        runner.write(out/"ACTIVE.json",dict(kind="driver",status="starting",started_unix=time.time()))
        with (attempt/"driver.log").open("x") as log:
            proc=subprocess.Popen([sys.executable,"-B","-m","tools.review_main_run_20261023","execute","--output",str(out)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            def cleanup():
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid,signal.SIGTERM)
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait()
                    except ProcessLookupError:
                        pass
            atexit.register(cleanup)
            state.update(driver_pid=proc.pid,guard_pid=os.getpid())
            try:
                with (attempt/"resources.jsonl").open("x") as history:
                    while proc.poll() is None:
                        family=[psutil.Process(os.getpid())]+psutil.Process(os.getpid()).children(recursive=True)
                        rss=0
                        for process in family:
                            try: rss+=process.memory_info().rss
                            except psutil.Error: pass
                        sizes=[];large=large_log=0
                        for path in out.rglob("*"):
                            try:
                                if not path.is_file(): continue
                                size=path.stat().st_size
                            except FileNotFoundError: continue
                            sizes.append(size)
                            if path.parent.name in ("sender","receiver") and path.suffix==".json":large=max(large,size)
                            if path.suffix==".log":large_log=max(large_log,size)
                        live=dict(unix=time.time(),elapsed_seconds=time.monotonic()-started,rss_gib=rss/2**30,
                            available_gib=psutil.virtual_memory().available/2**30,
                            swap_growth_gib=max(0,psutil.swap_memory().used-initial_swap)/2**30,
                            artifact_mib=sum(sizes)/2**20,largest_record_mib=large/2**20,largest_log_mib=large_log/2**20,
                            disk_free_gib=shutil.disk_usage(out).free/2**30,
                            active=runner.read(out/"ACTIVE.json") if (out/"ACTIVE.json").exists() else {})
                        state.update(elapsed_seconds=live["elapsed_seconds"],peak_rss_gib=max(state["peak_rss_gib"],live["rss_gib"]),
                            maximum_artifact_mib=max(state["maximum_artifact_mib"],live["artifact_mib"]),
                            maximum_swap_growth_gib=max(state["maximum_swap_growth_gib"],live["swap_growth_gib"]))
                        runner.write(attempt/"RESOURCES.json",state)
                        runner.write(out/"RESOURCES.json",state)
                        runner.write(out/"LIVE.json",live)
                        history.write(json.dumps(live)+"\n");history.flush()
                        reasons=requested_stop_reasons(out)
                        if reasons:
                            state.update(status="stopped",reasons=reasons)
                            cleanup()
                            break
                        time.sleep(5)
                code=proc.wait()
                state.update(returncode=code,status=state["status"] if state["status"]!="running" else "finished" if code==0 else "failed")
            finally:
                cleanup()
                state.update(elapsed_seconds=time.monotonic()-started,finished_unix=time.time())
                runner.write(attempt/"RESOURCES.json",state)
                runner.write(out/"RESOURCES.json",state)
                from tools.review_main_report_20261023 import analyze
                analyze(out)
        print(json.dumps(state),flush=True)
        if state["status"]!="finished":raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","prepare-smoke","preflight","run","execute","worker","score","analyze"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--directory",type=Path)
    parser.add_argument("--model")
    parser.add_argument("--role",choices=("sender","receiver"))
    args=parser.parse_args()
    os.environ.update(THREAD_ENV)
    out=(args.output or (SMOKE_OUT if args.phase=="prepare-smoke" else DEFAULT_OUT)).resolve()
    out.relative_to(ROOT)
    if args.phase.startswith("prepare"):
        prepare(out,smoke=args.phase=="prepare-smoke")
    elif args.phase=="preflight":preflight(out)
    elif args.phase=="run":guard(out)
    elif args.phase=="execute":execute(out)
    elif args.phase=="worker":work(out,args.directory,args.model,args.role)
    elif args.phase=="score":score(out,args.directory)
    else:
        from tools.review_main_report_20261023 import analyze
        analyze(out)


if __name__=="__main__":
    main()
