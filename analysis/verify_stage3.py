"""Recompute stage-three evidence from existing records, CPU only.

No historical main() is invoked. Writes only this study directory. Model
likelihoods are read from completed scores, never generated or re-scored.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from tools import independent_work_audit_20261029 as previous
from tools.expanded_records_20261021 import read_record
from research_notes.capacity_gap_20261026.run_io import read as sealed_read
from research_notes.capacity_gap_redesign_20261026.metrics import balanced_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INPUT_HASHES = {}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(2**20):
            h.update(block)
    return h.hexdigest()


def track(path):
    value = file_hash(path)
    INPUT_HASHES[str(path.relative_to(ROOT))] = value
    return value


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    track(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=2e-10, abs_tol=1e-8), (a, b)


def aggregate(rows):
    """Independent formulas for the key published statistics."""
    nbytes = sum(r["bytes"] for r in rows)
    old = sum(r["original_bytes"] for r in rows)
    bits = sum(r.get("strict_verified_bits") or 0 for r in rows)
    text_bits = sum((r.get("strict_verified_bits") or 0) for r in rows if r["text_renderable"])
    seconds = math.fsum((r.get("sender_seconds") or 0) + (r.get("receiver_seconds") or 0) for r in rows)
    scored = [r for r in rows if r.get("nll_total") is not None and (r.get("evaluator_tokens") or 0) > 0]
    ntokens = sum(r["evaluator_tokens"] for r in scored)
    total_nll = math.fsum(r["nll_total"] for r in scored)
    stego = rows[0]["method"] != "ordinary"
    return dict(n=len(rows), scored=len(scored), exact=sum(r.get("exact") is True for r in rows),
                receiver_planned=sum(r["method"] != "ordinary" for r in rows),
                sender_failures=sum(r["sender_status"] != "ok" for r in rows),
                unrenderable=sum(r["sender_status"] == "ok" and not r["text_renderable"] for r in rows),
                empty_body=sum(r["scorer_status"] == "empty_generated_body" for r in rows),
                bytes=nbytes, original_bytes=old, verified_bits=bits,
                byte_bpb=bits/nbytes if nbytes and stego else None,
                valid_text_bpb=text_bits/nbytes if nbytes and stego else None,
                original_bpb=bits/old if old and stego else None,
                sender_receiver_ms_per_byte=1000*seconds/nbytes if nbytes else None,
                verified_bits_per_sender_receiver_sec=bits/seconds if seconds else None,
                corpus_ppl=math.exp(total_nll/ntokens) if ntokens else None)


def frozen_inputs(rows, recorded_code):
    locations, configs, conditions, references = {}, [], {}, {}
    prompts = defaultdict(set)
    for batch, base in previous.SOURCES.items():
        design = sealed_read(base / "DESIGN.json")
        assert sealed_read(base / "RESOURCES.json")["status"] == "finished"
        assert sealed_read(base / "COMPLETE.json")["planned_cases"] == design["cases"]
        for name in ("DESIGN.json", "DATASET_MANIFEST.json", "COMPLETE.json", "RESOURCES.json"):
            track(base / name)
        for shard in design["shards"]:
            directory = base / "runs" / shard["name"]
            manifest_path = directory / "manifest.json"
            manifest = sealed_read(manifest_path)
            track(manifest_path)
            c = manifest["configuration"]
            model = shard["model"]
            assert c["source"]["sampling"]["temperature"] == c["source"]["sampling"]["top_p"] == 1
            assert c["source"]["sampling"]["top_k"] == shard["top_k"]
            assert c["stopping"]["mode"] == "token_budget" and c["stopping"]["max_new_tokens"] == 256
            assert c["message"]["delivery"] == "prefix_stream"
            assert c["randomness"]["stream_policy"] == "independent_cases_v1"
            assert c["source"]["kv_cache"] == "cpu_snapshots"
            assert c["source"]["languages"][model] == ("en" if model == "llama_instruct" else "zh")
            condition = dict(source=c["source"], stopping=c["stopping"], message=c["message"])
            coord = model, shard["top_k"]
            if coord in conditions:
                assert condition == conditions[coord]
            else:
                conditions[coord] = condition
            for variant in c["variants"]:
                if variant["kind"] == "ordinary":
                    continue
                assert variant["coder"] == "imec_prefix_exact"
                assert variant["coder_parameters"] == dict(block_size=6, window_size=3,
                                                          coupling_strategy="greedy", schedule="head_first")
                if variant["method"] == "syncpool":
                    assert variant["implementation"] == "shared_planner_v1"
                if shard["top_k"] == 0:
                    assert variant["method"] != "stepwise_chat_v1"
            for prompt in c["inputs"]["prompts"]:
                assert prompt["language"] == c["source"]["languages"][model] and prompt["assistant_prefix"]
                prompts[model].add(prompt["id"])
            for cid in shard["case_ids"]:
                locations[batch, shard["top_k"], cid] = directory
            configs.append((batch, model, shard["top_k"]))
            references[model] = c["model_revisions"][model]
    assert len(configs) == 352 and all(len(v) == 32 for v in prompts.values())
    current, missing, changed = 0, [], []
    for name, expected in recorded_code.items():
        p = ROOT / name
        if not p.exists():
            missing.append(name)
        elif file_hash(p) != expected:
            changed.append(name)
        else:
            current += 1
    manifest = sealed_read(previous.SOURCES["main"] / "DATASET_MANIFEST.json")
    return locations, dict(shards=len(configs), prompts_per_model={m:len(v) for m,v in prompts.items()},
        code_files_per_snapshot=len(recorded_code), current_code_matches=current,
        current_code_missing=missing, current_code_differences=changed,
        source_models={m:{k:s[k] for k in ("repo_id", "revision", "tokenizer_revision", "dtype")} for m,s in references.items()},
        dataset_sources=[{k:s.get(k) for k in ("dataset", "revision", "language", "split", "sha256")} for s in manifest["sources"]],
        document_selection=manifest["selection_rule"], source_scope=manifest["scope"],
        input_message_policy="Same random-input law, method-separated actual messages and protocol randomness; same seed is a replicate label")


def cell_recompute(rows):
    buckets = defaultdict(list)
    for r in rows:
        for view in ("pooled", r["source_batch"]):
            buckets[view, r["model"], r["top_k"], r["variant"]].append(r)
    calculated = [dict(view=v, model=m, top_k=k, variant=a, **aggregate(part))
                  for (v,m,k,a),part in sorted(buckets.items())]
    existing = {(r["view"],r["model"],int(r["top_k"]),r["variant"]):r for r in
                csv_read(ROOT / "reports/independent_work_audit_20261029/CORRECTED_CELLS.csv")}
    count = 0
    for r in calculated:
        old = existing[r["view"],r["model"],r["top_k"],r["variant"]]
        for field,value in r.items():
            if field in ("view", "model", "top_k", "variant"):
                continue
            if value is None:
                assert old[field] == ""
            else:
                close(value, old[field])
            count += 1
    assert len(calculated) == len(existing) == 308
    write_csv("RECOMPUTED_CELLS.csv", calculated)
    return buckets, calculated, count


def deep_case(row, locations):
    directory = locations[row["source_batch"],row["top_k"],row["case_id"]]
    sender = read_record(directory / "sender" / (row["case_id"] + ".json"))
    raw = previous.body_bytes(sender["result"])
    assert sender["status"] == "ok" and len(raw) == row["bytes"]
    if row["method"] == "ordinary":
        return dict(batch=row["source_batch"],case_id=row["case_id"],model=row["model"],top_k=row["top_k"],
                    bytes=len(raw),confirmed_bits=None,valid_utf8=sender["result"]["text"] is not None,
                    public_input_hash_verified=None,sender_receiver_equal=None,
                    role="ordinary; no steganographic receiver is planned")
    receiver = read_record(directory / "receiver" / (row["case_id"] + ".json"))
    public_path = directory / "public" / (row["case_id"] + ".json")
    assert sender["status"] == receiver["status"] == "ok"
    assert sender["pid"] != receiver["pid"]
    public_hash = track(public_path)
    assert sender["public_sha256"] == receiver["input_sha256"] == public_hash
    a,b = sender["result"], receiver["result"]
    assert all(a[k] == b[k] for k in ("bits", "trace", "text", "state_sha256", "coder_state", "native_token_sequence"))
    assert len(raw) == row["bytes"] and len(a["bits"]) == row["strict_verified_bits"]
    return dict(batch=row["source_batch"],case_id=row["case_id"],model=row["model"],top_k=row["top_k"],
                bytes=len(raw),confirmed_bits=len(a["bits"]),valid_utf8=a["text"] is not None,
                public_input_hash_verified=True,sender_receiver_equal=True)


def anomalies(rows, fixes, locations):
    bad = [r for r in rows if r["sender_status"] == "ok" and not r["text_renderable"]]
    checked = [deep_case(r, locations) for r in bad]
    assert len(bad) == len(fixes) == 17
    assert sum(f["corrected_bytes"] - f["original_bytes"] for f in fixes) == 19937
    examples = []
    for model in ("llama_instruct", "qwen_instruct"):
        candidates = sorted([r for r in rows if r["source_batch"] == "main" and r["model"] == model
                             and r["top_k"] == 0 and r["variant"] == "las_imec" and r["bytes"] > 0],key=lambda r:r["case_id"])
        examples.append(deep_case(candidates[0], locations))
    failures = []
    for r in rows:
        if r["sender_status"] == "ok":
            continue
        d = locations[r["source_batch"],r["top_k"],r["case_id"]]
        s = read_record(d / "sender" / (r["case_id"] + ".json"))
        assert s["status"] == "error"
        failures.append(dict(batch=r["source_batch"],case_id=r["case_id"],error=s["error"],receiver_status=r["receiver_status"]))
    return dict(invalid_utf8_cases_rechecked=checked, selected_valid_examples=examples, failed_senders=failures,
                invalid_suffix_only=sum(f["incomplete_utf8_suffix_only"] for f in fixes),
                byte_correction_total=19937,
                zero_payload_successes=sum(r["method"] != "ordinary" and r.get("exact") is True
                                           and not (r.get("strict_verified_bits") or 0) for r in rows))


def pure_ledger():
    old_cases = csv_read(ROOT / "reports/paper_rethink_20261029/PURE_LAS_TRACE_CASES.csv")
    output = []
    for index, old in enumerate(old_cases, 1):
        path = ROOT / old["sender_record"]
        assert track(path) == old["sender_file_sha256"]
        # The archival reader takes the original .json path and locates .gz.
        original_path = path.with_suffix("") if path.suffix == ".gz" else path
        result = read_record(original_path)["result"]
        identity, group, expected, realized_increments = [], [], [], []
        for step in result["trace"]:
            value = step["selected_group_exact_mass"] / step["selected_group_mass"] * step["exact_identity_entropy_bits"]
            close(value, step["exact_identity_information_loss_bits"])
            identity.append(max(0., value))
            group.append(step["group_entropy_bits"])
            expected.append(step["coder"]["expected_information_bits"])
            realized_increments.append(step["coder"]["posterior_information_revealed_bits"])
        state = result["coder_state"]
        information = state["head_block"] * state["block_bits"] + state["block_bits"] * state["window_blocks"] - math.fsum(state["entropies"])
        close(information, math.fsum(realized_increments))
        bits = len(result["bits"])
        pending = information - bits
        row = {k:old[k] for k in ("source_batch", "model", "prompt_id", "case_id")}
        row.update(top_k=int(old["top_k"]),seed=int(old["seed"]),
                   identity_upper_bits=math.fsum(identity),group_entropy_bits=math.fsum(group),
                   expected_coder_bits=math.fsum(expected),realized_information_bits=information,
                   verified_bits=bits,pending_information_bits=pending)
        for field in ("identity_upper_bits","group_entropy_bits","expected_coder_bits","realized_information_bits","verified_bits","pending_information_bits"):
            close(row[field],old[field])
        output.append(row)
        if index % 32 == 0:
            print(f"Pure LAS traces checked: {index}/{len(old_cases)}",flush=True)
    assert len(output) == 704
    buckets = defaultdict(list)
    for row in output:
        buckets[row["model"],row["top_k"]].append(row)
    cells = []
    for (model,k),part in sorted(buckets.items()):
        repeat_counts = Counter(r["prompt_id"] for r in part)
        assert len(repeat_counts) == 32 and len(set(repeat_counts.values())) == 1
        fields = ("identity_upper_bits","group_entropy_bits","expected_coder_bits","realized_information_bits","verified_bits","pending_information_bits")
        sums = {f:math.fsum(r[f] for r in part) for f in fields}
        h = sums["group_entropy_bits"]
        cells.append(dict(model=model,top_k=k,n=len(part),prompts=32,**sums,
                          identity_upper_over_group=sums["identity_upper_bits"]/h,
                          expected_coder_over_group=sums["expected_coder_bits"]/h,
                          verified_over_group=sums["verified_bits"]/h))
    old_cells = {(r["model"],int(r["top_k"])):r for r in csv_read(ROOT / "reports/paper_rethink_20261029/PURE_LAS_LEDGER_CELLS.csv") if r["view"] == "pooled"}
    for row in cells:
        for field in ("identity_upper_over_group","expected_coder_over_group","verified_over_group"):
            close(row[field],old_cells[row["model"],row["top_k"]][field])
    write_csv("PURE_LAS_RECOMPUTED_CASES.csv",output)
    write_csv("PURE_LAS_RECOMPUTED_CELLS.csv",cells)
    return dict(traces=704,cells=cells,scope="All recorded pure-LAS sender traces reread and sealed hashes checked; no model likelihoods recomputed")


def bootstrap_arrays(rows, prompts, index):
    fields = ("strict_verified_bits","bytes","sender_seconds","receiver_seconds","nll_total","evaluator_tokens")
    totals = {p:defaultdict(float) for p in prompts}
    for r in rows:
        for f in fields:
            totals[r["prompt_id"]][f] += r.get(f) or 0
    arrays = {f:np.asarray([totals[p][f] for p in prompts]) for f in fields}
    sampled = {f:arr[index].sum(axis=1) for f,arr in arrays.items()}
    with np.errstate(divide="ignore",invalid="ignore"):
        return dict(bpb=sampled["strict_verified_bits"]/sampled["bytes"],
                    ms=1000*(sampled["sender_seconds"]+sampled["receiver_seconds"])/sampled["bytes"],
                    nll=sampled["nll_total"]/sampled["evaluator_tokens"])


def interval(values):
    finite = values[np.isfinite(values)]
    return [float(v) for v in np.quantile(finite,[.025,.975])]


def matched_contrast(rows,view,model,k,variant,baseline):
    selected = [r for r in rows if r["model"]==model and r["top_k"]==k and
                (view=="pooled" or (view=="confirmation" and r["source_batch"]!="main") or r["source_batch"]==view)]
    arm = {(r["prompt_id"],r["seed"]):r for r in selected if r["variant"]==variant}
    ref = {(r["prompt_id"],r["seed"]):r for r in selected if r["variant"]==baseline}
    coords = sorted(set(arm)&set(ref))
    a,b = [arm[c] for c in coords],[ref[c] for c in coords]
    prompts = sorted({p for p,s in coords})
    seed = int(hashlib.sha256(repr((view,model,k,variant,baseline)).encode()).hexdigest()[:8],16)
    index=np.random.default_rng(seed).integers(0,len(prompts),size=(10000,len(prompts)))
    aa,bb = bootstrap_arrays(a,prompts,index),bootstrap_arrays(b,prompts,index)
    av,bv=aggregate(a),aggregate(b)
    capacity = av["byte_bpb"]/bv["byte_bpb"] if av["byte_bpb"] is not None and bv["byte_bpb"] else None
    capacity_ci=interval(aa["bpb"]/bb["bpb"]) if capacity is not None else [None,None]
    time_ratio=av["sender_receiver_ms_per_byte"]/bv["sender_receiver_ms_per_byte"]
    ppl_ratio=av["corpus_ppl"]/bv["corpus_ppl"]
    ppl_ci=interval(np.exp(aa["nll"]-bb["nll"]))
    return dict(view=view,model=model,top_k=k,variant=variant,baseline=baseline,matched_n=len(coords),prompts=len(prompts),
                capacity_ratio=capacity,capacity_ci95_low=capacity_ci[0],capacity_ci95_high=capacity_ci[1],
                time_per_byte_ratio=time_ratio,ppl_ratio_corpus=ppl_ratio,
                ppl_ci95_low=ppl_ci[0],ppl_ci95_high=ppl_ci[1],bootstrap_seed=seed,bootstrap_draws=10000,
                scored_arm=av["scored"],scored_control=bv["scored"])


def comparisons(rows,buckets):
    main=[]
    for model in ("llama_instruct","qwen_instruct"):
        for k in (16,32,128,256,0):
            main.append(matched_contrast(rows,"main",model,k,"las_imec","syncpool_imec"))
    selected=[]
    for model,var,base in [("llama_instruct","adaptive_0500_imec","las_imec"),
                           ("llama_instruct","adaptive_4000_imec","syncpool_imec"),
                           ("qwen_instruct","adaptive_0125_imec","las_imec"),
                           ("qwen_instruct","adaptive_1000_imec","las_imec"),
                           ("qwen_instruct","adaptive_4000_imec","syncpool_imec")]:
        selected.append(matched_contrast(rows,"pooled",model,0,var,base))
    ppl=[]
    for view in ("main","pooled","confirmation"):
        for model,k in [("llama_instruct",0),("llama_instruct",256),("qwen_instruct",0)]:
            ppl.append(matched_contrast(rows,view,model,k,"las_imec","ordinary"))
    points=[]
    for (view,model,k,variant),part in sorted(buckets.items()):
        if view!="main" or k!=0 or variant=="ordinary":
            continue
        prompts=sorted({r["prompt_id"] for r in part})
        assert len(part)==len(prompts)==32
        seed=int(hashlib.sha256((model+variant).encode()).hexdigest()[:8],16)
        index=np.random.default_rng(seed).integers(0,32,size=(10000,32))
        draws=bootstrap_arrays(part,prompts,index)
        bpb_ci=interval(draws["bpb"]);time_ci=interval(draws["ms"])
        points.append(dict(model=model,variant=variant,**aggregate(part),
                           bpb_ci95_low=bpb_ci[0],bpb_ci95_high=bpb_ci[1],
                           ms_ci95_low=time_ci[0],ms_ci95_high=time_ci[1],
                           bootstrap_seed=seed,bootstrap_draws=10000,
                           family="filtered_source" if variant in ("basic_imec","mwis_imec") else "same_source_ideal_contract"))
    write_csv("MAIN_MATCHED_LAS_SYNCPOOL.csv",main)
    write_csv("SELECTED_POOLED_MATCHED.csv",selected)
    write_csv("PPL_MAIN_POOLED_CONFIRMATION.csv",ppl)
    write_csv("FULL_MAIN_FIGURE_POINTS.csv",points)
    return dict(main=main,selected_pooled=selected,ppl=ppl,
                inference_scope="Exploratory uncorrected prompt-cluster bootstrap; paired coordinates, independent method random streams; not equivalence or population certification")


def old_figure_audit(cells):
    base=ROOT/"reports/review_main_targeted_20261025/merged_all_20261029"
    cap=csv_read(base/"MERGED_CELLS.csv");timing=csv_read(base/"TIME_BY_TAU.csv")
    time={(r["model"],int(r["top_k"]),r["variant"]):r for r in timing}
    corrected={(r["model"],r["top_k"],r["variant"]):r for r in cells if r["view"]=="pooled"}
    output=[]
    for r in cap:
        if r["source_batch"]!="merged" or not (r["variant"] in ("las_imec","syncpool_imec") or r["variant"].startswith("adaptive_")):
            continue
        key=r["model"],int(r["top_k"]),r["variant"]
        if key not in time:
            continue
        new=corrected[key]
        old_bpb=float(r["verified_bpb"]);old_ms=1000*float(time[key]["sec_per_visible_byte"])
        output.append(dict(model=key[0],top_k=key[1],variant=key[2],n=new["n"],
                           old_bpb=old_bpb,corrected_bpb=new["byte_bpb"],valid_text_bpb=new["valid_text_bpb"],
                           old_ms_per_byte=old_ms,corrected_ms_per_byte=new["sender_receiver_ms_per_byte"],
                           changed=not math.isclose(old_bpb,new["byte_bpb"],rel_tol=1e-10,abs_tol=1e-10)))
    write_csv("ACTIVE_FIGURE_POINT_AUDIT.csv",output)
    return dict(points=len(output),changed_points=sum(r["changed"] for r in output),sample_sizes=dict(Counter(r["n"] for r in output)),
                old_figure_sha256=track(base/"TIME_CAPACITY_SCAN.svg"),
                note="Source figure and old tables are preserved; only this study's table and plot are created")


def same_text():
    base=ROOT/"reports/capacity_gap_controlled_20261026"
    plan=sealed_read(base/"SAME_TEXT_1000_PLAN.json")
    manifest=sealed_read(base/"PAIRS.json")
    summary=sealed_read(base/"SAME_TEXT_1000_SUMMARY.json")
    for name in ("SAME_TEXT_1000_PLAN.json","PAIRS.json","SAME_TEXT_1000_SUMMARY.json"):
        track(base/name)
    streams=[s for s in manifest["streams"] if s["role"] in ("anchor","alternative")]
    values=defaultdict(lambda:defaultdict(list));seeds=set();counts=Counter();valid=0
    for stream in streams:
        for repeat in range(plan["draw_limits_by_state"][stream["state_id"]]):
            path=base/"samples"/stream["stream_id"]/f"r{repeat:03d}.json.gz"
            r=sealed_read(path);track(path)
            assert r["status"]=="ok" and r["role"]==stream["role"] and r["state_id"]==stream["state_id"]
            assert r["contract"]["stream_id"]==stream["stream_id"] and r["contract"]["repeat"]==repeat
            seeds.add(r["contract"]["seed"]);counts[r["model"]]+=1;valid+=bool(r["valid_utf8"])
            ll=r["prefix_log_probabilities_by_role"]["256"]
            values[stream["state_id"]][stream["role"]].append([ll["anchor"],ll["alternative"]])
    assert len(streams)==48 and len(values)==24 and len(seeds)==sum(counts.values())==1000
    receipt=sealed_read(base/"SAME_TEXT_1000_RECEIPT.json")
    for old in receipt["preserved_samples"]:
        assert file_hash(base/old["path"])==old["sha256"]
    recalculated=[]
    for pair in summary["pairs"]:
        rows=values[pair["state_id"]]
        point=[]
        for prior in (pair["analysis_prior"],[.5,.5]):
            point.append(.5*sum(statistics.mean(balanced_score(prior,ll)[0] for ll in rows[role]) for role in ("anchor","alternative")))
        old=pair["curves"]["256"]
        assert math.isclose(point[0],old["actual_prior"]["jsd_bits"],rel_tol=1e-9,abs_tol=1e-18)
        assert math.isclose(point[1],old["equal_prior"]["jsd_bits"],rel_tol=1e-9,abs_tol=1e-12)
        recalculated.append(dict(model=pair["model"],stratum=pair["stratum"],state_id=pair["state_id"],
                                 weighted_jsd_bits=point[0],equal_prior_jsd_bits=point[1]))
    groups=[]
    for model,stratum in sorted({(r["model"],r["stratum"]) for r in recalculated}):
        part=[r for r in recalculated if (r["model"],r["stratum"])==(model,stratum)]
        groups.append(dict(model=model,stratum=stratum,states=len(part),
                           weighted_jsd_bits=statistics.mean(r["weighted_jsd_bits"] for r in part),
                           equal_prior_jsd_bits=statistics.mean(r["equal_prior_jsd_bits"] for r in part)))
    write_csv("SAME_TEXT_256_RECOMPUTED.csv",groups)
    return dict(records=1000,states=24,history_streams=48,counts_by_model=dict(counts),valid_utf8=valid,
                retained_384_hashes=len(receipt["preserved_samples"]),groups=groups,
                scope="Recorded two-history 256-token-window likelihoods; estimator recomputed, model cross likelihoods not re-scored")


def main():
    print("Reading hash-checked shard metrics and frozen configurations",flush=True)
    rows,fixes,hashes,code=previous.collect()
    INPUT_HASHES.update(hashes)
    locations,config=frozen_inputs(rows,code)
    buckets,cells,cell_checks=cell_recompute(rows)
    anomalous=anomalies(rows,fixes,locations)
    write_csv("BYTE_CORRECTIONS_RECHECKED.csv",fixes)
    comparisons_result=comparisons(rows,buckets)
    figure=old_figure_audit(cells)
    print("Checking the 1000 planned same-text records",flush=True)
    same=same_text()
    ledger=pure_ledger()
    with gzip.open(OUT/"CORRECTED_CASES.json.gz","wt") as stream:
        json.dump(rows,stream,ensure_ascii=False,allow_nan=False)
    overall=dict(records=len(rows),source_batches=dict(Counter(r["source_batch"] for r in rows)),
                 successful_senders=sum(r["sender_status"]=="ok" for r in rows),
                 planned_receivers=sum(r["method"]!="ordinary" for r in rows),
                 exact=sum(r.get("exact") is True for r in rows),
                 scorer_statuses=dict(Counter(r["scorer_status"] for r in rows)),
                 empty_bodies=sum(r["scorer_status"]=="empty_generated_body" for r in rows),
                 independent_recomputed_cell_fields=cell_checks)
    assert overall["records"]==6080 and overall["exact"]==5375 and overall["planned_receivers"]==5376
    source_files=[Path(__file__).resolve(),ROOT/"tools/independent_work_audit_20261029.py",
                  ROOT/"tools/review_main_report_20261023.py",ROOT/"stegatext_runtime/experiments/native_ppl.py"]
    for p in source_files:
        track(p)
    evidence=dict(study_date="2026-10-30",timezone="Asia/Shanghai",scope="CPU-only stage-three evidence study",
                  totals=overall,configuration=config,anomalies=anomalous,
                  comparisons=comparisons_result,active_figure=figure,same_text=same,pure_las_ledger=ledger)
    write_json("EVIDENCE.json",evidence);write_json("INPUT_HASHES.json",INPUT_HASHES)
    print(json.dumps(dict(totals=overall,shards=config["shards"],old_figure=figure,pure_las_traces=ledger["traces"]),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
