"""Small, model-free main-table metrics and completion accounting."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import statistics

from stegatext_runtime.experiments.config import ROOT, cases, fingerprint, resolve
from stegatext_runtime.experiments import runner
from tools.expanded_records_20261021 import raw_digest, read_record


def optional_record(path):
    return read_record(path) if path.exists() or path.with_suffix(".json.gz").exists() else None


def collect(out, directory, job):
    config=runner.read(directory/"manifest.json")["configuration"] if (directory/"manifest.json").exists() else resolve(ROOT/job["config"])
    docs={d["id"]:d for d in runner.read(out/"DATASET_MANIFEST.json")["documents"]}
    audit={(a["model"],a["prompt_id"]):a for a in runner.read(out/"PROMPT_AUDIT.json")["rows"]}
    results={r["case_id"]:r for r in runner.read(directory/"results.json")["rows"]} if (directory/"results.json").exists() else {}
    scores={s["case_id"]:s for s in runner.read(directory/"ppl/scores.json")["rows"]} if (directory/"ppl/scores.json").exists() else {}
    rows=[]
    for case in cases(config):
        identifier=case["id"]
        v=case["variant"]
        ordinary=v["kind"]=="ordinary"
        sender=optional_record(directory/"sender"/(identifier+".json"))
        receiver=optional_record(directory/"receiver"/(identifier+".json")) if not ordinary else None
        result=(sender or {}).get("result",{})
        summary=results.get(identifier,{})
        score=scores.get(identifier,{})
        seq=result.get("native_token_sequence",{})
        perf=(sender or {}).get("performance",{})
        rperf=(receiver or {}).get("performance",{})
        text=result.get("text")
        exact=summary.get("exact")
        record=dict(case_id=identifier,model=case["model"],top_k=job["top_k"],prompt_id=case["prompt"]["id"],
            batch_index=job.get("batch_index",0),
            variant=v["id"],method=v.get("method","ordinary"),tau=v.get("tau"),seed=case["seed"],
            full_subset=docs[case["prompt"]["id"]].get("full_subset",False),
            sender_status=(sender or {}).get("status","unattempted"),
            receiver_status=(receiver or {}).get("status","unattempted") if not ordinary else None,
            error=summary.get("error") or (sender or {}).get("error") or (receiver or {}).get("error"),
            scorer_error=score.get("error"),
            exact=exact,strict_verified_bits=(summary.get("strict_verified_payload_bits",0) if exact is True else 0) if not ordinary else None,
            sender_confirmed_bits=result.get("total_confirmed_bits"),
            bytes=len(text.encode()) if text is not None else 0,body_tokens=seq.get("body_token_count"),
            natural_eos=result.get("natural_eos"),terminal_reason=result.get("terminal_reason"),
            text=text,scorer_status=score.get("evaluator_status","not_scored"),
            scorer_seconds=score.get("scorer_wall_seconds"),
            nll_total=score.get("nll_total"),evaluator_tokens=score.get("evaluator_tokens",0),ppl=score.get("ppl"),
            group_entropy_bits=result.get("group_entropy_total_bits"),
            protocol_decisions=result.get("total_protocol_decisions"),source_queries=result.get("total_model_queries"),
            retained_frontier_max=result.get("retained_frontier_max"),filtered_path_kl_total_bits=result.get("filtered_path_kl_total_bits"),
            final_visible_text_kl_status="not_estimated",
            sender_seconds=perf.get("wall_seconds",0),receiver_seconds=rperf.get("wall_seconds",0),
            actual_model_forward_calls=perf.get("actual_model_forward_calls"),peak_cuda_gib=perf.get("peak_cuda_gib"),
            expected_coder_information_bits=None,realized_information_bits=None,pending_information_bits=None,
            adaptive_las_actions=0,positive_u_sync_actions=0,synchronized_uncertainty_bits=0.,audit_errors=[])
        try:
            if sender and sender["status"]=="ok":
                assert seq["prompt_token_ids"]==audit[case["model"],case["prompt"]["id"]]["prompt_token_ids"], "prompt IDs"
                assert seq["source_token_count"]<=config["stopping"]["max_new_tokens"], "token cap"
                if score:
                    assert score["sender_record_sha256"]==raw_digest(directory/"sender"/(identifier+".json")), "PPL provenance"
                    if score.get("evaluator_status")=="scored":
                        assert score["native_sequence_sha256"]==fingerprint(seq), "native sequence hash"
                        assert score["evaluator_tokens"]==seq["body_token_count"], "PPL token count"
                if exact is True:
                    assert receiver and receiver["status"]=="ok" and sender["pid"]!=receiver["pid"], "receiver isolation"
                    assert all(result[k]==receiver["result"][k] for k in ("bits","text","trace","coder_state","native_token_sequence")), "receiver replay"
                trace=result.get("trace",[])
                if not ordinary:
                    coder=[t["coder"] for t in trace if "coder" in t]
                    assert len(coder)==len(trace), "coder trace"
                    assert all(c["selected_window_index"]==0 and c["exact_marginals"] for c in coder), "head-first exact marginals"
                    assert all(c[k]==0 for c in coder for k in ("row_normalization_error","column_normalization_error","coupling_mass_error")), "coupling marginals"
                    state=result["coder_state"]
                    information=state["head_block"]*state["block_bits"]+state["block_bits"]*state["window_blocks"]-math.fsum(state["entropies"])
                    pending=information-result["bits_count"]
                    assert math.isclose(information,math.fsum(c["posterior_information_revealed_bits"] for c in coder),abs_tol=1e-8), "information identity"
                    assert -1e-8<=pending<=18+1e-8, "pending information bound"
                    record.update(expected_coder_information_bits=math.fsum(c["expected_information_bits"] for c in coder),
                        realized_information_bits=information,pending_information_bits=pending)
                    if v["method"] in ("las","syncpool","adaptive"):
                        positive=[t for t in trace if t["kernel"]=="fullsync" and t["U_post"]>0]
                        record.update(adaptive_las_actions=sum(t["kernel"]=="las" for t in trace),
                            positive_u_sync_actions=len(positive),synchronized_uncertainty_bits=math.fsum(t["U_post"] for t in positive))
                        if v["method"]=="adaptive":
                            assert all(t["kernel"]==("las" if t["U_post"]>=v["tau"] else "fullsync") for t in trace), "adaptive threshold"
                            assert record["synchronized_uncertainty_bits"]<=v["tau"]*len(positive)+1e-8, "adaptive loss bound"
                entropy=score.get("entropy_body",{})
                record.update(conditional_entropy_nats=entropy.get("conditional_entropy_nats"),
                    surprisal_residual_nats=entropy.get("surprisal_residual_nats"),
                    sampler_normalization_tax_nats=score["nll_total"]-entropy["normalized_sampler_nll"] if entropy and score.get("nll_total") is not None else None)
        except (AssertionError,KeyError,TypeError,ValueError) as exc:
            record["audit_errors"].append(f"{type(exc).__name__}: {exc}")
        rows.append(record)
    return rows


def aggregate(rows):
    scored=[r for r in rows if r.get("nll_total") is not None and r.get("evaluator_tokens",0)>0]
    tokens=sum(r["evaluator_tokens"] for r in scored)
    nll=math.fsum(r["nll_total"] for r in scored)
    log_ppls=[r["nll_total"]/r["evaluator_tokens"] for r in scored]
    size=sum(r.get("bytes",0) for r in rows)
    bits=sum(r.get("strict_verified_bits") or 0 for r in rows)
    seconds=sum(r.get("sender_seconds",0)+r.get("receiver_seconds",0) for r in rows)
    lengths=[r["body_tokens"] for r in rows if r.get("body_tokens") is not None]
    h=math.fsum(r.get("group_entropy_bits") or 0 for r in rows)
    c=math.fsum(r.get("expected_coder_information_bits") or 0 for r in rows)
    return dict(planned=len(rows),sender_status=dict(Counter(r["sender_status"] for r in rows)),
        receiver_status=dict(Counter(r["receiver_status"] for r in rows if r.get("receiver_status") is not None)),
        exact_receivers=sum(r.get("exact") is True for r in rows),audit_errors=sum(bool(r.get("audit_errors")) for r in rows),
        scored=len(scored),nll_total=nll,evaluator_tokens=tokens,ppl=math.exp(nll/tokens) if tokens else None,
        sentence_mean_log_ppl=statistics.mean(log_ppls) if log_ppls else None,
        sentence_log_ppl_sd=statistics.stdev(log_ppls) if len(log_ppls)>1 else None,
        sentence_geometric_mean_ppl=math.exp(statistics.mean(log_ppls)) if log_ppls else None,
        strict_verified_bits=bits,bytes=size,verified_bpb=bits/size if size and rows[0]["method"]!="ordinary" else None,
        verified_bpt=bits/sum(lengths) if lengths and sum(lengths) and rows[0]["method"]!="ordinary" else None,
        sender_receiver_seconds=seconds,verified_bits_per_sender_receiver_second=bits/seconds if seconds else None,
        scorer_seconds=math.fsum(r.get("scorer_seconds") or 0 for r in rows),
        median_body_tokens=statistics.median(lengths) if lengths else None,
        below128=sum(x<128 for x in lengths),natural_eos=sum(bool(r.get("natural_eos")) for r in rows),
        group_entropy_bits=h,expected_coder_information_bits=c,coder_c_over_h=c/h if h else None,
        adaptive_las_actions=sum(r.get("adaptive_las_actions",0) for r in rows),
        positive_u_sync_actions=sum(r.get("positive_u_sync_actions",0) for r in rows),
        synchronized_uncertainty_bits=math.fsum(r.get("synchronized_uncertainty_bits",0) for r in rows))


def cell_metrics(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["model"],r["top_k"],r["variant"]].append(r)
    cells=[]
    for (model,k,variant),part in sorted(groups.items()):
        cells.append(dict(model=model,top_k=k,variant=variant,**aggregate(part)))
    controls={(c["model"],c["top_k"]):c for c in cells if c["variant"]=="ordinary"}
    las={(c["model"],c["top_k"]):c for c in cells if c["variant"]=="las_imec"}
    for cell in cells:
        control=controls.get((cell["model"],cell["top_k"]),{})
        reference=las.get((cell["model"],cell["top_k"]),{})
        # Ratios are only descriptive; do not compare differently sized partial batches.
        complete=cell["scored"]==cell["planned"]==control.get("scored")==control.get("planned")
        cell["ppl_ratio_to_ordinary"]=cell["ppl"]/control["ppl"] if complete and cell["ppl"] and control.get("ppl") else None
        attempted=all(c.get("sender_status",{}).get("unattempted",0)==0 and c.get("receiver_status",{}).get("unattempted",0)==0 for c in (cell,reference))
        cell["capacity_ratio_to_las"]=cell["verified_bpb"]/reference["verified_bpb"] if attempted and cell["verified_bpb"] is not None and reference.get("verified_bpb") and cell["planned"]==reference["planned"] else None
    return cells


def observation_summary(rows, *, complete):
    counts=Counter(r["sender_status"] for r in rows)
    attempted=sum(v for k,v in counts.items() if k!="unattempted")
    return dict(planned=len(rows),attempted_senders=attempted,successful_senders=counts["ok"],
        failed_senders=sum(v for k,v in counts.items() if k not in ("ok","unattempted")),
        not_attempted=len(rows)-attempted,
        planned_receivers=sum(r["method"]!="ordinary" for r in rows),
        exact_receivers=sum(r["exact"] is True for r in rows),
        scored=sum(r["scorer_status"]=="scored" for r in rows),
        audit_errors=sum(bool(r["audit_errors"]) for r in rows),complete=complete,
        cells=cell_metrics(rows))


def write_csv(path, rows):
    if rows:
        with path.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)


def analyze(out):
    design=runner.read(out/"DESIGN.json")
    rows=[]
    for job in design["shards"]:
        directory=out/"runs"/job["name"]
        if (directory/"COMPLETE.json").exists():
            marker=runner.read(directory/"COMPLETE.json")
            assert runner.file_hash(directory/"metrics.json")==marker["sha256"]["metrics.json"]
            part=runner.read(directory/"metrics.json")["rows"]
        else:
            part=collect(out,directory,job)
        rows.extend(dict(r,batch_index=job.get("batch_index",0)) for r in part)
    full_groups=defaultdict(list)
    for r in rows:
        if r["full_subset"]:full_groups[r["model"],r["top_k"],r["variant"]].append(r)
    common=[dict(model=m,top_k=k,variant=v,**aggregate(p)) for (m,k,v),p in sorted(full_groups.items())]
    dest=out/"analysis"
    dest.mkdir(parents=True,exist_ok=True)
    summary=dict(mode=design["mode"],**observation_summary(rows,complete=(out/"COMPLETE.json").exists()),
        full_common_subset=common,
        scope="Partial batches retain denominators; PPL is not visible-text KL; no distribution-equivalence claim; no external fact checking")
    assert summary["planned"]==design["cases"] and summary["planned_receivers"]==design["stego_cases"]
    batches=[]
    for batch in design.get("batches",[]):
        index=batch["batch_index"]
        batch_dest=dest/"batches"/f"batch_{index+1:02d}"
        batch_dest.mkdir(parents=True,exist_ok=True)
        complete=(out/"batches"/f"batch_{index+1:02d}"/"COMPLETE.json").exists()
        part=[r for r in rows if r["batch_index"]==index]
        current=dict(batch_index=index,prompt_start=batch["prompt_start"],prompt_stop=batch["prompt_stop"],
            **observation_summary(part,complete=complete))
        cumulative=observation_summary([r for r in rows if r["batch_index"]<=index],
            complete=complete and all(b["complete"] for b in batches))
        assert current["planned"]==batch["cases"] and current["planned_receivers"]==batch["stego_cases"]
        runner.write(batch_dest/"SUMMARY.json",current)
        runner.write(batch_dest/"CUMULATIVE_SUMMARY.json",cumulative)
        write_csv(batch_dest/"CELLS.csv",current["cells"])
        write_csv(batch_dest/"CUMULATIVE_CELLS.csv",cumulative["cells"])
        batches.append(current)
    summary["batches"]=batches
    runner.write(dest/"SUMMARY.json",summary)
    runner.write(dest/"CASES.json",dict(rows=rows))
    write_csv(dest/"CASES.csv",rows)
    write_csv(dest/"CELLS.csv",summary["cells"])
    batch_text="\n".join(
        f"- Batch {b['batch_index']+1} (documents {b['prompt_start']+1}-{b['prompt_stop']}): "
        f"attempted {b['attempted_senders']}/{b['planned']}, independently recovered exactly {b['exact_receivers']}/{b['planned_receivers']}, "
        f"scored {b['scored']}, batch complete: {b['complete']}." for b in batches)
    (dest/"REPORT.md").write_text(
        f"# Main comparison run summary\n\nMode: {design['mode']}. Planned {design['cases']}, attempted {summary['attempted_senders']}, successful senders {summary['successful_senders']}, failed {summary['failed_senders']}, not attempted {summary['not_attempted']}.\n\n"
        f"Independent receivers recovered exactly {summary['exact_receivers']}/{design['stego_cases']}; scored {summary['scored']}; audit errors {summary['audit_errors']}. Complete: {summary['complete']}.\n\n"
        f"{batch_text}\n\n"
        "Per-cell metrics are in CELLS.csv and SUMMARY.json; per-case data are in CASES.csv. CELLS.csv under batches/batch_XX covers that batch only, and CUMULATIVE_CELLS.csv covers all batches up to it (10/20/30/32 per cell). PPL aggregates total body NLL over total tokens. Failures, short texts, extreme cases, and unrun cases are kept. The KL of the final text is not estimated.\n")
    return summary
