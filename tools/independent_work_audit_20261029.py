"""CPU-only audit of immutable completed main/repeat experiment records.

No generation, sample deletion, or modification of frozen experiment inputs.
Read hashed shard metrics, recover zeroed UTF-8 byte denominators, and retain
both discovery and confirmation summaries. Intervals are exploratory, paired
by document rather than claims of common-random-number pairing or equivalence.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/independent_work_audit_20261029"
SOURCES = {
    "main": ROOT / "reports/review_main_batched_20261024",
    "repeat_01": ROOT / "reports/review_main_targeted_20261025/repeat_01",
    "repeat_02": ROOT / "reports/review_main_targeted_20261025/repeat_02",
}


def read(path):
    if not path.exists():
        path = path.with_suffix(".json.gz")
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        return json.load(f)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def body_bytes(result):
    wire = bytes.fromhex(result["visible_bytes_hex"])
    sequence = result["native_token_sequence"]
    # result carries the body; the verified native sequence includes natural
    # or synthetic EOS when present. Ordinary at a cap has no synthetic EOS.
    full_wire = wire + (b"\x00\x00" if sequence["natural_eos"] or sequence["synthetic_boundary_eos"] else b"")
    assert hashlib.sha256(full_wire).hexdigest() == sequence["wire_sha256"]
    if "total_visible_bytes" in result:
        assert len(wire) == result["total_visible_bytes"]
    assert len(wire) % 2 == 0 and all(b == 1 for b in wire[::2])
    return wire[1::2]


def collect():
    rows, fixes, manifests, metrics_hashes = [], [], [], {}
    configs = defaultdict(list)
    for source, base in SOURCES.items():
        design = read(base / "DESIGN.json")
        source_count = 0
        for job in design["shards"]:
            directory = base / "runs" / job["name"]
            marker = read(directory / "COMPLETE.json")
            path = directory / "metrics.json"
            value = digest(path)
            assert value == marker["sha256"]["metrics.json"]
            metrics_hashes[str(path.relative_to(ROOT))] = value
            manifest = read(directory / "manifest.json")
            manifests.append(manifest)
            config = manifest["configuration"]
            for prompt in config["inputs"]["prompts"]:
                configs[job["model"], prompt["id"]].append(prompt)
            for original in read(path)["rows"]:
                row = dict(original, source_batch=source, top_k=job["top_k"])
                row["original_bytes"] = row["bytes"]
                row["text_renderable"] = row["text"] is not None
                if row["sender_status"] == "ok" and row["text"] is None:
                    sender = read(directory / "sender" / (row["case_id"] + ".json"))
                    raw = body_bytes(sender["result"])
                    try:
                        raw.decode("utf-8", errors="strict")
                    except UnicodeDecodeError as error:
                        suffix_only = error.reason == "unexpected end of data" and error.end == len(raw)
                    else:
                        raise AssertionError("Missing text despite renderable UTF-8")
                    row["bytes"] = len(raw)
                    fixes.append(dict(source_batch=source, case_id=row["case_id"],
                        model=row["model"], top_k=row["top_k"], variant=row["variant"],
                        original_bytes=row["original_bytes"], corrected_bytes=len(raw),
                        incomplete_utf8_suffix_only=suffix_only,
                        strict_verified_bits=row["strict_verified_bits"]))
                row["strict_text_delivery_bits"] = (row["strict_verified_bits"] or 0) if row["text_renderable"] else 0
                rows.append(row)
                source_count += 1
        assert source_count == design["cases"]
    keys = [(r["model"], r["top_k"], r["prompt_id"], r["variant"], r["seed"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert all(all(p == ps[0] for p in ps) for ps in configs.values())
    code = manifests[0]["execution"]["code"]
    assert all(m["execution"]["code"] == code for m in manifests)
    return rows, fixes, metrics_hashes, code


def aggregate(rows):
    scores = [r for r in rows if r.get("nll_total") is not None and (r.get("evaluator_tokens") or 0) > 0]
    nll = math.fsum(r["nll_total"] for r in scores)
    tokens = sum(r["evaluator_tokens"] for r in scores)
    logs = [r["nll_total"] / r["evaluator_tokens"] for r in scores]
    size = sum(r["bytes"] for r in rows)
    original_size = sum(r["original_bytes"] for r in rows)
    bits = sum(r.get("strict_verified_bits") or 0 for r in rows)
    text_bits = sum(r["strict_text_delivery_bits"] for r in rows)
    send = sum(r.get("sender_seconds") or 0 for r in rows)
    receive = sum(r.get("receiver_seconds") or 0 for r in rows)
    forwards = sum(r.get("actual_model_forward_calls") or 0 for r in rows)
    lengths = [r["body_tokens"] for r in rows if r.get("body_tokens") is not None]
    h = sum(r.get("group_entropy_bits") or 0 for r in rows)
    c = sum(r.get("expected_coder_information_bits") or 0 for r in rows)
    decisions = sum(r.get('protocol_decisions') or 0 for r in rows)
    las_actions = sum(r.get('adaptive_las_actions') or 0 for r in rows)
    positive_sync = sum(r.get('positive_u_sync_actions') or 0 for r in rows)
    by_prompt = defaultdict(list)
    for r in scores:
        by_prompt[r['prompt_id']].append(r['nll_total']/r['evaluator_tokens'])
    residual_df = sum(len(v)-1 for v in by_prompt.values())
    residual_ss = sum(sum((x-statistics.mean(v))**2 for x in v) for v in by_prompt.values())
    nonordinary = rows[0]["method"] != "ordinary"
    return dict(n=len(rows), scored=len(scores), exact=sum(r.get("exact") is True for r in rows),
        receiver_planned=sum(r["method"] != "ordinary" for r in rows),
        sender_failures=sum(r["sender_status"] != "ok" for r in rows),
        unrenderable=sum(r["sender_status"] == "ok" and not r["text_renderable"] for r in rows),
        empty_body=sum(r.get("scorer_status") == "empty_generated_body" for r in rows),
        bytes=size, original_bytes=original_size, verified_bits=bits,
        byte_bpb=bits/size if size and nonordinary else None,
        valid_text_bpb=text_bits/size if size and nonordinary else None,
        original_bpb=bits/original_size if original_size and nonordinary else None,
        sender_ms_per_byte=1000*send/size if size else None,
        sender_receiver_ms_per_byte=1000*(send+receive)/size if size else None,
        sender_receiver_sec_per_case=(send+receive)/len(rows),
        sender_forward_calls_per_byte=forwards/size if size else None,
        verified_bits_per_sender_receiver_sec=bits/(send+receive) if send+receive else None,
        group_entropy_bits=h, expected_coder_bits=c,
        coder_over_group_entropy=c/h if h else None,
        delivery_over_group_entropy=bits/h if h else None,
        corpus_ppl=math.exp(nll/tokens) if tokens else None,
        geometric_ppl=math.exp(statistics.mean(logs)) if logs else None,
        pooled_logppl_variance=statistics.variance(logs) if len(logs)>1 else None,
        within_prompt_logppl_variance=residual_ss/residual_df if residual_df else None,
        protocol_decisions=decisions, las_actions=las_actions, positive_u_sync_actions=positive_sync,
        las_fraction_all_decisions=las_actions/decisions if decisions else None,
        las_fraction_vs_positive_sync=las_actions/(las_actions+positive_sync) if las_actions+positive_sync else None,
        median_body_tokens=statistics.median(lengths) if lengths else None,
        below128=sum(x<128 for x in lengths),
        natural_eos=sum(r.get("natural_eos") is True for r in rows))


def contrasts(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["model"],row["top_k"],row["variant"]][row["prompt_id"],row["seed"]] = row
    rng = np.random.default_rng(20261029)
    intervals=[]
    for (model,k,variant), arm in sorted(groups.items()):
        for baseline in ["ordinary", "las_imec", "syncpool_imec"]:
            if baseline == variant or (baseline != "ordinary" and variant == "ordinary"):
                continue
            control = groups.get((model,k,baseline),{})
            coords = sorted(set(arm)&set(control))
            if not coords:
                continue
            aa=[arm[x] for x in coords];bb=[control[x] for x in coords]
            a,b=aggregate(aa),aggregate(bb)
            entry=dict(model=model,top_k=k,variant=variant,baseline=baseline,
                matched_n=len(coords),prompts=len({p for p,s in coords}),
                capacity_ratio=a['byte_bpb']/b['byte_bpb'] if a['byte_bpb'] is not None and b['byte_bpb'] else None,
                time_per_byte_ratio=a['sender_receiver_ms_per_byte']/b['sender_receiver_ms_per_byte'],
                ppl_ratio_corpus=a['corpus_ppl']/b['corpus_ppl'] if a['corpus_ppl'] and b['corpus_ppl'] else None,
                ppl_ci95_low=None,ppl_ci95_high=None,
                paired_geometric_ratio=None,geometric_ci95_low=None,geometric_ci95_high=None,
                complete_scored_pairs=0,missing_body_scores_in_matched_arms=len(coords)*2-a['scored']-b['scored'])
            ps=sorted({p for p,s in coords})
            index=rng.integers(0,len(ps),size=(10000,len(ps)))
            def vectors(g):
                vn,vt,vs,vc=[],[],[],[]
                for p in ps:
                    rr=[g[x] for x in coords if x[0]==p and g[x].get('nll_total') is not None and (g[x].get('evaluator_tokens') or 0)>0]
                    vn.append(sum(r['nll_total'] for r in rr));vt.append(sum(r['evaluator_tokens'] for r in rr))
                return np.array(vn),np.array(vt)
            an,at=vectors(arm);bn,bt=vectors(control)
            with np.errstate(divide='ignore',invalid='ignore'):
                delta=an[index].sum(1)/at[index].sum(1)-bn[index].sum(1)/bt[index].sum(1)
            finite=delta[np.isfinite(delta)]
            if len(finite):
                low,high=np.quantile(finite,[.025,.975]);entry['ppl_ci95_low']=math.exp(float(low));entry['ppl_ci95_high']=math.exp(float(high))
            dd=[];counts=[]
            for p in ps:
                ds=[arm[x]['nll_total']/arm[x]['evaluator_tokens']-control[x]['nll_total']/control[x]['evaluator_tokens']
                    for x in coords if x[0]==p and (arm[x].get('evaluator_tokens') or 0)>0 and (control[x].get('evaluator_tokens') or 0)>0]
                dd.append(sum(ds));counts.append(len(ds))
            dd=np.array(dd);counts=np.array(counts)
            if counts.sum():
                entry['complete_scored_pairs']=int(counts.sum());entry['paired_geometric_ratio']=math.exp(float(dd.sum()/counts.sum()))
                den=counts[index].sum(1);draw=dd[index].sum(1)/den
                low,high=np.quantile(draw[np.isfinite(draw)],[.025,.975])
                entry['geometric_ci95_low']=math.exp(float(low));entry['geometric_ci95_high']=math.exp(float(high))
            intervals.append(entry)
    return intervals


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows,fixes,metrics_hashes,code=collect()
    groups=defaultdict(list)
    for r in rows:
        for view in ['pooled',r['source_batch']]:
            groups[view,r['model'],r['top_k'],r['variant'],r['tau']].append(r)
    cells=[dict(view=view,model=m,top_k=k,variant=v,tau=t,**aggregate(rs))
        for (view,m,k,v,t),rs in sorted(groups.items(),key=lambda z:str(z[0]))]
    save_csv('CORRECTED_CELLS.csv',cells)
    save_csv('BYTE_CORRECTIONS.csv',fixes)
    ci=contrasts(rows)
    save_csv('POOLED_MATCHED_COMPARISONS.csv',ci)
    confirm=contrasts([r for r in rows if r['source_batch']!='main'])
    save_csv('CONFIRMATION_MATCHED_COMPARISONS.csv',confirm)
    overview=dict(total_records=len(rows),shards_verified=len(metrics_hashes),
        source_rows=dict(Counter(r['source_batch'] for r in rows)),
        failed_senders=[{k:r.get(k) for k in ['source_batch','model','top_k','case_id','error']} for r in rows if r['sender_status']!='ok'],
        scored=sum(r['scorer_status']=='scored' for r in rows),empty_body=sum(r['scorer_status']=='empty_generated_body' for r in rows),
        exact=sum(r.get('exact') is True for r in rows),planned_receivers=sum(r['method']!='ordinary' for r in rows),
        byte_correction_records=len(fixes),byte_correction_total=sum(r['corrected_bytes']-r['original_bytes'] for r in fixes),
        all_null_text_errors_are_incomplete_suffix=all(r['incomplete_utf8_suffix_only'] for r in fixes),
        runtime_code_files_equal_across_all_shards=len(code),
        audit_errors=sum(bool(r['audit_errors']) for r in rows),
        inference_scope='Exploratory paired document-cluster bootstrap (10000 draws), no multiple-testing or selection correction, no equivalence certification.',
        byte_scope='All unescaped body bytes, including unrenderable suffixes; valid_text_bpb credits zero bits for unrenderable outputs.',
        time_scope='sender_seconds + receiver_seconds excludes model load and PPL; forward_calls is sender ONLY.',
        prompt_scope='32 original prompts per model; cells have one or three seeds; repeats are conditional repeats, not new documents.',
        hashes=metrics_hashes)
    (OUT/'AUDIT.json').write_text(json.dumps(overview,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in overview.items() if k!='hashes'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
