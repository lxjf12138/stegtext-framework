"""CPU-only progress audit of completed shards; never edits frozen run inputs.

Correct the old summary's zero byte denominator for an unrenderable UTF-8
output. Byte delivery and valid text delivery are reported separately, retaining
every output, native-token PPL, and the original immutable experiment records.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

from stegatext_runtime.experiments import runner
from tools.expanded_records_20261021 import read_record
from tools.review_main_report_20261023 import cell_metrics, write_csv


def output_bytes(result):
    """Undo the existing byte codec's tags and terminal, without text repair."""
    wire=bytes.fromhex(result["visible_bytes_hex"])
    assert len(wire)==result["total_visible_bytes"]
    while wire.endswith(b"\x00\x00"):
        wire=wire[:-2]
    assert len(wire)%2==0 and all(x==1 for x in wire[::2]), "Invalid escaped byte framing"
    return wire[1::2]


def correct_unrenderable(row, sender):
    result=sender["result"]
    raw=output_bytes(result)
    incomplete_suffix=0
    try:
        raw.decode("utf-8",errors="strict")
    except UnicodeDecodeError as exc:
        if exc.reason=="unexpected end of data" and exc.end==len(raw):
            incomplete_suffix=exc.end-exc.start
    else:
        raise AssertionError("Missing text despite renderable UTF-8; inspect separately")
    return dict(row,bytes=len(raw),original_reported_bytes=row["bytes"],
        text_renderable=False,framing_status=result["framing_status"],
        incomplete_utf8_suffix_bytes=incomplete_suffix,
        strict_text_delivery_bits=0 if row["method"]!="ordinary" else None)


def summarize_cells(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[row["model"],row["top_k"],row["variant"]].append(row)
    cells=cell_metrics(rows)
    for cell in cells:
        part=groups[cell["model"],cell["top_k"],cell["variant"]]
        denominator=sum(r["bytes"] for r in part)
        text_bits=sum(r.get("strict_text_delivery_bits") or 0 for r in part)
        cell.update(unrenderable_outputs=sum(not r["text_renderable"] for r in part),
            byte_delivery_bpb=cell["verified_bpb"],
            valid_text_delivery_bpb=text_bits/denominator if denominator and part[0]["method"]!="ordinary" else None)
    return cells


def analyze(out):
    design=runner.read(out/"DESIGN.json")
    rows=[];fixes=[];hashes={};planned=set();seconds=defaultdict(float);docs=defaultdict(set)
    completed_shards=0
    for job in design["shards"]:
        for case_id in job["case_ids"]:
            planned.add((job["model"],job["top_k"],case_id.rsplit("__",1)[-1]))
        directory=out/"runs"/job["name"]
        if not (directory/"COMPLETE.json").exists():
            continue
        completed_shards+=1
        metrics=directory/"metrics.json"
        marker=runner.read(directory/"COMPLETE.json")
        digest=runner.file_hash(metrics)
        assert digest==marker["sha256"]["metrics.json"]
        hashes[str(metrics.relative_to(out))]=digest
        for original in runner.read(metrics)["rows"]:
            row=dict(original,text_renderable=original["text"] is not None,
                strict_text_delivery_bits=original["strict_verified_bits"] if original["text"] is not None else 0)
            if row["sender_status"]=="ok" and row["text"] is None:
                row=correct_unrenderable(row,read_record(directory/"sender"/(row["case_id"]+".json")))
                fixes.append({k:row[k] for k in ("case_id","model","top_k","variant","original_reported_bytes",
                    "bytes","framing_status","incomplete_utf8_suffix_bytes","strict_verified_bits","exact")})
            rows.append(row)
            docs[job["model"],job["top_k"]].add(row["prompt_id"])
        state=runner.read(directory/"STAGES.json")
        seconds[job["model"],job["top_k"]]+=sum(v for k,v in state.items() if k.endswith("_wall_seconds"))
    groups=defaultdict(set)
    for row in rows:
        groups[row["model"],row["top_k"],row["variant"]].add(row["prompt_id"])
    common={}
    for model in sorted({key[0] for key in planned}):
        common[model]=sorted(set.intersection(*(groups[key] for key in planned if key[0]==model)))
    matched=[r for r in rows if r["prompt_id"] in common[r["model"]]]
    resource=runner.read(out/"RESOURCES.json")
    first_batch_hours=sum(10*value/len(docs[key]) for key,value in seconds.items())/3600
    summary=dict(created_unix=time.time(),completed_shards=completed_shards,
        completed_senders=len(rows),successful_senders=sum(r["sender_status"]=="ok" for r in rows),
        exact_receivers=sum(r["exact"] is True for r in rows),
        scored=sum(r["scorer_status"]=="scored" for r in rows),
        completed_shard_audit_errors=sum(bool(r["audit_errors"]) for r in rows),
        byte_denominator_corrections=fixes,common_prompt_ids=common,
        cells=summarize_cells(rows),common_document_cells=summarize_cells(matched),
        elapsed_hours=resource["elapsed_seconds"]/3600,
        first_batch_linear_stage_estimate_hours=first_batch_hours,
        first_batch_linear_remaining_hours=max(0,first_batch_hours-resource["elapsed_seconds"]/3600),
        estimate_scope="Mean observed stage time per document and model/k; no confidence bound; changing output lengths affect cost",
        metric_scope="Byte BPB counts all unescaped body bytes including incomplete UTF-8 tails; valid-text BPB assigns zero text-delivered bits to unrenderable output while retaining its bytes; PPL remains native-token based",
        input_metrics_sha256=hashes,script_sha256=runner.file_hash(Path(__file__).resolve()))
    dest=out/"analysis/progress_audit"
    runner.write(dest/"SUMMARY.json",summary)
    runner.write(dest/"CASES.json",dict(rows=rows))
    write_csv(dest/"CELLS.csv",summary["cells"])
    write_csv(dest/"COMMON_DOCUMENT_CELLS.csv",summary["common_document_cells"])
    (dest/"REPORT.md").write_text(
        "# Progress and denominator check of completed shards\n\n"
        f"Completed {completed_shards} shards, {len(rows)} generations, {summary['exact_receivers']} independent exact receptions, and {summary['scored']} PPL scores. "
        f"Original audit errors: {summary['completed_shard_audit_errors']}; byte denominators corrected here: {len(fixes)}.\n\n"
        "The frozen records and the main driver are unchanged. This directory is a separately corrected snapshot that this script can update. "
        "The earlier driver summary wrote zero bytes when text=null while still counting the bits, which can inflate BPB; this table recovers the actual body bytes from the verifiable visible_bytes_hex, excluding escape tags and the EOS marker. "
        "byte_delivery_bpb counts all actual bytes; valid_text_delivery_bpb assigns zero text-delivered bits to output that cannot be rendered as UTF-8 text and keeps its byte denominator. Original PPL and recovery results are kept; no sample is deleted or replaced.\n\n"
        f"Prompts completed by every k and method: { {m:len(v) for m,v in common.items()} }. "
        "Fair comparisons use COMMON_DOCUMENT_CELLS.csv; CELLS.csv keeps all documents completed so far for each condition. Later samples are not yet complete, so PPL convergence cannot be claimed from this snapshot.\n")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("reports/review_main_batched_20261024"))
    args=parser.parse_args()
    result=analyze(args.output.resolve())
    print({k:result[k] for k in ("completed_shards","completed_senders","exact_receivers","scored",
        "byte_denominator_corrections","first_batch_linear_remaining_hours")})
