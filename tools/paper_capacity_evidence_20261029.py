"""Read completed pure-LAS traces to separate protocol and coder losses.

CPU-only descriptive analysis. Does not generate samples or alter experiments.
All source rows come from the hash-checked main/repeat audit collector.
"""
from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path

from independent_work_audit_20261029 import SOURCES, aggregate, collect, read


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/paper_rethink_20261029"


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, _, _, _ = collect()
    selected = [r for r in rows if r["variant"] == "las_imec"]
    directory_index = {}
    for source, base in SOURCES.items():
        for shard in read(base / "DESIGN.json")["shards"]:
            for cid in shard["case_ids"]:
                directory_index[source, shard["top_k"], cid] = base / "runs" / shard["name"]
    cases = []
    for i, row in enumerate(selected, 1):
        assert row["sender_status"] == "ok" and row["exact"] is True
        path = directory_index[row["source_batch"], row["top_k"], row["case_id"]] / "sender" / (row["case_id"] + ".json")
        if not path.exists():
            path = path.with_suffix(".json.gz")
        record = read(path)
        assert record["status"] == "ok"
        result = record["result"]
        assert math.isclose(result["group_entropy_total_bits"], row["group_entropy_bits"], rel_tol=1e-12, abs_tol=1e-8), (
            row["source_batch"], row["case_id"], result["group_entropy_total_bits"], row["group_entropy_bits"])
        bound, multi = 0.0, 0
        for step in result["trace"]:
            value = step["exact_identity_information_loss_bits"]
            assert math.isfinite(value) and value >= -1e-8
            a = step["selected_group_exact_mass"] / step["selected_group_mass"]
            assert math.isclose(value, a * step["exact_identity_entropy_bits"], rel_tol=1e-10, abs_tol=1e-10)
            bound += max(0.0, value)
            multi += step["exact_candidate_count"] > 1
        cases.append(dict(source_batch=row["source_batch"], model=row["model"], top_k=row["top_k"],
            prompt_id=row["prompt_id"], seed=row["seed"], case_id=row["case_id"],
            decisions=len(result["trace"]), multi_exact_steps=multi,
            identity_upper_bits=bound, group_entropy_bits=row["group_entropy_bits"],
            expected_coder_bits=row["expected_coder_information_bits"],
            realized_information_bits=row["realized_information_bits"],
            verified_bits=row["strict_verified_bits"], pending_information_bits=row["pending_information_bits"],
            bytes=row["bytes"], sender_record=str(path.relative_to(ROOT)),
            sender_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if i % 100 == 0:
            print(f"Read {i}/{len(selected)} pure-LAS traces", flush=True)
    buckets = defaultdict(list)
    raw_buckets = defaultdict(list)
    for c, r in zip(cases, selected):
        for view in [c["source_batch"], "pooled"]:
            key = view, c["model"], c["top_k"]
            buckets[key].append(c)
            raw_buckets[key].append(r)
    cells = []
    for (view, model, k), part in sorted(buckets.items(), key=lambda item: str(item[0])):
        prompt_counts = Counter(r["prompt_id"] for r in part)
        assert len(prompt_counts) == 32 and len(set(prompt_counts.values())) == 1
        sums = {f: math.fsum(r[f] for r in part) for f in ["identity_upper_bits", "group_entropy_bits",
            "expected_coder_bits", "realized_information_bits", "verified_bits", "pending_information_bits"]}
        h = sums["group_entropy_bits"]
        eta = sums["identity_upper_bits"] / h
        a = aggregate(raw_buckets[view, model, k])
        cells.append(dict(view=view, model=model, top_k=k, n=len(part), prompts=32,
            repeats_per_prompt=next(iter(prompt_counts.values())),
            decisions=sum(c["decisions"] for c in part), multi_exact_steps=sum(c["multi_exact_steps"] for c in part),
            **sums, identity_upper_over_group=eta,
            plugin_ideal_fraction=1 / (1 + eta),
            expected_coder_over_group=sums["expected_coder_bits"] / h,
            verified_over_group=sums["verified_bits"] / h,
            information_sampling_residual_bits=sums["realized_information_bits"] - sums["expected_coder_bits"],
            byte_bpb=a["byte_bpb"], ms_per_byte=a["sender_receiver_ms_per_byte"]))
    write_csv("PURE_LAS_TRACE_CASES.csv", cases)
    write_csv("PURE_LAS_LEDGER_CELLS.csv", cells)
    print(f"Wrote {len(cases)} traces and {len(cells)} ledger cells to {OUT}")
    for c in cells:
        if c["view"] == "pooled":
            print(f"{c['model']} k={c['top_k']} n={c['n']}: "
                f"sum(aH)/sum(Hq)={c['identity_upper_over_group']:.8%}, "
                f"C/H={c['expected_coder_over_group']:.4%}, B/H={c['verified_over_group']:.4%}")


if __name__ == "__main__":
    main()
