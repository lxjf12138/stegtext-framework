"""Read-only, CPU-only LAS capacity diagnostics from completed main shards.

The exact identity entropy is an upper bound on local synchronization loss,
not a measurement of complete-future JSD. Sample ratios do not certify an
ideal population bound or the numerical implementation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path


def analyze(base: Path):
    snapshot_path = base / "analysis/progress_audit/SUMMARY.json"
    snapshot_raw = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_raw)
    cases = json.loads((base / "analysis/progress_audit/CASES.json").read_text())["rows"]
    design = json.loads((base / "DESIGN.json").read_text())
    directories = {(j["model"], j["top_k"], cid): base / "runs" / j["name"]
                   for j in design["shards"] for cid in j["case_ids"]}
    rows = []
    for case in cases:
        if case["variant"] != "las_imec":
            continue
        if case["prompt_id"] not in snapshot["common_prompt_ids"][case["model"]]:
            continue
        directory = directories[case["model"], case["top_k"], case["case_id"]]
        path = directory / "sender" / (case["case_id"] + ".json")
        actual = path if path.exists() else path.with_suffix(".json.gz")
        opener = gzip.open if actual.suffix == ".gz" else open
        with opener(actual, "rb") as stream:
            raw = stream.read(128 * 2**20 + 1)
        assert len(raw) <= 128 * 2**20, actual
        record = json.loads(raw)
        assert record["status"] == "ok" and case["exact"] is True, actual
        result = record["result"]
        assert math.isclose(result["group_entropy_total_bits"], case["group_entropy_bits"],
                            abs_tol=1e-8, rel_tol=1e-12)
        bound = 0.0
        missing = negative = multi = 0
        for step in result["trace"]:
            value = step.get("exact_identity_information_loss_bits")
            multi += (step.get("exact_candidate_count") or 0) > 1
            if value is None:
                missing += 1
                continue
            assert math.isfinite(value) and value >= -1e-8, (actual, value)
            negative += value < 0
            a = step["selected_group_exact_mass"] / step["selected_group_mass"]
            assert math.isclose(value, a * step["exact_identity_entropy_bits"],
                                abs_tol=1e-10, rel_tol=1e-10)
            bound += max(0.0, value)
        rows.append(dict(model=case["model"], top_k=case["top_k"],
                         prompt_id=case["prompt_id"], case_id=case["case_id"],
                         record_sha256=record["record_sha256"],
                         group_entropy_bits=case["group_entropy_bits"],
                         exact_identity_bound_bits=bound if missing == 0 else None,
                         missing_steps=missing, rounded_negative_steps=negative,
                         decisions=len(result["trace"]), multi_exact_steps=multi,
                         verified_bits=case["strict_verified_bits"],
                         expected_coder_information_bits=case["expected_coder_information_bits"],
                         realized_information_bits=case["realized_information_bits"],
                         pending_information_bits=case["pending_information_bits"],
                         bytes=case["bytes"], text_renderable=case["text_renderable"]))
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["top_k"]].append(row)
    cells = []
    for (model, k), part in sorted(groups.items()):
        def total(key):
            return math.fsum(r[key] for r in part)
        h = total("group_entropy_bits")
        bound = (total("exact_identity_bound_bits")
                 if all(r["exact_identity_bound_bits"] is not None for r in part) else None)
        cells.append(dict(model=model, top_k=k, cases=len(part),
                          decisions=sum(r["decisions"] for r in part),
                          multi_exact_steps=sum(r["multi_exact_steps"] for r in part),
                          missing_steps=sum(r["missing_steps"] for r in part),
                          group_entropy_bits=h, exact_identity_bound_bits=bound,
                          identity_bound_over_group_entropy=bound / h if h and bound is not None else None,
                          verified_over_group_entropy=total("verified_bits") / h if h else None,
                          expected_coder_over_group_entropy=total("expected_coder_information_bits") / h if h else None,
                          information_sampling_residual_bits=total("realized_information_bits") - total("expected_coder_information_bits"),
                          pending_information_bits=total("pending_information_bits"),
                          verified_bits=total("verified_bits"), bytes=total("bytes")))
    dest = base / "analysis/capacity_gap_20261026"
    dest.mkdir(parents=True, exist_ok=True)
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  progress_snapshot_sha256=hashlib.sha256(snapshot_raw).hexdigest(),
                  progress_snapshot_created_unix=snapshot["created_unix"],
                  scope="Completed common-document LAS paths; empirical path sums, not population certification; no model generation",
                  local_bound="Delta_LAS <= a * weighted_full_future_JSD <= a * H(exact_identity)",
                  negative_roundoff_policy="Values in [-1e-8,0) rounded up to zero; counts retained per case",
                  cells=cells, cases=rows)
    (dest / "SUMMARY.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for name, records in [("CELLS.csv", cells), ("CASES.csv", rows)]:
        with (dest / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    for cell in cells:
        print(json.dumps(cell, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/review_main_batched_20261024"))
    analyze(parser.parse_args().output)
