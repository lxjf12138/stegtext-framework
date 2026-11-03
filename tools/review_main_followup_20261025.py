"""Queue fixed-size, targeted replications after the entire frozen main run.

Only this companion and new output directories are changed. The original
worker, scorer, model settings, prompts, and stopping protocol are reused.
Selection uses the final main table; neither selection nor stopping depends
on making a replication agree with a preferred result.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import defaultdict

from stegatext_runtime.experiments import runner
from stegatext_runtime.experiments.config import ROOT, cases, read_tree, resolve
from tools.controlled_comparison.reproduce import write_config
from tools import review_main_run_20261023 as main_run
from tools.review_main_progress_audit_20261024 import analyze, summarize_cells
from tools.review_main_report_20261023 import write_csv

SELF = Path(__file__).resolve()
DEFAULT_QUEUE = ROOT / "reports/review_main_targeted_20261025"
SEEDS = [470101, 470102]
CONTROLS = {"ordinary", "las_imec", "syncpool_imec"}
K_ORDER = [16, 32, 128, 256, 0]


def eligible(variant):
    return variant in {"las_imec", "syncpool_imec"} or variant.startswith("adaptive_")


def select_targets(cells):
    """Screen matched cells, separating theoretical and empirical concerns."""
    by = defaultdict(dict)
    for cell in cells:
        by[cell["model"], cell["top_k"]][cell["variant"]] = cell
    reasons = defaultdict(list)

    def add(coord, variants, kind, **evidence):
        reasons[coord].append(dict(kind=kind, variants=sorted(variants), **evidence))

    for coord, group in sorted(by.items()):
        ordinary, las, sp = (group.get(v) for v in ("ordinary", "las_imec", "syncpool_imec"))
        if not ordinary or not las or not sp:
            continue
        for variant, cell in group.items():
            if not eligible(variant) or cell["planned"] != ordinary["planned"]:
                continue
            if cell.get("ppl") and ordinary.get("ppl"):
                ratio = cell["ppl"] / ordinary["ppl"]
                if abs(math.log(ratio)) > math.log(1.10):
                    add(coord, {variant}, "ppl_distribution_diagnostic", ratio=ratio,
                        note="10% is a diagnostic trigger, not a theorem or equivalence test")
            if variant.startswith("adaptive_"):
                tau = int(variant.split("_")[1]) / 1000
                capacity = cell.get("byte_delivery_bpb")
                if tau <= .5 and capacity is not None and las.get("byte_delivery_bpb"):
                    ratio = capacity / las["byte_delivery_bpb"]
                    if ratio < .9:
                        add(coord, {variant}, "empirical_capacity_retention", ratio=ratio,
                            note="90% retention is a practical target, not a mathematical guarantee")
        if las["group_entropy_bits"] / las["planned"] < sp["group_entropy_bits"] / sp["planned"]:
            add(coord, {"las_imec", "syncpool_imec"}, "ideal_entropy_budget_diagnostic")
        adaptive = sorted((v for v in group if v.startswith("adaptive_")),
                          key=lambda v: int(v.split("_")[1]))
        for lo, hi in zip(adaptive, adaptive[1:]):
            a, b = group[lo].get("byte_delivery_bpb"), group[hi].get("byte_delivery_bpb")
            if a and b is not None and b > 1.05 * a:
                add(coord, {lo, hi}, "empirical_threshold_reversal", ratio=b / a,
                    note="No global threshold monotonicity theorem is assumed")
    for model in sorted({m for m, _ in by}):
        for lo, hi in zip(K_ORDER, K_ORDER[1:]):
            a, b = by.get((model, lo), {}), by.get((model, hi), {})
            for variant in sorted(set(a) & set(b)):
                if variant != "las_imec" and not variant.startswith("adaptive_"):
                    continue
                x, y = a[variant].get("byte_delivery_bpb"), b[variant].get("byte_delivery_bpb")
                if x and y is not None and y < .95 * x:
                    for k in (lo, hi):
                        add((model, k), {variant}, "empirical_k_reversal", k_pair=[lo, hi],
                            ratio=y / x, note="No universal BPB monotonicity in k is assumed")
    return [dict(model=model, top_k=k,
                 variants=sorted(CONTROLS | {v for r in rs for v in r["variants"]}), reasons=rs)
            for (model, k), rs in sorted(reasons.items())]


def parent_finished(parent):
    marker, resources = parent / "COMPLETE.json", parent / "RESOURCES.json"
    if not marker.exists() or not resources.exists():
        return False
    return runner.read(resources).get("status") == "finished"


def init_queue(queue, parent):
    parent_design = main_run.preflight(parent)
    queue.mkdir(parents=True, exist_ok=True)
    value = dict(parent=str(parent.relative_to(ROOT)), parent_design_sha256=runner.file_hash(parent / "DESIGN.json"),
        companion_sha256=runner.file_hash(SELF), seeds=SEEDS,
        start_condition="Entire 32-document main matrix complete AND main guard status finished",
        repetitions="Exactly two fresh repetitions per original prompt; both fixed rounds always complete",
        selection="Final corrected main cells only; provisional preview never selects executable jobs",
        controls=sorted(CONTROLS), ppl_trigger_factor=1.10, low_tau_capacity_trigger=.90,
        empirical_reversal_factor=1.05, ppl_precision_factor=1.10, capacity_precision_factor=1.05,
        next_precision_checkpoints_per_prompt=[4, 8],
        further_repetitions="Next fixed-size block is considered after precision review, never until agreement",
        preserve="Every failure, extreme value, short output, and unrenderable byte output; no selective retry",
        execution="Existing serial single-GPU driver, one CPU thread, independent receivers, gzip; no resource/time cutoff",
        main_cases=parent_design["cases"], status="waiting_for_main")
    path = queue / "PLAN.json"
    if path.exists():
        before = runner.read(path)
        assert {k: v for k, v in before.items() if k != "record_sha256"} == value, "Queue plan changed"
    else:
        runner.write(path, value)
    return value


def prepare_round(parent, out, targets, seed):
    """Materialize a preflightable run without touching frozen main inputs."""
    if (out / "DESIGN.json").exists():
        return main_run.preflight(out)
    source_design = main_run.preflight(parent)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("DATASET_MANIFEST.json", "PROMPT_AUDIT.json"):
        shutil.copyfile(parent / name, out / name)
    shards = []
    for target in targets:
        model, k = target["model"], target["top_k"]
        original_job = next(j for j in source_design["shards"] if j["model"] == model and j["top_k"] == k)
        original = ROOT / original_job["parent_config"]
        base = read_tree(original)
        assert seed not in base["inputs"]["seeds"], "Replication reuses a main seed"
        original_prompts = copy.deepcopy(base["inputs"]["prompts"])
        base["name"] = f"targeted_{model}_k{k}_s{seed}"
        base["inputs"]["seeds"] = [seed]
        base["variants"] = [v for v in base["variants"] if v["id"] in target["variants"]]
        assert {v["id"] for v in base["variants"]} == set(target["variants"])
        parent_path = out / "parents" / (base["name"] + ".toml")
        write_config(parent_path, base)
        assert read_tree(parent_path)["inputs"]["prompts"] == original_prompts
        for placement in main_run.shard_slices(len(original_prompts), k):
            start, stop = placement["prompt_start"], placement["prompt_stop"]
            config = copy.deepcopy(base)
            config["name"] += f"_b{placement['batch_index']+1:02d}_p{start:02d}"
            config["inputs"]["prompts"] = original_prompts[start:stop]
            path = out / "configs" / (config["name"] + ".toml")
            write_config(path, config)
            expanded = list(cases(resolve(path)))
            shards.append(dict(name=config["name"], config=str(path.relative_to(ROOT)),
                config_sha256=runner.file_hash(path), parent_config=str(parent_path.relative_to(ROOT)),
                **placement, prompt_ids=[p["id"] for p in config["inputs"]["prompts"]],
                model=model, top_k=k, cases=len(expanded), case_ids=[c["id"] for c in expanded],
                stego_cases=sum(c["variant"]["kind"] != "ordinary" for c in expanded)))
    shards.sort(key=lambda j: (j["batch_index"], j["comparison_block"], K_ORDER.index(j["top_k"]), j["model"], j["part_index"]))
    design = copy.deepcopy(source_design)
    design.pop("record_sha256", None)
    design.update(shards=shards, cases=sum(j["cases"] for j in shards),
        stego_cases=sum(j["stego_cases"] for j in shards), batches=main_run.batch_plan(shards),
        input_sha256={name: runner.file_hash(out / name) for name in ("DATASET_MANIFEST.json", "PROMPT_AUDIT.json")},
        status="Targeted fixed-size replication prepared, no generation launched",
        followup=dict(parent=str(parent.relative_to(ROOT)), seed=seed, targets=targets,
                      scope="Confirmation is reported separately from the discovery main table"))
    runner.write(out / "DESIGN.json", design)
    return main_run.preflight(out)


def finish_report(queue, rounds):
    """Report all confirmation data, including deviations and uncertainty."""
    import numpy as np
    rows = []
    for out in rounds:
        analyze(out)
        rows.extend(runner.read(out / "analysis/progress_audit/CASES.json")["rows"])
    dest = queue / "analysis"
    cells = summarize_cells(rows)
    runner.write(dest / "CASES.json", dict(rows=rows))
    write_csv(dest / "CELLS.csv", cells)
    groups = defaultdict(dict)
    for row in rows:
        groups[row["model"], row["top_k"], row["variant"]][row["prompt_id"], row["seed"]] = row
    intervals = []
    rng = np.random.default_rng(20261025)
    for (model, k, variant), group in sorted(groups.items()):
        if variant == "ordinary":
            continue
        control = groups[model, k, "ordinary"]
        las = groups[model, k, "las_imec"]
        coordinates = sorted(control)
        # Missing/failed scores are retained in rows and explicitly prevent a CI.
        if set(coordinates) != set(group) or any(not r.get("evaluator_tokens") or r.get("nll_total") is None
                                                for g in (group, control) for r in g.values()):
            intervals.append(dict(model=model, top_k=k, variant=variant, status="unavailable_retained_failures"))
            continue
        prompts = sorted({p for p, _ in coordinates})
        index = rng.integers(0, len(prompts), size=(10000, len(prompts)))

        def sums(g, field):
            return np.array([sum((g[p, s].get(field) or 0) for p0, s in coordinates if p0 == p) for p in prompts])

        mn, mt = sums(group, "nll_total"), sums(group, "evaluator_tokens")
        on, ot = sums(control, "nll_total"), sums(control, "evaluator_tokens")
        delta = mn[index].sum(axis=1) / mt[index].sum(axis=1) - on[index].sum(axis=1) / ot[index].sum(axis=1)
        low, high = np.quantile(delta, [.025, .975])
        mb, mz = sums(group, "strict_verified_bits"), sums(group, "bytes")
        lb, lz = sums(las, "strict_verified_bits"), sums(las, "bytes")
        mz_draw, lb_draw, lz_draw = (v[index].sum(axis=1) for v in (mz, lb, lz))
        if np.any(mz_draw <= 0) or np.any(lb_draw <= 0) or np.any(lz_draw <= 0):
            ci = None
        else:
            cap = (mb[index].sum(axis=1) / mz_draw) / (lb_draw / lz_draw)
            ci = np.quantile(cap, [.025, .975])
        precise = float(high - low) / 2 <= math.log(1.10)
        ratio = math.exp(float(mn.sum() / mt.sum() - on.sum() / ot.sum()))
        intervals.append(dict(model=model, top_k=k, variant=variant, observations=len(group),
            ppl_ratio=ratio, ppl_ratio_ci95=[math.exp(float(low)), math.exp(float(high))],
            ppl_precision_met=precise, capacity_ratio_ci95=[float(v) for v in ci] if ci is not None else None,
            capacity_precision_met=bool(ci is not None and ci[0] > 0 and math.log(ci[1] / ci[0]) / 2 <= math.log(1.05)),
            ppl_status=("precision_unresolved_consider_next_fixed_block" if not precise else
                    "persistent_difference_requires_cause_audit" if abs(math.log(ratio)) > math.log(1.10) else
                    "small_difference_at_target_precision")))
    runner.write(dest / "SUMMARY.json", dict(cells=cells, intervals=intervals,
        generated=len(rows), independent_receivers=sum(r.get("exact") is True for r in rows),
        interval_scope="Exploratory paired document-cluster bootstrap, 10000 draws; no equivalence or zero-KL certification",
        main_data_included=False, stopping="Both predeclared repetitions completed regardless of agreement",
        next_precision_checkpoints_per_prompt=[4, 8]))


def watch(queue):
    plan = runner.read(queue / "PLAN.json")
    parent = ROOT / plan["parent"]
    assert plan["companion_sha256"] == runner.file_hash(SELF), "Queued companion changed"
    assert plan["parent_design_sha256"] == runner.file_hash(parent / "DESIGN.json")
    with (queue / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not parent_finished(parent):
            status = runner.read(parent / "RESOURCES.json").get("status")
            if status in {"failed", "stopped"} or (queue / "CANCEL_REQUESTED.json").exists():
                runner.write(queue / "STATUS.json", dict(status="waiting_cancelled_or_parent_not_finished", parent_status=status))
                return
            runner.write(queue / "STATUS.json", dict(status="waiting_for_entire_main", pid=os.getpid(), parent_status=status))
            time.sleep(30)
        if (queue / "CANCEL_REQUESTED.json").exists():
            return
        main_run.preflight(parent)
        audit = analyze(parent)
        assert audit["completed_senders"] == plan["main_cases"]
        assert audit["completed_shard_audit_errors"] == 0, "Investigate main audit failures before replication"
        selection_path = queue / "FINAL_SELECTION.json"
        if selection_path.exists():
            selection = runner.read(selection_path)
            assert selection["main_metrics_sha256"] == audit["input_metrics_sha256"]
        else:
            selection = dict(targets=select_targets(audit["cells"]),
                main_metrics_sha256=audit["input_metrics_sha256"], selected_after_entire_main=True)
            runner.write(selection_path, selection)
        targets = selection["targets"]
        if not targets:
            runner.write(queue / "STATUS.json", dict(status="no_targets_under_predeclared_rules"))
            return
        rounds = []
        for index, seed in enumerate(plan["seeds"], 1):
            if (queue / "CANCEL_REQUESTED.json").exists():
                return
            out = queue / f"repeat_{index:02d}"
            prepare_round(parent, out, targets, seed)
            runner.write(queue / "STATUS.json", dict(status="running_targeted_repetitions", repeat=index, seed=seed, output=str(out)))
            if not parent_finished(out):
                subprocess.run([sys.executable, "-B", "-m", "tools.review_main_run_20261023", "run", "--output", str(out)],
                               cwd=ROOT, check=True)
            rounds.append(out)
        finish_report(queue, rounds)
        runner.write(queue / "STATUS.json", dict(status="fixed_repetitions_complete_precision_review_available",
            seeds=plan["seeds"], report=str(queue / "analysis/SUMMARY.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("init", "preview", "watch"))
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--parent", type=Path, default=main_run.DEFAULT_OUT)
    args = parser.parse_args()
    os.environ.update(main_run.THREAD_ENV)
    queue, parent = args.queue.resolve(), args.parent.resolve()
    queue.relative_to(ROOT)
    parent.relative_to(ROOT)
    if args.phase == "init":
        init_queue(queue, parent)
        print("Fixed follow-up policy saved; no model loaded.")
    elif args.phase == "preview":
        audit = analyze(parent)
        targets = select_targets(audit["cells"])
        preview = dict(provisional=True, executable=False, targets=targets,
            planned_senders_if_selected_now=2 * 32 * sum(len(t["variants"]) for t in targets),
            note="Final selection is recomputed only after the full main run finishes")
        runner.write(queue / "PREVIEW.json", preview)
        print(json.dumps(dict(conditions=len(targets), provisional_senders=preview["planned_senders_if_selected_now"])))
    else:
        try:
            watch(queue)
        except Exception as exc:
            runner.write(queue / "STATUS.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
            raise


if __name__ == "__main__":
    main()
