"""Reuse the selected 64 documents for every k/method; CPU preparation only."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import shutil

from stegatext_runtime.experiments import runner
from stegatext_runtime.experiments.config import ROOT, cases, fingerprint, read_tree, resolve
from stegatext_runtime.experiments.randomness import case_nonce, descriptor
from tools.controlled_comparison.reproduce import write_config


PREVIOUS = ROOT / "research_notes/baseline_experiment_design_20261022/main_table_gt610_preparation_20261023"
OUT = ROOT / "research_notes/baseline_experiment_design_20261022/main_table_uniform_preparation_20261023"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def prepare():
    if OUT.exists():
        raise RuntimeError("Preparation exists; do not overwrite frozen inputs")
    previous=json.loads((PREVIOUS/"PREPARATION.json").read_text())
    for name,digest in previous["file_sha256"].items():
        assert runner.file_hash(PREVIOUS/name)==digest, name
    data=json.loads((PREVIOUS/"DATASET_MANIFEST.json").read_text())
    for doc in data["documents"]:
        doc["full_subset"]=True
    for selection in data["selection"]:
        selection["full_rows"]=list(selection["selected_rows"])
    data.update(
        selection_rule="Retain every previously selected IMDB/GT610 document in its original order; all k and methods use all 32 documents per language",
        matched_inputs_across_k=True,
        source_manifest_sha256=runner.file_hash(PREVIOUS/"DATASET_MANIFEST.json"))
    OUT.mkdir(parents=True)
    (OUT/".gitignore").write_text("experiment_root.key\n")
    save(OUT/"DATASET_MANIFEST.json",data)
    shutil.copyfile(PREVIOUS/"PROMPT_AUDIT.json",OUT/"PROMPT_AUDIT.json")
    # Preserve the existing experimental random root: changing allocation must
    # not silently resample any of the already specified cases.
    key=OUT/"experiment_root.key"
    with os.fdopen(os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"wb") as handle:
        handle.write((PREVIOUS/"experiment_root.key").read_bytes())
    jobs,rows=[],[]
    for job in previous["jobs"]:
        parent_path=ROOT/job["config"]
        assert runner.file_hash(parent_path)==job["config_sha256"]
        base=read_tree(parent_path)
        parent=resolve(parent_path)
        selected=[d for d in data["documents"] if d["language"]==job["language"]]
        assert len(selected)==32
        base["inputs"]["prompts"]=[dict(id=d["id"],language=d["language"],text=d["user_message"],assistant_prefix=d["prefix"]) for d in selected]
        base["randomness"]["file"]=str(key)
        path=OUT/"configs"/parent_path.name
        write_config(path,base)
        config=resolve(path)
        expanded=list(cases(config))
        assert len(expanded)==(352 if job["top_k"]==0 else 384)
        # All common cases retain their full prompt, settings and streams.
        source={"sha256":fingerprint(config["source"])}
        common={c["id"]:c for c in expanded}
        before,after=copy.deepcopy(parent),copy.deepcopy(config)
        for value in (before,after):
            value["inputs"].pop("prompts")
            value["randomness"].pop("file")
        assert before==after, "Parameters other than prompt allocation changed"
        for original in cases(parent):
            case=common[original["id"]]
            assert case==original
            assert case_nonce(parent,original,source)==case_nonce(config,case,source)
            if case["variant"]["kind"]=="ordinary":
                assert descriptor(parent,original,source)==descriptor(config,case,source)
        jobs.append(dict(job,documents=32,config=str(path.relative_to(ROOT)),
            config_sha256=runner.file_hash(path),cases=len(expanded)))
        rows.extend(dict(config=path.name,model=job["model"],language=job["language"],top_k=job["top_k"],
            prompt_id=c["prompt"]["id"],seed=c["seed"],variant=c["variant"]["id"],case_id=c["id"]) for c in expanded)
    assert len(rows)==3776 and sum(r["variant"]!="ordinary" for r in rows)==3456
    assert len({(r["config"],r["case_id"]) for r in rows})==len(rows)
    with (OUT/"CASES.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plan=copy.deepcopy(previous)
    plan.update(status="matched_inputs_prepared_no_main_generation_launched",jobs=jobs,
        preparation_script_sha256=runner.file_hash(Path(__file__)),
        source_preparation_sha256=runner.file_hash(PREVIOUS/"PREPARATION.json"),
        supersedes_candidate=str(PREVIOUS.relative_to(ROOT)),
        full_documents_per_language=32,generated_cases=3776,independent_receivers=3456,
        matched_inputs_across_k=True,
        allocation="All 32 fixed documents per language, identical prompts/order/repeat coordinates for every k and method; only Stepwise Full excluded",
        main_wall_hours_limit=None,
        execution="Not launched. Serial worker, CPU thread limits, resource observation and lossless gzip; no automatic wall-time or resource-threshold stops",
        unresolved="Full cost incompletely measured. Earlier 32/4 allocation timing does not apply to this 32/32 matrix; report per-method timing estimates",
        file_sha256={name:runner.file_hash(OUT/name) for name in ("DATASET_MANIFEST.json","PROMPT_AUDIT.json","CASES.csv")})
    save(OUT/"PREPARATION.json",plan)
    print(json.dumps(dict(status=plan["status"],documents_per_language=32,full_documents_per_language=32,
        generated_cases=3776,independent_receivers=3456,old_common_case_streams_preserved=True)))


if __name__=="__main__":
    prepare()
