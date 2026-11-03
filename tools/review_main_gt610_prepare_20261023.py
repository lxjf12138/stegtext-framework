"""Prepare the selected IMDB/GT610 main-table candidate on CPU only.

Keeps the earlier candidate intact. Development outputs never enter the main
table; document selection uses source text, labels and hashes only.
"""
from collections import Counter, defaultdict, deque
import copy
import json
from pathlib import Path

import tools.review_main_prepare_20261023 as scaffold
from stegatext_runtime.experiments.review_prompts import PREFILL_INSTRUCTIONS, clean_review, review_prefix
from tools.review_long_sweep_20261022 import shingles

ROOT = scaffold.ROOT
PREVIOUS = scaffold.OUT
OUT = ROOT / "research_notes/baseline_experiment_design_20261022/main_table_gt610_preparation_20261023"
DOMAIN = "review-main-gt610-candidate-20261023-v2"
SOURCE = ROOT / "research_notes/baseline_experiment_design_20261022/web_dataset_survey_20261023/samples/gt610_douban.parquet"
REVISION = "d1c97233eb7bb0db8417cd6ef731c411707e12ef"
SOURCE_SHA = "f14a897a68cf0a3cfc1de8335a1b433f6939c57f0935cb24c47b2930e0be81db"


def select_documents():
    import pyarrow.parquet as pq

    sha = scaffold.sha
    assert sha(SOURCE.read_bytes()) == SOURCE_SHA
    earlier = json.loads((PREVIOUS / "DATASET_MANIFEST.json").read_text())
    data = copy.deepcopy(earlier)
    english = [d for d in data["documents"] if d["language"] == "en"]
    assert len(english) == 32
    exclusions = scaffold.PRIOR + [PREVIOUS / "DATASET_MANIFEST.json",
        ROOT / "reports/review_dataset_quality_20261023/DATASET_MANIFEST.json"]
    prior = [d for p in exclusions for d in json.loads(p.read_text())["documents"] if d["language"] == "zh"]
    prior_texts = [clean_review(d["original_text"]) for d in prior]
    prior_full = {sha(s) for s in prior_texts}
    prior_prefix = {d["prefix_sha256"] for d in prior}
    prior_shingles = [shingles(s) for s in prior_texts]
    rows = pq.read_table(SOURCE).to_pylist()
    counts = Counter(total_rows=len(rows))
    candidates, seen_full, seen_prefix = [], set(), set()
    for index, row in enumerate(rows):
        raw = row["text"]
        text = clean_review(raw)
        if len(text) < 256:
            continue
        counts["length_eligible"] += 1
        prefix = review_prefix(raw, "zh")
        full_hash, prefix_hash = sha(text), sha(prefix)
        if full_hash in seen_full or prefix_hash in seen_prefix:
            counts["duplicate"] += 1
            continue
        seen_full.add(full_hash)
        seen_prefix.add(prefix_hash)
        if full_hash in prior_full or prefix_hash in prior_prefix:
            counts["excluded_prior"] += 1
            continue
        identity = f"GT610/douban:{REVISION}:train:{index}"
        candidates.append(dict(id=f"gt610_main_{index:05d}", document_id=identity,
            row_index=index, language="zh", original_text=raw, original_text_sha256=sha(raw),
            cleaned_text_sha256=full_hash, cleaned_codepoints=len(text), prefix=prefix,
            prefix_sha256=prefix_hash, prefix_codepoints=len(prefix),
            unused_suffix_codepoints=len(text)-len(prefix), label=int(row["label"]), movie_name=None,
            selection_hash=sha(DOMAIN+":"+identity+":"+full_hash)))
    candidates.sort(key=lambda d:d["selection_hash"])
    pools = defaultdict(deque)
    for d in candidates:
        pools[d["label"]].append(d)
    chosen, chosen_shingles, rejected = [], [], []
    for _ in range(16):
        for label in (0, 1):
            while pools[label]:
                d = pools[label].popleft()
                ss = shingles(clean_review(d["original_text"]))
                if any(len(ss & other)/len(ss | other) >= .8 for other in prior_shingles+chosen_shingles):
                    rejected.append(d["document_id"])
                    continue
                chosen.append(d)
                chosen_shingles.append(ss)
                break
            else:
                raise ValueError(f"Insufficient unseen GT610 documents for label {label}")
    assert Counter(d["label"] for d in chosen) == {0:16, 1:16}
    assert Counter(d["label"] for d in chosen[:4]) == {0:2, 1:2}
    assert not {d["prefix_sha256"] for d in chosen} & prior_prefix
    assert len({d["prefix_sha256"] for d in chosen}) == 32
    for rank, d in enumerate(chosen):
        d.update(dataset="GT610/douban", revision=REVISION, config="default", split="train",
            original_field="text", cleaning_version="html-br-entity-whitespace-v1", selection_rank=rank,
            full_subset=rank<4, user_message=PREFILL_INSTRUCTIONS["zh"], assistant_prefix=d["prefix"])
    data["documents"] = english+chosen
    data["sources"] = [s for s in data["sources"] if s["language"] == "en"] + [dict(
        name="gt610_douban", dataset="GT610/douban", revision=REVISION, config="default", split="train",
        language="zh", field="text", path="douban_reviews.parquet", size=SOURCE.stat().st_size,
        sha256=SOURCE_SHA, local_file=str(SOURCE.relative_to(ROOT)),
        url=f"https://huggingface.co/datasets/GT610/douban/resolve/{REVISION}/douban_reviews.parquet")]
    data["selection"] = [s for s in data["selection"] if s["language"] == "en"] + [dict(
        language="zh", dataset="GT610/douban", counts=dict(counts), minimum_cleaned_codepoints=256,
        source_movies=None, strata=dict(Counter(str(d["label"]) for d in chosen)), group_order=["0","1"],
        per_stratum_cap=16, selected_rows=[d["row_index"] for d in chosen], full_rows=[d["row_index"] for d in chosen[:4]],
        rejected_near_duplicates=rejected, label_policy="16 per supplied label; labels hidden; no film-count claim",
        near_duplicate_rule="12-codepoint shingle Jaccard >=0.8 against prior/selected full texts")]
    data.update(selection_version=DOMAIN, selection_independent_of_generated_results=True,
        selection_rule="English inputs retained from earlier unrun candidate; Chinese SHA256 order within labels, alternating 0/1; first four form Full subset",
        prior_manifests={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in exclusions},
        scope="Length-conditioned movie-review continuation; GT610 contains short comments and has no movie IDs. No claim of PPL stability from source selection.")
    return data


if __name__ == "__main__":
    scaffold.OUT, scaffold.DOMAIN = OUT, DOMAIN
    scaffold.select_documents = select_documents
    scaffold.prepare()
    path = OUT / "PREPARATION.json"
    plan = json.loads(path.read_text())
    plan.update(dataset_selection_script_sha256=scaffold.sha(Path(__file__).read_bytes()),
        supersedes_candidate=str(PREVIOUS.relative_to(ROOT)),
        dataset_status="IMDB and GT610 selected for main table; THUCNews reserved for supplemental genre coverage",
        quality_screen_status="Ongoing Chinese Qwen k128 development pilot; English and other k are not newly quality-validated")
    scaffold.save(path, plan)
