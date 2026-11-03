"""Prepare review main-table inputs/configs on CPU; never launch generation.

These are candidate inputs, separate from every historical frozen run. Runtime
cost limits still need finalizing, especially for Full Basic/MWIS reception.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

from stegatext_runtime.experiments.config import ROOT, cases, read_tree, resolve
from stegatext_runtime.experiments.review_prompts import (
    PREFILL_INSTRUCTIONS, PREFILL_VERSION, clean_review, review_prefix,
)
from tools.controlled_comparison.reproduce import write_config
from tools.review_long_sweep_20261022 import MIN_LENGTH, shingles, variants


OUT = ROOT / "research_notes/baseline_experiment_design_20261022/main_table_preparation_20261023"
PRIOR = [ROOT / f"reports/{name}/DATASET_MANIFEST.json" for name in (
    "review_instruct_pilot_20261022", "review_prefill_pilot_20261022",
    "review_long_sweep_20261022")]
KS = [16, 32, 128, 256, 0]
SEED = 470001
DOMAIN = "review-main-candidate-20261023-v1"


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def select_documents():
    import pyarrow.parquet as pq

    previous = [json.loads(path.read_text()) for path in PRIOR]
    sources = previous[-1]["sources"]
    documents, selections = [], []
    for spec in sources:
        language = spec["language"]
        path = ROOT / spec["local_file"]
        assert sha(path.read_bytes()) == spec["sha256"]
        if language == "en":
            rows = pq.read_table(path).to_pylist()
        else:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        used = {d["document_id"]: d for p in previous for d in p["documents"]
                if d["language"] == language}
        excluded_full = {d["cleaned_text_sha256"] for d in used.values()}
        excluded_prefix = {d["prefix_sha256"] for d in used.values()}
        prior_shingles = [shingles(clean_review(d["original_text"])) for d in used.values()]
        seen_full, seen_prefix, eligible = set(), set(), []
        counts = Counter(total_rows=len(rows))
        for index, row in enumerate(rows):
            raw = row[spec["field"]]
            cleaned = clean_review(raw)
            if len(cleaned) < MIN_LENGTH[language]:
                continue
            counts["length_eligible"] += 1
            if language == "zh" and row["Fake"] != "0":
                continue
            counts["label_eligible"] += 1
            prefix = review_prefix(raw, language)
            full_hash, prefix_hash = sha(cleaned), sha(prefix)
            if full_hash in seen_full or prefix_hash in seen_prefix:
                counts["duplicate"] += 1
                continue
            seen_full.add(full_hash)
            seen_prefix.add(prefix_hash)
            if full_hash in excluded_full or prefix_hash in excluded_prefix:
                counts["used_document_or_prefix"] += 1
                continue
            identity = f"{spec['dataset']}:{spec['revision']}:{spec['split']}:{index}"
            eligible.append(dict(id=f"{spec['name']}_main_{index:05d}", document_id=identity,
                row_index=index, language=language, original_text=raw,
                original_text_sha256=sha(raw), cleaned_text_sha256=full_hash,
                cleaned_codepoints=len(cleaned), prefix=prefix, prefix_sha256=prefix_hash,
                prefix_codepoints=len(prefix), unused_suffix_codepoints=len(cleaned)-len(prefix),
                label=row.get("label", row.get("Fake")), movie_name=row.get("movie_name"),
                selection_hash=sha(DOMAIN + ":" + identity + ":" + full_hash)))
        eligible.sort(key=lambda d: d["selection_hash"])
        groups = defaultdict(deque)
        for d in eligible:
            group = str(d["label"]) if language == "en" else d["movie_name"]
            groups[group].append(d)
        group_order = sorted(groups, key=lambda g: sha(DOMAIN + ":stratum:" + g))
        cap = 16 if language == "en" else 4
        chosen, chosen_shingles, chosen_counts = [], [], Counter()
        rejected_near = []
        while len(chosen) < 32:
            before = len(chosen)
            for group in group_order:
                if chosen_counts[group] >= cap:
                    continue
                while groups[group]:
                    d = groups[group].popleft()
                    ss = shingles(clean_review(d["original_text"]))
                    if any(len(ss & other) / len(ss | other) >= .8
                           for other in prior_shingles + chosen_shingles):
                        rejected_near.append(d["document_id"])
                        continue
                    chosen.append(d)
                    chosen_shingles.append(ss)
                    chosen_counts[group] += 1
                    break
                if len(chosen) == 32:
                    break
            if len(chosen) == before:
                raise ValueError(f"Insufficient documents after prespecified selection: {language}")
        assert len({d["prefix_sha256"] for d in chosen}) == 32
        assert len({d["cleaned_text_sha256"] for d in chosen}) == 32
        if language == "en":
            assert Counter(d["label"] for d in chosen) == {0: 16, 1: 16}
            assert Counter(d["label"] for d in chosen[:4]) == {0: 2, 1: 2}
        else:
            assert len(chosen_counts) == 9 and max(chosen_counts.values()) <= 4
            assert len({d["movie_name"] for d in chosen[:4]}) == 4
        for rank, d in enumerate(chosen):
            d.update(dataset=spec["dataset"], revision=spec["revision"], config=spec["config"],
                split=spec["split"], original_field=spec["field"],
                cleaning_version="html-br-entity-whitespace-v1",
                selection_rank=rank, full_subset=rank < 4,
                user_message=PREFILL_INSTRUCTIONS[language], assistant_prefix=d["prefix"])
        documents.extend(chosen)
        selections.append(dict(language=language, dataset=spec["dataset"], counts=dict(counts),
            source_movies=len({r["movie_name"] for r in rows}) if language == "zh" else None,
            minimum_cleaned_codepoints=MIN_LENGTH[language], strata=dict(chosen_counts),
            group_order=group_order, per_stratum_cap=cap, selected_rows=[d["row_index"] for d in chosen],
            full_rows=[d["row_index"] for d in chosen[:4]],
            excluded_prior_documents=sorted(used), rejected_near_duplicates=rejected_near,
            near_duplicate_rule="Jaccard of 12-codepoint shingles >=0.8 against prior/selected full texts",
            label_policy="16 per sentiment label; labels hidden" if language == "en" else
                         "Fake=0 only; labels hidden; differs from unfiltered development runs"))
    return dict(sources=sources, selection=selections, documents=documents,
        selection_version=DOMAIN, selection_independent_of_generated_results=True,
        selection_rule="SHA256 order within strata, round-robin across frozen strata; first four selected documents form Full subset",
        suffix_policy="Never supplied to model or treated as the unique correct continuation",
        prior_manifests={str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in PRIOR},
        scope="Length-conditioned review-prefix comparison. Douban is a nine-film short-comment corpus, not long-form reviews.")


def prepare():
    from transformers import AutoTokenizer
    from stegatext_runtime.models import MODEL_IDS
    from stegatext_runtime.prompts import AssistantPrefillPromptPolicy

    if OUT.exists():
        raise RuntimeError("Preparation directory already exists; inspect it instead of replacing inputs")
    data = select_documents()
    OUT.mkdir(parents=True)
    (OUT / ".gitignore").write_text("experiment_root.key\n")
    save(OUT / "DATASET_MANIFEST.json", data)
    locks = json.loads((ROOT / "environment/models.lock.json").read_text())["models"]
    prompt_rows, models = [], []
    for model, language in [("llama_instruct", "en"), ("qwen_instruct", "zh")]:
        lock = next(s for s in locks if s["repo_id"] == MODEL_IDS[model])
        model_path = ROOT / lock["local_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        models.append({k: lock[k] for k in ("repo_id", "revision", "tokenizer_revision", "dtype", "template_options")})
        for doc in [d for d in data["documents"] if d["language"] == language]:
            policy = AssistantPrefillPromptPolicy(doc["prefix"], template_options=lock["template_options"])
            rendered = policy.render_text(tokenizer, doc["user_message"])
            ids = policy.render(tokenizer, doc["user_message"])[0].tolist()
            messages = [dict(role="user", content=doc["user_message"]),
                        dict(role="assistant", content=doc["prefix"])]
            expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                continue_final_message=True, **lock["template_options"])
            assert ids == expected == tokenizer.encode(rendered, add_special_tokens=False)
            assert rendered.endswith(doc["prefix"]) and rendered.count(doc["prefix"]) == 1
            prompt_rows.append(dict(model=model, prompt_id=doc["id"], messages=messages,
                rendered_prompt=rendered, prompt_token_ids=ids, prompt_token_count=len(ids),
                rendered_prompt_sha256=sha(rendered), chat_template_sha256=sha(tokenizer.chat_template),
                template_options=lock["template_options"], continue_final_message=True,
                add_generation_prompt=False, tokenizer_revision=lock["tokenizer_revision"],
                tokenizer_files_sha256={name: sha((model_path / name).read_bytes())
                    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
                    if (model_path / name).is_file()}))
    save(OUT / "PROMPT_AUDIT.json", dict(template_version=PREFILL_VERSION, rows=prompt_rows))
    key_path = OUT / "experiment_root.key"
    descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(os.urandom(32).hex() + "\n")
    base = read_tree(ROOT / "configs/experiments/token_budget_T1.toml")
    base["randomness"] = dict(mode="key_file", file=str(key_path),
        stream_policy="independent_cases_v1", experiment_id=DOMAIN)
    base["execution"] = dict(order="hash_interleaved_v1", order_seed=20261023)
    jobs, case_rows = [], []
    for model, language in [("llama_instruct", "en"), ("qwen_instruct", "zh")]:
        for k in KS:
            name = f"review_main_{model}_k{k}_s{SEED}"
            config = copy.deepcopy(base)
            config["name"] = name
            config["source"].update(models=[model], languages={model: language})
            config["source"]["sampling"] = dict(do_sample=True, temperature=1., top_p=1., top_k=k,
                repetition_penalty=1., min_p=0., typical_p=1.)
            selected = [d for d in data["documents"] if d["language"] == language
                        and (k != 0 or d["full_subset"])]
            config["inputs"] = dict(ordinary_seed_policy="independent_conditions_v1", seeds=[SEED],
                prompts=[dict(id=d["id"], language=language, text=d["user_message"],
                              assistant_prefix=d["prefix"]) for d in selected])
            config["variants"] = [v for v in variants() if k != 0 or v["id"] != "stepwise_imec"]
            path = OUT / "configs" / (name + ".toml")
            write_config(path, config)
            expanded = list(cases(resolve(path)))
            assert len(expanded) == (44 if k == 0 else 384)
            jobs.append(dict(model=model, language=language, top_k=k, documents=len(selected),
                config=str(path.relative_to(ROOT)), config_sha256=sha(path.read_bytes()), cases=len(expanded)))
            for case in expanded:
                case_rows.append(dict(config=path.name, model=model, language=language, top_k=k,
                    prompt_id=case["prompt"]["id"], seed=case["seed"], variant=case["variant"]["id"],
                    case_id=case["id"]))
    assert len(case_rows) == 3160
    assert sum(r["variant"] != "ordinary" for r in case_rows) == 2896
    # Runtime case IDs are scoped to a config; k lives in that config.
    assert len({(r["config"], r["case_id"]) for r in case_rows}) == 3160
    with (OUT / "CASES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    plan = dict(status="candidate_inputs_prepared_no_generation_launched_runtime_cost_limits_pending",
        preparation_script_sha256=sha(Path(__file__).read_bytes()), jobs=jobs, models=models,
        top_k=KS, full_means_top_k_zero=True, tau=[.125, .25, .5, 1., 2., 4.],
        documents_per_language=32, full_documents_per_language=4, repeats_per_document=1,
        seed=SEED, generated_cases=3160, independent_receivers=2896, model_k_variant_cells=118,
        coder=dict(name="imec_prefix_exact", block_size=6, window_size=3,
                   coupling_strategy="greedy", schedule="head_first"),
        stopping=base["stopping"], payload=base["message"],
        full_scope="No probability cutoff; existing byte_codec token-validity mask remains disclosed",
        ppl="Normal main-table metric; sampling is a working assumption; no separate PPL study scheduled",
        followup="Repeat original prompts/settings for fluctuating cells and corresponding Ordinary; new independent streams, separate batch summaries, equal document weighting",
        execution="Not launched. One GPU worker, CPU thread limits, bounded records and lossless gzip required",
        main_wall_hours_limit=36,
        unresolved="Chinese Full and Full independent reception costs; finalize external stage/resource limits before generation",
        file_sha256={name: sha((OUT / name).read_bytes())
                     for name in ("DATASET_MANIFEST.json", "PROMPT_AUDIT.json", "CASES.csv")})
    save(OUT / "PREPARATION.json", plan)
    print(json.dumps(dict(status=plan["status"], generated_cases=3160, independent_receivers=2896,
        selection=[{k:s[k] for k in ("language", "strata", "selected_rows", "full_rows")} for s in data["selection"]],
        prompt_tokens={m:[min(r["prompt_token_count"] for r in prompt_rows if r["model"] == m),
                         max(r["prompt_token_count"] for r in prompt_rows if r["model"] == m)]
                       for m in ("llama_instruct", "qwen_instruct")}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
