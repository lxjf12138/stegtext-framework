"""Change only the public prompt; retain official-aligned sampling parameters."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from stegatext_runtime import Source
from stegatext_runtime.protocol import encode
from research_notes.full_validation.suite import ROOT, PROMPT, ZH_PROMPT, bits_for, fixture_key, write
from research_notes.full_validation.cache import accelerate
from research_notes.full_validation.report import quality
from research_notes.problem_resolution.text_rules import complete_sentences, sentence_end


HERE = ROOT/"research_notes/problem_resolution"
OUT = ROOT/"runs/problem_resolution_v1"
DRIFT = re.compile(r"Answer questions|What is the purpose of|be going to|Write an article|What is the author['’]s|阅读理解|回答下列|请回答|翻译成|下列问题", re.I)


def metrics(text, language):
    compact = re.sub(r"\s+", "", text)
    ngrams = [compact[i:i+12] for i in range(max(0, len(compact)-11))]
    return dict(**quality(text), substantive_complete_sentences=len(complete_sentences(text, language)),
                sentence_end=sentence_end(text, language), drift_markers=DRIFT.findall(text),
                repeated_12char_fraction=1-len(set(ngrams))/len(ngrams) if ngrams else 0.)


def run(model):
    source = accelerate(Source.load(model, "stego"))
    for language, old_prompt in (("en", PROMPT), ("zh", ZH_PROMPT)):
        new_prompt = (HERE/"prompts"/f"news_{language}.txt").read_text().rstrip("\n")
        for seed in (11, 29):
            for mode in ("ordinary", "adaptive"):
                # Same random fixture within a prompt pair, different from
                # every other case. This is a controlled prompt comparison.
                key = fixture_key(model, f"prompt-ablation:{language}:{seed}:{mode}")
                for variant, prompt in (("old", old_prompt), ("news", new_prompt)):
                    name = f"{language}__{seed}__{mode}__{variant}"
                    path = OUT/model/"prompt_ablation"/(name+".json")
                    if path.exists():
                        continue
                    if mode == "ordinary":
                        result = source.generate(prompt, seed=seed, max_new_tokens=384)
                    else:
                        packet, result = encode(source, prompt, bits_for(key), key=key,
                                                policy="adaptive", tau=.5, max_decisions=384)
                        result = dict(text=result["text"], source=packet["source"], natural_eos=result["natural_eos"],
                                      decisions=result["steps"], bits_count=result["bits_count"])
                    record = {**result, "model":model, "case":name, "language":language, "seed":seed, "mode":mode,
                              "prompt_variant":variant, "prompt":prompt,
                              "review":metrics(result["text"], language)}
                    write(path, record)
                    print(json.dumps(dict(model=model, case=name, characters=len(record["text"]), review=record["review"]), ensure_ascii=False), flush=True)


def summary():
    rows = []
    samples = ["# Prompt comparison texts\n", "Each old/news pair changes only the public prompt; the model, sampling parameters, message, and random fixture are fixed. Ordinary samples from the shared HF probability source; adaptive uses the corrected DisCop coder. The budgets are 384 new tokens and 384 protocol decisions, so the samples do not have equal length. Automatic markers are for screening only.\n"]
    for model in ("qwen", "llama"):
        for path in sorted((OUT/model/"prompt_ablation").glob("*.json")):
            r = json.loads(path.read_text())
            rows.append({k:v for k,v in r.items() if k in ("model", "case", "language", "seed", "mode", "prompt_variant", "review", "decisions", "bits_count")})
            samples += [f"## {model} / {r['case']}\n", f"Prompt:\n\n```text\n{r['prompt']}\n```\n", f"Continuation:\n\n```text\n{r['text']}\n```\n", json.dumps(r['review'], ensure_ascii=False)+"\n"]
    write(HERE/"prompt_results.json", dict(rows=rows, scope="Small controlled public-prompt comparison; no human equivalence or broad quality claim."))
    (HERE/"PROMPT_TEXT_SAMPLES.md").write_text("\n".join(samples))
    print(json.dumps(dict(cases=len(rows), drift_flagged=sum(bool(r['review']['drift_markers']) for r in rows))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("qwen", "llama"))
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    summary() if a.summary else run(a.model)
