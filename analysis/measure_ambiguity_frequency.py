#!/usr/bin/env python3
"""Ambiguity frequency on the main batch's ordinary-sampling texts.

Replays every recorded ordinary sequence under the main-batch sampler (same
model files, top-k, validity mask and prompt) and, at each body step, checks
whether the sampled token's codec bytes stand in a prefix relation with another
token in the sampler support. This is the token-level frequency used by
SyncPool and ReTokSync. It also checks whether re-encoding the visible text
reproduces the sampled IDs (the text-level inconsistency rate used by Stepwise).

Run from the repository root after the main comparison:
  source environment/activate.sh
  python analysis/measure_ambiguity_frequency.py
"""
from __future__ import annotations

import bisect
import csv
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STEGATEXT = Path(os.environ.get('STEGTEXT_PROJECT_ROOT', ROOT))
RUNS = STEGATEXT/'reports/review_main_batched_20261024/runs'
OUT = ROOT/'evidence/derived/experiments_20261101'
MODELS = ('llama_instruct', 'qwen_instruct')
KS = (16, 32, 128, 256, 0)
FIELDS = ('model', 'top_k', 'prompt_id', 'case_id', 'sender_record_sha256', 'body_steps',
          'ambiguous_steps', 'longer_candidate_steps', 'shorter_candidate_steps',
          'mean_support', 'retokenization_status', 'retokenization_equal',
          'max_selected_probability_abs_error', 'support_count_mismatches')


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        return json.load(f)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrefixIndex:
    """Prefix relations among codec byte strings; empty strings never relate."""

    def __init__(self, table):
        self.table = table
        self.order = np.array(sorted(range(len(table)), key=lambda i: table[i]), dtype=np.int64)
        self.keys = [table[i] for i in self.order]
        self.exact = {}
        for i, b in enumerate(table):
            if b:
                self.exact.setdefault(b, []).append(i)

    def extension_range(self, b):
        lo = bisect.bisect_left(self.keys, b)
        stem = b.rstrip(b'\xff')
        if not stem:
            return lo, len(self.keys)
        upper = stem[:-1] + bytes([stem[-1] + 1])
        return lo, bisect.bisect_left(self.keys, upper)

    def relations(self, token, support):
        """Return (has_longer_or_equal, has_shorter) for another supported token."""
        b = self.table[token]
        if not b:
            raise ValueError('Body token has no codec bytes')
        lo, hi = self.extension_range(b)
        ordered = support[self.order[lo:hi]]
        longer = int(ordered.sum()) - int(support[token]) > 0
        shorter = any(support[c] for j in range(1, len(b)) for c in self.exact.get(b[:j], ()))
        return longer, shorter


def run_directories(model, k):
    return sorted(RUNS.glob(f'review_main_{model}_k{k}_s470001_b*_p*'))


@torch.inference_mode()
def measure(source, index, prompt, record):
    from research_notes.inference_conformance.hf_reference import HFReference
    from stegatext_runtime.prompts import configure_prompt
    result = record['result']
    seq = result['native_token_sequence']
    # The source manifest covers the assistant prefill, so configure it first.
    configure_prompt(source, prompt.get('assistant_prefix'))
    if seq['status'] != 'verified' or seq['source_sha256'] != source.manifest()['sha256']:
        raise ValueError('Sequence is not a verified record of this source')
    ids, eos = seq['output_token_ids'], set(seq['eos_token_ids'])
    steps = result['ordinary_sampling']
    assert len(steps) == len(ids)
    ref = HFReference(source.backend, source.config, validity_mask=source.validity)
    ref.bind_prompt(seq['prompt_token_ids'])
    n = source.backend.codec.vocab_size
    history = list(seq['prompt_token_ids'])
    counts = dict(body_steps=0, ambiguous_steps=0, longer_candidate_steps=0, shorter_candidate_steps=0)
    support_total, max_err, mismatches = 0, 0.0, 0
    for token, step in zip(ids, steps):
        probabilities = ref.probabilities(history)
        p = probabilities[:n].double().cpu().numpy()
        support = p > 0
        max_err = max(max_err, abs(float(p[token]) - step['selected_token_probability']))
        mismatches += int(support.sum()) != step['source_support_count']
        if token not in eos:
            longer, shorter = index.relations(token, support)
            counts['body_steps'] += 1
            counts['ambiguous_steps'] += longer or shorter
            counts['longer_candidate_steps'] += longer
            counts['shorter_candidate_steps'] += shorter
            support_total += int(support.sum())
        history.append(token)
    body = [t for t in ids if t not in eos]
    if result.get('framing_status') != 'valid_utf8':
        status, equal = 'not_valid_utf8', ''
    else:
        try:
            status = 'compared'
            equal = int(source.retokenize_text(prompt['text'], result['text']) == body)
        except ValueError as exc:
            # Re-encoding merged the first body bytes into the public prefill: a mismatch.
            if 'prompt boundary' not in str(exc):
                raise
            status, equal = 'boundary_merged', 0
    return dict(counts, mean_support=support_total/max(counts['body_steps'], 1),
                retokenization_status=status, retokenization_equal=equal,
                max_selected_probability_abs_error=max_err, support_count_mismatches=mismatches)


def main():
    from stegatext_runtime.experiments.runner import load_source
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT/'AMBIGUITY_FREQUENCY_DOCS.csv'
    done = set()
    if out_csv.exists():
        done = {(r['model'], int(r['top_k']), r['case_id']) for r in csv.DictReader(out_csv.open())}
    writer_file = out_csv.open('a', newline='')
    writer = csv.DictWriter(writer_file, FIELDS)
    if not done:
        writer.writeheader()
    torch.set_num_threads(1)
    started = time.time()
    for model in MODELS:
        for k in KS:
            dirs = run_directories(model, k)
            assert dirs, (model, k)
            source, index = None, None
            for directory in dirs:
                config = read(directory/'manifest.json')['configuration']
                assert config['source']['sampling']['top_k'] == k
                prompts = {p['id']: p for p in config['inputs']['prompts']}
                for path in sorted((directory/'sender').glob('*__ordinary.json*')):
                    record = read(path)
                    if record['status'] != 'ok' or (model, k, record['case_id']) in done:
                        continue
                    if source is None:
                        source = load_source(config, model)
                        index = PrefixIndex(source.backend.codec.token_bytes_table)
                    prompt_id = record['case_id'].split('__')[1]
                    row = dict(model=model, top_k=k, prompt_id=prompt_id, case_id=record['case_id'],
                               sender_record_sha256=sha256(path),
                               **measure(source, index, prompts[prompt_id], record))
                    writer.writerow(row)
                    writer_file.flush()
                    print(f"{time.time()-started:7.0f}s {model} k={k} {prompt_id} "
                          f"amb={row['ambiguous_steps']}/{row['body_steps']} retok={row['retokenization_equal']} "
                          f"err={row['max_selected_probability_abs_error']:.2e} mism={row['support_count_mismatches']}",
                          flush=True)
            if source is not None:
                del source, index
                gc.collect()
                torch.cuda.empty_cache()
    writer_file.close()
    (OUT/'AMBIGUITY_FREQUENCY_RUN.json').write_text(json.dumps(dict(
        script_sha256=sha256(Path(__file__)), runs=str(RUNS),
        definition=('A body step is ambiguous when another token in the sampler support has codec bytes '
                    'that extend, equal, or are a proper prefix of the sampled token bytes.'),
        text_level='Visible text re-encoded with the public prompt boundary equals the sampled body IDs.',
        rows_sha256=sha256(out_csv), wall_seconds=time.time()-started), indent=2))


if __name__ == '__main__':
    main()
