"""Prompt-conditioned PPL on saved runtime IDs; never re-encode visible text.

The primary score uses raw, full-vocabulary generating-model logits. Sampler
likelihood is reported separately. Synthetic budget delimiters are not scored.
"""
from __future__ import annotations

import fcntl
import gc
import hashlib
import math
from pathlib import Path

import torch

from research_notes.inference_conformance.hf_reference import HFReference
from stegatext_runtime.native_tokens import VERSION

METRIC = 'native-path-generating-model-prompt-conditioned-v1'


def summary(losses):
    total = math.fsum(losses)
    mean = total/len(losses) if losses else None
    return dict(tokens=len(losses), nll_total=total if losses else None, log_ppl=mean,
                ppl=math.exp(mean) if mean is not None and mean < 700 else None)


@torch.inference_mode()
def score_sequence(source, sequence, *, reference=None):
    if sequence.get('version') != VERSION or sequence.get('status') != 'verified' or not sequence.get('wire_reconstruction_verified'):
        raise ValueError('PPL requires a verified recorded runtime token sequence')
    if sequence['source_sha256'] != source.manifest()['sha256']:
        raise ValueError('Native IDs must use the same model/tokenizer/source contract')
    prompt, ids = sequence['prompt_token_ids'], sequence['output_token_ids']
    if not prompt:
        raise ValueError('Native PPL requires the actual nonempty rendered prompt')
    eos = sequence['eos_token_ids']
    ended = bool(ids and ids[-1] in eos)
    if ended != sequence['natural_eos'] or any(t in eos for t in ids[:-1]):
        raise ValueError('Invalid recorded EOS')
    wire_ids = ids + ([sequence['wire_output_token_ids'][-1]] if sequence['synthetic_boundary_eos'] else [])
    if wire_ids != sequence['wire_output_token_ids']:
        raise ValueError('Internal delimiter is not isolated from the scored tokens')
    if sequence['body_token_count'] != len(ids)-int(ended):
        raise ValueError('Native body token count mismatch')
    if sequence['source_token_count'] != len(ids) or hashlib.sha256(
            source.backend.codec.tokens_to_bytes(wire_ids)).hexdigest() != sequence['wire_sha256']:
        raise ValueError('Native IDs no longer realize the verified wire')
    ref = reference or HFReference(source.backend, source.config, validity_mask=source.validity)
    ref.bind_prompt(prompt)
    history, rows = list(prompt), []
    for index, token in enumerate(ids):
        logits = ref.next_logits(history)
        raw_nll = -torch.log_softmax(logits.double(), dim=-1)[token].item()
        probabilities = ref.probabilities(history, logits=logits)
        probability = probabilities[token].item()
        if probability <= 0 or not math.isfinite(raw_nll):
            raise ValueError('Recorded native token has zero source support or invalid raw likelihood')
        argmax = int(probabilities.argmax().item())
        rows.append(dict(index=index, token_id=token, is_eos=token in eos,
                         model_nll=raw_nll, sampler_nll=-math.log(probability),
                         sampler_probability=probability, sampler_argmax_id=argmax,
                         sampler_argmax_probability=float(probabilities[argmax].item()),
                         selected_is_argmax=token == argmax))
        history.append(token)
    body = [r for r in rows if not r['is_eos']]
    main = summary([r['model_nll'] for r in body])
    return dict(metric=METRIC, input_kind='recorded_runtime_token_ids',
                evaluator_status='scored' if body else 'empty_generated_body',
                conditioning='actual rendered prompt and actual preceding output IDs; prompt excluded from loss',
                logits='raw full-vocabulary model; no top-k/temperature/validity processors',
                eos_policy='primary excludes natural EOS; inclusive score separate; synthetic delimiter never scored',
                evaluator_tokens=main['tokens'], nll_total=main['nll_total'],
                log_ppl=main['log_ppl'], ppl=main['ppl'],
                eos_inclusive=summary([r['model_nll'] for r in rows]),
                sampler_body=summary([r['sampler_nll'] for r in body]),
                non_argmax_body_tokens=sum(not r['selected_is_argmax'] for r in body), per_token=rows)


def evaluate(run_dirs, output_dir):
    from .config import ROOT, fingerprint
    from .runner import file_hash, load_source, read, write
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = dict(metric=METRIC, sequence_version=VERSION, retokenization=False,
                    evaluator='same generating model and tokenizer revision as each source',
                    conditioning='actual prompt', eos='body primary; natural EOS inclusive secondary; internal delimiter excluded',
                    logits='raw full vocabulary', history_schedule='fixed prompt prefill then single token replay',
                    source_sha256=file_hash(Path(__file__)))
    write(output_dir/'contract.json', contract)
    lock_path = ROOT/'.cache/experiment_gpu0.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}
    with lock_path.open('a') as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        source, source_key = None, None
        for directory in map(Path, run_dirs):
            config = read(directory/'manifest.json')['configuration']
            for row in read(directory/'results.json')['rows']:
                path = directory/'sender'/(row['case_id']+'.json')
                sender = read(path)
                seq = sender.get('result', {}).get('native_token_sequence')
                item = dict(run_directory=str(directory.resolve()), case_id=row['case_id'],
                            model=row['model'], variant=row['variant'], prompt_id=row['prompt_id'],
                            seed=row['seed'], censored=row['censored'], sender_record_sha256=file_hash(path), metric=METRIC)
                if sender['status'] != 'ok' or seq is None or seq['status'] != 'verified':
                    item.update(evaluator_status='unavailable_no_verified_native_sequence',
                                evaluator_tokens=0, ppl=None, log_ppl=None, nll_total=None)
                else:
                    new_key = fingerprint([row['model'], config['source']])
                    if source_key != new_key:
                        if source is not None:
                            del source
                            gc.collect()
                            torch.cuda.empty_cache()
                        source = load_source(config, row['model'])
                        source_key = new_key
                    from stegatext_runtime.prompts import configure_prompt
                    prompt = next(p for p in config['inputs']['prompts'] if p['id'] == row['prompt_id'])
                    configure_prompt(source, prompt.get('assistant_prefix'))
                    key = fingerprint(seq)
                    if key not in cache:
                        try:
                            cache[key] = score_sequence(source, seq)
                        except Exception as exc:
                            cache[key] = dict(evaluator_status='evaluation_error', error=f'{type(exc).__name__}: {exc}',
                                              evaluator_tokens=0, ppl=None, log_ppl=None, nll_total=None)
                    item.update(cache[key], native_sequence_sha256=key)
                rows.append(item)
                write(output_dir/'scores.json', dict(metric=METRIC, contract_sha256=fingerprint(contract), rows=rows))
                print(f"NATIVE PPL {len(rows)} {row['case_id']} {item['evaluator_status']}", flush=True)
    return rows
