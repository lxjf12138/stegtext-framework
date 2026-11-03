"""Legacy retokenized visible-text evaluator; not native generation-path PPL.

Kept for reproducing historical reports. New CLI scoring defaults to native_ppl.
"""
from __future__ import annotations

import fcntl
import hashlib
import math
from pathlib import Path

import torch

from .config import ROOT, fingerprint
from .runner import file_hash, read, write


def windows(count, context=512, stride=256):
    """Yield input bounds and prediction bounds in [initial]+body coordinates."""
    if count < 0 or not 1 <= stride < context:
        raise ValueError("Require count >= 0 and 1 <= stride < context")
    done = 0
    while done < count:
        end = min(count, done+stride)
        begin = max(0, end+1-context)
        yield begin, end+1, done-begin, end-begin
        done = end


@torch.inference_mode()
def score_ids(model, ids, initial, *, context=512, stride=256, device="cuda:0"):
    if not ids:
        return dict(evaluator_status="empty_visible_text", evaluator_tokens=0,
                    nll_total=None, log_ppl=None, ppl=None)
    sequence = [initial]+ids
    losses = []
    for begin, end, lo, hi in windows(len(ids), context, stride):
        values = torch.tensor([sequence[begin:end]], device=device)
        logits = model(values, use_cache=False).logits[0, lo:hi].float()
        targets = values[0, lo+1:hi+1]
        losses.append(torch.nn.functional.cross_entropy(logits, targets, reduction="sum").double().item())
    total = math.fsum(losses)
    ell = total/len(ids)
    return dict(evaluator_status="scored", evaluator_tokens=len(ids), nll_total=total,
                log_ppl=ell, ppl=math.exp(ell) if ell < 700 else None)


def evaluate(run_dirs, output_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = __import__('json').loads((ROOT/'environment/models.lock.json').read_text())
    spec = next(s for s in lock['models'] if s['repo_id'] == 'Qwen/Qwen3-4B-Base')
    folder = ROOT/spec['local_path']
    contract = dict(metric='legacy-retokenized-visible-text-base-v1', retokenization=True,
                    repo_id=spec['repo_id'], revision=spec['revision'], tokenizer_revision=spec['tokenizer_revision'],
                    context=512, stride=256, dtype='bfloat16', attention='sdpa',
                    conditioning='single BOS else first EOS; standalone visible body; no prompt or terminal EOS',
                    logits='unmodified full-vocabulary; every body token once',
                    files={p.name: file_hash(p) for p in sorted(folder.glob('*.json'))},
                    weights={p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(folder.glob('*.safetensors'))},
                    source_sha256=file_hash(Path(__file__)),
                    model_lock_sha256=file_hash(ROOT/'environment/models.lock.json'))
    write(output_dir/'contract.json', contract)
    lock_path = ROOT/'.cache/experiment_gpu0.lock'
    with lock_path.open('a') as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, dtype=torch.bfloat16,
                    attn_implementation='sdpa', device_map={'': 'cuda:0'}).eval()
        initial = tokenizer.bos_token_id
        if initial is None:
            initial = tokenizer.eos_token_id
        if initial is None:
            raise ValueError('Evaluator requires a fixed initial token')
        write(output_dir/'initial_token.json', dict(initial_token_id=initial, contract_sha256=fingerprint(contract)))
        rows, cache = [], {}
        for directory in map(Path, run_dirs):
            results = read(directory/'results.json')
            for row in results['rows']:
                sender = read(directory/'sender'/(row['case_id']+'.json'))
                body = sender.get('result', {}).get('text')
                item = dict(run_directory=str(directory.resolve()), case_id=row['case_id'],
                            model=row['model'], variant=row['variant'],
                            prompt_id=row['prompt_id'], seed=row['seed'],
                            censored=row['censored'], sender_record_sha256=file_hash(directory/'sender'/(row['case_id']+'.json')))
                if sender['status'] != 'ok' or body is None:
                    item.update(evaluator_status='unavailable_invalid_or_failed', evaluator_tokens=0,
                                nll_total=None, log_ppl=None, ppl=None, nll_per_byte=None)
                else:
                    digest = hashlib.sha256(body.encode()).hexdigest()
                    if digest not in cache:
                        ids = tokenizer.encode(body, add_special_tokens=False)
                        try:
                            value = score_ids(model, ids, initial)
                            value.update(nll_per_byte=value['nll_total']/len(body.encode()) if body else None,
                                         roundtrip_text_changed=tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != body)
                        except Exception as exc:
                            value = dict(evaluator_status='evaluation_error', error=f'{type(exc).__name__}: {exc}',
                                         evaluator_tokens=0, nll_total=None, log_ppl=None, ppl=None, nll_per_byte=None)
                        cache[digest] = value
                    item.update(cache[digest], text_sha256=digest, text_utf8_bytes=len(body.encode()))
                rows.append(item)
                write(output_dir/'scores.json', dict(contract_sha256=fingerprint(contract), rows=rows))
                print(f"PPL {len(rows)} {row['case_id']} {item['evaluator_status']}", flush=True)
    return rows
