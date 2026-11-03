"""Read-only native scoring with entropy and surprisal calibration diagnostics."""
from __future__ import annotations

import argparse
import fcntl
import gc
import math
from pathlib import Path

import torch

from research_notes.inference_conformance.hf_reference import HFReference
from stegatext_runtime.experiments.config import ROOT, fingerprint
from stegatext_runtime.experiments.native_ppl import METRIC, score_sequence
from stegatext_runtime.experiments.runner import file_hash, load_source, read, write

TILTS = (-.25, -.125, -.0625, .0625, .125, .25)
DIAGNOSTIC_VERSION = 'native-entropy-surprisal-v1'


class EntropyReference:
    """Observe the same probabilities used by the existing native scorer."""
    def __init__(self, reference):
        self.reference, self.diagnostics = reference, []

    def bind_prompt(self, ids):
        return self.reference.bind_prompt(ids)

    def next_logits(self, ids):
        return self.reference.next_logits(ids)

    def probabilities(self, ids, *, logits):
        original = self.reference.probabilities(ids, logits=logits)
        positive = original[original > 0].double()
        mass = positive.sum()
        q = positive / mass
        log_q = q.log()
        entropy = -(q*log_q).sum()
        variance = (q*(-log_q-entropy).square()).sum()
        self.diagnostics.append(dict(
            sampler_probability_sum=float(mass.item()),
            sampler_entropy_nats=float(entropy.item()),
            sampler_surprisal_variance_nats2=float(variance.item()),
            sampler_support=int(q.numel()),
            sampler_surprisal_log_mgfs=[float(torch.logsumexp((1-theta)*log_q, 0).item())
                                       for theta in TILTS]))
        return original


def enriched_score(source, sequence, *, reference=None):
    reference = reference or HFReference(source.backend, source.config, validity_mask=source.validity)
    observer = EntropyReference(reference)
    result = score_sequence(source, sequence, reference=observer)
    assert len(observer.diagnostics) == len(result['per_token'])
    for row, diagnostic in zip(result['per_token'], observer.diagnostics):
        row.update(diagnostic)
        row['normalized_sampler_nll'] = row['sampler_nll']+math.log(row['sampler_probability_sum'])
        row['surprisal_residual_nats'] = row['normalized_sampler_nll']-row['sampler_entropy_nats']
        row['surprisal_log_e_increments'] = [theta*row['normalized_sampler_nll']-mgf
            for theta,mgf in zip(TILTS,row['sampler_surprisal_log_mgfs'])]
    body = [r for r in result['per_token'] if not r['is_eos']]
    result['diagnostics_version'] = DIAGNOSTIC_VERSION
    result['surprisal_tilts'] = list(TILTS)
    for label, selected in [('body',body),('inclusive',result['per_token'])]:
        result['entropy_'+label] = dict(tokens=len(selected),
            conditional_entropy_nats=math.fsum(r['sampler_entropy_nats'] for r in selected),
            normalized_sampler_nll=math.fsum(r['normalized_sampler_nll'] for r in selected),
            surprisal_residual_nats=math.fsum(r['surprisal_residual_nats'] for r in selected),
            conditional_variance_nats2=math.fsum(r['sampler_surprisal_variance_nats2'] for r in selected),
            log_e_components=[math.fsum(r['surprisal_log_e_increments'][i] for r in selected)
                              for i in range(len(TILTS))])
    return result


def evaluate(directory, output):
    output.mkdir(parents=True,exist_ok=True)
    config = read(directory/'manifest.json')['configuration']
    contract = dict(metric=METRIC, diagnostic_version=DIAGNOSTIC_VERSION,
        source_sha256=file_hash(Path(__file__)), base_scorer_sha256=file_hash(ROOT/'stegatext_runtime/experiments/native_ppl.py'),
        tilts=TILTS, sampler_entropy='normalized actual processed source probabilities; natural log units',
        calibration='EOS-inclusive predictable surprisal mixture; body residual is descriptive only',
        source_contract='same locked native scorer; diagnostics observe probabilities without changing them')
    write(output/'contract.json',contract)
    previous = read(output/'scores.json')['rows'] if (output/'scores.json').exists() else []
    saved = {r['case_id']:r for r in previous}
    source = None
    with (ROOT/'.cache/experiment_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for row in read(directory/'results.json')['rows']:
            path = directory/'sender'/(row['case_id']+'.json')
            digest = file_hash(path)
            if row['case_id'] in saved:
                assert saved[row['case_id']]['sender_record_sha256']==digest
                continue
            sender = read(path)
            assert sender['status']=='ok'
            if source is None: source = load_source(config,row['model'])
            score = enriched_score(source,sender['result']['native_token_sequence'])
            item = dict(run_directory=str(directory.resolve()),case_id=row['case_id'],model=row['model'],
                variant=row['variant'],prompt_id=row['prompt_id'],seed=row['seed'],censored=row['censored'],
                sender_record_sha256=digest,**score)
            saved[row['case_id']] = item
            previous.append(item)
            write(output/'scores.json',dict(metric=METRIC,contract_sha256=fingerprint(contract),rows=previous))
            print(f"SCORED {len(previous)} {row['case_id']} {score['evaluator_status']}",flush=True)
        if source is not None: del source
        gc.collect()
        torch.cuda.empty_cache()
    return previous


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    evaluate(args.run_dir,args.output_dir)
