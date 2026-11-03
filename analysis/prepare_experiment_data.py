#!/usr/bin/env python3
"""Derive aggregate manuscript evidence offline; never run or rewrite experiments."""
from pathlib import Path
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('llama_instruct', 'qwen_instruct')
KS = (16, 32, 128, 256, 0)
TAUS = {'las_imec': 0, 'adaptive_0125_imec': .125,
        'adaptive_0250_imec': .25, 'adaptive_0500_imec': .5,
        'adaptive_1000_imec': 1, 'adaptive_2000_imec': 2,
        'adaptive_4000_imec': 4}
VARIANTS = (*TAUS, 'syncpool_imec', 'mwis_imec', 'basic_imec', 'stepwise_imec')
DRAWS = 10000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sums(rows):
    return np.array([[r['strict_verified_bits'] or 0, r['bytes'],
                      r['body_tokens'] or 0,
                      (r['sender_seconds'] or 0) + (r['receiver_seconds'] or 0),
                      r['strict_text_delivery_bits'] or 0,
                      r['actual_model_forward_calls'] or 0] for r in rows], float)


def metrics(t):
    b, y, n, s, v, c = np.moveaxis(t, -1, 0)
    return {'native_body_bpt': b/n, 'byte_bpb': b/y,
            'valid_text_bpb': v/y, 'sender_receiver_ms_per_byte': 1000*s/y,
            'sender_receiver_ms_per_body_token': 1000*s/n,
            'confirmed_bits_per_sender_receiver_second': b/s,
            'sender_forward_calls_per_body_token': c/n}


def boot_indices(key, n):
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return seed, np.random.default_rng(seed).integers(0, n, (DRAWS, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    src = args.source_root
    raw_rel = 'reports/project_understanding_stage3_20261030/CORRECTED_CASES.json.gz'
    raw = src/raw_rel
    hashes = json.loads((src/'reports/revision_readiness_20261030/INPUT_HASHES.json').read_text())
    assert digest(raw) == hashes[raw_rel], 'Corrected input changed since frozen audit'
    cases = json.loads(gzip.decompress(raw.read_bytes()))
    assert len(cases) == 6080
    assert len({(r['source_batch'], r['top_k'], r['case_id']) for r in cases}) == 6080
    main_cases = [r for r in cases if r['source_batch'] == 'main']
    out = ROOT/'evidence/derived/experiments_20261101'
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for model in MODELS:
        for k in KS:
            for v in VARIANTS:
                rows = sorted([r for r in main_cases if (r['model'], r['top_k'], r['variant'])
                               == (model, k, v)], key=lambda r: r['prompt_id'])
                if not rows:
                    assert k == 0 and v == 'stepwise_imec'
                    continue
                assert len(rows) == len({r['prompt_id'] for r in rows}) == 32
                groups[model, k, v] = rows
    assert len(groups) == 108
    frozen = list(csv.DictReader((ROOT/'evidence/snapshots/MAIN_ALL_METHODS_OPERATING_POINTS.csv').open()))
    point_rows, ci_rows = [], []
    checked_fields = 0
    for key, rows in groups.items():
        model, k, variant = key
        total = sums(rows).sum(axis=0)
        reference = next(r for r in frozen if (r['model'], int(r['top_k']), r['variant']) == key)
        values = metrics(total)
        for field, value in values.items():
            assert math.isclose(value, float(reference[field]), rel_tol=1e-12, abs_tol=1e-12), (key, field)
            checked_fields += 1
        point = dict(reference)
        point.update(values)
        point.update(invalid_utf8_success=sum(r['sender_status']=='ok' and not r['text_renderable'] for r in rows),
                     sender_failed=sum(r['sender_status']!='ok' for r in rows))
        point_rows.append(point)
        seed, indices = boot_indices('main/'+ '/'.join(map(str, key)), len(rows))
        for field, values in metrics(sums(rows)[indices].sum(axis=1)).items():
            lo, hi = np.quantile(values, [.025, .975])
            ci_rows.append(dict(model=model, top_k=k, variant=variant, metric=field,
                                ci95_low=float(lo), ci95_high=float(hi),
                                bootstrap_seed=seed, bootstrap_draws=DRAWS, n_documents=32))
    write_csv(out/'MAIN_POINTS.csv', point_rows)
    write_csv(out/'MAIN_POINT_INTERVALS.csv', ci_rows)

    recoveries = []
    for family in ('LAS', 'SyncPool', 'MWIS', 'Basic', 'Stepwise', 'All steganographic'):
        def selected(r):
            v = r['variant']
            return v in TAUS if family == 'LAS' else (
                v != 'ordinary' if family == 'All steganographic' else v == family.lower()+'_imec')
        for cohort in ('main', 'repeat_01', 'repeat_02', 'all'):
            rows = [r for r in cases if selected(r) and (cohort == 'all' or r['source_batch'] == cohort)]
            recoveries.append(dict(family=family, cohort=cohort, planned=len(rows),
                sender_success=sum(r['sender_status'] == 'ok' for r in rows),
                exact=sum(r['exact'] is True for r in rows),
                zero_payload_exact=sum(r['exact'] is True and r['strict_verified_bits'] == 0 for r in rows),
                invalid_utf8=sum(r['sender_status'] == 'ok' and not r['text_renderable'] for r in rows),
                wrong_after_success=sum(r['sender_status'] == 'ok' and r['exact'] is not True for r in rows)))
    write_csv(out/'RECOVERY.csv', recoveries)

    comparisons, ledger = [], []
    for model in MODELS:
        for k in KS:
            a, b = groups[model,k,'las_imec'], groups[model,k,'syncpool_imec']
            assert [r['prompt_id'] for r in a] == [r['prompt_id'] for r in b]
            ma, mb = metrics(sums(a).sum(axis=0)), metrics(sums(b).sum(axis=0))
            seed, indices = boot_indices(f'paired/main/{model}/{k}', 32)
            ba, bb = metrics(sums(a)[indices].sum(axis=1)), metrics(sums(b)[indices].sum(axis=1))
            for metric in ('native_body_bpt', 'byte_bpb', 'sender_receiver_ms_per_body_token'):
                lo, hi = np.quantile(ba[metric]/bb[metric], [.025, .975])
                comparisons.append(dict(model=model, top_k=k, metric=metric,
                    las0_over_shared_syncpool=ma[metric]/mb[metric],
                    ci95_low=float(lo), ci95_high=float(hi), bootstrap_seed=seed,
                    bootstrap_draws=DRAWS, n_documents=32))
            fields = ('group_entropy_bits','expected_coder_information_bits',
                      'realized_information_bits','strict_verified_bits','pending_information_bits')
            totals = {field: sum(r[field] or 0 for r in a) for field in fields}
            tokens = sum(r['body_tokens'] or 0 for r in a)
            assert math.isclose(totals['realized_information_bits'],
                totals['strict_verified_bits'] + totals['pending_information_bits'], rel_tol=1e-12)
            ledger.append(dict(model=model, top_k=k, n=32, body_tokens=tokens, **totals,
                               delivered_over_group=totals['strict_verified_bits']/totals['group_entropy_bits']))
    write_csv(out/'MAIN_LAS0_SYNCPOOL_COMPARISONS.csv', comparisons)
    write_csv(out/'MAIN_LAS0_LEDGER.csv', ledger)

    gates = []
    for (model,k,v), rows in groups.items():
        if v not in TAUS:
            continue
        count = sum(r['positive_u_sync_actions'] or 0 for r in rows)
        uncertainty = sum(r['synchronized_uncertainty_bits'] or 0 for r in rows)
        tau = TAUS[v]
        assert uncertainty <= tau*count + 1e-7
        gates.append(dict(model=model, top_k=k, tau_bits=tau, n=32,
            positive_u_whole_group_updates=count, synchronized_uncertainty_bits=uncertainty,
            tau_times_count_bits=tau*count,
            max_retained_frontier=max(r['retained_frontier_max'] or 0 for r in rows)))
    write_csv(out/'MAIN_GATE_DIAGNOSTICS.csv', gates)

    ordinary = {}
    for model in MODELS:
        for k in KS:
            rows = sorted([r for r in main_cases if (r['model'], r['top_k'], r['variant'])
                           == (model, k, 'ordinary')], key=lambda r: r['prompt_id'])
            assert len(rows) == len({r['prompt_id'] for r in rows}) == 32
            ordinary[model, k] = rows

    def quantiles(x):
        return [float(v) for v in np.quantile(x, [.025, .975])]

    def quality_sums(rows):
        # bits, body tokens, sampler entropy bits, filtering KL bits, NLL nats, scored tokens
        return np.array([[r['strict_verified_bits'] or 0, r['body_tokens'] or 0,
                          (r.get('conditional_entropy_nats') or 0)/math.log(2),
                          r.get('filtered_path_kl_total_bits') or 0,
                          r.get('nll_total') or 0, r.get('evaluator_tokens') or 0]
                         for r in rows], float)

    def corpus_ppl(t):
        return np.exp(t[..., 4]/t[..., 5])

    quality = []
    for (model, k, variant), rows in list(groups.items()) + [
            ((m, k, 'ordinary'), rows) for (m, k), rows in ordinary.items()]:
        base = ordinary[model, k]
        assert [r['prompt_id'] for r in rows] == [r['prompt_id'] for r in base]
        a, o = quality_sums(rows), quality_sums(base)
        seed, indices = boot_indices(f'quality/main/{model}/{k}/{variant}', 32)
        ba, bo = a[indices].sum(axis=1), o[indices].sum(axis=1)
        ta, to = a.sum(axis=0), o.sum(axis=0)
        row = dict(model=model, top_k=k, variant=variant, corpus_ppl=float(corpus_ppl(ta)),
                   ordinary_corpus_ppl=float(corpus_ppl(to)),
                   entropy_bits_per_body_token=ta[2]/ta[1])
        if variant == 'ordinary':
            row.update({name: '' for name in (
                'ppl_ratio_to_ordinary', 'ppl_ratio_ci95_low', 'ppl_ratio_ci95_high',
                'utilization', 'utilization_ci95_low', 'utilization_ci95_high',
                'kl_bits_per_body_token', 'kl_ci95_low', 'kl_ci95_high')})
        else:
            ratio = quantiles(corpus_ppl(ba)/corpus_ppl(bo))
            util = quantiles(ba[:, 0]/ba[:, 2])
            kl = quantiles(ba[:, 3]/ba[:, 1])
            row.update(ppl_ratio_to_ordinary=float(corpus_ppl(ta)/corpus_ppl(to)),
                       ppl_ratio_ci95_low=ratio[0], ppl_ratio_ci95_high=ratio[1],
                       utilization=ta[0]/ta[2], utilization_ci95_low=util[0],
                       utilization_ci95_high=util[1], kl_bits_per_body_token=ta[3]/ta[1],
                       kl_ci95_low=kl[0], kl_ci95_high=kl[1])
        row.update(bootstrap_seed=seed, bootstrap_draws=DRAWS, n_documents=32)
        quality.append(row)
    las256 = next(r for r in quality if (r['model'], r['top_k'], r['variant'])
                  == ('llama_instruct', 256, 'las_imec'))
    assert round(las256['ppl_ratio_to_ordinary'], 3) == 1.495, las256
    write_csv(out/'MAIN_QUALITY.csv', quality)

    def mechanism_sums(rows):
        return np.array([[r['strict_verified_bits'] or 0, r['body_tokens'] or 0,
                          r['group_entropy_bits'] or 0, r['synchronized_uncertainty_bits'] or 0,
                          r['positive_u_sync_actions'] or 0, r['protocol_decisions'] or 0]
                         for r in rows], float)

    def discarded_share(t):
        return t[..., 3]/(t[..., 2] + t[..., 3])

    def bpt_gain(x, y):
        return (x[..., 0]/x[..., 1])/(y[..., 0]/y[..., 1])

    mechanism = []
    for model in MODELS:
        for k in KS:
            a = mechanism_sums(groups[model, k, 'las_imec'])
            b = mechanism_sums(groups[model, k, 'syncpool_imec'])
            # Same draws as the paired LAS(0)/SyncPool comparison, so the BPT-ratio intervals agree.
            seed, indices = boot_indices(f'paired/main/{model}/{k}', 32)
            ba, bb = a[indices].sum(axis=1), b[indices].sum(axis=1)
            ta, tb = a.sum(axis=0), b.sum(axis=0)
            rounds, share, gain = (quantiles(bb[:, 4]/bb[:, 5]), quantiles(discarded_share(bb)),
                                   quantiles(bpt_gain(ba, bb)))
            mechanism.append(dict(model=model, top_k=k,
                syncpool_rounds_with_longer_token=tb[4]/tb[5],
                syncpool_rounds_ci95_low=rounds[0], syncpool_rounds_ci95_high=rounds[1],
                syncpool_group_bits_per_token=tb[2]/tb[1],
                syncpool_discarded_bits_per_token=tb[3]/tb[1],
                discarded_share=discarded_share(tb), discarded_share_ci95_low=share[0],
                discarded_share_ci95_high=share[1], predicted_gain=1/(1-discarded_share(tb)),
                las0_group_bits_per_token=ta[2]/ta[1],
                bpt_gain=bpt_gain(ta, tb), bpt_gain_ci95_low=gain[0], bpt_gain_ci95_high=gain[1],
                bootstrap_seed=seed, bootstrap_draws=DRAWS, n_documents=32))
    paired = {(r['model'], r['top_k']): r for r in comparisons if r['metric'] == 'native_body_bpt'}
    for r in mechanism:
        c = paired[r['model'], r['top_k']]
        assert all(math.isclose(r[a], c[b], rel_tol=1e-12) for a, b in (
            ('bpt_gain', 'las0_over_shared_syncpool'), ('bpt_gain_ci95_low', 'ci95_low'),
            ('bpt_gain_ci95_high', 'ci95_high'))), r
    write_csv(out/'MAIN_MECHANISM.csv', mechanism)

    relative = []
    for model in MODELS:
        for k in KS:
            base = sums(groups[model, k, 'las_imec'])
            seed, indices = boot_indices(f'threshold/main/{model}/{k}', 32)
            mb, bb = metrics(base.sum(axis=0)), metrics(base[indices].sum(axis=1))
            for variant in (*TAUS, 'syncpool_imec'):
                rows = sums(groups[model, k, variant])
                mv, bv = metrics(rows.sum(axis=0)), metrics(rows[indices].sum(axis=1))
                row = dict(model=model, top_k=k, variant=variant,
                           tau_bits=TAUS.get(variant, float('inf')))
                for name, field in (('capacity', 'native_body_bpt'),
                                    ('time', 'sender_receiver_ms_per_body_token')):
                    lo, hi = quantiles(bv[field]/bb[field])
                    row.update({f'{name}_ratio': mv[field]/mb[field],
                                f'{name}_ci95_low': lo, f'{name}_ci95_high': hi})
                relative.append(dict(row, bootstrap_seed=seed, bootstrap_draws=DRAWS, n_documents=32))
    write_csv(out/'THRESHOLD_RELATIVE.csv', relative)

    # Ambiguity replay on the same ordinary texts (analysis/measure_ambiguity_frequency.py).
    docs = list(csv.DictReader((out/'AMBIGUITY_FREQUENCY_DOCS.csv').open()))
    assert len(docs) == 320 and len({(r['model'], r['top_k'], r['case_id']) for r in docs}) == 320
    assert all(float(r['max_selected_probability_abs_error']) == 0 and
               r['support_count_mismatches'] == '0' for r in docs), 'Replay differs from the record'
    ambiguity = []
    for model in MODELS:
        for k in KS:
            rows = sorted([r for r in docs if r['model'] == model and int(r['top_k']) == k],
                          key=lambda r: r['prompt_id'])
            assert [r['prompt_id'] for r in rows] == [r['prompt_id'] for r in ordinary[model, k]]
            compared = [r['retokenization_status'] in ('compared', 'boundary_merged') for r in rows]
            t = np.array([[int(r['ambiguous_steps']), int(r['body_steps']),
                           int(r['longer_candidate_steps']), int(r['shorter_candidate_steps']),
                           c, c and r['retokenization_equal'] == '0',
                           float(r['mean_support'])*int(r['body_steps'])]
                          for r, c in zip(rows, compared)], float)
            seed, indices = boot_indices(f'ambiguity/main/{model}/{k}', 32)
            bt, tt = t[indices].sum(axis=1), t.sum(axis=0)
            share, retok = quantiles(bt[:, 0]/bt[:, 1]), quantiles(bt[:, 5]/bt[:, 4])
            ambiguity.append(dict(model=model, top_k=k, documents=len(rows), body_steps=int(tt[1]),
                ambiguous_step_share=tt[0]/tt[1], ambiguous_ci95_low=share[0],
                ambiguous_ci95_high=share[1],
                longer_candidate_share=tt[2]/tt[1], shorter_candidate_share=tt[3]/tt[1],
                mean_support=tt[6]/tt[1], retokenization_compared=int(tt[4]),
                retokenization_mismatches=int(tt[5]),
                retokenization_mismatch_share=tt[5]/tt[4],
                retokenization_ci95_low=retok[0], retokenization_ci95_high=retok[1],
                boundary_merged=sum(r['retokenization_status'] == 'boundary_merged' for r in rows),
                bootstrap_seed=seed, bootstrap_draws=DRAWS))
    write_csv(out/'AMBIGUITY_FREQUENCY.csv', ambiguity)

    # Hardware evidence comes from the same 352 frozen run manifests.
    inventory = json.loads((src/'reports/project_understanding_stage3_20261030/INPUT_HASHES.json').read_text())
    hardware, software, manifest_hashes = collections.Counter(), collections.defaultdict(set), {}
    for rel in inventory:
        if not rel.endswith('/metrics.json'):
            continue
        p = (src/rel).with_name('manifest.json')
        manifest_hashes[str(p.relative_to(src))] = digest(p)
        if str(p.relative_to(src)) in inventory:
            assert digest(p) == inventory[str(p.relative_to(src))]
        def scan(obj):
            if isinstance(obj, dict):
                for k,v in obj.items():
                    if k == 'gpu': hardware[v] += 1
                    if k in ('python','torch','transformers','platform'): software[k].add(str(v))
                    scan(v)
            elif isinstance(obj,list):
                for v in obj: scan(v)
        scan(json.loads(p.read_text()))
    assert sum(hardware.values()) == 352 and len(hardware) == 1
    receipt = dict(date='2026-11-02', source_root=str(src), raw_path=raw_rel,
        raw_sha256=digest(raw), raw_hash_matches_frozen_audit=True,
        cases=len(cases), main_cases=len(main_cases), main_stego_points=len(groups),
        main_documents_per_point=32, frozen_E12_numeric_fields_matched=checked_fields,
        bootstrap=dict(draws=DRAWS, method='prompt-cluster percentile, deterministic per-cell seeds',
            main_only=True, paired_comparisons='resample document coordinates jointly; method random streams differ',
            interpretation='Exploratory finite-document uncertainty; no multiplicity or source-selection correction'),
        hardware=dict(hardware), software={k:sorted(v) for k,v in software.items()},
        manifest_sha256=manifest_hashes,
        sender_errors=[{k:r[k] for k in ('case_id','model','top_k','source_batch','variant','error')}
                       for r in cases if r['sender_status'] != 'ok'],
        numpy_version=np.__version__,
        outputs={str((out/name).relative_to(ROOT)):digest(out/name) for name in (
            'MAIN_POINTS.csv','MAIN_POINT_INTERVALS.csv','RECOVERY.csv',
            'MAIN_LAS0_SYNCPOOL_COMPARISONS.csv','MAIN_LAS0_LEDGER.csv','MAIN_GATE_DIAGNOSTICS.csv',
            'MAIN_QUALITY.csv','MAIN_MECHANISM.csv','THRESHOLD_RELATIVE.csv',
            'AMBIGUITY_FREQUENCY_DOCS.csv','AMBIGUITY_FREQUENCY.csv')},
        ambiguity_replay=json.loads((out/'AMBIGUITY_FREQUENCY_RUN.json').read_text()),
        new_model_inference=('Ambiguity replay only: recorded ordinary IDs re-scored under the '
                             'main-batch sampler; no new sampling'),
        source_files_modified=False,
        scope='Aggregate-only regeneration from corrected frozen cases; no new algorithm or cryptographic certification')
    (out/'PROVENANCE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('cases','main_cases','main_stego_points',
          'frozen_E12_numeric_fields_matched','hardware','sender_errors')},ensure_ascii=False))


if __name__ == '__main__':
    main()
