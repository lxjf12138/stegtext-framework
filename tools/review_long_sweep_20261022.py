"""Frozen longer-review development sweep; one resumable model/seed/k shard.

The generation, disambiguation and coding kernels are unchanged. Completed
records are scored with their assistant prefill, audited, then losslessly
archived. Missing-only resume never replaces an existing failed observation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from stegatext_runtime.experiments.config import ROOT, cases, fingerprint, read_tree, resolve
from stegatext_runtime.experiments.review_prompts import (
    PREFILL_INSTRUCTIONS, PREFILL_VERSION, clean_review, review_prefix,
)
from stegatext_runtime.experiments.runner import code_state, environment, file_hash, read, run, write
from tools.controlled_comparison.reproduce import write_config
from tools.expanded_records_20261021 import archive, raw_digest, read_record

OUT = ROOT / 'reports/review_long_sweep_20261022'
PREVIOUS = ROOT / 'reports/review_prefill_pilot_20261022'
KS = [32, 128, 256, 512, 1024]
TAUS = [.125, .25, .5, 1., 2., 4.]
SEEDS = [430001, 430002]
MIN_LENGTH = {'en': 2000, 'zh': 256}
SCRIPTS = ['tools/review_long_sweep_20261022.py', 'tools/review_long_guard_20261022.py',
           'tools/expanded_records_20261021.py', 'tools/expanded_native_ppl_20261021.py',
           'tools/controlled_comparison/resume.py', 'tools/controlled_comparison/reproduce.py']


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def variants():
    result = [dict(id='ordinary', kind='ordinary')]
    for method, short in [('basic_bytes_v1', 'basic'), ('mwis_bytes_v1', 'mwis'),
                          ('stepwise_chat_v1', 'stepwise'), ('syncpool', 'syncpool'), ('las', 'las')]:
        value = dict(id=short + '_imec', kind='planner', method=method, coder='imec_prefix_exact',
                     coder_parameters=dict(block_size=6, window_size=3,
                                           coupling_strategy='greedy', schedule='head_first'))
        if method == 'syncpool':
            value['implementation'] = 'shared_planner_v1'
        result.append(value)
    for tau in TAUS:
        result.append(dict(id=f'adaptive_{int(tau * 1000):04d}_imec', kind='planner',
                           method='adaptive', coder='imec_prefix_exact', tau=tau,
                           coder_parameters=dict(block_size=6, window_size=3,
                                                 coupling_strategy='greedy', schedule='head_first')))
    return result


def shingles(text):
    return {text[i:i+12] for i in range(max(1, len(text)-11))}


def near_duplicate(text, previous):
    a = shingles(text)
    for other in previous:
        b = shingles(other)
        if len(a & b) / len(a | b) >= .8:
            return True
    return False


def select_documents():
    import pyarrow.parquet as pq
    previous = read(PREVIOUS / 'DATASET_MANIFEST.json')
    documents, selections = [], []
    for spec in previous['sources']:
        path = ROOT / spec['local_file']
        assert file_hash(path) == spec['sha256']
        if spec['language'] == 'en':
            rows = pq.read_table(path).to_pylist()
        else:
            with path.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
        language = spec['language']
        old = [d for d in previous['documents'] if d['language'] == language]
        excluded_full = {d['cleaned_text_sha256'] for d in old}
        excluded_prefix = {d['prefix_sha256'] for d in old}
        old_texts = [clean_review(d['original_text']) for d in old]
        seen_full, seen_prefix, eligible, lengths = set(), set(), [], []
        for index, row in enumerate(rows):
            raw = row[spec['field']]
            cleaned = clean_review(raw)
            lengths.append(len(cleaned))
            if len(cleaned) < MIN_LENGTH[language]:
                continue
            prefix = review_prefix(raw, language)
            full_hash, prefix_hash = digest(cleaned), digest(prefix)
            if full_hash in seen_full or prefix_hash in seen_prefix:
                continue
            if full_hash in excluded_full or prefix_hash in excluded_prefix:
                continue
            seen_full.add(full_hash)
            seen_prefix.add(prefix_hash)
            identity = f"{spec['dataset']}:{spec['revision']}:{spec['split']}:{index}"
            eligible.append(dict(document_id=identity, row_index=index, original_text=raw,
                original_text_sha256=digest(raw), cleaned_text_sha256=full_hash,
                cleaned_codepoints=len(cleaned), prefix=prefix, prefix_sha256=prefix_hash,
                prefix_codepoints=len(prefix), unused_suffix_codepoints=len(cleaned)-len(prefix),
                label=row.get('label', row.get('Fake')), movie_name=row.get('movie_name'),
                selection_hash=digest('review-long-sweep-dev-v1:' + identity + ':' + full_hash)))
        eligible.sort(key=lambda d: d['selection_hash'])
        chosen, chosen_movies, rejected_near = [], set(), []
        for doc in eligible:
            movie = doc['movie_name']
            if movie is not None and movie in chosen_movies:
                continue
            if near_duplicate(clean_review(doc['original_text']), old_texts + [
                    clean_review(d['original_text']) for d in chosen]):
                rejected_near.append(doc['document_id'])
                continue
            chosen.append(doc)
            if movie is not None:
                chosen_movies.add(movie)
            if len(chosen) == 2:
                break
        assert len(chosen) == 2
        ordered_lengths = sorted(lengths)
        selections.append(dict(dataset=spec['dataset'], total_rows=len(rows),
            length_unit='cleaned Unicode codepoints', minimum_codepoints=MIN_LENGTH[language],
            eligible_unique_rows=len(eligible), selected_rows=[d['row_index'] for d in chosen],
            selected_lengths=[d['cleaned_codepoints'] for d in chosen],
            length_quantiles={str(q):ordered_lengths[int((len(lengths)-1)*q)] for q in [0,.5,.9,.95,.99,1]},
            excluded_previous_documents=[d['document_id'] for d in old],
            rejected_near_duplicates=rejected_near,
            near_duplicate_rule='Jaccard of 12-codepoint shingles >=0.8 against prior/selected full texts',
            order='SHA256(review-long-sweep-dev-v1:document_id:cleaned_hash)',
            label_policy='No label balancing or filtering; labels never reach the model',
            movie_policy='At most one selected document per available movie_name',
            scope='Length-conditioned development subset; not representative of all reviews'))
        for doc in chosen:
            doc.update(id=f"{spec['name']}_longdev_{doc['row_index']:05d}", language=language,
                dataset=spec['dataset'], revision=spec['revision'], config=spec['config'], split=spec['split'],
                original_field=spec['field'], cleaning_version='html-br-entity-whitespace-v1',
                template_version=PREFILL_VERSION, user_message=PREFILL_INSTRUCTIONS[language],
                assistant_prefix=doc['prefix'])
            documents.append(doc)
    return dict(sources=previous['sources'], selection=selections, documents=documents,
        douban_scope='The existing corpus contains short comments (max 358 codepoints) with spam/genuine labels; this is its >=256-codepoint tail, not a new long-form corpus.',
        selection_independent_of_outputs=True, previous_pilot_pooled=False)


def prepare():
    from transformers import AutoTokenizer
    from stegatext_runtime.models import MODEL_IDS
    from stegatext_runtime.prompts import AssistantPrefillPromptPolicy
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'DESIGN.json').exists():
        raise RuntimeError('Design already frozen; inspect or resume without replacing inputs')
    data = select_documents()
    write(OUT / 'DATASET_MANIFEST.json', data)
    locks = json.loads((ROOT / 'environment/models.lock.json').read_text())['models']
    audit = []
    for model, language in [('qwen_instruct','zh'), ('llama_instruct','en')]:
        spec = next(s for s in locks if s['repo_id'] == MODEL_IDS[model])
        tokenizer = AutoTokenizer.from_pretrained(ROOT / spec['local_path'], local_files_only=True)
        for doc in [d for d in data['documents'] if d['language'] == language]:
            messages = [dict(role='user',content=doc['user_message']),
                        dict(role='assistant',content=doc['prefix'])]
            policy = AssistantPrefillPromptPolicy(doc['prefix'], template_options=spec['template_options'])
            rendered = policy.render_text(tokenizer,doc['user_message'])
            ids = policy.render(tokenizer,doc['user_message'])[0].tolist()
            expected = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,
                                                     continue_final_message=True,**spec['template_options'])
            assert ids == expected == tokenizer.encode(rendered,add_special_tokens=False)
            assert rendered.endswith(doc['prefix']) and rendered.count(doc['prefix']) == 1
            audit.append(dict(model=model,prompt_id=doc['id'],messages=messages,rendered_prompt=rendered,
                prompt_token_ids=ids,prompt_token_count=len(ids),rendered_prompt_sha256=digest(rendered),
                chat_template_sha256=digest(tokenizer.chat_template),template_options=spec['template_options'],
                model_revision=spec['revision'],tokenizer_revision=spec['tokenizer_revision'],
                continue_final_message=True,add_generation_prompt=False))
    write(OUT / 'PROMPT_AUDIT.json',dict(template_version=PREFILL_VERSION,rows=audit))
    key = OUT / 'experiment_root.key'
    with key.open('x') as handle:
        handle.write(os.urandom(32).hex()+'\n')
    key.chmod(0o600)
    base = read_tree(ROOT/'configs/experiments/token_budget_T1.toml')
    base['randomness'] = dict(mode='key_file',file=str(key),stream_policy='independent_cases_v1',
                              experiment_id='review-long-sweep-dev-20261022')
    base['execution'] = dict(order='hash_interleaved_v1',order_seed=20261022)
    base['variants'] = variants()
    jobs = []
    for seed in SEEDS:
        for k in KS:
            for model,language in [('qwen_instruct','zh'),('llama_instruct','en')]:
                name=f'review_long_{model}_k{k}_s{seed}'
                config=copy.deepcopy(base)
                config.update(name=name)
                config['source'].update(models=[model],languages={model:language})
                config['source']['sampling']['top_k']=k
                config['inputs']=dict(ordinary_seed_policy='independent_conditions_v1',seeds=[seed],
                    prompts=[dict(id=d['id'],language=language,text=d['user_message'],assistant_prefix=d['prefix'])
                             for d in data['documents'] if d['language']==language])
                path=OUT/'configs'/(name+'.toml')
                write_config(path,config)
                expanded=list(cases(resolve(path)))
                assert len(expanded)==24 and len({c['id'] for c in expanded})==24
                jobs.append(dict(name=name,model=model,language=language,top_k=k,seed=seed,
                    config=str(path.relative_to(ROOT)),config_sha256=file_hash(path),cases=len(expanded)))
    assert sum(j['cases'] for j in jobs)==480
    design=dict(frozen_before_generation=True,created_unix=time.time(),total_cases=480,
        ordinary_cases=40,stego_cases=440,independent_receivers=440,jobs=jobs,
        documents_per_language=2,repeats_per_document=2,seeds=SEEDS,top_k=KS,adaptive_tau=TAUS,
        endpoints=dict(las='tau=0',syncpool='tau=infinity; shared_planner_v1'),
        template_version=PREFILL_VERSION,sampler=dict(temperature=1.,top_p=1.,do_sample=True,repetition_penalty=1.),
        max_new_tokens=256,minimum_new_tokens=0,stop='Natural EOS or common source-token cap; no retries for short generations',
        coding=dict(name='imec_prefix_exact',block_size=6,window_size=3,coupling_strategy='greedy',schedule='head_first'),
        scope='Exploratory development scan; only two documents per model and two draws per document/arm/k. No confirmatory equality, monotonicity or capacity claim.',
        theory='U_post is uncertainty in bits, not original candidate entropy. tau increases synchronization. Actual delivered BPB need not be monotone in k or tau.',
        randomness='Fresh root; independent case/message/protocol/Ordinary streams. Seed coordinates do not imply paired token trajectories.',
        resource_limits=dict(cpu_threads=1,process_tree_rss_gib=12,minimum_system_available_gib=4,
            maximum_swap_growth_gib=.5,artifact_gib=4,maximum_record_mib=128,
            minimum_disk_free_gib=20,total_wall_seconds=28800,shard_wall_seconds=2700),
        archive='Per shard after score/audit: verified exact-byte gzip of sender/receiver records; original raw hashes remain valid.',
        failure_policy='Preserve all completed errors/zeros/EOS/censoring; halt and diagnose failures; resume missing records only.',
        selection_policy='Fixed length thresholds, deduplication and hash order before any generation; no outcome-based replacement or threshold winner selection.',
        dataset_manifest_sha256=file_hash(OUT/'DATASET_MANIFEST.json'),prompt_audit_sha256=file_hash(OUT/'PROMPT_AUDIT.json'))
    write(OUT/'DESIGN.json',design)
    write(OUT/'PREFLIGHT.json',dict(code=code_state(),environment=environment(),
        design_sha256=file_hash(OUT/'DESIGN.json'),scripts={s:file_hash(ROOT/s) for s in SCRIPTS},
        prior_prefill_verification_sha256=file_hash(PREVIOUS/'VERIFICATION.json')))
    report()
    print(json.dumps(dict(prepared=True,cases=480,shards=20,selection=data['selection']),ensure_ascii=False),flush=True)


def score_job(job):
    import torch
    from stegatext_runtime.experiments.native_ppl import METRIC
    from stegatext_runtime.experiments.runner import load_source
    from stegatext_runtime.prompts import configure_prompt
    from tools.expanded_native_ppl_20261021 import enriched_score, DIAGNOSTIC_VERSION, TILTS
    import fcntl
    jobroot=OUT/'jobs'/job['name']
    directory=Path(read(jobroot/'RUN.json')['directory'])
    config=read(directory/'manifest.json')['configuration']
    output=jobroot/'ppl'
    contract=dict(metric=METRIC,diagnostic_version=DIAGNOSTIC_VERSION,tilts=list(TILTS),
        source_sha256=file_hash(Path(__file__)),enriched_scorer_sha256=file_hash(ROOT/'tools/expanded_native_ppl_20261021.py'),
        base_scorer_sha256=file_hash(ROOT/'stegatext_runtime/experiments/native_ppl.py'),
        conditioning='Actual assistant-prefilled prompt and recorded runtime IDs; prompt excluded from loss',
        primary='Raw full-vocabulary body NLL; natural EOS inclusive secondary; sampler likelihood and entropy separate',
        retokenization=False)
    if (output/'contract.json').exists():
        assert read(output/'contract.json')==contract
    else:
        write(output/'contract.json',contract)
    saved=read(output/'scores.json')['rows'] if (output/'scores.json').exists() else []
    by_id={r['case_id']:r for r in saved}
    assert len(by_id)==len(saved)
    source=None
    with (ROOT/'.cache/experiment_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for row in read(directory/'results.json')['rows']:
            path=directory/'sender'/(row['case_id']+'.json')
            sender_hash=raw_digest(path)
            if row['case_id'] in by_id:
                assert by_id[row['case_id']]['sender_record_sha256']==sender_hash
                continue
            sender=read_record(path)
            assert sender['status']=='ok'
            if source is None:
                source=load_source(config,job['model'])
            prompt=next(p for p in config['inputs']['prompts'] if p['id']==row['prompt_id'])
            configure_prompt(source,prompt['assistant_prefix'])
            seq=sender['result']['native_token_sequence']
            score=enriched_score(source,seq)
            item=dict(case_id=row['case_id'],model=row['model'],prompt_id=row['prompt_id'],
                seed=row['seed'],variant=row['variant'],sender_record_sha256=sender_hash,
                native_sequence_sha256=fingerprint(seq),**score)
            saved.append(item)
            by_id[item['case_id']]=item
            write(output/'scores.json',dict(metric=METRIC,contract_sha256=fingerprint(contract),rows=saved))
            print(f"SCORED {len(saved)}/{job['cases']} {row['case_id']} {score['evaluator_status']}",flush=True)
    assert len(saved)==job['cases']
    assert all(s['evaluator_status'] in ('scored','empty_generated_body') for s in saved)
    write(output/'COMPLETE.json',dict(scores_sha256=file_hash(output/'scores.json'),cases=len(saved)))


def audit_job(job,directory,score_dir=None):
    score_dir=score_dir or directory.parents[2]/'ppl'
    scores={r['case_id']:r for r in read(score_dir/'scores.json')['rows']}
    docs={d['id']:d for d in read(OUT/'DATASET_MANIFEST.json')['documents']}
    audit={(r['model'],r['prompt_id']):r for r in read(OUT/'PROMPT_AUDIT.json')['rows']}
    metrics=[]
    for row in read(directory/'results.json')['rows']:
        case_id=row['case_id'];path=directory/'sender'/(case_id+'.json')
        sender=read_record(path);r=sender['result'];score=scores[case_id]
        assert sender['status']=='ok' and row['error'] is None
        seq=r['native_token_sequence']
        assert seq['prompt_token_ids']==audit[job['model'],row['prompt_id']]['prompt_token_ids']
        assert score['sender_record_sha256']==raw_digest(path)
        assert score['native_sequence_sha256']==fingerprint(seq)
        assert [t['token_id'] for t in score['per_token']]==seq['output_token_ids']
        assert score['evaluator_tokens']==seq['body_token_count']
        assert seq['source_token_count']<=256
        v=row['variant'];ordinary=v['kind']=='ordinary';trace=r.get('trace',[])
        information=expected=pending=None
        sync_positive=0;sync_u=0.;retain=0
        if not ordinary:
            receiver=read_record(directory/'receiver'/(case_id+'.json'))
            assert row['exact'] is True and receiver['status']=='ok' and sender['pid']!=receiver['pid']
            assert all(r[k]==receiver['result'][k] for k in
                       ('bits','bits_count','text','state_sha256','trace','coder_state','native_token_sequence'))
            assert receiver['result']['receiver_consumed']
            public_path=directory/'public'/(case_id+'.json');public=read(public_path)
            assert sender['public_sha256']==receiver['input_sha256']==file_hash(public_path)
            assert public['assistant_prefix']==docs[row['prompt_id']]['prefix']
            coder=[t['coder'] for t in trace if 'coder' in t]
            assert len(coder)==len(trace)
            assert all(t['selected_window_index']==0 and t['exact_marginals'] for t in coder)
            assert all(t[k]==0 for t in coder for k in
                       ('row_normalization_error','column_normalization_error','coupling_mass_error'))
            state=r['coder_state']
            information=state['head_block']*state['block_bits']+state['block_bits']*state['window_blocks']-math.fsum(state['entropies'])
            assert math.isclose(information,math.fsum(t['posterior_information_revealed_bits'] for t in coder),abs_tol=1e-8)
            pending=information-r['bits_count']
            assert -1e-8<=pending<=18+1e-8
            expected=math.fsum(t['expected_information_bits'] for t in coder)
            if v['method'] in ('las','syncpool','adaptive'):
                assert all(t['posterior_mass_error_after']<2e-6 for t in trace)
                if v['method']=='adaptive':
                    assert all(t['kernel']==('las' if t['U_post']>=v['tau'] else 'fullsync') for t in trace)
                retain=sum(t['kernel']=='las' for t in trace)
                positive=[t for t in trace if t['kernel']=='fullsync' and t['U_post']>0]
                sync_positive=len(positive)
                sync_u=math.fsum(t['U_post'] for t in positive)
                if v['method']=='adaptive':
                    assert sync_u<=v['tau']*sync_positive+1e-8
        losses=sorted([t['model_nll'] for t in score['per_token'] if not t['is_eos']],reverse=True)
        body=score['entropy_body'];tokens=score['evaluator_tokens']
        performance=row.get('performance') or {}
        b=row.get('strict_verified_payload_bits') if not ordinary else None
        nbytes=(row.get('output') or {}).get('text_utf8_bytes')
        assert nbytes==len(r['text'].encode('utf-8'))
        assert ordinary or b==len(r['bits'])
        metrics.append(dict(case_id=case_id,job=job['name'],model=job['model'],top_k=job['top_k'],
            prompt_id=row['prompt_id'],seed=row['seed'],variant=v['id'],method=v.get('method','ordinary'),tau=v.get('tau'),
            exact=row['exact'],bits=b,bytes=nbytes,source_tokens=seq['source_token_count'],
            common_evaluation_tokens=r.get('evaluation_token_count'),termination=row['terminal_reason'],
            text=r['text'],group_entropy_bits=r.get('group_entropy_total_bits'),
            expected_coder_information_bits=expected,realized_posterior_information_bits=information,pending_information_bits=pending,
            ppl=score['ppl'],nll_total=score['nll_total'],evaluator_tokens=tokens,
            conditional_entropy_nats=body['conditional_entropy_nats'],surprisal_residual_nats=body['surprisal_residual_nats'],
            sampler_normalization_tax_nats=(score['nll_total']-body['normalized_sampler_nll']) if tokens else None,
            max_token_nll=losses[0] if losses else None,
            top3_token_nll_share=math.fsum(losses[:3])/score['nll_total'] if tokens and score['nll_total'] else None,
            adaptive_las_actions=retain,positive_u_sync_actions=sync_positive,synchronized_uncertainty_bits=sync_u,
            protocol_decisions=len(trace),retained_frontier_max=r.get('retained_frontier_max'),
            filtered_path_kl_total_bits=r.get('filtered_path_kl_total_bits'),
            final_visible_text_kl_status='not_estimated',
            sender_seconds=performance.get('wall_seconds'),receiver_seconds=(row.get('receiver_performance') or {}).get('wall_seconds'),
            actual_model_forward_calls=performance.get('actual_model_forward_calls'),peak_cuda_gib=performance.get('peak_cuda_gib')))
    assert len(metrics)==job['cases']
    return metrics


def report():
    design=read(OUT/'DESIGN.json')
    done=[];all_rows=[]
    for job in design['jobs']:
        path=OUT/'metrics'/job['name']/'COMPLETE.json'
        if path.exists():
            marker=read(path)
            metric=path.with_name('cases.json')
            assert marker['metrics_sha256']==file_hash(metric)
            done.append(job)
            all_rows.extend(read(metric)['rows'])
    groups=defaultdict(list)
    for row in all_rows:
        groups[row['model'],row['top_k'],row['variant']].append(row)
    cells=[]
    for (model,k,variant),rows in sorted(groups.items()):
        nbytes=sum(r['bytes'] for r in rows)
        bits=sum(r['bits'] or 0 for r in rows)
        ntokens=sum(r['evaluator_tokens'] for r in rows)
        nll=math.fsum(r['nll_total'] or 0 for r in rows)
        ppl=math.exp(nll/ntokens) if ntokens else None
        values=[r['ppl'] for r in rows if r['ppl'] is not None]
        ordinary=rows[0]['method']=='ordinary'
        cells.append(dict(model=model,top_k=k,variant=variant,method=rows[0]['method'],tau=rows[0]['tau'],
            cases=len(rows),scored_cases=len(values),documents=len({r['prompt_id'] for r in rows}),
            verified_bits=None if ordinary else bits,bytes=nbytes,bpb=bits/nbytes if nbytes and not ordinary else None,
            ppl=ppl,ppl_median=statistics.median(values) if values else None,ppl_max=max(values) if values else None,
            nll_total=nll,evaluator_tokens=ntokens,mean_source_tokens=statistics.mean(r['source_tokens'] for r in rows),
            natural_eos_cases=sum(r['termination']=='natural_eos' for r in rows),
            actual_forwards_per_byte=sum(r['actual_model_forward_calls'] or 0 for r in rows)/nbytes if nbytes else None,
            sender_seconds_per_byte=sum(r['sender_seconds'] or 0 for r in rows)/nbytes if nbytes else None,
            H=math.fsum(r['group_entropy_bits'] or 0 for r in rows) if not ordinary else None,
            C=math.fsum(r['expected_coder_information_bits'] or 0 for r in rows) if not ordinary else None,
            I=math.fsum(r['realized_posterior_information_bits'] or 0 for r in rows) if not ordinary else None,
            pending=math.fsum(r['pending_information_bits'] or 0 for r in rows) if not ordinary else None,
            las_actions=sum(r['adaptive_las_actions'] for r in rows),
            positive_u_sync_actions=sum(r['positive_u_sync_actions'] for r in rows),
            synchronized_uncertainty_bits=math.fsum(r['synchronized_uncertainty_bits'] for r in rows)))
    write(OUT/'CELLS.json',dict(cells=cells))
    write(OUT/'PROGRESS.json',dict(finished_shards=len(done),total_shards=len(design['jobs']),
        completed_cases=len(all_rows),total_cases=design['total_cases'],updated_unix=time.time()))
    lines=['# Development sweep of k and the A-LAS threshold on longer reviews','',
        f"Scored, independently recovered, and archived: {len(all_rows)}/{design['total_cases']} generations, {len(done)}/{len(design['jobs'])} shards.",'',
        'Two fixed development documents per language, two independent repeats per document and condition. IMDB documents have at least 2000 characters after cleaning; Douban reviews have at least 256 characters (the longest in the corpus has 358, so this is not a long-review corpus). Documents are selected by length, deduplication, and identity hash, exclude the previous batch, and are not replaced based on outputs.',
        'The fixed instruction and assistant prefill are kept; English openings have at most 256 characters and Chinese openings 64. The unused remainder of a long document is not given to the model, and a long document does not guarantee a long output. At most 256 new source tokens; natural EOS is allowed, with no minimum length and no resampling of short outputs.',
        'k = 32,128,256,512,1024; tau = 0.125,0.25,0.5,1,2,4 bits; LAS is tau=0 and SyncPool is the immediate-synchronization endpoint. Temperature 1, top-p=1. Ordinary, Basic, MWIS, and Stepwise are compared at every k; Ordinary has no tau axis.',
        'All methods use greedy head_first iMEC with 6-bit blocks and a 3-block window. Fresh random root and independent per-case random domains; equal repeat indices do not imply identical token paths across methods.',
        'This is a development parameter sweep with only 2 documents / 4 generations per cell; it does not establish distribution equivalence, monotonicity of BPB in k, or a generally optimal threshold. Later formal data must exclude these documents and near duplicates.',
        'BPB = total strictly recovered bits / total new UTF-8 bytes; PPL = exp(total body NLL / total body tokens), from the original full-vocabulary logits and the actual prefilled context, excluding the prompt. PPL is not the KL of the output distribution.',
        'Each generation also records PPL, length, EOS, extreme token contributions, H/C/I/B, unconfirmed information, U and threshold actions, and actual forwards and time. The total synchronized U only checks the per-step bound and is not a cross-policy difference in delivered payload.','',
        '| Model | k | Method/threshold | n | BPB | PPL | Mean new tokens | Natural EOS |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    def fmt(x): return '—' if x is None else f'{x:.4f}'
    for c in cells:
        label=f"adaptive τ={c['tau']:g}" if c['method']=='adaptive' else c['method']
        lines.append(f"| {c['model']} | {c['top_k']} | {label} | {c['cases']} | {fmt(c['bpb'])} | {fmt(c['ppl'])} | {c['mean_source_tokens']:.1f} | {c['natural_eos_cases']} |")
    lines+=['','Run status is in `LIVE.json` / `RESOURCES.json`; each shard keeps `AUDIT.json`, per-token PPL scores, and `.json.gz` records verified against raw-byte hashes. Earlier small batches are not included in this table.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(completed_cases=len(all_rows),total_cases=design['total_cases'],finished_shards=len(done))),flush=True)


def execute():
    pre=read(OUT/'PREFLIGHT.json');design=read(OUT/'DESIGN.json')
    assert code_state()==pre['code'] and environment()==pre['environment']
    assert file_hash(OUT/'DESIGN.json')==pre['design_sha256']
    assert all(file_hash(ROOT/s)==h for s,h in pre['scripts'].items())
    for job in design['jobs']:
        if (OUT/'STOP_REQUESTED.json').exists():
            raise RuntimeError('Stop requested; retained records support missing-only resume')
        assert code_state()==pre['code']
        assert all(file_hash(ROOT/s)==h for s,h in pre['scripts'].items())
        marker=OUT/'metrics'/job['name']/'COMPLETE.json'
        if marker.exists():
            saved=read(marker)
            assert file_hash(marker.with_name('cases.json'))==saved['metrics_sha256']
            assert (Path(saved['directory'])/'ARCHIVE_COMPLETE.json').exists()
            continue
        write(OUT/'ACTIVE.json',dict(job=job['name'],phase='generation_and_receiver',started_unix=time.time()))
        config=ROOT/job['config'];assert file_hash(config)==job['config_sha256']
        jobroot=OUT/'jobs'/job['name']
        found=list((jobroot/'gpu'/job['name']).glob('*/manifest.json'))
        if found:
            assert len(found)==1
            directory=found[0].parent
            if (directory/'results.json').exists():
                rows=read(directory/'results.json')['rows']
            else:
                from tools.controlled_comparison.resume import finish_pending
                directory,rows=finish_pending(config,directory)
        else:
            directory,rows=run(config,jobroot/'gpu')
        write(jobroot/'RUN.json',dict(directory=str(directory)))
        failures=[r for r in rows if r['sender_status']!='ok' or (r['variant']['kind']!='ordinary' and r['exact'] is not True)]
        if failures:
            write(OUT/'FIRST_FAILURE.json',dict(job=job['name'],failures=failures))
            raise RuntimeError('Recorded failures retained; stop and diagnose before continuing')
        assert len(rows)==job['cases']
        write(OUT/'ACTIVE.json',dict(job=job['name'],phase='native_ppl',started_unix=time.time()))
        score_dir=jobroot/'ppl'
        if not (score_dir/'COMPLETE.json').exists():
            with (jobroot/'ppl.log').open('a') as log:
                subprocess.run([sys.executable,'-B',str(Path(__file__)),'score','--job',job['name']],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read(score_dir/'COMPLETE.json')['scores_sha256']==file_hash(score_dir/'scores.json')
        write(OUT/'ACTIVE.json',dict(job=job['name'],phase='audit_and_archive',started_unix=time.time()))
        metrics=audit_job(job,directory,score_dir)
        write(marker.with_name('cases.json'),dict(rows=metrics))
        write(marker.with_name('AUDIT.json'),dict(passed=True,cases=len(metrics),
            independent_receiver_exact=sum(r['exact'] is True for r in metrics),
            assistant_prompt_ids_verified=True,sender_score_raw_hashes_verified=True,
            exact_coupling_marginals_verified=True,head_first_information_identity_verified=True,
            adaptive_threshold_actions_verified=True,score_excludes_public_prefill=True))
        archive(directory)
        write(marker,dict(directory=str(directory),metrics_sha256=file_hash(marker.with_name('cases.json')),
                           scores_sha256=file_hash(score_dir/'scores.json'),cases=len(metrics)))
        report()
        print('FINISHED '+job['name'],flush=True)
    report()
    write(OUT/'COMPLETE.json',dict(completed=True,cases=design['total_cases'],finished_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','execute','score','report'])
    parser.add_argument('--job')
    args=parser.parse_args()
    if args.action=='score':
        job=next(j for j in read(OUT/'DESIGN.json')['jobs'] if j['name']==args.job)
        score_job(job)
    else:
        {'prepare':prepare,'execute':execute,'report':report}[args.action]()
