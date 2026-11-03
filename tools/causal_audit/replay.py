"""Replay frozen GPU supports, states and coder draws on CPU without model files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from perf_research_planner import make_jump_planner, make_bulk_planner
from round5_exact import make_exact_planner
from round4_2_common import CounterRNG
from stegtext.v2.core.types import MethodContext, PathRef, PromptState, TokenDistribution
from stegtext.v2.methods.las import LASState, RetainedPath
from stegtext.starter.adaptive import select_kernel
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegatext_runtime.discop import DisCopSession
from stegatext_runtime.experiments.engine import message_for
from tools.causal_audit.finite import Belief, Atom, compare_state, F
from tools.causal_audit.real import inspect_event
from tools.causal_audit.reward import symbolic_integral, tree_metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/causal_audit_20261018'


def path(ids):
    result = PathRef.root()
    for token in ids:
        result = result.append(token)
    return result


def restore(view):
    rows = view['retained']
    # The capture serialized semantic signatures in path order. Recover the
    # production residual ordering only when tie ordering is unambiguous.
    assert len({r[1] for r in rows}) == len(rows), 'Residual tie: order was not saved; do not claim RNG replay'
    retained = tuple(RetainedPath(path(ids), bytes.fromhex(residual), mass, terminal)
                     for ids, residual, mass, terminal in sorted(rows, key=lambda r:r[1]))
    return LASState(path(view['path']), view['alpha'], retained,
                    bytes.fromhex(view['visible']), view['terminal'])


def belief(view):
    return Belief(tuple(view['path']), F(view['alpha']),
                  tuple(Atom(tuple(ids), bytes.fromhex(residual), F(mass), terminal)
                        for ids,residual,mass,terminal in view['retained']),
                  bytes.fromhex(view['visible']), view['terminal'])


def main(directory=OUT):
    torch.set_num_threads(1)
    contract = json.loads((directory/'gpu_contract.json').read_text())
    checked = 0
    for name, digest in contract['code'].items():
        actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        assert actual == digest, (name, actual, digest)
        checked += 1
    for name, digest in contract['audit_sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    rows = []
    for filename in sorted((directory/'raw').glob('*.json')):
        data = json.loads(filename.read_text())
        key = bytes.fromhex(data['key_hex'])
        offset = 0
        stream = message_for(contract['base_config'], contract['seed'])
        source = None
        if data['method'] != 'syncpool':
            backend = ToyBackend(ToyBackendConfig(
                {int(t):bytes.fromhex(value) for t,value in data['sparse_token_bytes'].items()},
                {}, eos_ids=data['eos_ids']))
            backend.codec_index = backend.index
            ids = tuple(data['prompt_ids'])
            source = SimpleNamespace(backend=backend, prompt=PromptState(path(ids),ids,b''))
            ctx = MethodContext(backend.codec,source.prompt,'encode')
            methods = [make_jump_planner(backend.index), make_bulk_planner(backend.index),
                       make_exact_planner(backend.index,'block')]
        errors = dict(probabilities=0.,mass=0.,entropy=0.,naive_probabilities=0.)
        transitions = 0
        for row in data['states']:
            step = row['step']
            # Use the exact serialized FP32 values passed to the live coder.
            used, action = DisCopSession().encode(row['probabilities'], stream[offset:],
                                                  rng=CounterRNG(key,'coder',step))
            decoded = DisCopSession().decode(row['probabilities'],action,
                                             rng=CounterRNG(key,'coder',step))
            assert action == row['action'] and used == decoded and len(used) == row['consumed_bits']
            assert data['result']['bits'][offset:offset+len(used)] == used
            offset += len(used)
            if source is None:
                continue
            before, after = restore(row['before']), restore(row['after'])
            probs = torch.zeros(backend.vocab_size,dtype=torch.float64)
            for token, value in row['fresh_support']:
                probs[token] = value
            ds = [TokenDistribution(probs)] if before.expansion_mass else []
            for method in methods:
                plan = method.plan(before,ds,ctx)
                assert [r['key'].hex() for r in plan.opaque['records']] == row['group_keys']
                assert max(abs(a-float(b)) for a,b in zip(row['probabilities'],plan.probs)) < 3e-7
                advance = method.advance_fullsync if row['kernel']=='fullsync' else method.advance
                updated = advance(before,plan,action,CounterRNG(key,'advance',step),ctx)
                compare_state(updated,belief(row['after']))
                transitions += 1
            plan = methods[0].plan(before,ds,ctx)
            tau = 0. if data['method']=='las' else data['packet']['variant']['tau']
            assert select_kernel(row['u_selected'],tau) == row['kernel']
            event = dict(before=before,after=after,plan=plan,distributions=ds,
                         action=action,kernel=row['kernel'],step=step)
            for name,value in inspect_event(source,key,event).items():
                errors[name] = max(errors[name],value)
        assert offset == data['result']['bits_count']
        rows.append(dict(file=filename.name,steps=len(data['states']),bits=offset,
                         transitions=transitions,max_independent_errors=errors))
    vectors = json.loads((directory/'reward_vectors.json').read_text())
    for vector in vectors:
        p = vector['probabilities']; metrics = tree_metrics(p)
        for bit in ('0','1'):
            integral = symbolic_integral(p,bit*256)
            assert abs(integral['expected_bits']-metrics['expected_bits']) < 1e-11
    result = dict(all_passed=True,model_files_loaded=False,device='cpu',runtime_hashes=checked,
                  cases=rows,reward_vectors=len(vectors),
                  scope='Frozen supports and semantic states, original coder RNG and three optimized CPU planners; no LLM logits recomputation')
    (directory/'offline_replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=OUT)
    main(parser.parse_args().directory)
