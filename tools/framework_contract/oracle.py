"""Coder-free exact finite AR oracle. No production grouping/state helpers."""
from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction as F
from itertools import combinations
import math

import torch

from tests.finite_source import FiniteSource, ReplayUniform
from stegatext_runtime.experiments.engine import method
from stegtext.v2.core.types import MethodContext
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.model.tokenizer_codec import EOS_STEGA


class TreeSource(FiniteSource):
    def __init__(self, fixture='mixed'):
        super().__init__()
        if fixture == 'sentence':
            tokens = {0:b'Hello world.',1:b'Hello world. Additional text',2:EOS_STEGA}
            self.transitions = {():[(0,F(1,2)),(1,F(1,2))], (0,):[(2,F(1))],(1,):[(2,F(1))]}
            eos = [2]
        else:
            tokens = {0:b'a',1:b'ab',2:b'a',3:b'b',4:b'bc',5:b'c',6:EOS_STEGA,7:EOS_STEGA}
            eos = [6,7]
            self.transitions = {():[(0,F(1,4)),(1,F(1,2)),(2,F(1,4))],
                (0,):[(3,F(1,2)),(4,F(1,4)),(6,F(1,4))],
                (2,):[(5,F(1,2)),(6,F(1,2))],
                (1,):[(5,F(1,2)),(6,F(1,2))],
                (0,3):[(5,F(1,2)),(6,F(1,2))],
                (0,4):[(6,F(1,2)),(7,F(1,2))],
                (0,3,5):[(6,F(1))],(1,5):[(6,F(1))],(2,5):[(7,F(1))]}
            if fixture == 'thirds':
                self.transitions[()] = [(0,F(1,3)),(1,F(1,3)),(2,F(1,3))]
            if fixture == 'utf8':
                tokens.update({0:b'\xe4',1:'中'.encode(),2:b'\xe4',3:b'\xb8\xad',4:b'\xb8\xad!',5:b'!'})
        self.tokens, self.eos_ids = tokens, frozenset(eos)
        self.backend = ToyBackend(ToyBackendConfig(tokens,{},eos_ids=eos))
        self.backend.codec_index = self.backend.index
        self.backend.codec.text_to_bytes = lambda text:text.encode()


@dataclass(frozen=True)
class State:
    base: tuple = ()
    fresh: F = F(1)
    retained: tuple = ()  # (path, residual, probability, EOS)
    visible: bytes = b''
    terminal: bool = False


def groups(atoms):
    # A key is minimal in the prefix partial order. Derive by all-pairs
    # relation instead of the production lexical sweep.
    strings = set(a[1] for a in atoms)
    keys = sorted(x for x in strings if not any(x != y and x.startswith(y) for y in strings))
    return [(key, [a for a in atoms if a[1].startswith(key)]) for key in keys]


def atoms(source, state):
    fresh = [(state.base+(t,),source.tokens[t],state.fresh*p,t in source.eos_ids)
             for t,p in source.transitions.get(state.base,[]) if p and state.fresh]
    return fresh+list(state.retained)


def successors(state, key, rows, las):
    total = sum(a[2] for a in rows)
    exact = [a for a in rows if a[1] == key] if las else rows
    exact_mass = sum(a[2] for a in exact)
    strict = tuple((p,r[len(key):],w/total,e) for p,r,w,e in rows if r != key) if las else ()
    out = defaultdict(F)
    for path,residual,weight,eos in exact:
        next_state = State(path, F(0) if eos else exact_mass/total if las else F(1),
                           strict, state.visible+(key if las else residual),eos)
        out[next_state] += weight/exact_mass
    return out


def signature(state):
    return (state.base, state.fresh, tuple(sorted(state.retained)),state.visible,state.terminal)


def assert_state(actual, expected, las):
    assert actual.resolved_bytes == expected.visible and actual.terminal == expected.terminal
    path = actual.base_path if las else actual.path
    assert tuple(path.token_ids()) == expected.base
    if las:
        assert math.isclose(actual.expansion_mass,float(expected.fresh),rel_tol=2e-12,abs_tol=0)
        rows = sorted((tuple(r.path.token_ids()),r.residual,r.mass,r.terminal) for r in actual.retained)
        target = sorted(expected.retained)
        assert len(rows) == len(target)
        for a,b in zip(rows,target):
            assert a[:2] == b[:2] and a[3] == b[3] and math.isclose(a[2],float(b[2]),rel_tol=2e-12,abs_tol=0)


def raw_law(source, path=(), data=b'', mass=F(1), sentence=False):
    if (path and path[-1] in source.eos_ids) or (sentence and b'.' in data):
        return {(path,data):mass}
    law = defaultdict(F)
    for t,p in source.transitions[path]:
        visible = b'' if t in source.eos_ids else source.tokens[t]
        for y,w in raw_law(source,path+(t,),data+visible,mass*p,sentence).items():
            law[y] += w
    return dict(law)


def audit(fixture='mixed', name='las', sentence=False, budget=None, *, source=None):
    source = TreeSource(fixture) if source is None else source
    m = method(source,name)
    las = name != 'syncpool'
    ctx = MethodContext(source.backend.codec,source.bind(''),'encode')
    checked = [0]
    law = defaultdict(F)

    def visit(actual, state, weight, depth):
        assert depth < 16
        assert_state(actual,state,las)
        checked[0] += 1
        if state.terminal or (sentence and b'.' in state.visible) or (budget is not None and depth >= budget):
            data = state.visible[:-len(EOS_STEGA)] if state.terminal else state.visible
            law[(state.base,data)] += weight
            return
        a = atoms(source,state)
        gs = groups(a)
        paths = ([actual.base_path] if actual.expansion_mass else []) if las else m.queries(actual)
        plan = m.plan(actual,[source.distribution(p) for p in paths],ctx)
        actual_keys = [r['key'] for r in plan.opaque['records']] if las else plan.opaque['keys']
        assert actual_keys == [k for k,_ in gs]
        for action,(key,rows) in enumerate(gs):
            mass = sum(row[2] for row in rows)
            assert math.isclose(float(plan.probs[action]),float(mass),rel_tol=2e-12,abs_tol=0)
            if las and name == 'las':
                record = plan.opaque['records'][action]
                assert abs(record['exact_mass']-float(sum(r[2] for r in rows if r[1] == key))) < 2e-12
                assert abs(record['strict_mass']-float(sum(r[2] for r in rows if r[1] != key))) < 2e-12
            expected = successors(state,key,rows,las)
            # Include all subset sums: covers every ordering of representative
            # atoms without importing production representative order.
            exact = [r[2] for r in rows if r[1] == key] if las else [r[2] for r in rows]
            total = sum(exact)
            cuts = {F(0),F(1)}
            for n in range(1,len(exact)):
                cuts.update(sum(c)/total for c in combinations(exact,n))
            cuts = sorted(cuts)
            actual_law, representatives = defaultdict(F), {}
            for left,right in zip(cuts,cuts[1:]):
                nxt = m.advance(actual,plan,action,ReplayUniform([(left+right)/2]),ctx)
                matches = [s for s in expected if s.base == tuple((nxt.base_path if las else nxt.path).token_ids())]
                assert len(matches) == 1
                e = matches[0]
                assert_state(nxt,e,las)
                actual_law[e] += right-left
                representatives[e] = nxt
            assert actual_law == expected
            for nxt,p in expected.items():
                visit(representatives[nxt],nxt,weight*mass*p,depth+1)
    visit(m.start(ctx.prompt),State(),F(1),0)
    assert sum(law.values()) == 1
    if not sentence and budget is None:
        assert dict(law) == raw_law(source)
    visible = defaultdict(F)
    for (_,data),p in law.items(): visible[data] += p
    return dict(fixture=fixture,method=name,states=checked[0],law={k.hex():str(v) for k,v in visible.items()},
                terminal_path_law_verified=not sentence and budget is None, tolerance=2e-12, mass_comparison='relative with zero absolute tolerance; positive atoms cannot match zero',
                oracle='Fraction all-pairs prefix relation, independent representative law; coder-free')


def main():
    import json
    from pathlib import Path
    rows = [audit(f,m) for f in ['mixed','thirds','utf8','sentence'] for m in ['las','las_local','syncpool']]
    rows += [audit('mixed',m,budget=2) for m in ['las','las_local','syncpool']]
    Path('reports/framework_contract_20261018/audit/FINITE_SOURCE_ORACLE.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(contracts=len(rows),states=sum(r['states'] for r in rows))))

if __name__ == '__main__': main()
