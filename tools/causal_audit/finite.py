"""Independent rational tree/belief oracle; never uses planner grouping helpers."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import json
import math
from pathlib import Path
import random

import torch

from perf_research_planner import make_jump_planner, make_bulk_planner
from round5_exact import make_exact_planner
from stegtext.v2.methods.las import LASState, RetainedPath, NaiveLASMethod
from stegtext.v2.core.types import PathRef, MethodContext, TokenDistribution
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.model.tokenizer_codec import EOS_STEGA

OUT = Path(__file__).resolve().parents[2]/'reports/causal_audit_20261018'
TOL = 5e-12


@dataclass(frozen=True)
class Atom:
    path: tuple
    residual: bytes
    mass: F
    terminal: bool = False


@dataclass(frozen=True)
class Belief:
    path: tuple = ()
    alpha: F = F(1)
    retained: tuple = ()
    visible: bytes = b''
    terminal: bool = False


class Tree:
    def __init__(self, seed, depth=3, chinese=False, tiny=False, wide=0):
        self.seed = seed
        self.depth = depth
        raw = [b'a', b'a', b'ab', b'abc', b'abcd', b'ax',
               b'b', b'b', b'bc', b'bcd', b'c', b'c', b'cd', b'd', b'x']
        if chinese:
            # Deliberately split UTF-8 across tokens; wire law includes bytes.
            raw = [v.replace(b'a', b'\xe4').replace(b'b', b'\xb8').replace(b'c', b'\xad') for v in raw]
        self.tokens = dict(enumerate(raw+[EOS_STEGA, EOS_STEGA]))
        self.eos = {15, 16}
        self.wide = wide
        for i in range(wide):
            self.tokens[17+i] = b'a'+b'b'*(1+i//3)+[b'', b'c', b'd'][i%3]
        rng = random.Random(seed)
        self.transitions = {}
        def grow(history):
            if len(history) >= depth:
                self.transitions[history] = [(15, F(1, 4)), (16, F(3, 4))]
                return
            choices = [0, 1, 2, 3, 4, 5] if not history else (
                [6, 7, 8, 9, 15] if len(history) == 1 else [10, 11, 12, 13, 15])
            if not history: choices = choices+list(range(17, 17+wide))
            if seed % 3 == 1 and history:
                # Different histories have genuinely different predictive laws.
                choices = rng.sample(choices, 3)
            weights = [rng.randint(1, 7) for _ in choices]
            if tiny and not history:
                weights = [2**35, 3, 7, 2, 1, 5]
            total = sum(weights)
            self.transitions[history] = [(t, F(w, total)) for t, w in zip(choices, weights)]
            for t in choices:
                if t not in self.eos: grow(history+(t,))
        grow(())
        self.backend = ToyBackend(ToyBackendConfig(self.tokens, {}, eos_ids=sorted(self.eos)))
        self.prompt = self.backend.render_prompt('')

    def distribution(self, path):
        p = torch.zeros(len(self.tokens), dtype=torch.float64)
        for t, w in self.transitions[tuple(path)]: p[t] = float(w)
        return TokenDistribution(p)

    @lru_cache(None)
    def cover(self, history=()):
        """Raw AR paths only; independent of beliefs and every LAS method."""
        result = defaultdict(F)
        for token, mass in self.transitions[history]:
            if token in self.eos:
                result[self.tokens[token]] += mass
            else:
                for tail, probability in self.cover(history+(token,)).items():
                    result[self.tokens[token]+tail] += mass*probability
        return dict(result)

    def describe(self):
        return dict(seed=self.seed, depth=self.depth, wide=self.wide,
                    tokens={str(t): v.hex() for t, v in self.tokens.items()}, eos=sorted(self.eos),
                    transitions=[dict(history=h, choices=[(t, str(p)) for t, p in law])
                                 for h, law in self.transitions.items()])


def candidates(tree, belief):
    fresh = [Atom(belief.path+(t,), tree.tokens[t], belief.alpha*p, t in tree.eos)
             for t, p in tree.transitions[belief.path] if belief.alpha*p > 0] if belief.alpha else []
    # Ordering is a declared sampling convention, not a planner grouping call.
    fresh.sort(key=lambda a: (a.residual, a.path[-1]))
    return fresh+list(belief.retained)


def oracle_groups(tree, belief):
    rows = candidates(tree, belief)
    values = set(a.residual for a in rows)
    # Quadratic strict-prefix definition; no index, interval, CDF or merge reuse.
    keys = sorted(v for v in values if not any(v != u and v.startswith(u) for u in values))
    return [(key, [a for a in rows if a.residual.startswith(key)]) for key in keys]


def oracle_next(belief, key, rows, selected, fullsync):
    total = sum(a.mass for a in rows)
    if fullsync:
        return Belief(selected.path, F(0) if selected.terminal else F(1), (),
                      belief.visible+selected.residual, selected.terminal)
    exact = sum(a.mass for a in rows if a.residual == key)
    strict = [Atom(a.path, a.residual[len(key):], a.mass/total, a.terminal)
              for a in rows if a.residual != key]
    strict.sort(key=lambda a: a.residual)  # stable fresh-before-retained ties
    return Belief(selected.path, F(0) if selected.terminal else exact/total,
                  tuple(strict), belief.visible+key, selected.terminal)


def signature(state):
    return dict(path=state.base_path.token_ids(), alpha=float(state.expansion_mass),
                visible=state.resolved_bytes.hex(), terminal=state.terminal,
                retained=sorted((a.path.token_ids(), a.residual.hex(), float(a.mass), a.terminal)
                                for a in state.retained))


def expected_signature(b):
    return dict(path=b.path, alpha=float(b.alpha), visible=b.visible.hex(), terminal=b.terminal,
                retained=sorted((a.path, a.residual.hex(), float(a.mass), a.terminal) for a in b.retained))


def compare_state(actual, expected):
    a, b = signature(actual), expected_signature(expected)
    assert (a['path'], a['visible'], a['terminal']) == (b['path'], b['visible'], b['terminal']), (a, b)
    assert math.isclose(a['alpha'], b['alpha'], rel_tol=TOL, abs_tol=TOL), (a, b)
    assert len(a['retained']) == len(b['retained']), (a, b)
    for x, y in zip(a['retained'], b['retained']):
        assert x[:2] == y[:2] and x[3] == y[3] and math.isclose(x[2], y[2], rel_tol=TOL, abs_tol=TOL), (a, b)


class Uniform:
    def __init__(self, u): self.u = float(u)
    def random(self): return self.u


def kernel_audit(tree, tau=0.):
    index = tree.backend.index
    methods = {'jump': make_jump_planner(index), 'bulk': make_bulk_planner(index),
               'exact_block': make_exact_planner(index, 'block'), 'naive': NaiveLASMethod(index)}
    ctx = MethodContext(tree.backend.codec, tree.prompt, 'encode')
    counts = Counter()
    worst = dict(group_mass=0., posterior_entropy=0., terminal_probability=0.)
    cache = {}
    first_context = {}

    def walk(belief, state, depth):
        nonlocal first_context
        compare_state(state, belief)
        if belief.terminal: return {belief.visible: F(1)}, {belief.visible: 1.}
        if belief in cache: return cache[belief]
        assert depth <= tree.depth*max(map(len, tree.tokens.values()))+2
        assert counts['states'] < 40000, 'Resource bound, not a passed/pruned case'
        counts['states'] += 1
        ideal, actual = defaultdict(F), defaultdict(float)
        counts['max_frontier'] = max(counts['max_frontier'], len(belief.retained))
        ds = [tree.distribution(belief.path)] if belief.alpha else []
        groups = oracle_groups(tree, belief)
        # Reconstruct machine input masses independently, not from plan metadata.
        machine_atoms = ([Atom(belief.path+(t,), tree.tokens[t], F(float(state.expansion_mass)*float(ds[0].probs[t])), t in tree.eos)
                          for t, _ in tree.transitions[belief.path]] if ds else [])
        machine_atoms += [Atom(a.path.token_ids(), a.residual, F(float(a.mass)), a.terminal) for a in state.retained]
        plans = {name: m.plan(state, ds, ctx) for name, m in methods.items()}
        first_context = dict(belief=expected_signature(belief), depth=depth)
        for name, plan in plans.items():
            assert [r['key'] for r in plan.opaque['records']] == [g[0] for g in groups], name
            assert abs(sum(plan.probs.tolist())-1) < TOL
        assert sum(sum(a.mass for a in rows) for _, rows in groups) == 1
        for i, (key, rows) in enumerate(groups):
            counts['groups'] += 1
            total = sum(a.mass for a in rows)
            exact = [a for a in rows if a.residual == key]
            strict = [a for a in rows if a.residual != key]
            e = sum(a.mass for a in exact)
            assert exact and all(not k.startswith(key) and not key.startswith(k) for k, _ in groups if k != key)
            if len(exact)>1: counts['multi_exact'] += 1
            if strict and belief.retained: counts['interleaved_retained'] += 1
            if any(a.path[:-1] != belief.path for a in exact): counts['old_exact'] += 1
            probs = [e/total]+[a.mass/total for a in strict]
            u = -math.fsum(float(p)*math.log2(float(p)) for p in probs if p)
            fullsync = u < tau
            for name, plan in plans.items():
                err = abs(float(plan.probs[i])-float(total))
                worst['group_mass'] = max(worst['group_mass'], err)
                assert err < TOL, (name, key, total, plan.probs)
                if name != 'naive':
                    record = plan.opaque['records'][i]
                    for field, value in [('group_mass', total), ('exact_mass', e), ('strict_mass', total-e)]:
                        assert math.isclose(record[field], float(value), rel_tol=TOL, abs_tol=TOL), (name, field)
                    assert (record['exact_count'], record['strict_count']) == (len(exact), len(strict))
                    worst['posterior_entropy'] = max(worst['posterior_entropy'], abs(record['las_posterior_entropy_bits']-u))
                    assert abs(record['las_posterior_entropy_bits']-u) < TOL
                    # A threshold equality may straddle FP rounding. tau=0/.5/inf
                    # fixtures have no near-.5 cases; fail rather than hide any.
                    assert (record['las_posterior_entropy_bits'] < tau) == fullsync
            choices = rows if fullsync else exact
            choice_total = sum(a.mass for a in choices)
            cursor = F(0)
            for selected in choices:
                midpoint = (cursor+selected.mass/2)/choice_total
                cursor += selected.mass
                expected = oracle_next(belief, key, rows, selected, fullsync)
                new = {}
                for name, m in methods.items():
                    if fullsync and name == 'naive': continue  # no native FullSync API
                    advance = m.advance_fullsync if fullsync else m.advance
                    new[name] = advance(state, plans[name], i, Uniform(midpoint), ctx)
                    compare_state(new[name], expected)
                    counts['state_transition_comparisons'] += 1
                counts['fullsync' if fullsync else 'las'] += 1
                law, measured = walk(expected, new['jump'], depth+1)
                # Ideal rational kernel branch weight; machine probabilities
                # integrated as a normalized categorical, not a 53-bit RNG law.
                rational_weight = total*selected.mass/choice_total
                machine_choices = [a for a in machine_atoms if a.residual.startswith(key)
                                   and (fullsync or a.residual == key)]
                machine_selected = sum(a.mass for a in machine_choices
                                       if (a.path,a.residual)==(selected.path,selected.residual))
                machine_weight = float(plans['jump'].probs[i])*float(machine_selected/sum(a.mass for a in machine_choices))
                for y, p in law.items(): ideal[y] += rational_weight*p
                for y, p in measured.items(): actual[y] += machine_weight*p
        cache[belief] = dict(ideal), dict(actual)
        return cache[belief]

    initial = Belief()
    try:
        exact, actual = walk(initial, methods['jump'].start(tree.prompt), 0)
        cover = tree.cover()
        assert exact == cover, 'Independent rational LAS and cover differ'
        assert actual.keys() == cover.keys()
        worst['terminal_probability'] = max(abs(actual[y]-float(p)) for y, p in cover.items())
        assert worst['terminal_probability'] < TOL
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'FIRST_FAILURE.json').write_text(json.dumps(dict(tree=tree.describe(), tau=str(tau),
             first_state=first_context, error=repr(exc)), ensure_ascii=False, indent=2)+'\n')
        raise
    return dict(seed=tree.seed, depth=tree.depth, wide=tree.wide, tau=str(tau), all_passed=True,
                cover_outputs=len(cover), counts=dict(counts), max_absolute_error=worst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    specs = [(s, 3+(s==0), s==2, s==3, 0) for s in range(8)]+[(8,3,False,False,24),(9,3,False,False,48)]
    for seed, depth, chinese, tiny, wide in specs:
        tree = Tree(seed, depth, chinese, tiny, wide)
        for tau in [0., .5, math.inf]:
            result = kernel_audit(tree, tau)
            results.append(result)
            print(seed, tau, result['counts'], flush=True)
            (OUT/'finite_oracle.json').write_text(json.dumps(dict(all_passed=len(results)==len(specs)*3,
                 cases=results, tolerance=TOL,
                 scope='Independent rational AR and belief oracle; actual planner/advance per group and representative; ideal normalized categorical actions, not finite-RNG perfect security'),indent=2)+'\n')


if __name__ == '__main__': main()
