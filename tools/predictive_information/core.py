"""Finite visible-prefix laws and JS; no planner, model, or payload dependency."""
from collections import defaultdict
from functools import lru_cache
import math

EOS = 256


def entropy(values):
    return -math.fsum(float(p)*math.log2(float(p)) for p in values if p > 0)


def prefix_laws(components, kernel, realize, horizon=2):
    """Components contain history, residual SYMBOLS and terminal, not tokens.

    kernel(history) gives positive (token_id, normalized_probability) pairs.
    realize(token_id) returns nonempty bytes-as-symbols or (EOS,).
    EOS is absorbing: (byte, EOS) and (EOS, EOS) are distinct outcomes.
    Every positive token path is included; no beam/top-probability truncation.
    """
    @lru_cache(None)
    def tail(history, remaining):
        if remaining == 0:
            return {(): 1.}
        law = defaultdict(float)
        rows = kernel(history)
        if abs(math.fsum(p for _, p in rows)-1) > 1e-10:
            raise ValueError('Kernel is not normalized')
        for token, probability in rows:
            symbols = tuple(realize(token))
            if not symbols or probability <= 0:
                raise ValueError('Invisible token or nonpositive kernel row')
            for suffix, weight in complete(history+(token,), symbols, remaining).items():
                law[suffix] += probability*weight
        return dict(law)

    def complete(history, known, remaining):
        if EOS in known:
            stop = known.index(EOS)
            known = known[:stop]+(EOS,)*(remaining+1)
        if len(known) >= remaining:
            return {known[:remaining]: 1.}
        return {known+suffix: p for suffix, p in tail(history, remaining-len(known)).items()}

    laws = []
    for component in components:
        known = tuple(component['residual'])
        if component.get('terminal') and EOS not in known:
            known += (EOS,)
        law = complete(tuple(component['history']), known, horizon)
        if abs(math.fsum(law.values())-1) > 2e-10:
            raise ValueError('Visible law lost mass')
        laws.append(law)
    return laws


def js_information(weights, laws):
    if len(weights) != len(laws) or any(w < 0 for w in weights):
        raise ValueError('Invalid component mixture')
    total = math.fsum(weights)
    if total <= 0:
        raise ValueError('Zero component mass')
    q = [w/total for w in weights]
    mixture = defaultdict(float)
    for w, law in zip(q, laws):
        if abs(math.fsum(law.values())-1) > 2e-10:
            raise ValueError('Component law is not normalized')
        for z, p in law.items():
            mixture[z] += w*p
    direct = math.fsum(w*p*math.log2(p/mixture[z]) for w, law in zip(q, laws)
                       for z, p in law.items() if w > 0 and p > 0)
    difference = entropy(mixture.values())-math.fsum(w*entropy(law.values()) for w, law in zip(q, laws))
    upper = entropy(q)
    if abs(direct-difference) > 2e-10 or direct < -2e-10 or direct > upper+2e-10:
        raise ValueError('JS identities/bound failed')
    return dict(information=max(0., direct), entropy_difference=difference,
                component_entropy=upper, mixture_entropy=entropy(mixture.values()),
                conditional_entropy=math.fsum(w*entropy(law.values()) for w, law in zip(q, laws)),
                mixture=dict(mixture))


def marginal(law, horizon=1):
    result = defaultdict(float)
    for z, p in law.items():
        result[z[:horizon]] += p
    return dict(result)


def encode_law(law):
    return [[list(z), p] for z, p in sorted(law.items())]


def decode_law(rows):
    return {tuple(z): p for z, p in rows}
