"""Small acyclic sources, independent rational cover oracle, actual kernels."""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations
from types import SimpleNamespace

import torch

from stegatext_runtime.source import Source
from stegatext_runtime.discop import DisCopSession
from stegatext_runtime.experiments.engine import method
from research_notes.full_validation.suite import digest, rendered
from stegtext.v2.coders.discop import _normal, _tree
from stegtext.v2.core.types import MethodContext, TokenDistribution
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.model.tokenizer_codec import EOS_STEGA


class FiniteSource:
    generate = Source.generate

    def __init__(self, *, multiple_eos=False, chinese=False, endless=False):
        tokens = {0: b"a", 1: b"ab", 2: b"b", 3: EOS_STEGA}
        self.transitions = {(): [(0, F(1, 2)), (1, F(1, 2))],
                            (0,): [(2, F(1, 2)), (3, F(1, 2))],
                            (1,): [(3, F(1))], (0, 2): [(3, F(1))]}
        eos = [3]
        if multiple_eos:
            tokens[4] = EOS_STEGA
            eos.append(4)
            self.transitions[(0,)] = [(2, F(1, 2)), (3, F(1, 4)), (4, F(1, 4))]
            self.transitions[(1,)] = [(3, F(1, 2)), (4, F(1, 2))]
        if chinese:
            tokens = {0: b"\xe4", 1: "中".encode(), 2: b"\xb8\xad", 3: EOS_STEGA}
            self.transitions[(0,)] = [(2, F(1))]
        self.endless = endless
        self.backend = ToyBackend(ToyBackendConfig(tokens, {}, eos_ids=eos))
        self.backend.codec_index = self.backend.index
        self.backend.codec.text_to_bytes = lambda text: text.encode()
        self.config = SimpleNamespace(do_sample=True)
        self.eos_ids = frozenset(eos)
        self.reference = SimpleNamespace(forwards=0)
        self.tokenizer = SimpleNamespace(decode=lambda ids, **kw: b"".join(tokens[t] for t in ids).decode(errors="replace"))

    def bind(self, prompt):
        return self.backend.render_prompt(prompt)

    def probabilities(self, ids):
        self.reference.forwards += 1
        values = torch.zeros(self.backend.vocab_size, dtype=torch.float64)
        for token, p in ([(0, F(1))] if self.endless else self.transitions[tuple(ids)]):
            values[token] = float(p)
        return values

    def distribution(self, path):
        return TokenDistribution(self.probabilities(path.token_ids()))

    def manifest(self):
        return {"sha256": "finite-source", "scope": "CPU finite source"}


def cover_oracle(source, history=(), visible=b"", mass=F(1)):
    """Enumerate raw autoregressive transitions, without a stego method/coder."""
    if history and history[-1] in source.eos_ids:
        return {visible: mass}
    result = defaultdict(F)
    for token, p in source.transitions[history]:
        data = b"" if token in source.eos_ids else source.backend.codec.token_bytes(token)
        for y, q in cover_oracle(source, history+(token,), visible+data, mass*p).items():
            result[y] += q
    return dict(result)


class NeedUniform(Exception):
    pass


class BitCarrierSource(FiniteSource):
    """128 binary choices followed by EOS, for end-to-end finite frames."""
    def __init__(self):
        super().__init__()
        self.backend = ToyBackend(ToyBackendConfig({0: b"a", 1: b"b", 2: b"", 3: EOS_STEGA}, {}, eos_ids=[3]))
        self.backend.codec_index = self.backend.index
        self.backend.codec.text_to_bytes = lambda text: text.encode()

    def probabilities(self, ids):
        self.reference.forwards += 1
        return torch.tensor([.5, .5, 0., 0.] if len(ids) < 128 else [0., 0., 0., 1.], dtype=torch.float64)


class ReplayUniform:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        try:
            return float(next(self.values))
        except StopIteration:
            raise NeedUniform from None


@lru_cache(None)
def coder_law(probs, supplied):
    """Integrate actual DisCop.encode over all piecewise-constant RNG regions.

    Breakpoints come from its Huffman tree. No sampling frequency estimate is
    used. Float comparisons are checked against a rational cover with tolerance.
    """
    tree = _tree(_normal(probs))
    cuts = {F(0), F(1), F(1, 2)}

    def collect(node):
        if node.left is None:
            return
        cut = F(node.left.weight/(node.left.weight+node.right.weight))
        cuts.update((cut, (cut+F(1, 2)) % 1))
        collect(node.left)
        collect(node.right)
    collect(tree)
    edges = sorted(cuts)
    intervals = [(a, b) for a, b in zip(edges, edges[1:]) if a < b]
    law = defaultdict(F)

    def visit(values, weight):
        try:
            outcome = DisCopSession().encode(probs, supplied, rng=ReplayUniform(values))
        except NeedUniform:
            for a, b in intervals:
                visit(values+[(a+b)/2], weight*(b-a))
        else:
            law[outcome] += weight
    visit([], F(1))
    assert sum(law.values()) == 1
    return tuple(law.items())


def advance_regions(weights):
    # Every cumulative-mass threshold for any ordering/subset is included.
    # This avoids duplicating LAS or SyncPool's representative selection code.
    weights = [F(float(x)) for x in weights if x > 0]
    cuts = {F(0), F(1)}
    for n in range(1, len(weights)+1):
        for ids in combinations(range(len(weights)), n):
            total = sum(weights[i] for i in ids)
            for k in range(1, len(ids)):
                for subset in combinations(ids, k):
                    cuts.add(sum(weights[i] for i in subset)/total)
    edges = sorted(cuts)
    return [(b-a, (a+b)/2) for a, b in zip(edges, edges[1:])]


def actual_wire_law(source, name, bits="0"*4096, tau=.5, joint_payload=False):
    m = method(source, name)
    ctx = MethodContext(source.backend.codec, source.bind(""), "encode")
    selective = name in ("las", "fullsync", "adaptive")

    def visit(state, offset, depth):
        assert depth <= 12, "Finite source did not terminate"
        if state.terminal:
            visible = rendered(source.backend.codec, state).encode()
            return {(visible, offset) if joint_payload else visible: F(1)}
        paths = ([state.base_path] if state.expansion_mass > 0 else []) if selective else m.queries(state)
        ds = [source.distribution(path) for path in paths]
        plan = m.plan(state, ds, ctx)
        weights = ([float(x)*state.expansion_mass for x in ds[0].probs] if ds and selective else ds[0].probs.tolist() if ds else [])
        weights += list(getattr(getattr(state, "retained", None), "masses", ()))
        regions = advance_regions(weights)
        branches = defaultdict(F)
        states = {}
        for (used, action), p in coder_law(tuple(plan.probs.tolist()), bits[offset:offset+32]):
            advance = m.advance
            if name == "fullsync" or (name == "adaptive" and plan.opaque["records"][action]["las_posterior_entropy_bits"] < tau):
                advance = m.advance_fullsync
            for q, u in regions:
                new = advance(state, plan, action, ReplayUniform([u]), ctx)
                key = (digest(new), offset+len(used))
                branches[key] += p*q
                states[key] = new
        result = defaultdict(F)
        for key, weight in branches.items():
            for y, tail in visit(states[key], key[1], depth+1).items():
                result[y] += weight*tail
        return dict(result)
    return visit(m.start(ctx.prompt), 0, 0)
