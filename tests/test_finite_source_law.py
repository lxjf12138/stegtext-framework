from collections import defaultdict
from fractions import Fraction as F
import pytest
import torch

from tests.finite_source import FiniteSource, actual_wire_law, coder_law, cover_oracle


@pytest.mark.parametrize("method", ["syncpool", "las", "adaptive", "fullsync"])
@pytest.mark.parametrize("fixture", [{}, {"multiple_eos": True}, {"chinese": True}])
@pytest.mark.parametrize("pattern", ["0"*4096, "1"*4096])
def test_actual_coder_and_kernel_complete_wire_law_matches_exact_cover(method, fixture, pattern):
    source = FiniteSource(**fixture)
    oracle = cover_oracle(source)
    actual = actual_wire_law(source, method, pattern)
    assert sum(actual.values()) == 1
    assert actual.keys() == oracle.keys()
    assert all(abs(float(actual[y]-oracle[y])) < 1e-12 for y in oracle)


@pytest.mark.parametrize("probabilities", [(1.,), (.25, .75), (0., .25, .75), (.25, .25, .5), (2**-20, 1-2**-20)])
@pytest.mark.parametrize("pattern", ["0"*64, "1"*64, "00110101"*8])
def test_discop_fixed_message_conditional_action_law(probabilities, pattern):
    law = coder_law(probabilities, pattern)
    for action, p in enumerate(probabilities):
        actual = sum(weight for (_, a), weight in law if a == action)
        assert abs(float(actual)-p) < 1e-12


@pytest.mark.parametrize("fixture", [{}, {"multiple_eos": True}, {"chinese": True}])
def test_actual_ordinary_generation_matches_exact_path_oracle(monkeypatch, fixture):
    source = FiniteSource(**fixture)
    law = defaultdict(F)
    from stegatext_runtime.experiments.stopping import normalize
    stop = normalize(dict(mode="natural_eos_only"))

    def walk(history=(), mass=F(1)):
        if history and history[-1] in source.eos_ids:
            tokens = iter(history)
            monkeypatch.setattr(torch, "multinomial", lambda *a, **kw: torch.tensor([[next(tokens)]]))
            result = source.generate("", seed=11, stopping=stop)
            assert result["natural_eos"] and not result["censored"]
            law[result["text"].encode()] += mass
            return
        for token, p in source.transitions[history]:
            walk(history+(token,), mass*p)
    walk()
    assert dict(law) == cover_oracle(source)


def test_machine_rng_grid_deviation_is_measured_not_called_perfect_security():
    from math import ceil
    from stegatext_runtime.discop import DisCopSession
    from tests.finite_source import ReplayUniform
    # Enumerate the entire small RNG grid through the real coder.
    for bit in ("0", "1"):
        n = 256
        hits = sum(DisCopSession().encode((.3, .7), bit*8, rng=ReplayUniform([F(j, n)]))[1] == 0 for j in range(n))
        assert hits == ceil(F(.3)*n)
        assert abs(hits/n-.3) <= 1/n
    # On the real 53-bit grid, a positive interval smaller than one cell is
    # not sampled with its ideal real-valued probability. Check endpoints.
    n, tiny = 2**53, 2**-60
    assert F(ceil(F(tiny)*n), n) != F(tiny)
    assert DisCopSession().encode((tiny, 1.), "0"*8, rng=ReplayUniform([F(0)]))[1] == 0
    assert DisCopSession().encode((tiny, 1.), "0"*8, rng=ReplayUniform([F(1, n)]))[1] == 1
