"""Cached geometry must preserve tiny masses, changing boundaries and replay."""
import numpy as np
import pytest

import perf_research_planner as implementation
from perf_research_planner import IntervalSumCache
from round5_exact import stable_interval_sums


def test_cached_reductions_preserve_values_and_boundary_changes():
    cache = IntervalSumCache(capacity=2)
    for _ in range(2):
        for size in (4, 7):
            for lows, highs in [([], []), ([0, 1, 2], [1, 2, 4]),
                               ([0, 1, 2], [4, 3, 2]), ([size], [size])]:
                low, high = np.array(lows, dtype=np.int64), np.array(highs, dtype=np.int64)
                for scale in (1., 3.):
                    values = np.array([1., 1e-50, 0., 1e-100] + [0.]*(size-4)) * scale
                    actual = cache(values, low, high)
                    np.testing.assert_array_equal(actual, stable_interval_sums(values, low, high))
                    if lows == [0, 1, 2] and highs == [1, 2, 4]:
                        assert actual[1] > 0 and actual[2] > 0
                assert len(cache.geometry) <= 2
    assert cache(np.array([]), np.array([0]), np.array([0])).tolist() == [0.]


@pytest.mark.parametrize('name', ['las', 'adaptive', 'fullsync'])
@pytest.mark.parametrize('coder', ['discop_copy', 'imec_prefix_exact'])
def test_cached_protocol_matches_uncached_and_receiver(monkeypatch, name, coder):
    from stegatext_runtime.experiments.engine import planner
    from stegatext_runtime.experiments.engine_v2 import public_termination
    from tests.finite_source import FiniteSource
    from tests.test_token_budget import packet
    p = packet(name, coder, limit=4)
    # Exercise the same caching path on a fully enumerable small source.
    monkeypatch.setattr(implementation, '_INTERVAL_CACHE_MIN_ATOMS', float('inf'))
    before = planner(FiniteSource(), p, key=b'las-cache', stream='001101'*1024)
    monkeypatch.setattr(implementation, '_INTERVAL_CACHE_MIN_ATOMS', 0)
    after = planner(FiniteSource(), p, key=b'las-cache', stream='001101'*1024)
    public = dict(p, text=after['text'], natural_eos=after['natural_eos'],
                  termination=public_termination(after))
    received = planner(FiniteSource(), public, key=b'las-cache')
    for field in ('text', 'bits', 'trace', 'state_sha256', 'natural_eos', 'token_budget'):
        assert before[field] == after[field] == received[field]
    assert received['receiver_consumed']
