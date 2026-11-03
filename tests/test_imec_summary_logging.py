"""Logging must neither expand with the vocabulary nor change encoding."""
import json
import random

import pytest

from stegatext_runtime.imec import IMECSession
from stegatext_runtime.experiments.recording import distribution_trace


@pytest.mark.parametrize('strategy', ['greedy', 'prefix'])
@pytest.mark.parametrize('schedule', ['max_entropy', 'head_first'])
def test_summary_and_full_logging_have_identical_numerical_trajectories(strategy, schedule):
    sessions = [IMECSession(coupling_strategy=strategy, schedule=schedule,
                            drain_all=True, mode=mode)
                for mode in ('encode', 'encode', 'decode')]
    full, compact, decoder = sessions
    full.set_trace_policy(max_full_actions=None)
    compact.set_trace_policy(max_full_actions=0)
    decoder.set_trace_policy(max_full_actions=0)
    generator = random.Random(2510)
    message = ''.join(generator.choice('01') for _ in range(4096))
    offset = 0
    for step in range(20):
        probs = [generator.random() for _ in range(67)]
        a = full.encode(probs, message[offset:], rng=random.Random(step))
        b = compact.encode(probs, message[offset:], rng=random.Random(step))
        assert a == b
        assert decoder.decode(probs, a[1], rng=None) == a[0]
        assert full.state() == compact.state() == decoder.state()
        assert compact.last_trace == decoder.last_trace
        for k, v in compact.last_trace.items():
            if k != 'storage':
                assert full.last_trace[k] == v
        offset += len(a[0])
    assert full.finalize_confirmed_prefix() == compact.finalize_confirmed_prefix() == decoder.finalize_confirmed_prefix()


@pytest.mark.parametrize('count', [8192, 131072])
def test_rational_diagnostic_conversions_are_bounded_at_the_source(monkeypatch, count):
    import stegatext_runtime.imec as module
    real_str, calls = str, []
    def tracked(value):
        calls.append(1)
        return real_str(value)
    monkeypatch.setattr(module, 'str', tracked, raising=False)
    session = IMECSession()
    session.encode([1/count]*count, '01'*100, rng=random.Random(19))
    assert len(calls) <= 256  # prior/posterior over 64 messages + eight preview cells
    assert 'coupling_edges' not in session.last_trace
    assert 'group_probabilities' not in session.last_trace
    assert len(json.dumps(session.last_trace)) < 16000


def test_large_distribution_summaries_store_only_preview_and_scalars():
    import math
    import struct
    import hashlib
    from tools.controlled_comparison.statistics import normalized_entropy
    p = [1/(i+1) for i in range(16384)]
    summary = distribution_trace(p, 89)['group_distribution']
    assert summary['count'] == len(p)
    assert summary['head'] == [[j, p[j]] for j in range(8)]
    assert summary['weights_binary64le_sha256'] == hashlib.sha256(struct.pack('<'+'d'*len(p), *p)).hexdigest()
    assert summary['entropy_bits_normalized'] == pytest.approx(normalized_entropy(p), abs=1e-12)
    assert summary['selected_probability_normalized'] == p[89]/math.fsum(p)
    assert len(json.dumps(summary)) < 1500
    assert len(json.dumps([summary]*256)) < 384000
