"""Independent path-likelihood and final-history recording contracts."""
from fractions import Fraction
import math

import pytest
import torch

from stegatext_runtime.experiments.engine_v2 import planner, public_termination
from stegatext_runtime.experiments.native_ppl import score_sequence
from stegatext_runtime.native_tokens import record_sequence
from tests.finite_source import FiniteSource
from tests.test_token_budget import packet


@pytest.mark.parametrize('method', ['syncpool', 'las', 'fullsync', 'adaptive'])
@pytest.mark.parametrize('limit', [1, 4])
def test_final_history_is_runtime_path_and_receiver_reconstructs_it(method, limit):
    p = packet(method, limit=limit)
    paths = set()
    for seed in range(8):
        source = FiniteSource()
        visited = []
        s = planner(source, p, key=f'path{seed}'.encode(), stream='01'*4096,
                    observer=lambda **kw: visited.append(kw['after']))
        seq = s['native_token_sequence']
        final = visited[-1]
        path = final.base_path if hasattr(final, 'base_path') else final.path
        assert seq['wire_output_token_ids'] == list(path.token_ids())
        assert seq['status'] == 'verified'
        assert source.backend.codec.tokens_to_bytes(seq['wire_output_token_ids']) == final.resolved_bytes
        assert seq['synthetic_boundary_eos'] == (limit == 1)
        assert seq['source_token_count'] <= limit
        if limit == 1:
            assert len(seq['output_token_ids']) == 1
            assert seq['wire_output_token_ids'][-1] == 3
        else:
            assert seq['output_token_ids'][-1] == 3 and seq['natural_eos']
        public = dict(p, text=s['text'], natural_eos=s['natural_eos'], termination=public_termination(s))
        r = planner(FiniteSource(), public, key=f'path{seed}'.encode())
        assert r['native_token_sequence'] == seq
        paths.add(tuple(seq['output_token_ids']))
    # These paths can render the same text through distinct tokenizations.
    assert len(paths) >= 2


def test_recording_has_no_effect_on_actions_bits_randomness_or_queries(monkeypatch):
    from stegatext_runtime import native_tokens
    source = FiniteSource()
    first = planner(source, packet(limit=4), key=b'instrumentation', stream='01'*4096)
    queries = source.reference.forwards
    monkeypatch.setattr(native_tokens, 'record_state', lambda *args, **kw: {})
    source = FiniteSource()
    second = planner(source, packet(limit=4), key=b'instrumentation', stream='01'*4096)
    assert source.reference.forwards == queries
    assert all(first[k] == second[k] for k in ['text', 'bits', 'trace', 'state_sha256'])


def scoring_fixture():
    source = FiniteSource()
    # Any attempt to obtain IDs from the visible text must fail.
    source.tokenizer.encode = lambda *args, **kw: pytest.fail('Text retokenization is forbidden')
    class Reference:
        def bind_prompt(self, ids):
            assert ids == [4, 4]  # Prompt has two tokens; no prompt token is scored.
        def next_logits(self, ids):
            table = {(4, 4): [.2, .4, .3, .1],
                     (4, 4, 0): [.1, .1, .5, .3],
                     (4, 4, 0, 2): [.1, .1, .1, .7],
                     (4, 4, 1): [.1, .1, .1, .7]}
            return torch.tensor([math.log(p) for p in table[tuple(ids)]], dtype=torch.float64)
        def probabilities(self, ids, logits):
            return torch.softmax(logits, dim=-1)
    def sequence(ids, natural=True, synthetic=False):
        return record_sequence(source, [4, 4], ids, source.backend.codec.tokens_to_bytes(ids),
                               natural_eos=natural, synthetic_eos=synthetic, origin='runtime_test')
    return source, Reference(), sequence


def test_same_visible_text_has_distinct_native_path_ppl_and_correct_causal_alignment():
    source, ref, sequence = scoring_fixture()
    first, second = sequence([1, 3]), sequence([0, 2, 3])
    one, two = score_sequence(source, first, reference=ref), score_sequence(source, second, reference=ref)
    assert one['evaluator_tokens'] == 1 and two['evaluator_tokens'] == 2
    assert one['ppl'] == pytest.approx(1/.4)
    assert two['ppl'] == pytest.approx(1/math.sqrt(.2*.5))
    assert one['eos_inclusive']['nll_total'] == pytest.approx(-math.log(.4*.7))
    assert two['eos_inclusive']['nll_total'] == pytest.approx(-math.log(.2*.5*.7))
    assert first['output_token_ids'] != second['output_token_ids']


def test_synthetic_delimiter_is_never_scored_even_when_it_has_nonunit_model_likelihood():
    source, ref, sequence = scoring_fixture()
    seq = sequence([1, 3], natural=False, synthetic=True)
    value = score_sequence(source, seq, reference=ref)
    assert seq['output_token_ids'] == [1]
    assert value['eos_inclusive']['tokens'] == value['evaluator_tokens'] == 1
    assert value['ppl'] == pytest.approx(2.5)


def test_empty_body_has_undefined_ppl_and_still_scores_a_real_eos():
    source, ref, sequence = scoring_fixture()
    value = score_sequence(source, sequence([3]), reference=ref)
    assert value['ppl'] is None and value['evaluator_status'] == 'empty_generated_body'
    assert value['eos_inclusive']['ppl'] == pytest.approx(10)


@pytest.mark.parametrize('field,value', [('source_sha256', 'wrong-model'), ('status', 'unavailable_incomplete_history'),
                                        ('wire_reconstruction_verified', False), ('prompt_token_ids', []),
                                        ('body_token_count', 100), ('natural_eos', False)])
def test_corrupt_or_unverified_native_input_fails_closed(field, value):
    source, ref, sequence = scoring_fixture()
    seq = sequence([1, 3])
    seq[field] = value
    with pytest.raises(ValueError):
        score_sequence(source, seq, reference=ref)


def test_raw_model_ppl_is_distinct_from_processed_sampler_ppl():
    source, ref, sequence = scoring_fixture()
    ref.probabilities = lambda ids, logits: torch.tensor([0., 1., 0., 0.])
    value = score_sequence(source, sequence([1, 3], natural=False, synthetic=True), reference=ref)
    assert value['ppl'] == pytest.approx(2.5)
    assert value['sampler_body']['ppl'] == 1


def test_utf8_boundary_and_zero_token_watchdog_remain_explicit():
    source = FiniteSource(chinese=True)
    source.transitions[()] = [(0, Fraction(1))]
    s = planner(source, packet(limit=1), key=b'utf8', stream='01'*4096)
    assert s['text'] is None and s['visible_bytes_hex'] == 'e4'
    assert s['native_token_sequence']['output_token_ids'] == [0]
    assert s['native_token_sequence']['status'] == 'verified'
