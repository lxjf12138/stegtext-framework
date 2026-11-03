from itertools import product
import math

import pytest

from tools.predictive_information.core import EOS, entropy, js_information, marginal, prefix_laws


def fixed(history, residual=(), terminal=False):
    return dict(history=tuple(history), residual=tuple(residual), terminal=terminal)


def test_identical_kernels_have_no_identity_information():
    laws = prefix_laws([fixed((0,)), fixed((1,))], lambda h: [(2, .3), (3, .7)],
                       lambda t: (65, 66) if t == 2 else (67, 68))
    assert js_information([.1, .9], laws)['information'] == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize('q', [[.5, .5], [.1, .9], [.01, .09, .9]])
def test_residuals_alone_can_reveal_identity_without_model_calls(q):
    def no_kernel(h):
        raise AssertionError('Long enough residual must not query LM')
    laws = prefix_laws([fixed((j,), (65+j, 70)) for j in range(len(q))], no_kernel, lambda t: ())
    assert js_information(q, laws)['information'] == pytest.approx(entropy(q))


def test_first_byte_can_hide_information_visible_in_second_byte():
    laws = prefix_laws([fixed((0,), (65, 66)), fixed((1,), (65, 67))],
                       lambda h: [], lambda t: ())
    assert js_information([.3, .7], [marginal(p) for p in laws])['information'] == pytest.approx(0.)
    assert js_information([.3, .7], laws)['information'] == pytest.approx(entropy([.3, .7]))


def test_residual_then_model_bytes_and_absorbing_eos():
    laws = prefix_laws([fixed((0,), (65,)), fixed((1,), terminal=True)],
                       lambda h: [(2, .75), (3, .25)], lambda t: (66,) if t == 2 else (EOS,))
    assert laws == [{(65, 66): .75, (65, EOS): .25}, {(EOS, EOS): 1.}]
    assert js_information([.6, .4], laws)['information'] == pytest.approx(entropy([.6, .4]))


def test_recursive_byte_law_matches_independent_two_token_path_enumeration():
    kernels = {(): [(0, .4), (1, .6)], (0,): [(0, .2), (1, .8)], (1,): [(0, .7), (1, .3)]}
    symbols = {0: (65,), 1: (66,)}
    law = prefix_laws([fixed(())], lambda h: kernels[h], lambda t: symbols[t])[0]
    oracle = {(symbols[a]+symbols[b]): pa*dict(kernels[(a,)])[b]
              for a, pa in kernels[()] for b in (0, 1)}
    assert law == pytest.approx(oracle)


def test_multiple_eos_tokens_merge_as_one_visible_terminal_symbol():
    law = prefix_laws([fixed(())], lambda h: [(0, .2), (1, .3), (2, .5)],
                      lambda t: (EOS,) if t < 2 else (65, 66))[0]
    assert law == {(EOS, EOS): .5, (65, 66): .5}


def test_sparse_grid_of_mixtures_obeys_bound_and_prefix_data_processing():
    for a, b in product([.1, .5, .9], repeat=2):
        laws = [{(65, 66): a, (65, 67): 1-a}, {(65, 66): b, (66, 67): 1-b}]
        for weight in [.01, .1, .5, .9, .99]:
            short = js_information([weight, 1-weight], [marginal(p) for p in laws])
            long = js_information([weight, 1-weight], laws)
            assert 0 <= short['information'] <= long['information']+1e-12
            assert long['information'] <= long['component_entropy']+1e-12
