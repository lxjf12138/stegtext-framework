import math

import pytest

from tests.finite_source import FiniteSource, actual_wire_law, cover_oracle
from stegtext.starter.adaptive import select_kernel


@pytest.mark.parametrize('tau', [0., .25, .5, 1., 2., math.inf])
@pytest.mark.parametrize('fixture', [{}, {'multiple_eos': True}, {'chinese': True}])
def test_adaptive_grid_preserves_finite_source_wire_law(tau, fixture):
    source = FiniteSource(**fixture)
    actual = actual_wire_law(source, 'adaptive', '001101'*700, tau=tau)
    oracle = cover_oracle(source)
    assert actual.keys() == oracle.keys()
    assert all(abs(float(actual[y]-oracle[y])) < 1e-12 for y in oracle)


def test_threshold_endpoints_and_boundary():
    for u in (0., .25, .5, 1., 2.):
        assert select_kernel(u, 0) == 'las'
        assert select_kernel(u, math.inf) == 'fullsync'
        assert select_kernel(u, u) == 'las'
        assert select_kernel(u, u+.001) == 'fullsync'
