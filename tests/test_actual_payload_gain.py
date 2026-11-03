from fractions import Fraction as F

import pytest

from tests.finite_source import FiniteSource, actual_wire_law, cover_oracle


@pytest.mark.parametrize('bits', ['0'*4096,'1'*4096])
def test_actual_copy_coder_gain_can_be_positive_or_zero_despite_entropy_gain(bits):
    for q,sp_expected,las_expected in [(F(1,2),F(1,2),F(1,2)),(F(1,10),F(1,10),F(9,10))]:
        means={}
        for method in ['syncpool','las']:
            source=FiniteSource()
            source.transitions[(0,)]=[(2,q),(3,1-q)]
            joint=actual_wire_law(source,method,bits,joint_payload=True)
            oracle=cover_oracle(source)
            for y,p in oracle.items():
                assert abs(float(sum(m for (text,n),m in joint.items() if text==y)-p))<1e-12
            means[method]=sum(float(m)*n for (y,n),m in joint.items())
        assert means['syncpool']==pytest.approx(float(sp_expected),abs=1e-12)
        assert means['las']==pytest.approx(float(las_expected),abs=1e-12)
