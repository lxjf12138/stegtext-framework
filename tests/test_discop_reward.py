import math

import pytest

from tests.finite_source import coder_law
from stegtext.v2.coders.discop import _normal, _tree


@pytest.mark.parametrize('probs', [(.3, .7), (.25, .25, .5), (.125, .125, .25, .5), (.01, .09, .2, .7)])
@pytest.mark.parametrize('bits', ['0'*128,'1'*128,'00110101'*16])
def test_copy_disagreement_integral_matches_tree_reward_and_entropy_bound(probs, bits):
    root = _tree(_normal(probs))
    def reward(node):
        if node.left is None: return 0.
        return 2*min(node.left.weight,node.right.weight)+reward(node.left)+reward(node.right)
    integrated = sum(float(mass)*len(used) for (used,action),mass in coder_law(probs,bits))
    assert integrated == pytest.approx(reward(root),abs=1e-12)
    assert integrated <= -sum(p*math.log2(p) for p in probs)+1e-12
