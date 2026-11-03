import math
from types import SimpleNamespace

import pytest
import torch

from stegatext_runtime.experiments.ppl import score_ids, windows


@pytest.mark.parametrize('count', [0, 1, 255, 256, 257, 511, 512, 1025])
def test_sliding_windows_score_every_token_exactly_once(count):
    predictions = []
    for begin, end, lo, hi in windows(count):
        assert end-begin <= 512
        assert 0 <= lo < hi < end-begin
        predictions.extend(range(begin+lo+1, begin+hi+1))
    assert predictions == list(range(1, count+1))


def test_uniform_lm_has_known_ppl_and_nll():
    class Uniform:
        def __call__(self, values, **kwargs):
            return SimpleNamespace(logits=torch.zeros((*values.shape, 8)))
    result = score_ids(Uniform(), [1, 2, 3]*200, 0, device='cpu')
    assert result['ppl'] == pytest.approx(8, rel=1e-6)
    assert result['nll_total'] == pytest.approx(600*math.log(8), rel=1e-6)
    assert score_ids(Uniform(), [], 0, device='cpu')['ppl'] is None


def test_causal_next_token_alignment_across_window_boundaries():
    class CountingLM:
        def __call__(self, values, **kwargs):
            logits = torch.full((*values.shape,4), math.log(.1))
            logits.scatter_(-1, ((values+1)%4).unsqueeze(-1), math.log(.7))
            return SimpleNamespace(logits=logits)
    result = score_ids(CountingLM(), [(i+1)%4 for i in range(1000)], 0, device='cpu')
    assert result['log_ppl'] == pytest.approx(-math.log(.7), rel=1e-6)
