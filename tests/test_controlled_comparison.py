import math
from types import SimpleNamespace

import pytest
import torch

from stegatext_runtime.experiments.engine import planner
from stegatext_runtime.experiments.engine_v2 import instrumentation
from stegatext_runtime.experiments.runner import ordered_cases
from stegatext_runtime.experiments.config import resolve, ROOT, cases
from stegatext_runtime.experiments.token_budget import TokenBudgetSource
from tests.finite_source import FiniteSource, actual_wire_law
from tests.test_token_budget import packet


@pytest.mark.parametrize('tau',[0., .25, .5, 1., 2., math.inf])
@pytest.mark.parametrize('fixture',[{},dict(multiple_eos=True),dict(chinese=True)])
def test_adaptive_thresholds_preserve_token_capped_visible_law(tau,fixture):
    source=FiniteSource(**fixture)
    # The generic coder enumerator returns UTF-8-encoded Python strings. A
    # reversible Latin-1 transport here lets it enumerate even a partial UTF-8
    # byte string, without repairing or discarding the original bytes.
    source.backend.codec.bytes_to_text=lambda data:data.decode('latin1')
    law=actual_wire_law(TokenBudgetSource(source,1),'adaptive',tau=tau)
    law={k.decode().encode('latin1'):v for k,v in law.items()}
    expected={b'a':.5,b'ab':.5} if not fixture.get('chinese') else {b'\xe4':.5,'中'.encode():.5}
    assert {k:float(v) for k,v in law.items()}==expected


def test_filter_coverage_uses_original_mass_not_renormalized_action_probs():
    dist=SimpleNamespace(probs=torch.tensor([.25,.75],dtype=torch.float64))
    plan=SimpleNamespace(probs=torch.tensor([1.]),opaque={'members':[[0]]})
    row=instrumentation(plan,0,SimpleNamespace(),distributions=[dist])
    assert row['source_probability_mass_retained']==.25
    assert row['group_entropy_bits']==0
    assert row['source_entropy_bits'][0]==pytest.approx(-.25*math.log2(.25)-.75*math.log2(.75))


def test_entropy_instrumentation_for_both_coders_and_ordinary():
    for coder in ('discop_copy','imec_prefix_exact'):
        result=planner(FiniteSource(),packet(coder=coder,limit=4),key=b'coverage',stream='01'*4096)
        assert all('source_entropy_bits' in r and r['source_probability_mass_retained']==1 for r in result['trace'])
    p=packet(limit=4)
    ordinary=FiniteSource().generate('',seed=211,stopping=p['stopping'])
    assert len(ordinary['ordinary_sampling'])==len(ordinary['token_ids'])
    assert ordinary['ordinary_sampling'][0]['source_entropy_bits']==1


def test_execution_order_is_reproducible_and_keeps_exact_case_set():
    cfg=resolve(ROOT/'configs/experiments/controlled_T1_k32.toml')
    expected=list(cases(cfg))
    ordered=ordered_cases(cfg)
    assert ordered==ordered_cases(cfg)
    assert sorted(r['id'] for r in ordered)==sorted(r['id'] for r in expected)
    assert ordered!=expected
    assert len(ordered)==368
    cfg['execution']['order_seed']+=1
    assert ordered_cases(cfg)!=ordered


@pytest.mark.parametrize('top_k', [0, 512, 1024, 2048])
def test_high_topk_contract_keeps_explicit_T1_and_top_p1(tmp_path, top_k):
    template = (ROOT/'configs/experiments/controlled_T1_k256.toml').read_text()
    path = tmp_path/'higher_k.toml'
    text = template.replace('top_k = 256', f'top_k = {top_k}')
    path.write_text(text)
    config = resolve(path)
    assert config['source']['sampling']['top_k'] == top_k
    assert len(list(cases(config))) == 368
    for wrong in [text.replace('temperature = 1.0', 'temperature = 0.6'),
                  text.replace('top_p = 1.0', 'top_p = 0.9')]:
        path.write_text(wrong)
        with pytest.raises(ValueError, match='Controlled T1 contract'):
            resolve(path)
