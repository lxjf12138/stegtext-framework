import copy
from types import SimpleNamespace as NS
import pytest
from stegatext_runtime.experiments.config import resolve,cases
from stegatext_runtime.experiments.delivery import measure
from stegatext_runtime.experiments.stopping import contract
from stegatext_runtime.source import Source


def test_frozen_t1_matrix_and_explicit_operational_law():
    count=0
    for name,k in [('sentence_T1_coder_audit',32),('sentence_T1_coder_audit_k128',128),('sentence_T1_coder_audit_k256',256)]:
        cfg=resolve('configs/experiments/'+name+'.toml')
        assert cfg['source']['sampling']['temperature']==1.0 and cfg['source']['sampling']['top_p']==1.0
        assert cfg['source']['sampling']['top_k']==k
        for case in cases(cfg):
            c=contract(case['variant'],cfg['stopping'])
            assert c['visible_law']=='operational_method_dependent' and not c['strict_cover_law_claim']
            count+=1
    assert count==96


def test_runtime_sampling_guard_checks_reference_as_well_as_source():
    expected=dict(temperature=1.,top_p=1.,top_k=32,do_sample=True)
    s=NS(expected_sampling=expected,config=NS(**expected),reference=NS(config=NS(**expected)))
    Source.assert_sampling_contract(s)
    s.reference.config.temperature=.7
    with pytest.raises(ValueError,match='reference'):
        Source.assert_sampling_contract(s)


def test_zero_nonempty_and_wrong_prefix_separate():
    zero=measure('01','','',replay_exact=True)
    assert zero['prefix_exact'] and not zero['nonempty_payload'] and zero['strict_verified_payload_bits']==0
    nonzero=measure('01','0','0',replay_exact=True)
    assert nonzero['prefix_exact'] and nonzero['nonempty_payload'] and nonzero['strict_verified_payload_bits']==1
    assert not measure('01','1','1',replay_exact=True)['prefix_exact']


def test_common_evaluation_model_lock_is_plain_json_not_runner_envelope():
    from stegatext_runtime.experiments.runner import common_evaluation_spec
    from pathlib import Path
    spec=common_evaluation_spec()
    assert 'Qwen3-4B-Instruct' in spec['repo_id'] and Path(spec['local_path']).exists()


@pytest.mark.parametrize('value',[[],[{'case':1}],{'complete':True}])
def test_smoke_driver_writes_plain_json_lists_and_objects(tmp_path,value):
    import json
    from tools.framework_contract.run_smoke import write_json
    path=tmp_path/'index.json'
    write_json(path,value)
    assert json.loads(path.read_text())==value and not path.with_suffix('.tmp').exists()
