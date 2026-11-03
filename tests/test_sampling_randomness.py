"""Random-stream isolation, legacy replay and independent receiver contracts."""
import copy

import pytest
import torch

from stegatext_runtime.experiments.config import ROOT, cases, resolve
from stegatext_runtime.experiments.engine import send, receive
from stegatext_runtime.experiments.randomness import descriptor, INDEPENDENT
from stegatext_runtime.experiments.runner import conditions, planning
from tests.finite_source import FiniteSource
from tools.controlled_comparison.reproduce import single_config, write_config


def fixture():
    config = resolve(ROOT/'configs/experiments/token_budget_T1.toml')
    config['inputs']['ordinary_seed_policy'] = INDEPENDENT
    config['stopping']['max_new_tokens'] = 4
    case = next(c for c in cases(config) if c['variant']['kind'] == 'ordinary')
    case['prompt'] = dict(case['prompt'], text='', sha256='unused-by-toy')
    return config, case


@pytest.mark.parametrize('change', ['source', 'prompt', 'replicate', 'stopping'])
def test_conditions_get_distinct_reproducible_streams(change):
    config, case = fixture()
    manifest = dict(sha256='model-A-k256')
    before = descriptor(config, case, manifest)
    cfg, row, src = copy.deepcopy(config), copy.deepcopy(case), dict(manifest)
    if change == 'source': src['sha256'] = 'model-A-k1024'
    if change == 'prompt': row['prompt']['text'] += 'changed'
    if change == 'replicate': row['seed'] += 1
    if change == 'stopping': cfg['stopping']['max_new_tokens'] += 1
    after = descriptor(cfg, row, src)
    assert before == descriptor(config, case, manifest)
    assert before['condition_sha256'] != after['condition_sha256']
    assert before['torch_seed'] != after['torch_seed']
    assert 0 <= after['torch_seed'] < 2**63


def test_draw_is_local_and_does_not_touch_global_torch_rng_or_probabilities():
    config, case = fixture()
    source = FiniteSource()
    expected_probs = source.probabilities([]).clone()
    torch.manual_seed(98765)
    global_state = torch.random.get_rng_state().clone()
    _, first = send(source, case, config)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    torch.rand(30)  # Global draws and preceding cases cannot change this draw.
    _, second = send(FiniteSource(), case, config)
    assert first == second
    assert torch.equal(expected_probs, source.probabilities([]))
    assert first['seed'] == first['sampling_randomness']['torch_seed']
    assert first['sampling_randomness']['replicate_seed'] == case['seed']


@pytest.mark.parametrize('schema', [1, 2])
def test_legacy_without_new_policy_replays_old_global_generator_exactly(schema):
    config, case = fixture()
    config['schema_version'] = schema
    config['inputs'].pop('ordinary_seed_policy')
    if schema == 1:
        config['stopping'] = dict(max_steps=4, utf8_extension=32)
    # The original algorithm uses global manual_seed and multinomial.
    oracle = FiniteSource()
    torch.manual_seed(case['seed'])
    ids = []
    while len(ids) < 4:
        token = int(torch.multinomial(oracle.probabilities(ids)[None], num_samples=1)[0, 0])
        ids.append(token)
        if token in oracle.eos_ids: break
    expected_state = torch.random.get_rng_state().clone()
    _, actual = send(FiniteSource(), case, config)
    assert actual['token_ids'] == ids
    assert torch.equal(expected_state, torch.random.get_rng_state())
    assert 'sampling_randomness' not in actual


@pytest.mark.parametrize('schema', [1, 2])
def test_new_policy_applies_to_both_ordinary_entry_points(schema):
    config, case = fixture()
    config['schema_version'] = schema
    if schema == 1: config['stopping'] = dict(max_steps=4, utf8_extension=32)
    _, result = send(FiniteSource(), case, config)
    assert result['sampling_randomness'] == descriptor(config, case, FiniteSource().manifest())


@pytest.mark.parametrize('method', ['syncpool', 'las'])
def test_ordinary_policy_does_not_change_stego_protocol_keys_or_recovery(method):
    config, case = fixture()
    case['variant'] = dict(id=method, kind='planner', method=method, coder='discop_copy')
    legacy = copy.deepcopy(config)
    legacy['inputs'].pop('ordinary_seed_policy')
    packet, result = send(FiniteSource(), case, config)
    old_packet, old_result = send(FiniteSource(), case, legacy)
    assert packet == old_packet
    assert {k:v for k,v in result.items() if not k.endswith('_seconds')} == {
        k:v for k,v in old_result.items() if not k.endswith('_seconds')}
    recovered = receive(FiniteSource(), packet, config)
    assert recovered['bits'] == result['bits']
    assert recovered['state_sha256'] == result['state_sha256']


def test_single_case_slicing_and_variant_labels_keep_identical_stream(tmp_path):
    config = resolve(ROOT/'configs/experiments/token_budget_T1.toml')
    case = next(c for c in cases(config) if c['variant']['kind'] == 'ordinary')
    selected = single_config(config, case['id'])
    path = tmp_path/'single.toml'
    write_config(path, selected)
    sliced = resolve(path)
    only_case = next(cases(sliced))
    manifest = dict(sha256='actual-source-k256')
    expected = descriptor(config, case, manifest)
    assert sliced['inputs']['ordinary_seed_policy'] == INDEPENDENT
    assert expected == descriptor(sliced, only_case, manifest)
    only_case['variant'] = dict(id='renamed', kind='ordinary')
    only_case['prompt']['id'] = 'renamed'
    assert expected == descriptor(sliced, only_case, manifest)


def test_policy_is_validated_and_does_not_silently_change_legacy_configs(tmp_path):
    old = resolve(ROOT/'configs/experiments/smoke.toml')
    assert 'ordinary_seed_policy' not in old['inputs']
    assert planning(old)['ordinary_seed_policy'] == 'shared_replicate_v1'
    bad = tmp_path/'bad.toml'
    bad.write_text('extends="'+str(ROOT/'configs/experiments/smoke.toml')+'"\n'
                   '[inputs]\nordinary_seed_policy="typo"\n')
    with pytest.raises(ValueError, match='ordinary_seed_policy'): resolve(bad)


def test_conditions_publish_the_actual_sampling_stream():
    config, case = fixture()
    manifest = FiniteSource().manifest()
    audit = conditions(case, config, dict(environment={}, code={}), manifest)
    _, result = send(FiniteSource(), case, config)
    assert audit['sampling_randomness'] == result['sampling_randomness']
    assert audit['seed'] == case['seed']  # Logical paired replicate stays stable.


def test_mismatched_generator_seed_cannot_be_reported_as_another_seed():
    with pytest.raises(ValueError, match='Generator seed'):
        FiniteSource().generate('', max_new_tokens=4, seed=1,
                                generator=torch.Generator().manual_seed(2))
