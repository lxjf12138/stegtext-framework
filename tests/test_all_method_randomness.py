"""Case-separated messages/keys, native recovery and portable case selection."""
import copy
import hashlib

import pytest

from stegatext_runtime.experiments.config import ROOT, cases, resolve
from stegatext_runtime.experiments.engine import message_for, send, receive
from stegatext_runtime.experiments.randomness import case_nonce, descriptor, CASE_POLICY
from tools.controlled_comparison.reproduce import single_config, write_config
from tests.finite_source import FiniteSource


def fixture(tmp_path):
    raw = single_config(resolve(ROOT/'configs/experiments/token_budget_T1.toml'),
        'qwen_instruct__city_zh__211__ordinary')
    raw['randomness'] = dict(mode='key_file', file=str(tmp_path/'root.key'),
                            stream_policy=CASE_POLICY, experiment_id='test-independent')
    (tmp_path/'root.key').write_text(bytes(range(32)).hex())
    path = tmp_path/'input.toml'
    write_config(path, raw)
    config = resolve(path)
    config['stopping']['max_new_tokens'] = 4
    case = next(cases(config))
    case['prompt']['text'] = ''
    case['prompt']['sha256'] = hashlib.sha256(b'').hexdigest()
    case['variant'] = dict(id='las', kind='planner', method='las', coder='discop_copy')
    return config, case


@pytest.mark.parametrize('field', ['source', 'prompt', 'replicate', 'method', 'coder', 'tau', 'experiment'])
def test_every_experimental_axis_gets_distinct_message_and_protocol_domain(tmp_path, field):
    config, case = fixture(tmp_path)
    source = FiniteSource().manifest()
    before = case_nonce(config, case, source)
    cfg, row, src = copy.deepcopy(config), copy.deepcopy(case), dict(source)
    if field == 'source': src['sha256'] = 'other-model-or-top-k'
    if field == 'prompt': row['prompt']['text'] = 'other'
    if field == 'replicate': row['seed'] += 1
    if field == 'method': row['variant']['method'] = 'syncpool'
    if field == 'coder': row['variant']['coder'] = 'imec_prefix_exact'
    if field == 'tau': row['variant'].update(method='adaptive', tau=.5)
    if field == 'experiment': cfg['randomness']['experiment_id'] = 'other-experiment'
    after = case_nonce(cfg, row, src)
    a, b = message_for(config, case['seed'], context=before), message_for(cfg, row['seed'], context=after)
    assert before != after and a != b and len(a) == len(b) == 131072
    assert set(a) == {'0', '1'}


@pytest.mark.parametrize('method', ['syncpool', 'las', 'fullsync', 'adaptive'])
@pytest.mark.parametrize('coder', ['discop_copy', 'imec_prefix_exact'])
def test_sender_and_receiver_reconstruct_keys_without_secret_message_or_trace(tmp_path, method, coder):
    config, case = fixture(tmp_path)
    case['variant'].update(method=method, coder=coder)
    if method == 'adaptive': case['variant']['tau'] = .5
    if coder == 'imec_prefix_exact': case['variant']['coder_parameters'] = dict(block_size=6, window_size=3)
    packet, sender = send(FiniteSource(), case, config)
    assert 'message_randomness' not in packet and 'bits' not in packet and 'trace' not in packet
    received = receive(FiniteSource(), packet, config)
    assert sender['bits'] == received['bits']
    assert sender['native_token_sequence'] == received['native_token_sequence']
    assert sender['protocol_randomness'] == received['protocol_randomness']
    payload = message_for(config, case['seed'], context=packet['nonce'])
    assert payload.startswith(received['bits'])
    assert sender['message_randomness']['sha256'] == hashlib.sha256(payload.encode()).hexdigest()
    assert sender['message_randomness']['padded'] is False
    wrong = copy.deepcopy(packet)
    wrong['experimental_randomness']['experiment_id'] = 'wrong'
    with pytest.raises(ValueError, match='randomness contract'):
        receive(FiniteSource(), wrong, config)


def test_case_slicing_renaming_and_key_file_resolution_preserve_streams(tmp_path):
    config, case = fixture(tmp_path)
    config['inputs']['prompts'] = [case['prompt']]
    config['variants'] = [case['variant']]
    case = next(cases(config))
    # Avoid empty prompt validation; the fixture's toy source is irrelevant here.
    case['prompt']['text'] = config['inputs']['prompts'][0]['text'] = '测试'
    raw = single_config(config, case['id'])
    path = tmp_path/'single.toml'
    write_config(path, raw)
    sliced = resolve(path)
    single = next(cases(sliced))
    manifest = dict(sha256='actual-source')
    expected = case_nonce(config, case, manifest)
    assert case_nonce(sliced, single, manifest) == expected
    single['variant']['id'] = 'renamed'
    assert case_nonce(sliced, single, manifest) == expected
    assert message_for(config, case['seed'], context=expected) == message_for(sliced, single['seed'], context=expected)


def test_missing_context_or_changed_root_fails_and_ordinary_uses_experiment_root(tmp_path):
    config, case = fixture(tmp_path)
    with pytest.raises(ValueError, match='complete case context'): message_for(config, case['seed'])
    before = descriptor(config, case, FiniteSource().manifest())
    changed = copy.deepcopy(config)
    changed['randomness']['sha256'] = 'other-root'
    assert before['torch_seed'] != descriptor(changed, case, FiniteSource().manifest())['torch_seed']
    (tmp_path/'root.key').write_text(b'changed-key-32-bytes-padding-12345'.hex())
    with pytest.raises(ValueError, match='Key file changed'):
        message_for(config, case['seed'], context='case')


@pytest.mark.parametrize('coder', ['discop_copy', 'imec_prefix_exact'])
def test_cpu_coder_replay_uses_the_new_case_payload_domain(tmp_path, coder):
    from stegatext_runtime.experiments.runner import write
    from tools.controlled_comparison.replay import verify_case
    config, case = fixture(tmp_path)
    case['variant']['coder'] = coder
    if coder == 'imec_prefix_exact':
        case['variant']['coder_parameters'] = dict(block_size=6, window_size=3)
    packet, sent = send(FiniteSource(), case, config)
    recovered = receive(FiniteSource(), packet, config)
    folder = tmp_path/'run'
    ident = case['id']
    write(folder/'sender'/(ident+'.json'), dict(status='ok',result=sent))
    write(folder/'public'/(ident+'.json'), packet)
    write(folder/'receiver'/(ident+'.json'), dict(status='ok',result=recovered))
    row = dict(case_id=ident,seed=case['seed'],variant=case['variant'],exact=True)
    assert verify_case(folder,row,config)['status'] == 'coder_exact'
