"""Record new scheduling defaults without changing historical wire contracts."""
import json
import subprocess
import sys

import pytest

from stegatext_runtime.experiments.config import ROOT, resolve
from stegatext_runtime.experiments.engine import coder, message_for, send
from tests.finite_source import BitCarrierSource


def configuration(tmp_path, parameters=''):
    path = tmp_path / 'imec.toml'
    path.write_text(
        f'extends = "{ROOT / "configs/experiments/token_budget_T1_smoke.toml"}"\n'
        'name = "imec_defaults"\n'
        '[[variants]]\n'
        'id = "las_imec"\nkind = "planner"\nmethod = "las"\n'
        'coder = "imec_prefix_exact"\n' + parameters + '\n'
    )
    return resolve(path)


@pytest.mark.parametrize('parameters,block_size,window_size', [
    ('', 6, 3),
    ('coder_parameters = {block_size=2, window_size=2}', 2, 2),
])
def test_new_configuration_records_greedy_head_first(tmp_path, parameters, block_size, window_size):
    cfg = configuration(tmp_path, parameters)
    options = cfg['variants'][0]['coder_parameters']
    assert options == dict(block_size=block_size, window_size=window_size,
                           coupling_strategy='greedy', schedule='head_first')
    session = coder('imec_prefix_exact', 'encode', options)
    assert session.schedule == 'head_first' and session.drain_all


def test_explicit_schedule_and_legacy_defaults_are_preserved(tmp_path):
    cfg = configuration(tmp_path, 'coder_parameters = {block_size=6, window_size=3, schedule="max_entropy"}')
    assert coder('imec_prefix_exact', 'encode', cfg['variants'][0]['coder_parameters']).schedule == 'max_entropy'
    # Old public packets can omit the schedule. Their receiver must keep the
    # original interpretation instead of adopting a new configuration default.
    assert coder('imec_prefix_exact', 'decode', dict(block_size=6, window_size=3)).schedule == 'max_entropy'
    assert coder('imec_window_exact', 'decode').schedule == 'max_entropy'


def test_active_config_applies_same_schedule_to_both_methods():
    cfg = resolve(ROOT / 'configs/experiments/token_budget_T1_smoke.toml')
    variants = [v for v in cfg['variants'] if v.get('coder') == 'imec_prefix_exact']
    assert {v['method'] for v in variants} == {'las', 'syncpool'}
    assert all(v['coder_parameters'] == dict(block_size=6, window_size=3,
                                           coupling_strategy='greedy', schedule='head_first') for v in variants)
    historical = resolve(ROOT / 'configs/experiments/sentence_T1_coder_audit.toml')
    assert all(v['coder_parameters']['schedule'] == 'max_entropy'
               for v in historical['variants'] if v.get('coder') == 'imec_prefix_exact')


def test_new_default_roundtrip_from_public_packet_in_separate_process(tmp_path):
    cfg = configuration(tmp_path)
    case = dict(prompt=dict(text='', language='en'), seed=211, variant=cfg['variants'][0])
    public, sent = send(BitCarrierSource(), case, cfg)
    assert public['variant']['coder_parameters']['schedule'] == 'head_first'
    assert sent['bits'] and sent['bits'] == message_for(cfg, 211)[:sent['bits_count']]
    assert all(row['coder']['selected_window_index'] == 0 for row in sent['trace'])
    assert not ({'bits', 'trace', 'coder_state', 'token_ids'} & public.keys())
    script = (
        'import json,sys; from tests.finite_source import BitCarrierSource; '
        'import stegatext_runtime.experiments.engine as e; '
        'e.message_for=lambda *a,**kw: (_ for _ in ()).throw(AssertionError("Secret access")); '
        'p,c=json.load(sys.stdin); print(json.dumps(e.receive(BitCarrierSource(),p,c)))'
    )
    child = subprocess.run([sys.executable, '-B', '-c', script],
                           input=json.dumps([public, cfg]), text=True,
                           capture_output=True, cwd=ROOT, check=True)
    received = json.loads(child.stdout)
    assert received['receiver_consumed']
    assert all(received[k] == sent[k] for k in ('bits', 'trace', 'coder_state'))
