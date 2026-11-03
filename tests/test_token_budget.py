"""Compare capped generation to an independently truncated Fraction AR source."""
from collections import defaultdict
from fractions import Fraction as F
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from stegatext_runtime.experiments.config import ROOT, resolve, cases
from stegatext_runtime.experiments.engine import planner, send, receive
from stegatext_runtime.experiments.engine_v2 import public_termination
from stegatext_runtime.experiments.stopping import normalize, contract
from stegatext_runtime.experiments.token_budget import TokenBudgetSource
from tests.finite_source import FiniteSource, BitCarrierSource, actual_wire_law
from tools.framework_contract.oracle import TreeSource, audit


def stopped_cover(source, limit, path=(), visible=b'', mass=F(1)):
    # Direct ordinary generation, stopping BEFORE querying token N+1.
    if len(path) == limit or (path and path[-1] in source.eos_ids):
        return {visible: mass}
    out = defaultdict(F)
    for token, p in source.transitions[path]:
        data = b'' if token in source.eos_ids else source.tokens[token]
        for text, weight in stopped_cover(source, limit, path+(token,), visible+data, mass*p).items():
            out[text] += weight
    return dict(out)


def exact_capped_fixture(base, limit):
    # This independent transition table is read by the Fraction oracle. The
    # real planner instead uses TokenBudgetSource.distribution on the original
    # source, so expected and actual don't share the capping implementation.
    transitions = {}

    def visit(path):
        if path and path[-1] in base.eos_ids:
            return
        transitions[path] = ([(min(base.eos_ids), F(1))] if len(path) == limit
                             else base.transitions[path])
        if len(path) < limit:
            for token, _ in transitions[path]:
                visit(path+(token,))

    visit(())
    capped = TokenBudgetSource(base, limit)
    return SimpleNamespace(tokens=base.tokens, eos_ids=base.eos_ids,
                           transitions=transitions, backend=base.backend,
                           bind=capped.bind, distribution=capped.distribution)


@pytest.mark.parametrize('fixture', ['mixed', 'thirds', 'utf8', 'sentence'])
@pytest.mark.parametrize('limit', [1, 2, 3])
@pytest.mark.parametrize('name', ['las', 'las_local', 'syncpool'])
def test_fraction_law_matches_ordinary_source_token_cap(fixture, limit, name):
    source = TreeSource(fixture)
    expected = stopped_cover(source, limit)
    result = audit(fixture, name, source=exact_capped_fixture(source, limit))
    assert {bytes.fromhex(k): F(v) for k, v in result['law'].items()} == expected


def packet(name='las', coder='discop_copy', limit=1):
    variant = dict(id=name, kind='planner', method=name, coder=coder)
    if name == 'adaptive':
        variant['tau'] = .5
    if coder == 'imec_prefix_exact':
        variant['coder_parameters'] = dict(block_size=6, window_size=3)
    return dict(prompt='', language='en', message_delivery='prefix_stream', variant=variant,
                stopping=normalize(dict(mode='token_budget', max_new_tokens=limit,
                                        watchdog_protocol_decisions=256)))


@pytest.mark.parametrize('name', ['las', 'syncpool', 'adaptive', 'fullsync'])
@pytest.mark.parametrize('coder', ['discop_copy', 'imec_prefix_exact'])
@pytest.mark.parametrize('limit', [1, 2, 4])
def test_cap_roundtrip_and_natural_eos_are_distinct(name, coder, limit):
    p = packet(name, coder, limit)
    sender = planner(FiniteSource(), p, key=b'token-cap', stream='01'*4096)
    public = dict(p, text=sender['text'], natural_eos=sender['natural_eos'],
                  termination=public_termination(sender))
    receiver = planner(FiniteSource(), public, key=b'token-cap')
    assert receiver['receiver_consumed']
    assert all(sender[k] == receiver[k] for k in ['bits','trace','state_sha256','coder_state','token_budget'] if k in sender)
    assert sender['token_budget']['representative_generated_tokens'] <= limit
    if limit == 1:
        assert sender['terminal_reason'] == 'token_limit' and sender['censored']
        assert not sender['natural_eos']
    if limit == 4:
        assert sender['terminal_reason'] == 'natural_eos' and sender['natural_eos']
        assert not sender['censored']


def test_one_source_token_can_require_multiple_las_actions():
    p = packet(limit=1)
    outcomes = set()
    for seed in range(16):
        s = planner(FiniteSource(), p, key=f'cap{seed}'.encode(), stream='01'*4096)
        assert s['token_budget']['representative_generated_tokens'] == 1
        outcomes.add(s['text'])
    assert outcomes == {'a', 'ab'}
    source = TreeSource('sentence')
    law = audit('sentence', 'las', source=exact_capped_fixture(source, 1))['law']
    assert {bytes.fromhex(k):F(v) for k,v in law.items()} == {
        b'Hello world.':F(1,2), b'Hello world. Additional text':F(1,2)}


@pytest.mark.parametrize('name', ['las', 'syncpool', 'adaptive', 'fullsync'])
def test_actual_discop_capped_text_law(name):
    source = TokenBudgetSource(FiniteSource(), 1)
    law = actual_wire_law(source, name)
    assert law == {b'a':F(1,2), b'ab':F(1,2)}


def test_token_mode_never_checks_sentence_or_message_goals(monkeypatch):
    from stegatext_runtime.experiments import sentence, engine
    def forbidden(*args, **kwargs):
        raise AssertionError('Token-budget mode must not inspect punctuation/bit goals')
    monkeypatch.setattr(sentence, 'observation', forbidden)
    monkeypatch.setattr(engine, 'goal', forbidden)
    source = FiniteSource(endless=True)
    p = packet(limit=3)
    s = planner(source, p, key=b'not-sentence', stream='01'*4096)
    assert s['text'] == 'aaa' and s['terminal_reason'] == 'token_limit'
    ordinary = source.generate('', seed=211, stopping=p['stopping'])
    assert ordinary['text'] == 'aaa' and ordinary['token_budget'] == s['token_budget']


def test_cap_flushes_strict_residuals_but_never_repairs_utf8():
    p = packet(limit=1)
    for seed in range(8):
        s = planner(FiniteSource(chinese=True), p, key=f'utf{seed}'.encode(), stream='01'*4096)
        public = dict(p, text=s['text'], visible_bytes_hex=s['visible_bytes_hex'],
                      natural_eos=False, termination=public_termination(s))
        r = planner(FiniteSource(chinese=True), public, key=f'utf{seed}'.encode())
        assert r['receiver_consumed'] and r['visible_bytes_hex'] == s['visible_bytes_hex']
        assert s['visible_bytes_hex'] in ('e4', 'e4b8ad')
        assert s['terminal_reason'] == 'token_limit'
        if s['text'] is None:
            assert s['framing_status'] == 'invalid_utf8'


def test_eos_at_exact_limit_is_natural_and_no_model_query_beyond_cap():
    p = packet(limit=1)
    source = FiniteSource()
    source.transitions[()] = [(3, F(1))]
    s = planner(source, p, key=b'eos', stream='01'*4096)
    assert s['natural_eos'] and not s['token_budget']['token_limit_reached']
    source = FiniteSource(endless=True)
    s = planner(source, packet(limit=2), key=b'no-extra-query', stream='01'*4096)
    assert source.reference.forwards == 2
    assert s['steps'] == 3 and s['terminal_reason'] == 'token_limit'


def test_prompt_tokens_do_not_consume_generation_budget():
    from stegtext.v2.core.types import PathRef, PromptState
    source = FiniteSource(endless=True)
    path = PathRef.root().append(0).append(1).append(2)
    source.bind = lambda prompt: PromptState(path, (0,1,2), b'abb', prompt)
    p = packet(limit=2)
    p['prompt'] = 'nonempty prompt'
    s = planner(source, p, key=b'prompt-budget', stream='01'*4096)
    assert s['text'] == 'aa' and s['token_budget']['representative_generated_tokens'] == 2
    assert source.reference.forwards == 2


def test_resource_watchdog_is_separate_from_shared_token_budget():
    p = packet(limit=10)
    p['stopping']['watchdog_protocol_decisions'] = 1
    s = planner(FiniteSource(endless=True), p, key=b'watchdog', stream='01'*4096)
    assert s['terminal_reason'] == 'resource_watchdog' and s['censored']
    assert not s['token_budget']['source_terminal'] and not s['token_budget']['token_limit_reached']


def test_independent_receiver_uses_no_hidden_token_count():
    cfg = dict(schema_version=2, stopping=packet(limit=8)['stopping'],
               message=dict(mode='fixture', bits=131072, pattern='random', delivery='prefix_stream'),
               randomness=dict(mode='fixture'))
    case = dict(prompt=dict(text='', language='en'), seed=211,
                variant=packet(coder='imec_prefix_exact')['variant'])
    public, s = send(BitCarrierSource(), case, cfg)
    assert s['terminal_reason'] == 'token_limit'
    assert not ({'bits','trace','token_ids','coder_state'} & public.keys())
    script = ('import json,sys; from tests.finite_source import BitCarrierSource; '
              'from stegatext_runtime.experiments.engine import receive; '
              'p,c=json.load(sys.stdin); print(json.dumps(receive(BitCarrierSource(),p,c)))')
    completed = subprocess.run([sys.executable, '-B', '-c', script], input=json.dumps([public,cfg]),
                               text=True, capture_output=True, cwd=ROOT, check=True)
    r = json.loads(completed.stdout)
    assert r['receiver_consumed'] and r['bits'] == s['bits'] and r['trace'] == s['trace']
    public['termination']['token_budget']['token_limit_reached'] = False
    with pytest.raises((ValueError, RuntimeError)):
        receive(BitCarrierSource(), public, cfg)


@pytest.mark.parametrize('patch', [dict(sentences=1),dict(max_steps=8),dict(utf8_extension=2),
                                   dict(max_new_tokens=0),dict(max_new_tokens=True)])
def test_token_budget_rejects_incompatible_or_invalid_fields(patch):
    value = dict(mode='token_budget', max_new_tokens=256)
    value.update(patch)
    with pytest.raises(ValueError):
        normalize(value)


def test_new_config_uses_source_token_units_and_preserves_old_evidence():
    cfg = resolve(ROOT/'configs/experiments/token_budget_T1_smoke.toml')
    assert len(list(cases(cfg))) == 10
    for variant in cfg['variants']:
        c = contract(variant, cfg['stopping'])
        assert c['bounded_unit'] == 'source_generated_tokens' and not c['sentence_stopping']
    assert cfg['source']['sampling']['temperature'] == 1.
    assert cfg['message']['bits'] == 131072
    old = resolve(ROOT/'configs/experiments/sentence_T1_coder_audit.toml')
    assert old['stopping']['mode'] == 'sentence_boundary'
