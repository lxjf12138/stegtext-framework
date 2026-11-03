"""Open assistant messages, public replay and scoring share the same context."""
import hashlib
import json
import subprocess
import sys

import pytest

from tests.test_instruct_source import source
from tests.finite_source import BitCarrierSource
from stegatext_runtime.experiments.config import ROOT, resolve
from stegatext_runtime.experiments.engine import send, receive
from stegatext_runtime.experiments.stopping import normalize


def test_official_open_assistant_template_and_reset(source):
    instruction, prefix = 'Write a movie review.', 'The acting was excellent.'
    original = source.manifest()
    source.configure_assistant_prefix(prefix)
    expected = source.tokenizer.apply_chat_template(
        [dict(role='user', content=instruction), dict(role='assistant', content=prefix)],
        tokenize=True, add_generation_prompt=False, continue_final_message=True,
        **source.spec['template_options'])
    prompt = source.backend.render_prompt(instruction)
    assert list(prompt.token_ids) == expected
    rendered = source.backend.prompt_policy.render_text(source.tokenizer, instruction)
    assert rendered.endswith(prefix)
    assert rendered.count(prefix) == 1
    assert source.manifest()['message_roles'] == ['user', 'assistant']
    assert source.manifest()['continue_final_message']
    assert not source.manifest()['add_generation_prompt']
    assert source.manifest()['sha256'] != original['sha256']
    source.configure_assistant_prefix(None)
    assert source.manifest() == original


def test_receiver_retokenization_uses_prefilled_assistant_context(source):
    source.configure_assistant_prefix('The acting was excellent.')
    instruction, text = 'Write a movie review.', ' The pacing was good.'
    p = source.backend.render_prompt(instruction)
    rendered = source.backend.prompt_policy.render_text(source.tokenizer, instruction)
    whole = source.tokenizer.encode(rendered + text, add_special_tokens=False)
    assert whole[:len(p.token_ids)] == list(p.token_ids)
    assert source.retokenize_text(instruction, text) == whole[len(p.token_ids):]


class PrefillBitSource(BitCarrierSource):
    """Small source for the public protocol; real chat templates are tested above."""
    def __init__(self):
        super().__init__()
        self.prefill = None

    def configure_assistant_prefix(self, prefix=None):
        self.prefill = prefix

    def bind(self, prompt):
        from stegtext.v2.core.types import PromptState
        state = super().bind(prompt)
        if self.prefill is None:
            return state
        ids = tuple({'a': 0, 'b': 1}[c] for c in self.prefill)
        path = state.path
        for token in ids:
            path = path.append(token)
        return PromptState(path, state.token_ids + ids, self.prefill.encode(), prompt)

    def manifest(self):
        value = super().manifest()
        if self.prefill is not None:
            value['assistant_prefix_sha256'] = hashlib.sha256(self.prefill.encode()).hexdigest()
            value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return value


def test_prefill_public_packet_roundtrip_in_an_independent_receiver():
    cfg = dict(schema_version=2, stopping=normalize(dict(mode='token_budget', max_new_tokens=8)),
               message=dict(mode='fixture', bits=4096, pattern='random', delivery='prefix_stream'),
               randomness=dict(mode='fixture'))
    case = dict(seed=211, prompt=dict(text='', language='en', assistant_prefix='a'),
                variant=dict(id='las', kind='planner', method='las', coder='imec_prefix_exact',
                             coder_parameters=dict(block_size=6, window_size=3, schedule='head_first')))
    public, sent = send(PrefillBitSource(), case, cfg)
    assert public['assistant_prefix'] == 'a'
    assert sent['native_token_sequence']['prompt_token_ids'] == [0]
    assert sent['native_token_sequence']['source_token_count'] == 8
    script = (
        'import json,sys; from tests.test_assistant_prefill import PrefillBitSource; '
        'from stegatext_runtime.experiments.engine import receive; '
        'p,c=json.load(sys.stdin); print(json.dumps(receive(PrefillBitSource(),p,c)))'
    )
    child = subprocess.run([sys.executable, '-B', '-c', script], input=json.dumps([public, cfg]),
                           text=True, capture_output=True, cwd=ROOT, check=True)
    received = json.loads(child.stdout)
    assert received['receiver_consumed']
    assert received['bits'] == sent['bits']
    assert received['trace'] == sent['trace']
    public['assistant_prefix'] = 'b'
    with pytest.raises(ValueError, match='Source'):
        receive(PrefillBitSource(), public, cfg)


def test_config_records_prefill_in_prompt_identity(tmp_path):
    path = tmp_path / 'prefill.toml'
    base = f'extends = "{ROOT}/configs/experiments/token_budget_T1_smoke.toml"\n[inputs]\nprompts = '
    prompts = '[{id="en",language="en",text="Write a review.",assistant_prefix="Good acting."}, {id="zh",language="zh",text="写影评。",assistant_prefix="演得好。"}]\n'
    path.write_text(base + prompts)
    first = resolve(path)
    path.write_text(base + prompts.replace('Good acting.', 'Bad acting.'))
    second = resolve(path)
    assert first['inputs']['prompts'][0]['sha256'] != second['inputs']['prompts'][0]['sha256']
    assert first['inputs']['prompts'][1]['sha256'] == second['inputs']['prompts'][1]['sha256']
