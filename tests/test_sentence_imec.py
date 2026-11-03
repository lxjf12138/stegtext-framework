import json
import subprocess
import sys

import pytest

from tests.finite_source import FiniteSource, BitCarrierSource
from stegatext_runtime.experiments.engine import planner, send, receive
from stegatext_runtime.experiments.engine_v2 import public_termination
from stegatext_runtime.experiments.sentence import observation
from stegatext_runtime.experiments.stopping import normalize
from stegatext_runtime.experiments.delivery import measure
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.model.tokenizer_codec import EOS_STEGA


class SentenceSource(FiniteSource):
    def __init__(self, split_utf8=False):
        super().__init__()
        values = {0:b'One sentence. More',1:b'Other sentence. More',2:b'x',3:EOS_STEGA}
        if split_utf8:
            values = {0:b'First. \xe4',1:b'First. \xe4',2:b'\xb8\xad',3:EOS_STEGA}
            self.transitions = {():[(0,1)],(0,):[(2,1)],(0,2):[(3,1)]}
        self.backend = ToyBackend(ToyBackendConfig(values,{},eos_ids=[3]))
        self.backend.codec_index = self.backend.index
        self.backend.codec.text_to_bytes = lambda text:text.encode()


def packet(name='syncpool',coder='imec_window_exact'):
    variant = dict(id=name,kind='planner',method=name,coder=coder)
    if coder == 'imec_window_exact':
        variant['coder_parameters'] = dict(block_size=2,window_size=2)
    return dict(prompt='',language='en',message_delivery='prefix_stream',variant=variant,
                stopping=normalize(dict(mode='sentence_boundary',watchdog_protocol_decisions=16,watchdog_generated_tokens=16)))


@pytest.mark.parametrize('name',['syncpool','las'])
def test_sentence_overshoot_retains_decodable_last_action(name):
    source,p = SentenceSource(),packet(name)
    s = planner(source,p,key=b'sentence',stream='01'*256)
    assert s['terminal_reason'] == 'sentence_goal' and not s['censored']
    assert s['sentence_goal_reached'] and s['overshoot_utf8_bytes'] == 5
    public = dict(p,text=s['text'],natural_eos=False,termination=public_termination(s))
    r = planner(SentenceSource(),public,key=b'sentence')
    assert r['receiver_consumed'] and r['trace'] == s['trace'] and r['bits'] == s['bits']
    public['text'] = s['text'][:s['first_sentence_utf8_bytes']]
    with pytest.raises(RuntimeError):
        planner(SentenceSource(),public,key=b'sentence')


def test_ordinary_has_same_predicate_and_preserves_atomic_overshoot():
    stop = packet()['stopping']
    s = SentenceSource().generate('',seed=11,stopping=stop,language='en')
    assert s['terminal_reason'] == 'sentence_goal' and s['overshoot_utf8_bytes'] == 5


def test_utf8_completion_keeps_extra_bytes_without_repair():
    source,p = SentenceSource(True),packet()
    s = planner(source,p,key=b'sentence',stream='0'*512)
    assert s['steps'] == 2 and s['text'] == 'First. 中'
    assert s['overshoot_utf8_bytes'] == 4 and s['framing_status'] == 'valid_utf8'
    p['stopping']['watchdog_protocol_decisions'] = 1
    s = planner(SentenceSource(True),p,key=b'sentence',stream='0'*512)
    assert s['text'] is None and s['censored'] and s['visible_bytes_hex'].endswith('e4')


def test_eos_before_sentence_and_watchdog_are_distinct():
    p = packet()
    s = planner(FiniteSource(),p,key=b'sentence',stream='01'*256)
    assert s['terminal_reason'] == 'natural_eos' and not s['sentence_goal_reached'] and not s['censored']
    p['stopping']['watchdog_protocol_decisions'] = 2
    s = planner(FiniteSource(endless=True),p,key=b'sentence',stream='01'*256)
    assert s['censored'] and not s['sentence_goal_reached']


def test_imec_receiver_process_needs_only_public_packet():
    p = packet()
    cfg = dict(schema_version=2,stopping=normalize(dict(mode='natural_eos_only',watchdog_protocol_decisions=140)),
               message=dict(mode='fixture',bits=4096,pattern='random',delivery='prefix_stream'),randomness=dict(mode='fixture'))
    case = dict(prompt=dict(text='',language='en'),seed=19,variant=p['variant'])
    public,s = send(BitCarrierSource(),case,cfg)
    assert not ({'bits','trace','coder_state','token_ids'} & public.keys())
    script = 'import json,sys; from tests.finite_source import BitCarrierSource; from stegatext_runtime.experiments.engine import receive; p,c=json.load(sys.stdin); print(json.dumps(receive(BitCarrierSource(),p,c)))'
    r = subprocess.run([sys.executable,'-B','-c',script],input=json.dumps([public,cfg]),text=True,capture_output=True,check=True)
    result = json.loads(r.stdout)
    assert result['bits'] == s['bits'] and result['trace'] == s['trace'] and result['receiver_consumed']
    assert len(s['bits']) > 100 and result['coder_state'] == s['coder_state']


def test_sentence_abbreviation_and_chinese_byte_boundary():
    assert not observation('Dr. Smith','en')['sentence_goal_reached']
    assert observation('你好。下一句','zh')['first_sentence_utf8_bytes'] == 9


def test_strict_reliable_payload_does_not_credit_accidental_common_prefix():
    row = measure('001101','0011','0010',replay_exact=False)
    assert row['confirmed_prefix_bits'] == 3
    assert not row['strict_prefix_verified'] and row['strict_verified_payload_bits'] == 0
