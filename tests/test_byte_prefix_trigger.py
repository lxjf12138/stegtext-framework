"""The concrete whitespace and partial UTF-8 contracts used by SyncPool."""
import math
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from stegtext.v2.core.types import TokenDistribution
from stegtext.v2.core.errors import TokenizerContractError
from stegtext.v2.methods.syncpool import SyncPoolMethod
from stegtext.v2.model.tokenizer_codec import TokenizerCodec


@pytest.fixture
def codec():
    # Real ByteLevel decoder, including three individual bytes of UTF-8 中.
    vocab = {'Ġ': 0, 'Ġthe': 1, 'Ġhello': 2, 'Ġworld': 3,
             '!': 4, 'ä': 5, '¸': 6, 'Ń': 7, '<eos>': 8}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token='<eos>')
    return TokenizerCodec(fast, [8])


def plan(codec, probabilities):
    p = torch.zeros(codec.vocab_size, dtype=torch.float64)
    for token, mass in probabilities.items():
        p[token] = mass
    return SyncPoolMethod().plan(None, [TokenDistribution(p)], SimpleNamespace(codec=codec))


@pytest.mark.parametrize('space_mass', [0.01, 1e-12, 1e-30])
def test_positive_space_merges_all_space_prefixed_words_even_at_tiny_mass(codec, space_mass):
    result = plan(codec, {0: space_mass, 1: .49, 2: .30, 3: .20})
    assert result.opaque['members'] == [[0, 2, 1, 3]]
    assert result.opaque['keys'] == [b'\1 ']
    assert result.probs.tolist() == [1.]


def test_zero_mass_space_does_not_create_an_ambiguity_group(codec):
    result = plan(codec, {0: 0., 1: .49, 2: .30, 3: .20})
    assert len(result.probs) == 3
    assert all(len(group) == 1 for group in result.opaque['members'])
    h = -sum(p*math.log2(p) for p in result.probs.tolist())
    assert h > 1.


def test_space_group_cannot_merge_eos_or_non_space_punctuation(codec):
    result = plan(codec, {0: .01, 1: .29, 2: .20, 3: .20, 4: .10, 8: .20})
    masses = {frozenset(ids):float(p) for ids,p in zip(result.opaque['members'],result.probs)}
    assert masses[frozenset([0,1,2,3])] == pytest.approx(.7)
    assert masses[frozenset([4])] == pytest.approx(.1)
    assert masses[frozenset([8])] == pytest.approx(.2)


def test_utf8_fragments_are_preserved_instead_of_replacement_decoded(codec):
    assert [codec.token_bytes(i)[1::2] for i in (5,6,7)] == [b'\xe4',b'\xb8',b'\xad']
    assert codec.bytes_to_text(codec.tokens_to_bytes([5,6,7])) == '中'
    for i in (5,6,7):
        with pytest.raises(TokenizerContractError, match='valid UTF-8'):
            codec.bytes_to_text(codec.token_bytes(i))


def test_escape_preserves_prefixes_for_all_256_raw_byte_values():
    escape = TokenizerCodec._escape_bytes
    for byte in range(256):
        raw = bytes([byte])
        assert escape(raw+b' suffix').startswith(escape(raw))
        assert not escape(bytes([(byte+1)%256])+b' suffix').startswith(escape(raw))
