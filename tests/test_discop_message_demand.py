"""Message demand follows the selected copy-tree path, not action count."""
from math import ceil, log2

import pytest
import torch

from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.model.tokenizer_codec import EOS_STEGA
from stegatext_runtime.experiments.engine import send, receive
from tests.finite_source import FiniteSource
from tests.test_all_method_randomness import fixture


class UniformWords(FiniteSource):
    def __init__(self, n):
        super().__init__()
        self.n = n
        width = ceil(log2(n))
        tokens = {i: ("x" + format(i, f"0{width}b")).encode() for i in range(n)}
        tokens[n] = EOS_STEGA
        self.backend = ToyBackend(ToyBackendConfig(tokens, {}, eos_ids=[n]))
        self.backend.codec_index = self.backend.index
        self.backend.codec.text_to_bytes = lambda text: text.encode()
        self.eos_ids = frozenset([n])

    def probabilities(self, ids):
        self.reference.forwards += 1
        values = torch.zeros(self.n + 1, dtype=torch.float64)
        if ids:
            values[-1] = 1.
        else:
            values[:-1] = 1. / self.n
        return values


@pytest.mark.parametrize("n", [257, 131073])
def test_large_action_alphabet_uses_short_real_message_without_padding(tmp_path, n):
    config, case = fixture(tmp_path)
    config["message"]["bits"] = 128
    case["variant"].update(method="basic_bytes_v1", coder="discop_copy")
    packet, sender = send(UniformWords(n), case, config)
    receiver = receive(UniformWords(n), packet, config)
    # All equal weights imply Huffman depths floor(log2 n) or ceil(log2 n).
    # DisCop reads at most one bit per tree edge, regardless of action count.
    assert 0 < sender["bits_count"] <= ceil(log2(n))
    assert sender["bits"] == receiver["bits"]
    assert sender["native_token_sequence"] == receiver["native_token_sequence"]
    assert sender["message_input"]["padding_consumed"] == 0
    assert sender["message_input"]["max_window_required_bits"] == sender["bits_count"]


def test_actual_discop_shortage_is_rejected_without_padding(tmp_path):
    config, case = fixture(tmp_path)
    config["message"]["bits"] = 1
    case["variant"].update(method="basic_bytes_v1", coder="discop_copy")
    with pytest.raises(ValueError, match="Insufficient valid supplied DisCop bits"):
        send(UniformWords(4), case, config)


def test_imec_keeps_its_real_window_requirement(tmp_path):
    config, case = fixture(tmp_path)
    config["message"]["bits"] = 17
    case["variant"].update(method="basic_bytes_v1", coder="imec_prefix_exact",
                           coder_parameters=dict(block_size=6, window_size=3))
    with pytest.raises(ValueError, match="Insufficient unpadded input"):
        send(UniformWords(4), case, config)
