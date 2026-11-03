import unittest
import random

import torch

from stegtext.v2.coders.discop import DisCop
from stegtext.v2.coders.imec import IMEC
from stegtext.v2.coders.range_ac import RangeAC
from stegtext.v2.core.errors import ProbabilityError, ReceiverMismatchError
from stegtext.v2.core.runner import StegoRunner
from stegtext.v2.core.rng import session_rng
from stegtext.v2.methods.consistency import ConsistencyMethod
from stegtext.v2.methods.direct import DirectMethod
from stegtext.v2.methods.las import LASMethod, NaiveLASMethod
from stegtext.v2.model.canonicality import ExactCanonicalityOracle
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.model.vocab_index import TokenByteIndex
from stegtext.v2.model.tokenizer_codec import EOS_STEGA, TokenizerCodec
from stegtext.v2.sampling.policies import FullSoftmax, TopK
from stegtext.v2.experiments.config import ExperimentConfig


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.mapping = {0: b"a", 1: b"b", 2: EOS_STEGA, 3: b"ab", 4: b"ac", 5: b"ad"}
        transitions = {
            (): ((0, 0.25), (1, 0.15), (2, 0.10), (3, 0.25), (4, 0.15), (5, 0.10)),
            (0,): ((2, 1.0),),
            (1,): ((2, 1.0),),
            (3,): ((2, 1.0),),
            (4,): ((2, 1.0),),
            (5,): ((2, 1.0),),
        }
        self.backend = ToyBackend(ToyBackendConfig(self.mapping, transitions, eos_ids=(2,)))

    def test_probability_validation_and_topk_support(self):
        with self.assertRaises(ProbabilityError):
            from stegtext.v2.core.types import TokenDistribution
            TokenDistribution(torch.tensor([float("nan"), 1.0]))
        dist = TopK(2)(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(dist.support_ids.numel(), 2)
        self.assertAlmostEqual(float(dist.probs.sum()), 1.0, places=12)
        masked = FullSoftmax()(torch.tensor([0.0, float("-inf"), -1.0]))
        self.assertEqual(float(masked.probs[1]), 0.0)
        bf16 = FullSoftmax()(torch.tensor([0.0, -1.0, -2.0], dtype=torch.bfloat16))
        self.assertEqual(bf16.probs.dtype, torch.float32)
        self.assertAlmostEqual(float(bf16.probs.sum().item()), 1.0, places=7)

    def test_hf_internal_terminal_symbol_is_prefix_disjoint(self):
        class ByteTokenizer:
            pieces = {0: "ÿ", 1: "A", 2: "<eos>"}

            def get_vocab(self):
                return {piece: token_id for token_id, piece in self.pieces.items()}

            def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
                return [self.pieces[int(token_id)] for token_id in ids]

            def decode(self, ids, **kwargs):
                return "".join(self.pieces[int(token_id)] for token_id in ids)

        codec = TokenizerCodec(ByteTokenizer(), eos_ids=(2,))
        raw_ff = codec.token_bytes(0)
        terminal = codec.token_bytes(2)
        self.assertEqual(raw_ff, b"\x01\xff")
        self.assertEqual(terminal, EOS_STEGA)
        self.assertFalse(raw_ff.startswith(terminal))
        self.assertFalse(terminal.startswith(raw_ff))
        self.assertEqual(codec.bytes_to_text(codec.text_to_bytes("Water")), "Water")

    def test_index_merges_multiple_static_intervals(self):
        index = TokenByteIndex([b"aa", b"ab", b"ba", b"bb"])
        dynamic = index.dynamic_groups([b"a"])
        self.assertEqual(dynamic[0][0], b"a")
        self.assertEqual(len(dynamic[0][1]), 2)

    def _roundtrip(self, method):
        runner = StegoRunner(self.backend, FullSoftmax(), method, DisCop())
        encoded = runner.encode("", "101001", rng=session_rng(b"v2-test"))
        decoded = runner.decode("", encoded.text, rng=session_rng(b"v2-test"))
        self.assertEqual(decoded.text, encoded.text)
        self.assertEqual(decoded.bits, encoded.emitted_bits)
        self.assertTrue(encoded.finished)

    def test_direct_roundtrip(self):
        self._roundtrip(DirectMethod())

    def test_consistency_roundtrip(self):
        self._roundtrip(ConsistencyMethod(ExactCanonicalityOracle(), initial_k=2))

    def test_las_naive_and_optimized_roundtrip(self):
        self._roundtrip(NaiveLASMethod(self.backend.index))
        self._roundtrip(LASMethod(self.backend.index))

    def test_coder_sessions_are_independent(self):
        coder = RangeAC(precision=24)
        a = coder.start("encode")
        b = coder.start("encode")
        bits_a, action_a = a.encode(torch.tensor([0.25, 0.75]), "101010", rng=None)
        bits_b, action_b = b.encode(torch.tensor([0.25, 0.75]), "101010", rng=None)
        self.assertEqual((bits_a, action_a), (bits_b, action_b))

    def test_all_coder_sessions_recover_random_actions(self):
        coders = (DisCop(), RangeAC(precision=24), IMEC(block_size=3, window_size=2))
        for coder in coders:
            for trial in range(12):
                random.seed(f"{type(coder).__name__}-{trial}")
                probs = torch.rand(5, dtype=torch.float64)
                probs /= probs.sum()
                payload = "".join(random.choice("01") for _ in range(32))
                seed = f"{type(coder).__name__}-{trial}".encode("ascii")
                encoded = coder.start("encode").encode(
                    probs, payload, rng=session_rng(seed)
                )
                decoded = coder.start("decode").decode(
                    probs, encoded[1], rng=session_rng(seed)
                )
                self.assertEqual(decoded, encoded[0], type(coder).__name__)

    def test_toml_mapping_preserves_method_and_coder_options(self):
        config = ExperimentConfig.from_mapping(
            {
                "model": {"path": "/tmp/model"},
                "method": {"name": "consistency", "initial_k": 4, "max_k": 32},
                "coder": {"name": "range_ac", "precision": 24},
            }
        )
        self.assertEqual(config.method_options, {"initial_k": 4, "max_k": 32})
        self.assertEqual(config.coder_options, {"precision": 24})


if __name__ == "__main__":
    unittest.main()
