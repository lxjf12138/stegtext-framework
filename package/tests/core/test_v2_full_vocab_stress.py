from __future__ import annotations

import unittest

import torch

from stegtext.v2.core.types import MethodContext
from stegtext.v2.methods.las import LASMethod
from stegtext.v2.model.tokenizer_codec import EOS_STEGA
from stegtext.v2.model.vocab_index import TokenByteIndex
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.sampling.policies import FullSoftmax


class FullVocabularyLASStressTests(unittest.TestCase):
    def test_dense_plan_uses_static_cumsum_and_path_refs(self) -> None:
        vocab = [bytes((65 + (i % 26),)) + (b"x" * (i // 26)) for i in range(512)]
        vocab[-1] = EOS_STEGA
        index = TokenByteIndex(vocab)
        backend = ToyBackend(
            ToyBackendConfig(
                token_bytes={i: value for i, value in enumerate(vocab)},
                transitions={(): tuple((i, 1.0 / len(vocab)) for i in range(len(vocab)))},
                eos_ids=(len(vocab) - 1,),
            )
        )
        method = LASMethod(index)
        prompt = backend.render_prompt("")
        state = method.start(prompt)
        ctx = MethodContext(codec=backend.codec, prompt=prompt, mode="encode")
        plan = method.plan(state, [FullSoftmax()(backend.next_logits(state.base_path))], ctx)

        self.assertTrue(ctx.stats["las_static_vocab"])
        self.assertGreater(ctx.stats["las_group_count"], 1)
        self.assertEqual(len(state.retained), 0)
        for record in plan.opaque["records"]:
            self.assertTrue(all(isinstance(token_id, int) for token_id in record["fresh_ids"]))
            self.assertNotIn("tokens", record)
            self.assertNotIn("history", record)
        self.assertEqual(plan.probs.ndim, 1)
        self.assertAlmostEqual(float(plan.probs.sum().item()), 1.0, places=12)
        self.assertFalse(any(isinstance(value, torch.Tensor) and value.ndim == 2 for value in plan.opaque["records"]))


if __name__ == "__main__":
    unittest.main()
