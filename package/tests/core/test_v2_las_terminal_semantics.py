from __future__ import annotations

import unittest

from stegtext.v2.core.types import MethodContext
from stegtext.v2.core.rng import session_rng
from stegtext.v2.methods.las import LASMethod
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.model.tokenizer_codec import EOS_STEGA
from stegtext.v2.model.vocab_index import TokenByteIndex
from stegtext.v2.sampling.policies import FullSoftmax


class LASV2TerminalTests(unittest.TestCase):
    def test_static_groups_are_prefix_free_with_terminal_extensions(self) -> None:
        index = TokenByteIndex([b"a", b"ab" + EOS_STEGA, b"ac"])
        keys = [group.key for group in index.base_groups]
        self.assertEqual(keys, [b"a"])
        for left in keys:
            for right in keys:
                if left != right:
                    self.assertFalse(right.startswith(left))

    def test_terminal_extension_is_retained_until_exact_prefix_resolves(self) -> None:
        backend = ToyBackend(
            ToyBackendConfig(
                token_bytes={1: b"a", 2: b"ab" + EOS_STEGA, 3: b"ac"},
                transitions={(): ((1, 0.4), (2, 0.6)), (1,): ((2, 1.0),)},
                eos_ids=(2,),
            )
        )
        method = LASMethod(backend.index)
        state = method.start(backend.render_prompt(""))
        ctx = MethodContext(codec=backend.codec, prompt=backend.render_prompt(""), mode="encode")
        plan = method.plan(state, [FullSoftmax()(backend.next_logits(state.base_path))], ctx)
        self.assertEqual(plan.opaque["records"][0]["key"], b"a")
        next_state = method.advance(state, plan, 0, session_rng(b"terminal"), ctx)
        self.assertFalse(next_state.terminal)
        self.assertEqual(len(next_state.retained), 1)
        self.assertEqual(next_state.retained[0].residual, b"b" + EOS_STEGA)
        self.assertAlmostEqual(next_state.retained[0].mass, 0.6 / 1.0)

    def test_duplicate_terminal_realizations_are_one_action(self) -> None:
        backend = ToyBackend(
            ToyBackendConfig(
                token_bytes={1: b"a" + EOS_STEGA, 2: b"a" + EOS_STEGA},
                transitions={(): ((1, 0.25), (2, 0.75))},
                eos_ids=(1, 2),
            )
        )
        method = LASMethod(backend.index)
        state = method.start(backend.render_prompt(""))
        prompt = backend.render_prompt("")
        ctx = MethodContext(codec=backend.codec, prompt=prompt, mode="encode")
        plan = method.plan(state, [FullSoftmax()(backend.next_logits(state.base_path))], ctx)
        self.assertEqual(len(plan.opaque["records"]), 1)
        self.assertEqual(plan.opaque["records"][0]["key"], b"a" + EOS_STEGA)
        state = method.advance(state, plan, 0, session_rng(b"terminal-duplicate"), ctx)
        self.assertTrue(state.terminal)
        self.assertEqual(method.output(state, ctx), b"a")


if __name__ == "__main__":
    unittest.main()
