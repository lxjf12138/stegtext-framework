from __future__ import annotations

import unittest

import torch

from stegtext.v2.coders.discop import DisCop
from stegtext.v2.coders.range_ac import RangeAC
from stegtext.v2.core.rng import session_rng
from stegtext.v2.core.runner import StegoRunner
from stegtext.v2.core.types import MethodContext
from stegtext.v2.core.errors import ReceiverMismatchError
from stegtext.v2.methods.consistency import ConsistencyMethod
from stegtext.v2.methods.direct import DirectMethod
from stegtext.v2.methods.las import LASMethod, NaiveLASMethod
from stegtext.v2.methods.mwis import MWISMethod
from stegtext.v2.methods.syncpool import SyncPoolMethod
from stegtext.v2.model.canonicality import ExactCanonicalityOracle
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.model.tokenizer_codec import EOS_STEGA
from stegtext.v2.sampling.policies import FullSoftmax
from stegtext.v2.eval.eos_matrix import assess_sentence, eos_diagnosis


class EndToEndSentenceTests(unittest.TestCase):
    """Exercise the actual multi-step sender/receiver protocol, including EOS."""

    def setUp(self) -> None:
        # The first step carries payload choices.  Every branch then produces
        # the same complete sentence continuation and terminates explicitly.
        self.backend = ToyBackend(
            ToyBackendConfig(
                token_bytes={
                    0: b"Hello",
                    1: b"Hi",
                    2: b" world",
                    3: b"!",
                    4: EOS_STEGA,
                },
                transitions={
                    (): ((0, 0.55), (1, 0.45)),
                    (0,): ((2, 1.0),),
                    (1,): ((2, 1.0),),
                    (0, 2): ((3, 1.0),),
                    (1, 2): ((3, 1.0),),
                    (0, 2, 3): ((4, 1.0),),
                    (1, 2, 3): ((4, 1.0),),
                },
                eos_ids=(4,),
            )
        )

    def _roundtrip(self, method, coder) -> None:
        seed = b"v2-complete-sentence"
        runner = StegoRunner(self.backend, FullSoftmax(), method, coder)
        encoded = runner.encode("", "101100111", rng=session_rng(seed), max_steps=16)
        self.assertTrue(encoded.finished)
        self.assertEqual(encoded.stop_reason, "eos")
        self.assertEqual(encoded.text, "Hello world!" if encoded.text.startswith("Hello") else "Hi world!")
        self.assertEqual(encoded.steps, 4)
        for key in ("model_queries", "model_queries_per_token", "model_forward_ms", "grouping_ms", "coder_ms", "total_ms", "peak_cpu_memory_bytes", "peak_gpu_memory_bytes", "ber", "exact_message", "bpt", "bps"):
            self.assertIn(key, encoded.stats)

        decoded = runner.decode("", encoded.text, rng=session_rng(seed), max_steps=16)
        self.assertTrue(decoded.finished)
        self.assertEqual(decoded.stop_reason, "eos")
        self.assertEqual(decoded.text, encoded.text)
        self.assertEqual(decoded.bits, encoded.emitted_bits)
        self.assertEqual(decoded.stats["target_remaining"], 0)
        with self.assertRaises(ReceiverMismatchError):
            runner.decode("", encoded.text + " trailing", rng=session_rng(seed), max_steps=16)

    def test_direct_sentence_with_discop(self) -> None:
        self._roundtrip(DirectMethod(), DisCop())

    def test_consistency_sentence_with_discop(self) -> None:
        self._roundtrip(ConsistencyMethod(ExactCanonicalityOracle(), initial_k=2), DisCop())

    def test_naive_las_sentence_with_discop(self) -> None:
        self._roundtrip(NaiveLASMethod(self.backend.index), DisCop())

    def test_optimized_las_sentence_with_range_ac(self) -> None:
        self._roundtrip(LASMethod(self.backend.index), RangeAC(precision=24))

    def test_eos_trace_records_terminal_selection(self) -> None:
        runner = StegoRunner(self.backend, FullSoftmax(), DirectMethod(), DisCop())
        encoded = runner.encode(
            "",
            "1",
            rng=session_rng(b"trace-eos"),
            max_tokens=16,
            collect_trace=True,
        )
        trace = encoded.stats["trace"]
        self.assertEqual(len(trace), encoded.steps)
        self.assertTrue(trace[-1]["terminal"])
        self.assertEqual(trace[-1]["active_token_id"], 4)
        self.assertEqual(trace[-1]["eos_rank"], 1)
        self.assertAlmostEqual(trace[-1]["eos_probability"], 1.0)
        self.assertEqual(encoded.stats["terminal_token_id"], 4)
        self.assertEqual(encoded.stats["latent_text"], encoded.text)
        diagnosis = eos_diagnosis(encoded.__dict__)
        self.assertEqual(diagnosis["summary"], "natural EOS selected")

    def test_sentence_assessment_rejects_truncation_and_repetition(self) -> None:
        self.assertTrue(
            assess_sentence("Water moves quietly through rivers toward the waiting sea.")["passed"]
        )
        incomplete = assess_sentence("Water moves quietly")
        self.assertFalse(incomplete["passed"])
        self.assertIn("missing terminal punctuation", incomplete["issues"])
        repeated = assess_sentence("Water water water water water water water water.")
        self.assertFalse(repeated["passed"])
        self.assertIn("repeated trigram", repeated["issues"])
        fragmented = assess_sentence("W flow down due t grav.")
        self.assertFalse(fragmented["passed"])
        self.assertEqual(fragmented["suspicious_fragments"], ["W", "t"])
        divergence = assess_sentence(
            "The crystal-c water supports marine lif species.",
            latent_text="The crystal-clear water supports marine life species.",
        )
        self.assertFalse(divergence["passed"])
        self.assertLess(divergence["latent_similarity"], 1.0)

    def test_sentence_assessment_separates_instruction_length(self) -> None:
        short = assess_sentence("Water flows downhill due to gravity.")
        self.assertTrue(short["passed"])
        self.assertFalse(short["instruction_passed"])
        self.assertEqual(short["issues"], [])

    def test_syncpool_emits_sampled_member_and_roundtrips(self) -> None:
        backend = ToyBackend(
            ToyBackendConfig(
                token_bytes={0: b"a", 1: b"ab", 2: EOS_STEGA},
                transitions={
                    (): ((0, 0.01), (1, 0.99)),
                    (0,): ((2, 1.0),),
                    (1,): ((2, 1.0),),
                },
                eos_ids=(2,),
            )
        )
        runner = StegoRunner(backend, FullSoftmax(), SyncPoolMethod(), DisCop())
        seed = b"syncpool-visible-group"
        encoded = runner.encode("", "1", rng=session_rng(seed), max_tokens=8)
        self.assertEqual(encoded.text, "ab")
        self.assertEqual(encoded.text, encoded.stats["latent_text"])
        self.assertEqual(encoded.stop_reason, "eos")
        decoded = runner.decode("", encoded.text, rng=session_rng(seed), max_tokens=8)
        self.assertEqual(decoded.text, encoded.text)
        self.assertEqual(decoded.bits, encoded.emitted_bits)
        self.assertEqual(decoded.stats["target_remaining"], 0)

    def test_duplicate_eos_ids_are_one_receiver_action(self) -> None:
        backend = ToyBackend(
            ToyBackendConfig(
                token_bytes={
                    0: b"Complete sentence.",
                    1: b"Alternative sentence.",
                    2: EOS_STEGA,
                    3: EOS_STEGA,
                },
                transitions={
                    (): ((0, 0.55), (1, 0.45)),
                    (0,): ((2, 0.2), (3, 0.8)),
                    (1,): ((2, 0.2), (3, 0.8)),
                },
                eos_ids=(2, 3),
            )
        )
        for method in (
            DirectMethod(),
            ConsistencyMethod(ExactCanonicalityOracle(), initial_k=4),
            MWISMethod(),
        ):
            with self.subTest(method=type(method).__name__):
                runner = StegoRunner(backend, FullSoftmax(), method, DisCop())
                seed = b"duplicate-eos-action"
                encoded = runner.encode("", "101", rng=session_rng(seed), max_tokens=8)
                decoded = runner.decode("", encoded.text, rng=session_rng(seed), max_tokens=8)
                self.assertEqual(encoded.stop_reason, "eos")
                self.assertEqual(decoded.stop_reason, "eos")
                self.assertEqual(decoded.bits, encoded.emitted_bits)
                self.assertEqual(decoded.stats["target_remaining"], 0)
                expected_duplicates = 0 if isinstance(method, MWISMethod) else 1
                self.assertEqual(encoded.stats["exact_duplicate_tokens"], expected_duplicates)

    def test_naive_and_optimized_las_match_step_by_step(self) -> None:
        naive = NaiveLASMethod(self.backend.index)
        optimized = LASMethod(self.backend.index)
        prompt_n = naive.start(self.backend.render_prompt(""))
        prompt_o = optimized.start(self.backend.render_prompt(""))
        ctx_n = type("Ctx", (), {"codec": self.backend.codec, "stats": {}})()
        ctx_o = type("Ctx", (), {"codec": self.backend.codec, "stats": {}})()

        for step in range(8):
            if naive.done(prompt_n) or optimized.done(prompt_o):
                self.assertEqual(naive.done(prompt_n), optimized.done(prompt_o))
                break
            dist_n = [FullSoftmax()(self.backend.next_logits(naive.queries(prompt_n)[0]))]
            dist_o = [FullSoftmax()(self.backend.next_logits(optimized.queries(prompt_o)[0]))]
            plan_n = naive.plan(prompt_n, dist_n, ctx_n)
            plan_o = optimized.plan(prompt_o, dist_o, ctx_o)
            torch.testing.assert_close(plan_n.probs, plan_o.probs, rtol=0.0, atol=0.0)
            self.assertEqual(
                [record["key"] for record in plan_n.opaque["records"]],
                [record["key"] for record in plan_o.opaque["records"]],
            )
            action = 0
            prompt_n = naive.advance(prompt_n, plan_n, action, session_rng(b"parity" + bytes([step])), ctx_n)
            prompt_o = optimized.advance(prompt_o, plan_o, action, session_rng(b"parity" + bytes([step])), ctx_o)
            self.assertEqual(prompt_n.base_path.token_ids(), prompt_o.base_path.token_ids())
            self.assertEqual(prompt_n.resolved_bytes, prompt_o.resolved_bytes)
            self.assertEqual(prompt_n.terminal, prompt_o.terminal)
            self.assertAlmostEqual(prompt_n.expansion_mass, prompt_o.expansion_mass, places=15)
            self.assertEqual(len(prompt_n.retained), len(prompt_o.retained))
            for left, right in zip(prompt_n.retained, prompt_o.retained):
                self.assertEqual(left.path.token_ids(), right.path.token_ids())
                self.assertEqual(left.residual, right.residual)
                self.assertAlmostEqual(left.mass, right.mass, places=15)
        else:
            self.fail("LAS parity loop did not reach EOS")

    def test_las_parity_exhausts_all_small_vocab_actions(self) -> None:
        backend = ToyBackend(
            ToyBackendConfig(
                token_bytes={
                    0: b"a",
                    1: b"b",
                    2: b"ab",
                    3: b"ac",
                    4: EOS_STEGA,
                },
                transitions={
                    (): ((0, 0.25), (1, 0.25), (2, 0.20), (3, 0.20), (4, 0.10)),
                    (0,): ((4, 1.0),),
                    (1,): ((4, 1.0),),
                    (2,): ((4, 1.0),),
                    (3,): ((4, 1.0),),
                },
                eos_ids=(4,),
            )
        )
        naive = NaiveLASMethod(backend.index)
        optimized = LASMethod(backend.index)
        prompt = backend.render_prompt("")

        def compare(state_n, state_o, depth: int) -> None:
            self.assertEqual(naive.done(state_n), optimized.done(state_o))
            if depth == 0 or naive.done(state_n):
                return
            ctx_n = MethodContext(codec=backend.codec, prompt=prompt, mode="encode")
            ctx_o = MethodContext(codec=backend.codec, prompt=prompt, mode="encode")
            dist = FullSoftmax()(backend.next_logits(state_n.base_path))
            plan_n = naive.plan(state_n, [dist], ctx_n)
            plan_o = optimized.plan(state_o, [dist], ctx_o)
            torch.testing.assert_close(plan_n.probs, plan_o.probs, rtol=0.0, atol=0.0)
            for left, right in zip(plan_n.opaque["records"], plan_o.opaque["records"]):
                self.assertEqual(left["key"], right["key"])
            for action in range(len(plan_n.opaque["records"])):
                rng_n = session_rng(b"exhaustive" + bytes([depth, action]))
                rng_o = session_rng(b"exhaustive" + bytes([depth, action]))
                next_n = naive.advance(state_n, plan_n, action, rng_n, ctx_n)
                next_o = optimized.advance(state_o, plan_o, action, rng_o, ctx_o)
                self.assertEqual(next_n.resolved_bytes, next_o.resolved_bytes)
                self.assertAlmostEqual(next_n.expansion_mass, next_o.expansion_mass, places=15)
                self.assertEqual(len(next_n.retained), len(next_o.retained))
                for retained_n, retained_o in zip(next_n.retained, next_o.retained):
                    self.assertEqual(retained_n.residual, retained_o.residual)
                    self.assertAlmostEqual(retained_n.mass, retained_o.mass, places=15)
                compare(next_n, next_o, depth - 1)

        compare(naive.start(prompt), optimized.start(prompt), 4)


if __name__ == "__main__":
    unittest.main()
