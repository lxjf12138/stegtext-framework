from __future__ import annotations

import unittest

import torch

from stegtext.components.coding.discop import DisCop
from stegtext.components.disambiguation.lookahead import LookAhead
from stegtext.components.source.base import EOS_STEGA
from stegtext.components.source.toy import ToySource, ToySourceConfig
from stegtext.components.source.vocab import TokenByteVocab
from stegtext.core.data import Candidate
from stegtext.core.engine import StegoEngine
from stegtext.core.grouping import group_by_prefix_bytes
from stegtext.core.rng import CSPRNG


class _FixedRNG:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _cand(vb: bytes, p: float, *, eos: bool, tid: int) -> Candidate:
    return Candidate(
        p=p,
        tokens=torch.tensor([tid], dtype=torch.long),
        vb=vb,
        is_eos=eos,
        prompt_len=0,
        prompt_byte_len=0,
    )


class TerminalGroupingTests(unittest.TestCase):
    def test_group_keys_are_prefix_free_with_terminal_candidates(self) -> None:
        cands = [
            _cand(b"a", 0.2, eos=False, tid=1),
            _cand(b"ab" + EOS_STEGA, 0.3, eos=True, tid=2),
            _cand(b"ac", 0.5, eos=False, tid=3),
        ]
        groups = group_by_prefix_bytes(cands).groups
        keys = [g.key for g in groups]
        for i, left in enumerate(keys):
            for j, right in enumerate(keys):
                if i == j:
                    continue
                self.assertFalse(right.startswith(left), (left, right))
        # The shortest nonterminal prefix owns both strict extensions.
        self.assertEqual(keys, [b"a"])

    def test_terminal_and_nonterminal_keys_do_not_double_match(self) -> None:
        terminal = _cand(b"a" + EOS_STEGA, 0.4, eos=True, tid=1)
        nonterminal = _cand(b"ab", 0.6, eos=False, tid=2)
        groups = group_by_prefix_bytes([terminal, nonterminal]).groups
        keys = [g.key for g in groups]
        target = terminal.vb
        matches = [key for key in keys if target.startswith(key)]
        self.assertEqual(matches, [terminal.vb])

    def test_terminal_extension_is_retained_not_selected(self) -> None:
        prefix = _cand(b"a", 0.4, eos=False, tid=1)
        terminal_extension = _cand(
            b"ab" + EOS_STEGA, 0.6, eos=True, tid=2
        )
        las = LookAhead()
        plan = las.plan([prefix, terminal_extension])

        child = _cand(b"ac", 1.0, eos=False, tid=3)

        def generate(_tokens, _prompt_len):
            return [child]

        out = las.advance(plan, 0, _FixedRNG(0.0), generate)
        self.assertIsInstance(out, list)
        self.assertIs(las.last_selected, prefix)
        self.assertIn(terminal_extension, out)
        self.assertAlmostEqual(sum(c.p for c in out), 1.0)
        self.assertAlmostEqual(terminal_extension.p, 0.6)
        self.assertAlmostEqual(child.p, 0.4)

    def test_terminal_only_group_resolves_to_singleton(self) -> None:
        left = _cand(b"a" + EOS_STEGA, 0.25, eos=True, tid=1)
        right = _cand(b"a" + EOS_STEGA, 0.75, eos=True, tid=2)
        las = LookAhead()
        plan = las.plan([left, right])

        def no_generate(_tokens, _prompt_len):
            self.fail("terminal-only group must not call source_generate")

        chosen = las.advance(plan, 0, _FixedRNG(0.9), no_generate)
        self.assertIsInstance(chosen, Candidate)
        self.assertTrue(chosen.is_eos)
        self.assertEqual(chosen.p, 1.0)

    def test_reference_mode_rejects_multi_representative_sampling(self) -> None:
        with self.assertRaises(ValueError):
            LookAhead(m_reps=2)

    def test_toy_sender_receiver_recover_exact_emitted_bits(self) -> None:
        mapping = {i: bytes([i]) for i in range(256)}
        mapping[256] = EOS_STEGA
        mapping[257] = b"ab"
        mapping[258] = b"abc"
        vocab = TokenByteVocab(mapping=mapping, vocab_size=259)
        transitions = {
            (): (
                (97, 0.25),
                (98, 0.20),
                (257, 0.20),
                (258, 0.15),
                (32, 0.10),
                (256, 0.10),
            )
        }

        def source() -> ToySource:
            return ToySource(
                ToySourceConfig(
                    vocab=vocab,
                    transitions=transitions,
                    end_token_ids=(256,),
                )
            )

        for trial in range(32):
            key = f"terminal-test-{trial}".encode()
            engine = StegoEngine(source(), LookAhead(), DisCop())
            payload = ("0100111010011011" * 20)[trial % 5 :]
            enc = engine.encode(
                "", payload, rng=CSPRNG(key=key), max_steps=20
            )
            engine.init()
            dec = engine.decode(
                "", enc.text, rng=CSPRNG(key=key), max_steps=20
            )
            self.assertEqual(dec.bits, enc.emitted_bits)


if __name__ == "__main__":
    unittest.main()
