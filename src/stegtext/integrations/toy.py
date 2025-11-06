"""Factory helpers for building the default toy source."""

from __future__ import annotations

from typing import Dict, Tuple

from ..components.source.base import EOS_STEGA
from ..components.source.toy import ToySource, ToySourceConfig
from ..components.source.vocab import TokenByteVocab

__all__ = ["make_toy_source"]


_DEFAULT_TRANSITIONS = {
    (): (
        (32, 0.40),
        (44, 0.06),
        (46, 0.08),
        (63, 0.06),
        (97, 0.20),
        (98, 0.15),
        (256, 0.05),
    )
}

# An alternative transition set that includes tokens with prefix ambiguity
# relative to the single-byte token for 'a' (97). We introduce token ids 257
# and 258 that map to b"ab" and b"abc" respectively so that visible bytes have
# proper prefix relationships across candidates.
_AMBIGUOUS_TRANSITIONS = {
    (): (
        (32, 0.34),   # space
        (44, 0.05),   # ','
        (46, 0.07),   # '.'
        (63, 0.05),   # '?'
        (97, 0.18),   # 'a'
        (98, 0.13),   # 'b'
        (257, 0.07),  # 'ab'
        (258, 0.06),  # 'abc'
        (256, 0.05),  # EOS
    )
}

# A wider ambiguity set that creates multiple overlapping prefix families.
_WIDE_AMBIGUOUS_SPECS = (
    # Single-byte baselines
    (32, 0.20),   # ' '
    (44, 0.03),   # ','
    (46, 0.04),   # '.'
    (63, 0.03),   # '?'
    (97, 0.10),   # 'a'
    (98, 0.07),   # 'b'
    (99, 0.05),   # 'c'
    (256, 0.05),  # EOS
    # 'a' family
    (257, 0.07),  # 'ab'
    (258, 0.05),  # 'abc'
    (259, 0.03),  # 'abcd'
    (260, 0.05),  # 'a '
    (261, 0.03),  # 'a,'
    (262, 0.03),  # 'a.'
    (263, 0.04),  # 'aa'
    (264, 0.02),  # 'aaa'
    # 'b' family
    (266, 0.04),  # 'ba'
    (267, 0.02),  # 'bab'
    (268, 0.01),  # 'baba'
    # whitespace / punctuation runs
    (269, 0.03),  # '  '
    (270, 0.02),  # '   '
    (271, 0.02),  # '. '
    (272, 0.02),  # '..'
    (273, 0.01),  # '...'
    (274, 0.02),  # ', '
)


def make_toy_source(
    *,
    transitions: Dict[Tuple[int, ...], Tuple[Tuple[int, float], ...]] | None = None,
    vocab_size: int = 300,
    prefix_ambiguity: bool | str = True,
) -> ToySource:
    """Return a ToySource.

    By default we include a small amount of prefix ambiguity in the visible
    bytes to exercise grouping/disambiguation code paths:
      - token 97 -> b"a"
      - token 257 -> b"ab"
      - token 258 -> b"abc"

    Set ``prefix_ambiguity=False`` to keep a strict 1:1 byte mapping only.
    """
    mapping = {i: bytes([i]) for i in range(256)}
    mapping[256] = EOS_STEGA

    mode_basic = (prefix_ambiguity is True) or (prefix_ambiguity == "basic")
    mode_wide = (prefix_ambiguity == "wide")

    if mode_basic or mode_wide:
        # Add multi-byte tokens that create prefix relationships with 'a'
        mapping[257] = b"ab"
        mapping[258] = b"abc"
        mapping[259] = b"abcd"
        mapping[260] = b"a "
        mapping[261] = b"a,"
        mapping[262] = b"a."
        mapping[263] = b"aa"
        mapping[264] = b"aaa"
        # 'b' family
        mapping[266] = b"ba"
        mapping[267] = b"bab"
        mapping[268] = b"baba"
        # whitespace / punctuation runs
        mapping[269] = b"  "
        mapping[270] = b"   "
        mapping[271] = b". "
        mapping[272] = b".."
        mapping[273] = b"..."
        mapping[274] = b", "

    # Choose transitions: explicit beats defaults; otherwise pick based on flag
    if transitions is None:
        if mode_wide:
            # Normalize wide specs to sum to 1 for neatness
            total = sum(p for _, p in _WIDE_AMBIGUOUS_SPECS)
            specs = tuple((tid, p / total) for tid, p in _WIDE_AMBIGUOUS_SPECS) if total > 0 else _WIDE_AMBIGUOUS_SPECS
            transitions = {(): specs}
        elif mode_basic:
            transitions = _AMBIGUOUS_TRANSITIONS
        else:
            transitions = _DEFAULT_TRANSITIONS

    # Ensure vocab size covers all explicit ids
    max_tid = max(mapping.keys()) if mapping else -1
    vocab_size = max(vocab_size, max_tid + 1)
    vocab = TokenByteVocab(mapping=mapping, vocab_size=vocab_size)

    cfg = ToySourceConfig(
        vocab=vocab,
        transitions=transitions,
        end_token_ids=(256,),
        prompt_to_tokens=None,
    )
    return ToySource(cfg)
