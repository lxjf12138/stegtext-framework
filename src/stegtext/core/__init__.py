from __future__ import annotations

# 核心数据与契约
from .data import (
    Candidate,
    Group,
    GroupingResult,
    SentenceStat,
    EncodeResult,
    DecodeResult,
)
from .contracts import (
    Plan,
    Disambiguator,
    Source,
    BaseCodingStrategy,
    SupportsRandom,
    AfterPickResult,  # 为兼容旧版 none.py
)

# 核心工具
from .prob import sanitize1d
from .rng import CSPRNG, session_rng, derive_rng
from .framing import pack_bits, parse_framed_bits, FrameParse, parse_framed_progress
from .payload import random_bitstring, text_to_bits, bits_to_text

# 引擎（避免在此处直接导入以打破 components.source.base ↔ core 的循环依赖）
# 如需使用，请从 `stegtext.core.engine` 显式导入 StegoEngine。
