
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    typical_p: Optional[float] = None
    max_candidates: int = 50  # how many candidates to expose to disambiguator
    span_length: int = 1      # tokens per candidate piece (>=1)
    include_eos: bool = True  # include eos if model provides
