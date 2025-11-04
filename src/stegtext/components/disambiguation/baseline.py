from __future__ import annotations

from typing import List, Dict, Any, Callable, Optional
import torch

from stegtext.components.source.base import EOS_STEGA

from ...core.data import Candidate, Group, GroupingResult
from ...core.contracts import Disambiguator, Plan
from ...core.prob import sanitize1d


class Baseline(Disambiguator):
    """
    Baseline disambiguator: no disambiguation, no grouping/merging.
    - plan: every candidate becomes a single-member group; group_probs are the
      candidates' normalized probabilities.
    - advance: expand only the chosen group's representative.

    This maximizes per-step entropy (upper bound on capacity) under the model's
    filtered distribution (top-k/top-p/temperature), and shares the same engine
    structure/contract as other disambiguators.
    """

    def __init__(self, **kwargs):
        self.last_selected: Optional[Candidate] = None

    def init(self) -> None:
        self.last_selected = None

    def reset(self) -> None:
        self.last_selected = None

    def plan(self, candidates: List[Candidate]) -> Plan:
        # One group per candidate; probability = candidate.p
        groups: List[Group] = [Group(c) for c in candidates]
        probs = [max(0.0, float(c.p)) for c in candidates]
        gp = sanitize1d(torch.tensor(probs if probs else [1.0], dtype=torch.float64))
        gr_out = GroupingResult(groups=groups, group_probs=gp, stats={})
        return Plan(groups=gr_out, group_probs=gp, meta={"mode": "baseline-raw"})

    def advance(
        self,
        plan: Plan,
        chosen_group_idx: int,
        rng,  # unused, for signature compatibility
        source_generate: Callable[[torch.LongTensor, Optional[int]], List[Candidate]],
    ) -> List[Candidate] | Candidate:
        g = plan.groups.groups[chosen_group_idx]
        rep = g.members[0]
        if rep.vb.endswith(EOS_STEGA):
            self.last_selected = rep
            return rep
        children = list(source_generate(rep.tokens, rep.prompt_len) or [])
        self.last_selected = rep
        return children

