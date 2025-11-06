from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional
import torch

from stegtext.components.source.base import EOS_STEGA

from ...core.data import Candidate
from ...core.grouping import group_by_prefix_bytes
from ...core.contracts import Disambiguator, Plan, SupportsRandom
from ...core.prob import sanitize1d


def _weighted_choice(indices: List[int], weights: torch.Tensor, rng: SupportsRandom) -> int:
    """在给定 indices 子集上，按 weights 采样返回一个“全局下标”。"""
    assert indices, "weighted_choice: indices must be non-empty"
    sub = torch.as_tensor([float(weights[i]) for i in indices], dtype=torch.float64)
    tot = float(sub.sum().item())
    if tot <= 0.0:
        r = float(rng.random()) * len(indices)
        return indices[min(int(r), len(indices) - 1)]
    r = float(rng.random()) * tot
    acc = 0.0
    for i, w in zip(indices, sub.tolist()):
        acc += max(0.0, float(w))
        if r <= acc:
            return i
    return indices[-1]


class SyncPool(Disambiguator):

    def __init__(self, **kwargs):
        self.last_selected: Optional[Candidate] = None

    def init(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def plan(self, candidates: List[Candidate]) -> Plan:
        gr = group_by_prefix_bytes(candidates)
        group_sums: List[float] = []
        meta_groups: List[Dict[str, Any]] = []

        for g in gr.groups:
            mem = g.members
            raw_p = torch.tensor([max(0.0, float(m.p)) for m in mem], dtype=torch.float64)
            group_sums.append(float(raw_p.sum().item()))
            meta_groups.append({"raw_p": raw_p})

        gp = sanitize1d(torch.tensor(group_sums, dtype=torch.float64))  # 组间给 coder
        meta: Dict[str, Any] = {"mode": "syncpool", "controls_next": True, "groups": meta_groups}
        return Plan(groups=gr, group_probs=gp, meta=meta)

    def advance(
        self,
        plan: Plan,
        chosen_group_idx: int,
        rng: SupportsRandom,
        source_generate: Callable[[torch.LongTensor, Optional[int]], List[Candidate]],
    ) -> List[Candidate] | Candidate:
        g = plan.groups.groups[chosen_group_idx]
        mem = g.members
        if g.key.endswith(EOS_STEGA):
            info = plan.meta["groups"][chosen_group_idx]
            raw_p = torch.as_tensor(info["raw_p"], dtype=torch.float64)
            weights = sanitize1d(raw_p)
            indices = list(range(len(mem)))
            chosen_idx = _weighted_choice(indices, weights, rng) if indices else 0
            chosen = mem[chosen_idx]
            chosen.p = float(round(float(weights[chosen_idx]), 12)) if len(indices) > 0 else 1.0
            return chosen

        info = plan.meta["groups"][chosen_group_idx]
        raw_p: torch.Tensor = torch.as_tensor(info["raw_p"], dtype=torch.float64)

        # 组内归一化（仅此组）
        intra = sanitize1d(raw_p)
        midx = _weighted_choice(list(range(len(mem))), intra, rng)
        chosen = mem[midx]

        # 展开
        children = list(source_generate(chosen.tokens, chosen.prompt_len) or [])
        self.last_selected = chosen
        return children
