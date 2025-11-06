from __future__ import annotations
from typing import List
import torch

from stegtext.components.source.base import EOS_STEGA
from .data import Candidate, Group, GroupingResult

def group_by_prefix_bytes(cands: List[Candidate]) -> GroupingResult:
    """按整句**字节**前缀分组"""
    non = cands
    # 这里使用简单 O(n^2) 插入法（候选数通常很小）；需要时可优化。
    groups: List[Group] = []
    # 先按 vb 排序，便于构建最短代表
    non_sorted = sorted(non, key=lambda c: c.vb)

    for c in non_sorted:
        placed = False
        if c.vb.endswith(EOS_STEGA):
            # Merge EOS variants that share the same visible bytes into a single group.
            for g in groups:
                if g.key == c.vb:
                    g.add(c)
                    placed = True
                    break
            if not placed:
                groups.append(Group(c))
            continue
        for g in groups:
            if g.key.endswith(EOS_STEGA):
                continue
            key = g.key
            if c.vb.startswith(key):
                g.add(c); placed = True; break
            if key.startswith(c.vb):
                g.add(c); g.key = c.vb; placed = True; break
        if not placed:
            groups.append(Group(c))

    # 组间概率：组内 p 之和
    probs = [sum(max(0.0, float(m.p)) for m in g.members) for g in groups]
    t = torch.tensor(probs, dtype=torch.float64)
    s = float(t.sum().item())
    gp = (t / s) if s > 0 else torch.full((len(groups) or 1,), 1.0/(len(groups) or 1), dtype=torch.float64)
    return GroupingResult(groups=groups, group_probs=gp, stats={})

