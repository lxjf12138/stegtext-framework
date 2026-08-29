from __future__ import annotations
from typing import List
import torch

from .data import Candidate, Group, GroupingResult


def group_by_prefix_bytes(cands: List[Candidate]) -> GroupingResult:
    """Partition candidates by the shortest visible-byte prefix.

    Terminal candidates participate in the same prefix relation as ordinary
    candidates. Their byte representation already carries the distinguished
    EOS marker supplied by the source vocabulary. Keeping one uniform prefix
    rule is important: special-casing EOS can create prefix-related group keys
    and make receiver-side group identification ambiguous.
    """
    groups: List[Group] = []

    # Lexicographic order places a byte string before all of its proper
    # extensions. The first unassigned candidate therefore provides the
    # shortest key for its prefix class.
    for c in sorted(cands, key=lambda item: item.vb):
        placed = False
        for g in groups:
            if c.vb.startswith(g.key):
                g.add(c)
                placed = True
                break
        if not placed:
            groups.append(Group(c))

    # Group keys are prefix-free by construction: if a later key extended an
    # earlier one, that candidate would have been inserted into the earlier
    # group instead of starting a new group.
    probs = [sum(max(0.0, float(m.p)) for m in g.members) for g in groups]
    t = torch.tensor(probs, dtype=torch.float64)
    s = float(t.sum().item())
    gp = (
        t / s
        if s > 0
        else torch.full(
            (len(groups) or 1,),
            1.0 / (len(groups) or 1),
            dtype=torch.float64,
        )
    )
    return GroupingResult(groups=groups, group_probs=gp, stats={})
