# from __future__ import annotations
# from typing import List, Dict, Any, Callable
# import torch

# from ...core.data import Candidate, Group, GroupingResult
# from ...core.contracts import Disambiguator, Plan
# from ...core.prob import sanitize1d


# class NoDisamb(Disambiguator):
#     """
#     基线：每个候选各自成组（整句语义），可见输出=成员的整句文本 v；
#     advance：仅展开被选成员的 tokens，归一化 children；不合并路径。
#     """

#     def __init__(self, **kwargs):
#         pass

#     def init(self) -> None:
#         return

#     def reset(self) -> None:
#         return

#     def plan(self, candidates: List[Candidate]) -> Plan:
#         groups: List[Group] = []
#         probs: List[float] = []
#         for c in candidates:
#             key = c.v if not c.is_eos else "<eos>"
#             groups.append(Group(key=key, members=[c]))
#             probs.append(max(0.0, float(c.p)))

#         gp = sanitize1d(torch.tensor(probs, dtype=torch.float64))
#         gr = GroupingResult(groups=groups, group_probs=gp, stats={})
#         return Plan(groups=gr, group_probs=gp, meta={"visible_by": "member_v", "controls_next": True, "mode": "none"})

#     def advance(
#         self,
#         plan: Plan,
#         chosen_group_idx: int,
#         rng,  # 未使用
#         source_generate: Callable[[torch.LongTensor], List[Candidate]],
#     ) -> List[Candidate]:
#         g = plan.groups.groups[chosen_group_idx]
#         chosen = g.members[0]
#         if chosen.is_eos or g.key == "<eos>":
#             return []

#         children = list(source_generate(chosen.tokens) or [])
#         if not children:
#             return []

#         total = sum(max(0.0, float(c.p)) for c in children) + 1e-12
#         inv = 1.0 / total
#         for c in children:
#             c.p = float(c.p) * inv
#         return children
