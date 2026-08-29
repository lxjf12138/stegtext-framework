# src/stegtext/components/disambiguation/lookahead.py
from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional
import os
import torch

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
        # 均匀兜底
        r = float(rng.random()) * len(indices)
        return indices[min(int(r), len(indices) - 1)]
    r = float(rng.random()) * tot
    acc = 0.0
    for i, w in zip(indices, sub.tolist()):
        acc += max(0.0, float(w))
        if r <= acc:
            return i
    return indices[-1]


class LookAhead(Disambiguator):
    """
    Look-ahead Sync
    - plan：仅做“前缀互斥分组 + 组间质量聚合”，组内仅记录 raw_p，不做组内归一化。
    - advance（只对被选组处理）：
        1) intra = normalize(raw_p)
        2) 在 S_prefix 上按 intra 抽样一个 ssync
        3) s = sum_{i in S_prefix} intra[i]
        4) children = source_generate(ssync.tokens)
        5) children 概率就地乘以 s
        6) S_partial 成员的 p 就地改写为 intra[i]
        7) 返回 S_partial(改好 p) + children（不做任何按可见文本的合并）
    """

    def __init__(self, m_reps: Optional[int] = None) -> None:
        """Create the theorem-covered single-representative LAS variant.

        ``m_reps`` is retained only for backwards-compatible configuration.
        The reference algorithm proved in the paper samples exactly one
        representative from ``S_prefix``; values other than one are rejected
        so experiments cannot silently run an unproved Monte-Carlo variant.
        """
        self._last_meta: Dict[str, Any] = {}
        self.last_selected: Optional[Candidate] = None
        if m_reps is None:
            try:
                m_reps = int(os.environ.get("LOOKAHEAD_M", "1").strip())
            except Exception:
                m_reps = 1
        self.m_reps: int = int(m_reps if m_reps is not None else 1)
        if self.m_reps != 1:
            raise ValueError(
                "LookAhead reference mode requires m_reps=1; "
                "multi-representative sampling is not covered by the LAS theorem"
            )

    # lifecycle
    def init(self) -> None:
        self._last_meta = {}
        self.last_selected = None

    def reset(self) -> None:
        self._last_meta = {}
        self.last_selected = None

    # -------- plan：分组 + 记录原始组内权重 raw_p（不做组内归一化） --------
    def plan(self, candidates: List[Candidate]) -> Plan:
        gr = group_by_prefix_bytes(candidates)

        group_sums: List[float] = []
        meta_groups: List[Dict[str, Any]] = []

        for g in gr.groups:
            mem = g.members  # 不复制，减少对象
            raw_p = torch.tensor([max(0.0, float(m.p)) for m in mem], dtype=torch.float64)
            group_sums.append(float(raw_p.sum().item()))
            sprefix_idx = [i for i, m in enumerate(mem) if (m.vb == g.key)]
            spartial_idx = [i for i in range(len(mem)) if i not in sprefix_idx]
            meta_groups.append({
                "key": g.key,
                "raw_p": raw_p,               # 仅记录，advance 时才归一化
                "sprefix_idx": sprefix_idx,
                "spartial_idx": spartial_idx,
            })
        gp = sanitize1d(torch.tensor(group_sums, dtype=torch.float64))  # 组间给 coder
        meta: Dict[str, Any] = {"mode": "lookahead", "controls_next": True, "groups": meta_groups}
        self._last_meta = meta
        return Plan(groups=gr, group_probs=gp, meta=meta)

    # -------- advance：只处理被选组；不创建多余 Candidate、绝不“按可见文本合并” --------
    def advance(
        self,
        plan: Plan,
        chosen_group_idx: int,
        rng: SupportsRandom,
        source_generate: Callable[[torch.LongTensor, Optional[int]], List[Candidate]],
    ) -> List[Candidate] | Candidate:
        g = plan.groups.groups[chosen_group_idx]
        mem = g.members  # List[Candidate]（当前组的成员，含完整 tokens）

        # A terminal-only group has no valid strict extension after EOS.  The
        # representative is sampled by the exact conditional group weights and
        # returned as a singleton terminal outcome.  Terminal candidates that
        # merely extend a shorter nonterminal group key remain in S_partial and
        # are retained instead of forcing premature termination.
        if mem and all(m.is_eos for m in mem):
            info = plan.meta["groups"][chosen_group_idx]
            raw_p = torch.as_tensor(info["raw_p"], dtype=torch.float64)
            weights = sanitize1d(raw_p)
            indices = list(range(len(mem)))
            chosen_idx = _weighted_choice(indices, weights, rng)
            chosen = mem[chosen_idx]
            chosen.p = 1.0
            self.last_selected = chosen
            return chosen

        info = plan.meta["groups"][chosen_group_idx]
        raw_p: torch.Tensor = torch.as_tensor(info["raw_p"], dtype=torch.float64)
        sprefix_idx: List[int] = info["sprefix_idx"]
        spartial_idx: List[int] = info["spartial_idx"]

        # (1) 仅对被选组做组内归一化
        intra = sanitize1d(raw_p)

        # (2) 在 S_prefix 上同步抽样代表（按 intra 权重）；last_selected 仅 1 条
        if sprefix_idx:
            midx = _weighted_choice(sprefix_idx, intra, rng)
            ssync = mem[midx]
        else:
            # 理论上非 eos 组应当有 S_prefix；兜底：当作空前缀不可展开，仅保留 S_partial
            ssync = None  # type: ignore[assignment]

        # (3) s = S_prefix 概率和（归一化后的）
        s = float(intra[sprefix_idx].sum().item()) if sprefix_idx else 0.0

        # (4) 展开：若 ssync 是 eos 或不存在，children 为空；否则继续生成
        if ssync is None:
            raise RuntimeError("LookAhead: unable to select sync candidate")
        if ssync.is_eos:
            raise RuntimeError(
                "LookAhead invariant violated: a terminal representative cannot "
                "be the exact-prefix member of a nonterminal group"
            )

        # Reference LAS expands exactly one synchronized representative.
        children: List[Candidate] = list(
            source_generate(ssync.tokens, ssync.prompt_len) or []
        )
        for c in children:
            c.p = float(c.p) * s

        # S_partial keeps its exact conditional mass.
        for i in spartial_idx:
            mem[i].p = float(intra[i])

        # (7) 组装下一轮候选：严格不按可见文本合并路径，直接拼接
        out: List[Candidate] = [mem[i] for i in spartial_idx]
        out.extend(children)

        # 轻度数值稳健（理论上 sum==1；这里容忍微小误差）
        total = sum(max(0.0, float(c.p)) for c in out)
        if not (0.999999 <= total <= 1.000001) and total > 0:
            scale = 1.0 / total
            for c in out:
                c.p = float(c.p) * scale

        self.last_selected = ssync
        return out
