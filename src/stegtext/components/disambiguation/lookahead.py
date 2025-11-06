# src/stegtext/components/disambiguation/lookahead.py
from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional
import os
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
        """Look-ahead disambiguator with optional multi-representative sampling.

        m_reps controls how many with-replacement samples are drawn from
        S_prefix to estimate the mixture for the next step:
          - m_reps <= 1 (default): legacy single-representative behavior.
          - m_reps > 1: sample m_reps times with replacement according to
            intra weights over S_prefix. For each sampled prefix, expand its
            children and multiply their probabilities by s/m where s is the
            total intra mass over S_prefix. This preserves per-child token
            prefixes and is an unbiased Monte-Carlo estimate of the full
            mixture. Larger m reduces variance.

        m_reps can also be provided via environment variable LOOKAHEAD_M.
        """
        self._last_meta: Dict[str, Any] = {}
        self.last_selected: Optional[Candidate] = None
        if m_reps is None:
            try:
                m_reps = int(os.environ.get("LOOKAHEAD_M", "1").strip())
            except Exception:
                m_reps = 1
        self.m_reps: int = int(m_reps if m_reps is not None else 1)

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
        if g.key.endswith(EOS_STEGA):
            info = plan.meta["groups"][chosen_group_idx]
            raw_p = torch.as_tensor(info["raw_p"], dtype=torch.float64)
            weights = sanitize1d(raw_p)
            indices = list(range(len(g.members)))
            chosen_idx = _weighted_choice(indices, weights, rng) if indices else 0
            chosen = g.members[chosen_idx]
            chosen.p = float(round(float(weights[chosen_idx]), 12)) if len(indices) > 0 else 1.0
            self.last_selected = chosen
            return chosen

        mem = g.members  # List[Candidate]（当前组的成员，含完整 tokens）
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
        children: List[Candidate] = []
        if not ssync.is_eos:
            if self.m_reps > 1 and sprefix_idx:
                # 有放回 m 次抽样；对相同前缀计数，仅前向一次，再按 count*(s/m) 赋权
                m = int(max(1, self.m_reps))
                per = (s / float(m)) if m > 0 else 0.0
                draw_cnt: Dict[int, int] = {}
                for _ in range(m):
                    ridx = _weighted_choice(sprefix_idx, intra, rng)
                    draw_cnt[ridx] = draw_cnt.get(ridx, 0) + 1
                if per > 0.0:
                    for ridx, cnt in draw_cnt.items():
                        rnode = mem[ridx]
                        ch = list(source_generate(rnode.tokens, rnode.prompt_len) or [])
                        w = per * float(cnt)
                        for c in ch:
                            c.p = float(round(float(c.p) * w, 12))
                            children.append(c)
            else:
                # 单代表路径：只展开 ssync，一次性整体乘以 s
                children = list(source_generate(ssync.tokens, ssync.prompt_len) or [])
                if children:
                    for c in children:
                        c.p = float(round(float(c.p) * s, 12))

        # (6) S_partial 的 p 改写为组内归一化后的 intra[i]
        for i in spartial_idx:
            mem[i].p = float(round(float(intra[i]), 12))

        # (7) 组装下一轮候选：严格不按可见文本合并路径，直接拼接
        out: List[Candidate] = [mem[i] for i in spartial_idx]
        out.extend(children)

        # 轻度数值稳健（理论上 sum==1；这里容忍微小误差）
        total = sum(max(0.0, float(c.p)) for c in out)
        if not (0.999999 <= total <= 1.000001) and total > 0:
            scale = 1.0 / total
            for c in out:
                c.p = float(c.p) * scale

        # 终止条件说明：
        # - 若 ssync 是 eos 且 S_partial 为空 → out==[]，上层会在本步已输出完毕后自然结束。
        self.last_selected = ssync
        if ssync.is_eos:
            return ssync
        return out
