from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional, Tuple
import math
import torch

from stegtext.components.source.base import EOS_STEGA

from ...core.data import Candidate, Group, GroupingResult
from ...core.contracts import Disambiguator, Plan
from ...core.prob import sanitize1d


# --------------------- 基础工具 ---------------------

def _is_eos_vb(vb: bytes) -> bool:
    return vb.endswith(EOS_STEGA)

def _conflict(a_vb: bytes, b_vb: bytes) -> bool:
    """
    两候选是否前缀冲突（含相等视为冲突）。
    论文一致性：不再对 <eos> 做免冲突特判。
    """
    return a_vb.startswith(b_vb) or b_vb.startswith(a_vb)

def _dedup_by_vb_keep_max(cands: List[Candidate]) -> List[Candidate]:
    """相同可见串只保留 p 最大的那个。"""
    best: Dict[bytes, Candidate] = {}
    for c in cands:
        k = c.vb
        if k not in best or float(c.p) > float(best[k].p):
            best[k] = c
    return list(best.values())

def _build_conflict_graph(pool: List[Candidate]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    构造冲突图（无向图）与连通分量。
    返回：
      - adj: 邻接表，adj[i] = 与 i 冲突的结点索引列表
      - components: 若干分量，每个是局部结点索引列表
    """
    n = len(pool)
    adj: List[List[int]] = [[] for _ in range(n)]
    # 为加速，按 vb 长度排序后仅相邻附近做 startswith 检查
    order = sorted(range(n), key=lambda i: len(pool[i].vb))
    for idx_a, i in enumerate(order):
        vb_i = pool[i].vb
        # 仅与长度 >= len(vb_i) 的候选检查前缀；再回填邻接
        for j in order[idx_a:]:
            if i == j:
                continue
            vb_j = pool[j].vb
            if _conflict(vb_i, vb_j):
                adj[i].append(j)
                adj[j].append(i)

    # 连通分量
    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        comp: List[int] = []
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return adj, comps

# --------------------- 分量级 MWIS 求解 ---------------------

_MAX_EXACT_SIZE = 14  # 小分量阈值：≤14 用精确 MWIS（分支定界）

def _mwis_exact(comp: List[int], adj: List[List[int]], weights: List[float]) -> List[int]:
    """
    分支定界的精确 MWIS（在单个连通分量上）。
    comp: 该分量的全局索引列表；返回值同为全局索引。
    """
    # 将分量映射到局部 [0..m-1]
    m = len(comp)
    if m == 0:
        return []
    idx_of = {g: i for i, g in enumerate(comp)}
    w = [weights[g] for g in comp]
    # 邻接 bitmask（局部）
    nb = [0] * m
    for gi in comp:
        i = idx_of[gi]
        mask = 0
        for gj in adj[gi]:
            if gj in idx_of:
                mask |= (1 << idx_of[gj])
        nb[i] = mask

    # 按权重降序的顺序做分支定界
    order = sorted(range(m), key=lambda i: w[i], reverse=True)

    best_weight = 0.0
    best_mask = 0

    # 预计算剩余权重上界（贪心上界）
    prefix_max = [0.0] * (m + 1)
    acc = 0.0
    for t in range(m):
        acc += w[order[t]]
        prefix_max[t + 1] = acc

    def dfs(pos: int, chosen_mask: int, banned_mask: int, cur_weight: float):
        nonlocal best_weight, best_mask
        # 上界剪枝：当前 + 剩余最大可能
        if cur_weight + (prefix_max[m] - prefix_max[pos]) <= best_weight + 1e-15:
            return
        if pos == m:
            if cur_weight > best_weight + 1e-15:
                best_weight = cur_weight
                best_mask = chosen_mask
            return
        u = order[pos]
        bit_u = 1 << u
        # 选 u（若不被禁）
        if not (banned_mask & bit_u):
            dfs(
                pos + 1,
                chosen_mask | bit_u,
                banned_mask | bit_u | nb[u],  # 选中 u 后禁掉 u 及其邻居
                cur_weight + w[u],
            )
        # 不选 u
        dfs(pos + 1, chosen_mask, banned_mask, cur_weight)

    dfs(0, 0, 0, 0.0)

    # 还原到全局索引
    out: List[int] = []
    for i in range(m):
        if best_mask & (1 << i):
            out.append(comp[i])
    return out

def _mwis_greedy_improved(comp: List[int], adj: List[List[int]], weights: List[float], pool: List[Candidate]) -> List[int]:
    """
    改进贪心：优先选择“更长的 vb”（减少短前缀吞噬）→ 概率大 → 局部 1-swap 提升。
    返回全局索引列表。
    """
    # 初选：按 (len(vb) DESC, p DESC) 排序
    comp_sorted = sorted(comp, key=lambda g: (len(pool[g].vb), float(weights[g])), reverse=True)
    selected: List[int] = []
    selected_set = set()
    for g in comp_sorted:
        if all((h not in adj[g]) for h in selected_set):
            selected.append(g)
            selected_set.add(g)

    # 1-swap 局部改进：尝试把一个未选结点替换掉其冲突的若干已选结点
    # 若 w(new) > sum w(conflicts) 且 new 不与 (selected - conflicts) 冲突，则接受
    remain = [g for g in comp if g not in selected_set]
    improved = True
    while improved:
        improved = False
        # 候选按权重降序尝试
        for g in sorted(remain, key=lambda x: weights[x], reverse=True):
            conflicts = [h for h in selected if h in adj[g]]
            if not conflicts:
                # 已经不冲突，直接加入
                selected.append(g)
                selected_set.add(g)
                remain.remove(g)
                improved = True
                break
            w_conf = sum(weights[h] for h in conflicts)
            if weights[g] > w_conf + 1e-15:
                # 检查与其它已选是否冲突
                other = [h for h in selected if h not in conflicts]
                if all(h not in adj[g] for h in other):
                    # 执行替换
                    for h in conflicts:
                        selected.remove(h)
                        selected_set.discard(h)
                        remain.append(h)
                    selected.append(g)
                    selected_set.add(g)
                    remain.remove(g)
                    improved = True
                    break
    return selected


# --------------------- 主策略：论文式 MWIS ---------------------

class MWIS(Disambiguator):
    """
    论文式 MWIS 消歧（接口保持不变）：
      - 仅在存在“前缀冲突”时裁剪；否则原样分组（不改分布）。
      - 冲突图连通分量内：小分量精确 MWIS，大分量改进贪心 + 1-swap。
      - 每个胜出候选单独成组；组间概率 = 胜出候选原始 p → sanitize1d。
      - 在 stats/meta 中记录 eta_a 与 KLD_c 以评估改动大小。
    """

    def __init__(self, **kwargs):
        self.last_selected: Optional[Candidate] = None

    def init(self) -> None:
        self.last_selected = None

    def reset(self) -> None:
        self.last_selected = None

    def _plan_winners(self, candidates: List[Candidate]) -> Tuple[List[Candidate], Dict[str, Any]]:
        # 去重（相同可见串仅保留最大概率者）
        pool = _dedup_by_vb_keep_max(candidates)
        n = len(pool)
        if n == 0:
            return [], {"eta_a": 0.0, "kld_c": float("inf"), "pruned_mass": 1.0}

        # 构图与分量
        adj, comps = _build_conflict_graph(pool)

        # 快路径：无任何冲突 → 原样（论文中的 minimal intervention）
        has_conflict = any(len(adj[i]) > 0 for i in range(n))
        if not has_conflict:
            kept = pool[:]  # 不裁剪
            total_mass = sum(max(0.0, float(c.p)) for c in pool) + 1e-18
            kept_mass = total_mass
            eta_a = kept_mass / total_mass
            kld_c = -math.log(eta_a + 1e-18)
            return kept, {"eta_a": eta_a, "kld_c": kld_c, "pruned_mass": 0.0}

        # 有冲突：按分量逐块求解 MWIS
        weights = [max(0.0, float(c.p)) for c in pool]
        chosen_idx: List[int] = []
        for comp in comps:
            if len(comp) <= _MAX_EXACT_SIZE:
                chosen_idx.extend(_mwis_exact(comp, adj, weights))
            else:
                chosen_idx.extend(_mwis_greedy_improved(comp, adj, weights, pool))

        # 兜底：极端情况下保证至少一个
        if not chosen_idx:
            # 选全局最大 p
            best_i = max(range(n), key=lambda i: weights[i])
            chosen_idx = [best_i]

        kept = [pool[i] for i in chosen_idx]

        # 指标（基于去重后的池）
        total_mass = sum(weights) + 1e-18
        kept_mass = sum(weights[i] for i in chosen_idx)
        eta_a = kept_mass / total_mass
        kld_c = -math.log(eta_a + 1e-18)

        stats = {
            "eta_a": float(eta_a),
            "kld_c": float(kld_c),
            "pruned_mass": float(max(0.0, total_mass - kept_mass)),
            "n_candidates_dedup": int(n),
            "n_components": int(len(comps)),
        }
        return kept, stats

    def plan(self, candidates: List[Candidate]) -> Plan:
        winners, stats = self._plan_winners(candidates)

        groups: List[Group] = []
        probs: List[float] = []
        meta_groups: List[Dict[str, Any]] = []

        for rep in winners:
            groups.append(Group(rep))
            probs.append(max(0.0, float(rep.p)))
            meta_groups.append({"rep_vb": rep.vb})

        gp = sanitize1d(torch.tensor(probs if probs else [1.0], dtype=torch.float64))
        gr_out = GroupingResult(groups=groups, group_probs=gp, stats={"mwis": stats})
        meta: Dict[str, Any] = {
            "groups": meta_groups,
            "mode": "mwis-paper",
            "eta_a": stats.get("eta_a", 1.0),
            "kld_c": stats.get("kld_c", 0.0),
        }
        return Plan(groups=gr_out, group_probs=gp, meta=meta)

    def advance(
        self,
        plan: Plan,
        chosen_group_idx: int,
        rng,  # 未使用，仅为接口对齐
        source_generate: Callable[[torch.LongTensor, Optional[int]], List[Candidate]],
    ) -> List[Candidate] | Candidate:
        g = plan.groups.groups[chosen_group_idx]
        rep = g.members[0]

        # 仍保留对 <eos> 的“直接终止”行为（接口与上层语义保持不变）
        if _is_eos_vb(rep.vb):
            self.last_selected = rep
            return rep

        children = list(source_generate(rep.tokens, rep.prompt_len) or [])
        if not children:
            self.last_selected = rep
            return rep

        # 归一化 children 概率（不合并路径）
        total = sum(max(0.0, float(c.p)) for c in children)
        if total > 0.0:
            inv = 1.0 / total
            for c in children:
                c.p = float(c.p) * inv

        self.last_selected = rep
        return children
