from __future__ import annotations
from typing import Sequence, Tuple, Any, List
import torch

from ...core.contracts import BaseCodingStrategy, SupportsRandom
from ...core.prob import sanitize1d
from .mec import minimum_entropy_coupling


def _shannon_entropy(p: torch.Tensor) -> float:
    p = torch.clamp(torch.as_tensor(p, dtype=torch.float64), min=0.0)
    s = float(p.sum().item())
    if s <= 0.0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float((-nz * torch.log2(nz)).sum().item())


def _choice_from_probs_torch(probs_1d: torch.Tensor, rng: SupportsRandom) -> int:
    if rng is None or not hasattr(rng, "random"):
        raise RuntimeError("IMEC.encode 需要外部 rng（实现 .random() -> [0,1)）。")
    arr = probs_1d.detach().to(dtype=torch.float64, device="cpu")
    cdf = torch.cumsum(arr, dim=0).clamp(max=1.0)
    u = float(rng.random())
    idx = int(torch.searchsorted(cdf, torch.tensor(u, dtype=torch.float64)).item())
    if idx >= int(arr.numel()):
        idx = int(arr.numel()) - 1
    return idx


def _column_entropies_and_argmax(Mcol: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    K, _ = Mcol.shape
    H = torch.empty(K, dtype=torch.float64)
    am = torch.empty(K, dtype=torch.long)
    for k in range(K):
        col = Mcol[k]
        s = float(col.sum().item())
        if s <= 0.0:
            H[k] = 0.0
            am[k] = 0
        else:
            p = (col / s).to(dtype=torch.float64)
            nz = p[p > 0]
            H[k] = float((-nz * torch.log2(nz)).sum().item())
            am[k] = int(torch.argmax(p).item())
    return H, am


class IMEC(BaseCodingStrategy):
    """
    Windowed IMEC（官方化的块级确认 + 窗口并行）：
      - 同时维护 window_size 个块的后验 belief。
      - 每步推进熵最大的块；只有“窗口最左块”坍缩且匹配时才发射其 B 位并左移窗口。
      - 列筛选：仅允许非坍缩列或“坍缩且与该块 m 匹配”的列，避免解码端提前读错。
    """

    def __init__(self, block_size: int = 6, window_size: int = 3, **_: Any):
        self.block_size = int(block_size)
        self.window_size = max(1, int(window_size))
        self.beliefs: List[torch.Tensor] | None = None

    # ----------------------- 生命周期 -----------------------
    def init(self) -> None:
        B = 1 << max(self.block_size, 1)
        uni = torch.ones(B, dtype=torch.float64) / B
        self.beliefs = [uni.clone() for _ in range(self.window_size)]

    def _ensure_inited(self):
        if self.beliefs is None:
            raise RuntimeError("IMEC 未初始化：请先调用 IMEC.init()。")

    def _ensure_window_len(self):
        assert self.beliefs is not None
        B = 1 << max(self.block_size, 1)
        while len(self.beliefs) < self.window_size:
            self.beliefs.append(torch.ones(B, dtype=torch.float64) / B)
        if len(self.beliefs) > self.window_size:
            self.beliefs = self.beliefs[: self.window_size]

    # ----------------------- 编码 --------------------------
    def encode(
        self,
        group_probs: Sequence[float] | torch.Tensor,
        bit_stream: str,
        *,
        rng: SupportsRandom,
    ) -> Tuple[str, int]:
        """
        返回 (emitted_bits, token_id)：
          - 常规返回 ("", token_id)
          - 仅在“窗口最左块坍缩且 bits_hat==bit_stream[:B]”时，返回该前缀 B 位并左移窗口。
        """
        self._ensure_inited()
        self._ensure_window_len()
        beliefs = self.beliefs  # type: ignore[assignment]
        Bbits = self.block_size
        B = 1 << max(Bbits, 1)
        q = sanitize1d(group_probs)

        # 目标索引：从 bit_stream 取前 window_size 个块
        targets = []
        for i in range(self.window_size):
            seg = bit_stream[i * Bbits : (i + 1) * Bbits]
            m = int(seg, 2) if seg else 0
            targets.append((seg, m))

        # 选择要推进的块：熵最大者
        ent = torch.tensor([_shannon_entropy(p) for p in beliefs], dtype=torch.float64)
        j = int(torch.argmax(ent).item())
        p_j = beliefs[j]
        to_read_j, m_j = targets[j]

        # MEC（对第 j 块）
        mec = minimum_entropy_coupling(
            p=p_j,
            q=q,
            select_row=m_j,
            select_col=None,
            mode="greedy",
            algo_atol=1e-7,
        )
        Mcol = mec["M_colfirst"]          # [K, B]
        row = mec["M_selected_row"]       # [K]
        row_sum = float(row.sum().item())

        # 列筛选（避免解码提前读错）
        H_cols, am_cols = _column_entropies_and_argmax(Mcol)
        eps = 1e-9
        col_sums = Mcol.sum(dim=1)
        has_mass = (col_sums > 0).to(dtype=torch.float64)
        non_collapse = (H_cols >= eps).to(dtype=torch.float64)
        collapse_match = ((H_cols < eps) & (am_cols == m_j)).to(dtype=torch.float64)
        safe_cols_mask = ((non_collapse + collapse_match) > 0).to(dtype=torch.float64)

        # 选列（两分支共享）
        if row_sum <= 0.0:
            # 行 m_j 无质量：不消费；从“有质量 ∧ 安全”的列按 q 采样推进
            pick_base = q * has_mass * safe_cols_mask
            pick_probs = sanitize1d(pick_base)
            if float(pick_probs.sum().item()) == 0.0:
                # 回退：在有质量列里选熵最大的列，尽量避免坍缩
                candidates = (has_mass > 0)
                if bool(candidates.any().item()):
                    k_star = int(torch.argmax(H_cols * candidates.to(dtype=torch.float64)).item())
                    token_id = k_star
                else:
                    token_id = _choice_from_probs_torch(q, rng)
            else:
                token_id = _choice_from_probs_torch(pick_probs, rng)
        else:
            row = row / row_sum
            pick_probs = sanitize1d(row * safe_cols_mask)
            if float(pick_probs.sum().item()) == 0.0:
                candidates = (has_mass > 0)
                if bool(candidates.any().item()):
                    k_star = int(torch.argmax(H_cols * candidates.to(dtype=torch.float64)).item())
                    token_id = k_star
                else:
                    token_id = _choice_from_probs_torch(row, rng)
            else:
                token_id = _choice_from_probs_torch(pick_probs, rng)

        # belief_j ← 选中列
        col = Mcol[token_id]
        col_sum = float(col.sum().item())
        if col_sum > 0.0:
            beliefs[j] = col / col_sum  # 更新第 j 块
        # 否则保持原 belief_j，不消费

        # 仅当前缀块（index=0）坍缩且匹配时发射，并左移窗口
        p0 = beliefs[0]
        if _shannon_entropy(p0) < eps:
            m_hat0 = int(torch.argmax(p0).item())
            bits_hat0 = format(m_hat0, f"0{Bbits}b")
            to_read0, m0 = targets[0]
            if bits_hat0 == to_read0:
                # 发射前缀块位，并左移窗口：弹出最左 belief，末尾补均匀
                emitted = bits_hat0
                beliefs.pop(0)
                beliefs.append(torch.ones(B, dtype=torch.float64) / B)
                return emitted, int(token_id)

        return "", int(token_id)

    # ----------------------- 解码 --------------------------
    def decode(
        self,
        group_probs: Sequence[float] | torch.Tensor,
        token_id: int,
        *,
        rng: SupportsRandom,  # 兼容签名；不使用
    ) -> str:
        """
        多块并行的解码：
          - 每步对“熵最大块”应用 MEC(select_col=token_id) 更新其 belief；
          - 仅当前缀块坍缩时输出该块 B 位并左移窗口。
        """
        self._ensure_inited()
        self._ensure_window_len()
        beliefs = self.beliefs  # type: ignore[assignment]
        Bbits = self.block_size
        B = 1 << max(Bbits, 1)
        q = sanitize1d(group_probs)

        # 选择要推进的块（和编码端同策略：熵最大者）
        ent = torch.tensor([_shannon_entropy(p) for p in beliefs], dtype=torch.float64)
        j = int(torch.argmax(ent).item())
        p_j = beliefs[j]

        mec = minimum_entropy_coupling(
            p=p_j,
            q=q,
            select_row=None,
            select_col=int(token_id),
            mode="greedy",
            algo_atol=1e-7,
        )
        col = mec["M_selected_col"]
        col_sum = float(col.sum().item())
        if col_sum > 0.0:
            beliefs[j] = col / col_sum

        # 若前缀块坍缩，则读出其位并左移窗口
        eps = 1e-9
        p0 = beliefs[0]
        if _shannon_entropy(p0) < eps:
            m_hat0 = int(torch.argmax(p0).item())
            bits0 = format(m_hat0, f"0{Bbits}b")
            beliefs.pop(0)
            beliefs.append(torch.ones(B, dtype=torch.float64) / B)
            return bits0

        return ""
