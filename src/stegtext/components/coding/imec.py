from __future__ import annotations
from typing import Sequence, Tuple, Any
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
    """
    用外部 rng.random() ∈ [0,1) 在离散分布中采样索引（Torch 版，不依赖 numpy）。
    """
    if rng is None or not hasattr(rng, "random"):
        raise RuntimeError("IMEC.encode 需要外部 rng（实现 .random() -> [0,1)）。")
    arr = probs_1d.detach().to(dtype=torch.float64, device="cpu")
    cdf = torch.cumsum(arr, dim=0).clamp(max=1.0)
    u = float(rng.random())
    idx = int(torch.searchsorted(cdf, torch.tensor(u, dtype=torch.float64)).item())
    if idx >= int(arr.numel()):
        idx = int(arr.numel()) - 1
    return idx


class IMEC(BaseCodingStrategy):
    """
    Iterative MEC 单步编码器（块后验），仅依赖外部 rng：
      - encode 采样 token_id 时必须使用传入的 rng.random()。
      - decode 不采样，无需使用 rng。
    约定：位流无限；init() 重置 belief 为均匀先验。
    """

    def __init__(self, block_size: int = 6, **_: Any):
        self.block_size = int(block_size)
        self.belief: torch.Tensor | None = None

    # ----------------------- 生命周期 -----------------------
    def init(self) -> None:
        B = 1 << max(self.block_size, 1)
        self.belief = torch.ones(B, dtype=torch.float64) / B

    def _ensure_inited(self):
        if self.belief is None:
            raise RuntimeError("IMEC 未初始化：请在会话开始调用 IMEC.init()。")

    # ----------------------- 编码 --------------------------
    def encode(
        self,
        group_probs: Sequence[float] | torch.Tensor,
        bit_stream: str,
        *,
        rng: SupportsRandom,
    ) -> Tuple[str, int]:
        self._ensure_inited()
        belief = self.belief  # type: ignore[assignment]

        # 位流无限：直接读 B 位作为 m（不在 coder 内补位）
        to_read = bit_stream[: self.block_size]
        m = int(to_read, 2) if to_read else 0

        q = sanitize1d(group_probs)

        mec = minimum_entropy_coupling(
            p=belief,
            q=q,
            select_row=m,
            select_col=None,
            mode="greedy",
            algo_atol=1e-7,
        )
        row = mec["M_selected_row"]
        row = row / row.sum() if float(row.sum().item()) > 0 else torch.ones_like(row) / max(1, row.numel())

        # 采样 token_id —— 仅使用外部 rng
        token_id = _choice_from_probs_torch(row, rng)

        # belief ← 选中列
        col = mec["M_colfirst"][token_id]
        col = col / col.sum() if float(col.sum().item()) > 0 else torch.ones_like(col) / max(1, col.numel())
        self.belief = col

        # 若后验熵≈0，确认消费该块位；否则暂不发射（为空串）
        if _shannon_entropy(self.belief) < 1e-9:
            B = 1 << max(self.block_size, 1)
            self.belief = torch.ones(B, dtype=torch.float64) / B
            return to_read, int(token_id)

        return "", int(token_id)

    # ----------------------- 解码 --------------------------
    def decode(
        self,
        group_probs: Sequence[float] | torch.Tensor,
        token_id: int,
        *,
        rng: SupportsRandom,  # 兼容签名；不使用
    ) -> str:
        self._ensure_inited()

        q = sanitize1d(group_probs)

        mec = minimum_entropy_coupling(
            p=self.belief,           # type: ignore[arg-type]
            q=q,
            select_row=None,
            select_col=int(token_id),
            mode="greedy",
            algo_atol=1e-7,
        )
        col = mec["M_selected_col"]
        col = col / col.sum() if float(col.sum().item()) > 0 else torch.ones_like(col) / max(1, col.numel())
        self.belief = col

        if _shannon_entropy(self.belief) < 1e-9:
            m_hat = int(torch.argmax(self.belief).item())
            bit_read = format(m_hat, f"0{self.block_size}b")
            B = 1 << max(self.block_size, 1)
            self.belief = torch.ones(B, dtype=torch.float64) / B
            return bit_read

        return ""
