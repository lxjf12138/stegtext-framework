from __future__ import annotations
from typing import Sequence, Tuple, Any, List
import numpy as np
import torch

from ...core.contracts import BaseCodingStrategy, SupportsRandom
from ...core.prob import sanitize1d


def _to_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


class RangeAC(BaseCodingStrategy):
    """
    固定精度算术编码（步进式嵌入），不使用随机数。
      - init() 重置内部区间为 [0, 2^precision)。
      - encode/decode 接口包含 rng 形参以与引擎一致，但不会使用它。
    核心行为未变。
    """

    def __init__(self, precision: int = 40, **_: Any):
        self.precision = int(precision)
        self._inited = False
        self.lower = 0
        self.upper = 0

    # ----------------------- 生命周期 -----------------------
    def init(self, rng: SupportsRandom | None = None) -> None:
        # 区间半开 [lower, upper)
        self.lower = 0
        self.upper = 1 << self.precision
        self._inited = True

    def _ensure_inited(self) -> None:
        if not self._inited:
            raise RuntimeError("RangeAC 未初始化：请在会话开始调用 RangeAC.init()。")

    # ---------------- 概率向量 → 整数分割 -------------------
    def _partition(self, probs: torch.Tensor) -> tuple[List[int], List[int]]:
        """
        把概率向量分配成覆盖 [lower, upper) 的整数宽度，和为 W (= upper-lower)。
        采用“最大余数法”实现：floor 后按小数部分从大到小补齐差额。
        """
        W = int(self.upper - self.lower)
        if W <= 0:
            return [], []

        p = sanitize1d(_to_tensor(probs))
        K = int(p.numel())
        if K == 0:
            return [], []

        P = p.cpu().numpy()  # float64
        raw = P * W
        base = np.floor(raw).astype(np.int64)
        total = int(base.sum())
        rem = W - total

        if rem > 0:
            frac = raw - base
            # 小数部分降序、索引升序保证稳定
            order = np.lexsort((np.arange(K), -frac))
            for i in order[:rem]:
                base[i] += 1
        elif rem < 0:
            # 理论上 floor 求和不会 > W，这里仅为数值兜底
            take = -rem
            frac = raw - base
            order = np.lexsort((np.arange(K), frac))  # 从小数部分最小的开始回收
            for i in order[:take]:
                if base[i] > 0:
                    base[i] -= 1

        ints = [int(x) for x in base.tolist()]
        bounds: List[int] = []
        run = self.lower
        for w in ints:
            run += int(w)
            bounds.append(run)
        return ints, bounds

    # --------------------- 公共前缀 ------------------------
    def _common_prefix_bits(self, low: int, up: int) -> str:
        if up <= low:
            return ""
        lb = f"{low:0{self.precision}b}"
        ub = f"{up - 1:0{self.precision}b}"
        k = 0
        for a, b in zip(lb, ub):
            if a == b:
                k += 1
            else:
                break
        return lb[:k]

    # ----------------------- 编码 --------------------------
    def encode(
        self,
        group_probs: Sequence[float] | torch.Tensor,
        bit_stream: str,
        *,
        rng: SupportsRandom,  # 兼容签名；不使用
    ) -> Tuple[str, int]:
        """
        返回：(本步发射的公共前缀位串, 选择的组索引 group_idx)
        依赖“位流无限”的前提：bit_stream 至少有 precision 位（由引擎的虚拟位流保证）。
        """
        self._ensure_inited()
        p = _to_tensor(group_probs)
        _, bounds = self._partition(p)
        if not bounds:
            return "", 0

        # 读取消息值（前 precision 位）
        peek = bit_stream[: self.precision]
        msg = int(peek, 2) if peek else 0
        msg = min(max(msg, self.lower), self.upper - 1)

        # 决定落在哪个子段
        idx = 0
        for i, b in enumerate(bounds):
            if msg < b:
                idx = i
                break

        new_low = self.lower if idx == 0 else bounds[idx - 1]
        new_up = bounds[idx]

        # 发射公共前缀并收缩区间
        bit_read = self._common_prefix_bits(new_low, new_up)

        low_b = f"{new_low:0{self.precision}b}"
        up_b = f"{(new_up - 1):0{self.precision}b}" if new_up > new_low else low_b
        k = len(bit_read)
        lb = int((low_b[k:] + "0" * k), 2)
        ub = int((up_b[k:] + "1" * k), 2)
        self.lower, self.upper = lb, ub + 1

        return bit_read, int(idx)

    # ----------------------- 解码 --------------------------
    def decode(
        self,
        group_probs: Sequence[float] | torch.Tensor,
        chosen_group_idx: int,
        *,
        rng: SupportsRandom,  # 兼容签名；不使用
    ) -> str:
        """
        返回：本步恢复的公共前缀位串（必须与 encode 一致）
        """
        self._ensure_inited()
        p = _to_tensor(group_probs)
        _, bounds = self._partition(p)
        if not bounds:
            return ""

        gi = max(0, min(int(chosen_group_idx), len(bounds) - 1))
        new_low = self.lower if gi == 0 else bounds[gi - 1]
        new_up = bounds[gi]

        bit_read = self._common_prefix_bits(new_low, new_up)

        low_b = f"{new_low:0{self.precision}b}"
        up_b = f"{(new_up - 1):0{self.precision}b}" if new_up > new_low else low_b
        k = len(bit_read)
        lb = int((low_b[k:] + "0" * k), 2)
        ub = int((up_b[k:] + "1" * k), 2)
        self.lower, self.upper = lb, ub + 1

        return bit_read
