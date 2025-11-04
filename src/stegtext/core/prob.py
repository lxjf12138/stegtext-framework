from __future__ import annotations
import torch


def sanitize1d(x) -> torch.Tensor:
    """
    概率向量的稳健化：
      - NaN/Inf -> 0
      - 负数截到 0
      - 总和<=0 时退化为均匀
      - 最终归一到 1（float64）
    """
    t = torch.as_tensor(x, dtype=torch.float64)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.clamp(t, min=0.0)
    s = float(t.sum().item())
    if s <= 0.0:
        n = int(t.numel()) or 1
        return torch.full((n,), 1.0 / n, dtype=torch.float64)
    return t / s
