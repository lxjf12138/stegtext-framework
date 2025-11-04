from __future__ import annotations
from typing import Dict, Optional
import torch


def _sanitize_probs(x: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float64)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.clamp(t, min=0.0)
    s = t.sum()
    if not torch.isfinite(s) or s <= 0:
        n = int(t.numel()) if int(t.numel()) > 0 else 1
        return torch.ones(n, dtype=torch.float64) / n
    return t / s


def minimum_entropy_coupling(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    select_row: Optional[int] = None,
    select_col: Optional[int] = None,
    # 兼容旧签名（未使用）
    mode: str | None = None,
    algo_atol: float | None = None,
) -> Dict[str, torch.Tensor]:
    """
    稳定的 MEC 贪心构造（矩形版本）：
      输入：p (B,), q (K,)
      输出：M (B, K)，以及便捷字段：
        * M_selected_row = M[select_row, :]
        * M_selected_col = M[:, select_col]
        * M_colfirst     = M.T
    """
    p = _sanitize_probs(p).clone()
    q = _sanitize_probs(q).clone()

    B = int(p.numel())
    K = int(q.numel())
    M = torch.zeros((B, K), dtype=torch.float64)

    pr = p.clone()
    qr = q.clone()
    eps = 1e-18

    while True:
        i = int(torch.argmax(pr).item()) if B > 0 else 0
        j = int(torch.argmax(qr).item()) if K > 0 else 0
        vi = float(pr[i].item()) if B > 0 else 0.0
        vj = float(qr[j].item()) if K > 0 else 0.0
        if vi <= eps and vj <= eps:
            break
        x = vi if vi < vj else vj
        if x <= 0.0:
            break
        M[i, j] += x
        pr[i] -= x
        qr[j] -= x

    out: Dict[str, torch.Tensor] = {"M": M}
    if select_row is not None:
        out["M_selected_row"] = M[int(select_row), :]
    if select_col is not None:
        out["M_selected_col"] = M[:, int(select_col)]
    out["M_colfirst"] = M.t()
    return out
