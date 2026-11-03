from __future__ import annotations

from typing import Optional

import torch

from ..core.errors import ProbabilityError


def _normalize(values: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if tensor.numel() == 0:
        raise ProbabilityError(f"MEC {name} distribution is empty")
    if not bool(torch.isfinite(tensor).all().item()) or bool((tensor < 0).any().item()):
        raise ProbabilityError(f"MEC {name} distribution contains invalid probabilities")
    total = float(tensor.sum().item())
    if total <= 0.0:
        raise ProbabilityError(f"MEC {name} distribution has zero mass")
    return tensor / total


def minimum_entropy_coupling(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    select_row: Optional[int] = None,
    select_col: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Construct the deterministic greedy minimum-entropy coupling."""
    p_norm = _normalize(p, "row")
    q_norm = _normalize(q, "column")
    rows, cols = int(p_norm.numel()), int(q_norm.numel())
    if select_row is not None and not 0 <= int(select_row) < rows:
        raise IndexError("MEC selected row is outside the coupling")
    if select_col is not None and not 0 <= int(select_col) < cols:
        raise IndexError("MEC selected column is outside the coupling")

    matrix = torch.zeros((rows, cols), dtype=torch.float64)
    remaining_rows = p_norm.clone()
    remaining_cols = q_norm.clone()
    tolerance = 1e-15
    while True:
        row = int(torch.argmax(remaining_rows).item())
        col = int(torch.argmax(remaining_cols).item())
        row_mass = float(remaining_rows[row].item())
        col_mass = float(remaining_cols[col].item())
        if row_mass <= tolerance or col_mass <= tolerance:
            row_total = float(remaining_rows.sum().item())
            col_total = float(remaining_cols.sum().item())
            if max(row_total, col_total) <= 1e-12:
                break
            raise ProbabilityError("MEC residual marginals became inconsistent")
        mass = min(row_mass, col_mass)
        if mass <= 0.0:
            raise ProbabilityError("MEC residual marginals became inconsistent")
        matrix[row, col] += mass
        remaining_rows[row] -= mass
        remaining_cols[col] -= mass

    matrix.clamp_min_(0.0)
    result: dict[str, torch.Tensor] = {"M": matrix, "M_colfirst": matrix.t()}
    if select_row is not None:
        result["M_selected_row"] = matrix[int(select_row), :]
    if select_col is not None:
        result["M_selected_col"] = matrix[:, int(select_col)]
    return result
