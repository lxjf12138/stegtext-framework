from __future__ import annotations

from typing import Any, Sequence

import torch

from ..core.errors import InvariantError, ProbabilityError
from .base import CoderSession
from .mec import minimum_entropy_coupling


def _normal(probs: Sequence[float] | torch.Tensor) -> torch.Tensor:
    p = torch.as_tensor(probs, dtype=torch.float64).flatten()
    if p.numel() == 0 or not bool(torch.isfinite(p).all().item()) or bool((p < 0).any().item()):
        raise ProbabilityError("IMEC probability vector is invalid")
    total = float(p.sum().item())
    if total <= 0.0:
        raise ProbabilityError("IMEC probability vector has zero mass")
    return p / total


def _entropy(p: torch.Tensor) -> float:
    values = p[p > 0]
    return float((-values * torch.log2(values)).sum().item()) if values.numel() else 0.0


def _sample(probs: torch.Tensor, rng: Any) -> int:
    if rng is None or not hasattr(rng, "random"):
        raise RuntimeError("IMEC requires an RNG")
    p = _normal(probs)
    value = float(rng.random())
    if not 0.0 <= value < 1.0:
        raise ValueError("RNG must return a value in [0, 1)")
    cdf = torch.cumsum(p, dim=0)
    return min(int(torch.searchsorted(cdf, torch.tensor(value, dtype=torch.float64)).item()), int(p.numel()) - 1)


def _column_stats(matrix_colfirst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    entropies: list[float] = []
    argmaxes: list[int] = []
    for column in matrix_colfirst:
        mass = float(column.sum().item())
        if mass <= 0.0:
            entropies.append(0.0)
            argmaxes.append(0)
            continue
        normalized = column / mass
        entropies.append(_entropy(normalized))
        argmaxes.append(int(torch.argmax(normalized).item()))
    return torch.tensor(entropies, dtype=torch.float64), torch.tensor(argmaxes, dtype=torch.long)


class _IMECSession:
    def __init__(self, block_size: int, window_size: int, mode: str) -> None:
        if mode not in {"encode", "decode"}:
            raise ValueError("IMEC session mode must be 'encode' or 'decode'")
        self.block_size = int(block_size)
        self.window_size = int(window_size)
        self.mode = mode
        block_count = 1 << max(self.block_size, 1)
        uniform = torch.ones(block_count, dtype=torch.float64) / block_count
        self.beliefs = [uniform.clone() for _ in range(self.window_size)]

    def _targets(self, bit_stream: str) -> list[tuple[str, int]]:
        width = self.block_size
        return [
            (segment, int(segment, 2) if segment else 0)
            for segment in (bit_stream[i * width : (i + 1) * width] for i in range(self.window_size))
        ]

    def _advance_index(self) -> int:
        entropies = torch.tensor([_entropy(p) for p in self.beliefs], dtype=torch.float64)
        return int(torch.argmax(entropies).item())

    def _safe_columns(self, matrix_colfirst: torch.Tensor, target: int) -> torch.Tensor:
        entropies, argmaxes = _column_stats(matrix_colfirst)
        has_mass = matrix_colfirst.sum(dim=1) > 0
        safe = has_mass & ((entropies >= 1e-9) | ((entropies < 1e-9) & (argmaxes == int(target))))
        if not bool(safe.any().item()):
            raise InvariantError("IMEC coupling has no receiver-safe column")
        return safe

    def _collapse_if_ready(self, target: tuple[str, int] | None) -> str:
        if _entropy(self.beliefs[0]) >= 1e-9:
            return ""
        value = int(torch.argmax(self.beliefs[0]).item())
        if target is not None and value != target[1]:
            return ""
        bits = format(value, f"0{self.block_size}b")
        self.beliefs.pop(0)
        block_count = 1 << max(self.block_size, 1)
        self.beliefs.append(torch.ones(block_count, dtype=torch.float64) / block_count)
        return bits

    def encode(self, group_probs, bit_stream: str, *, rng: Any) -> tuple[str, int]:
        q = _normal(group_probs)
        targets = self._targets(str(bit_stream))
        index = self._advance_index()
        _, target_row = targets[index]
        coupling = minimum_entropy_coupling(self.beliefs[index], q, select_row=target_row)
        selected_row = coupling["M_selected_row"]
        safe = self._safe_columns(coupling["M_colfirst"], target_row)
        row_mass = float(selected_row.sum().item())
        pick = selected_row / row_mass if row_mass > 0.0 else q
        pick = pick * safe.to(dtype=torch.float64)
        if float(pick.sum().item()) <= 0.0:
            raise InvariantError("IMEC selected row has no safe column")
        token_id = _sample(pick, rng)
        column = coupling["M_colfirst"][token_id]
        column_mass = float(column.sum().item())
        if column_mass <= 0.0:
            raise InvariantError("IMEC selected action has zero coupling mass")
        self.beliefs[index] = column / column_mass
        return self._collapse_if_ready(targets[0]), token_id

    def decode(self, group_probs, action: int, *, rng: Any) -> str:
        q = _normal(group_probs)
        index = self._advance_index()
        coupling = minimum_entropy_coupling(self.beliefs[index], q, select_col=int(action))
        column = coupling["M_selected_col"]
        mass = float(column.sum().item())
        if mass <= 0.0:
            raise InvariantError("IMEC action has zero coupling mass")
        self.beliefs[index] = column / mass
        return self._collapse_if_ready(None)


class IMEC:
    """Windowed IMEC payload coder with explicit independent sessions."""

    def __init__(self, block_size: int = 6, window_size: int = 3, **_: Any) -> None:
        if int(block_size) <= 0:
            raise ValueError("IMEC block_size must be positive")
        if int(window_size) <= 0:
            raise ValueError("IMEC window_size must be positive")
        self.block_size = int(block_size)
        self.window_size = int(window_size)

    def start(self, mode: str = "encode", *, rng: Any = None) -> CoderSession:
        return _IMECSession(self.block_size, self.window_size, mode)

