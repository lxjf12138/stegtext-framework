from __future__ import annotations

from typing import Any

import torch

from .base import CoderSession


class _RangeSession:
    def __init__(self, precision: int) -> None:
        if precision < 8:
            raise ValueError("arithmetic precision must be at least 8")
        self.precision = int(precision)
        self.lower = 0
        self.upper = 1 << self.precision

    def _bounds(self, probs: torch.Tensor) -> list[int]:
        p = probs.detach().to(dtype=torch.float64, device="cpu").flatten()
        if p.numel() == 0 or not torch.isfinite(p).all() or bool((p < 0).any()):
            raise ValueError("invalid arithmetic-coder probability vector")
        p = p / p.sum()
        width = self.upper - self.lower
        raw = p.numpy() * width
        base = raw.astype("int64")
        for idx in sorted(range(len(raw)), key=lambda i: (-(raw[i] - base[i]), i))[: width - int(base.sum())]:
            base[idx] += 1
        out, cur = [], self.lower
        for value in base.tolist():
            cur += int(value)
            out.append(cur)
        return out

    def _common(self, low: int, high: int) -> str:
        if high <= low:
            return ""
        lo = f"{low:0{self.precision}b}"
        hi = f"{high - 1:0{self.precision}b}"
        n = 0
        while n < len(lo) and lo[n] == hi[n]:
            n += 1
        return lo[:n]

    def _step(self, bounds: list[int], action: int) -> str:
        if not bounds:
            raise ValueError("empty arithmetic action set")
        idx = max(0, min(int(action), len(bounds) - 1))
        low = self.lower if idx == 0 else bounds[idx - 1]
        high = bounds[idx]
        bits = self._common(low, high)
        n = len(bits)
        lo_b = f"{low:0{self.precision}b}"
        hi_b = f"{max(low, high - 1):0{self.precision}b}"
        self.lower = int((lo_b[n:] + "0" * n), 2)
        self.upper = int((hi_b[n:] + "1" * n), 2) + 1
        return bits

    def encode(self, probs, bit_stream: str, *, rng: Any) -> tuple[str, int]:
        p = torch.as_tensor(probs, dtype=torch.float64).flatten()
        bounds = self._bounds(p)
        peek = bit_stream[: self.precision].ljust(self.precision, "0")
        value = int(peek, 2)
        idx = next((i for i, bound in enumerate(bounds) if value < bound), len(bounds) - 1)
        return self._step(bounds, idx), idx

    def decode(self, probs, action: int, *, rng: Any) -> str:
        return self._step(self._bounds(torch.as_tensor(probs, dtype=torch.float64)), action)


class RangeAC:
    def __init__(self, precision: int = 40, **_: Any) -> None:
        self.precision = int(precision)

    def start(self, mode: str = "encode", *, rng: Any = None) -> CoderSession:
        if mode not in {"encode", "decode"}:
            raise ValueError("RangeAC session mode must be 'encode' or 'decode'")
        return _RangeSession(self.precision)
