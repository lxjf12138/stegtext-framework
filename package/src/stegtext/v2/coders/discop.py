from __future__ import annotations

import heapq
from typing import Any, Sequence

import torch

from .base import CoderSession


def _normal(probs: Sequence[float] | torch.Tensor) -> list[float]:
    p = torch.as_tensor(probs, dtype=torch.float64).flatten()
    if p.numel() == 0 or not torch.isfinite(p).all() or bool((p < 0).any()):
        raise ValueError("invalid coder probability vector")
    s = float(p.sum().item())
    if s <= 0:
        raise ValueError("coder probability vector has zero mass")
    return [float(x) for x in (p / s).tolist()]


class _Node:
    __slots__ = ("weight", "left", "right", "parent", "index")

    def __init__(self, weight: float, index: int = -1) -> None:
        self.weight = float(weight)
        self.left = self.right = self.parent = None
        self.index = int(index)

    def __lt__(self, other: "_Node") -> bool:
        return self.weight < other.weight


def _tree(weights: list[float]) -> _Node:
    nodes = [_Node(w, i) for i, w in enumerate(weights)] or [_Node(1.0, 0)]
    heapq.heapify(nodes)
    while len(nodes) > 1:
        left, right = heapq.heappop(nodes), heapq.heappop(nodes)
        parent = _Node(left.weight + right.weight)
        parent.left, parent.right = left, right
        left.parent, right.parent = parent, parent
        heapq.heappush(nodes, parent)
    return nodes[0]


class _DisCopSession:
    @staticmethod
    def _u(rng: Any) -> float:
        if rng is None or not hasattr(rng, "random"):
            raise RuntimeError("DisCop requires an RNG")
        return float(rng.random())

    def encode(self, probs, bit_stream: str, *, rng: Any) -> tuple[str, int]:
        root = _tree(_normal(probs))
        if root.left is None:
            return "", root.index
        node, out, pos = root, [], 0
        while node.left is not None and node.right is not None:
            total = node.left.weight + node.right.weight
            cut = node.left.weight / total if total else 0.5
            u0 = self._u(rng)
            u1 = (u0 + 0.5) % 1.0
            if u0 < cut and u1 < cut:
                node = node.left
            elif u0 >= cut and u1 >= cut:
                node = node.right
            else:
                bit = bit_stream[pos] if pos < len(bit_stream) else "0"
                pos += 1
                out.append(bit)
                node = node.left if bit == "0" else node.right
        return "".join(out), node.index

    def decode(self, probs, action: int, *, rng: Any) -> str:
        weights = _normal(probs)
        root = _tree(weights)
        if root.left is None:
            return ""
        path: list[int] = []
        node = root
        target = int(action)
        leaves: dict[int, list[int]] = {}

        def walk(cur: _Node, bits: list[int]) -> None:
            if cur.left is None:
                leaves[cur.index] = list(bits)
                return
            walk(cur.left, bits + [0])
            walk(cur.right, bits + [1])

        walk(root, [])
        path = leaves.get(target)
        if path is None:
            raise ValueError(f"unknown DisCop action {target}")
        out: list[str] = []
        node = root
        for direction in path:
            total = node.left.weight + node.right.weight
            cut = node.left.weight / total if total else 0.5
            u0 = self._u(rng)
            u1 = (u0 + 0.5) % 1.0
            mixed = not ((u0 < cut and u1 < cut) or (u0 >= cut and u1 >= cut))
            if mixed:
                out.append(str(direction))
            node = node.left if direction == 0 else node.right
        return "".join(out)


class DisCop:
    """Stateless payload coder with an explicit per-run session."""

    def start(self, mode: str = "encode", *, rng: Any = None) -> CoderSession:
        if mode not in {"encode", "decode"}:
            raise ValueError("DisCop session mode must be 'encode' or 'decode'")
        return _DisCopSession()
