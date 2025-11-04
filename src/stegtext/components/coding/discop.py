from __future__ import annotations
from typing import Sequence, Tuple, Any, List
import heapq
import torch

from ...core.contracts import BaseCodingStrategy, SupportsRandom
from ...core.prob import sanitize1d


# ---------- Huffman 基础 ----------
class _Node:
    __slots__ = ("w", "l", "r", "p", "idx")

    def __init__(self, w: float, idx: int = -1):
        self.w = float(w)
        self.l: "_Node | None" = None
        self.r: "_Node | None" = None
        self.p: "_Node | None" = None
        self.idx = int(idx)

    def __lt__(self, other: "_Node"):
        return self.w < other.w


def _build_huffman(weights: Sequence[float]) -> Tuple[_Node, list[str]]:
    """
    构造 Huffman 树并返回（根，码本），码本是 idx -> bitstring。
    单元素时约定码字为 "0"。
    """
    if not weights:
        root = _Node(1.0, 0)
        return root, ["0"]

    nodes = [_Node(max(0.0, float(w)), i) for i, w in enumerate(weights)]
    heap = nodes[:]
    heapq.heapify(heap)

    if len(heap) == 1:
        root = heap[0]
        codes = [""] * len(weights)
        codes[root.idx] = "0"
        return root, codes

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        p = _Node(a.w + b.w)
        p.l, p.r = a, b
        a.p = p
        b.p = p
        heapq.heappush(heap, p)

    root = heap[0]
    # 生成码本
    codes = [""] * len(weights)
    for leaf in nodes:
        s = ""
        cur = leaf
        while cur.p is not None:
            s = ("0" if cur.p.l is cur else "1") + s
            cur = cur.p
        codes[leaf.idx] = s if s else "0"
    return root, codes


class DisCop(BaseCodingStrategy):
    """
    DisCop（双均匀数 + Huffman），仅依赖外部 rng：
      - encode/decode 仅使用传入的 rng.random() 产生 U(0,1)。
      - 不在 coder 内部做位流补齐（位流无限由引擎保证）。
    核心行为未变。
    """

    def __init__(self, **_: Any):
        # 无内部持久状态；不依赖 random/seed
        pass

    def init(self) -> None:
        return

    @staticmethod
    def _u01(rng: SupportsRandom) -> float:
        if rng is None or not hasattr(rng, "random"):
            raise RuntimeError("DisCop.encode/decode 需要外部 rng（实现 .random() -> [0,1)）。")
        return float(rng.random())

    # ----------------------- 编码 --------------------------
    def encode(self, group_probs: Sequence[float] | torch.Tensor, bit_stream: str, *, rng: SupportsRandom) -> Tuple[str, int]:
        p = sanitize1d(group_probs)
        weights = [float(x) for x in p.tolist()]
        root, _codes = _build_huffman(weights)

        # 叶子：直接返回该 idx
        if root.l is None and root.r is None:
            return "", (root.idx if root.idx >= 0 else 0)

        node = root
        cnt = 0
        out_bits: List[str] = []

        while node.l is not None and node.r is not None:
            lw = node.l.w if node.l else 0.0
            rw = node.r.w if node.r else 0.0
            denom = lw + rw
            mid = lw / denom if denom > 0 else 0.5

            u0 = self._u01(rng)
            u1 = (u0 + 0.5) % 1.0  # 反相相位；节省一次采样
            same_left = (u0 < mid) and (u1 < mid)
            same_right = (u0 >= mid) and (u1 >= mid)

            if same_left:
                node = node.l  # 无消费下行
            elif same_right:
                node = node.r
            else:
                # 消费一位消息位决定左右（位流无限由上层保证）
                b = bit_stream[cnt]
                out_bits.append(b)
                cnt += 1
                node = node.l if b == "0" else node.r

        leaf_idx = node.idx if node.idx >= 0 else 0
        return "".join(out_bits), int(leaf_idx)

    # ----------------------- 解码 --------------------------
    def decode(self, group_probs: Sequence[float] | torch.Tensor, group_idx: int, *, rng: SupportsRandom) -> str:
        p = sanitize1d(group_probs)
        weights = [float(x) for x in p.tolist()]
        root, codes = _build_huffman(weights)

        if root.l is None and root.r is None:
            return ""

        path = codes[int(group_idx)]
        node = root
        out_bits: List[str] = []

        for d in path:
            lw = node.l.w if node.l else 0.0
            rw = node.r.w if node.r else 0.0
            denom = lw + rw
            mid = lw / denom if denom > 0 else 0.5

            u0 = self._u01(rng)
            u1 = (u0 + 0.5) % 1.0
            same_left = (u0 < mid) and (u1 < mid)
            same_right = (u0 >= mid) and (u1 >= mid)

            if d == "0":
                if not same_left and not same_right:
                    out_bits.append("0")
                node = node.l if node.l is not None else node
            else:
                if not same_left and not same_right:
                    out_bits.append("1")
                node = node.r if node.r is not None else node

            if node.l is None and node.r is None:
                break

        return "".join(out_bits)
