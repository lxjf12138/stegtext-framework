"""DisCop copy-index rule (SP 2023, Algorithms 2–3).

The frozen starter's Huffman tree is retained. Only the bit-to-copy mapping is
corrected; this is not a claim of identical tree tie breaking to the Cython repo.
The probability proof assumes fresh ideal uniforms at each visited node.
"""
from __future__ import annotations

import math

from stegtext.v2.coders.discop import _normal, _tree


CODER_ID = "discop-copy-index-v1"


class DisCopSession:
    def __init__(self):
        self.last_trace = None

    @staticmethod
    def _directions(node, rng):
        if rng is None or not hasattr(rng, "random"):
            raise RuntimeError("DisCop requires an RNG")
        u = float(rng.random())
        if not 0 <= u < 1:
            raise ValueError("DisCop uniform must be in [0,1)")
        cut = node.left.weight / (node.left.weight + node.right.weight)
        rotated = u - .5 if u >= .5 else u + .5
        return int(u >= cut), int(rotated >= cut)

    def encode(self, probs, bit_stream, *, rng):
        weights = _normal(probs)
        node = _tree(weights)
        pos = 0
        depth = 0
        mixed_nodes = 0
        expected_bits = 0.0
        while node.left is not None:
            d0, d1 = self._directions(node, rng)
            cut = node.left.weight / (node.left.weight + node.right.weight)
            expected_bits += 2.0 * min(cut, 1.0 - cut)
            if d0 == d1:
                direction = d0
            else:
                mixed_nodes += 1
                if pos >= len(bit_stream) or bit_stream[pos] not in "01":
                    raise ValueError("Insufficient valid supplied DisCop bits")
                direction = (d0, d1)[int(bit_stream[pos])]
                pos += 1
            node = node.left if direction == 0 else node.right
            depth += 1
        probability = float(weights[node.index])
        self.last_trace = dict(version="discop-step-v1", root_leaf_count=len(weights),
                               tree_depth=depth, mixed_nodes=mixed_nodes,
                               expected_consumed_bits=expected_bits,
                               consumed_bits=pos, selected_action=int(node.index),
                               selected_probability=probability,
                               selected_information_bits=-math.log2(probability) if probability > 0 else None)
        return bit_stream[:pos], node.index

    def decode(self, probs, action, *, rng):
        weights = _normal(probs)
        root = _tree(weights)
        target = int(action)

        def find(node):
            if node.left is None:
                return [] if node.index == target else None
            for direction, child in enumerate((node.left, node.right)):
                tail = find(child)
                if tail is not None:
                    return [direction] + tail
            return None

        path = find(root)
        if path is None:
            raise ValueError(f"Unknown DisCop action {target}")
        node, bits = root, []
        depth = 0
        mixed_nodes = 0
        expected_bits = 0.0
        for direction in path:
            total = node.left.weight + node.right.weight
            cut = node.left.weight / total if total else 0.5
            expected_bits += 2.0 * min(cut, 1.0 - cut)
            d0, d1 = self._directions(node, rng)
            if d0 == d1:
                if direction != d0:
                    raise ValueError("DisCop action contradicts shared random copies")
            else:
                mixed_nodes += 1
                bits.append("0" if direction == d0 else "1")
            node = node.left if direction == 0 else node.right
            depth += 1
        probability = float(weights[target])
        self.last_trace = dict(version="discop-step-v1", root_leaf_count=len(weights),
                               tree_depth=depth, mixed_nodes=mixed_nodes,
                               expected_consumed_bits=expected_bits,
                               consumed_bits=len(bits), selected_action=target,
                               selected_probability=probability,
                               selected_information_bits=-math.log2(probability) if probability > 0 else None)
        return "".join(bits)
