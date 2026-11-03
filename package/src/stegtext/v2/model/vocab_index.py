from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class StaticGroup:
    key: bytes
    lo: int
    hi: int
    exact_positions: tuple[int, ...]


class TokenByteIndex:
    """One-time lexicographic prefix geometry for an entire vocabulary."""

    def __init__(self, token_bytes: Iterable[bytes], *, device: Optional[torch.device | str] = None) -> None:
        self.token_bytes = tuple(bytes(x) for x in token_bytes)
        self.vocab_size = len(self.token_bytes)
        order = sorted(range(self.vocab_size), key=lambda i: (self.token_bytes[i], i))
        self.lex_token_ids_cpu = np.asarray(order, dtype=np.int64)
        self._lex_bytes = [self.token_bytes[i] for i in order]
        self.empty_token_ids_cpu = np.asarray(
            [token_id for token_id, value in enumerate(self.token_bytes) if not value],
            dtype=np.int64,
        )
        target = device if device is not None else "cpu"
        self.lex_token_ids_gpu = torch.as_tensor(order, dtype=torch.long, device=target)
        exact_positions: dict[bytes, list[int]] = {}
        for position, token_id in enumerate(order):
            key = self.token_bytes[token_id]
            if key:
                exact_positions.setdefault(key, []).append(position)
        self.exact_ids = {
            key: np.asarray(positions, dtype=np.int64)
            for key, positions in exact_positions.items()
        }
        self.base_groups = self._build_groups()
        self._group_keys = [g.key for g in self.base_groups]

    def _build_groups(self) -> list[StaticGroup]:
        groups: list[StaticGroup] = []
        order = self.lex_token_ids_cpu.tolist()
        i = 0
        while i < len(order):
            key = self.token_bytes[order[i]]
            if not key:
                i += 1
                continue
            lo = i
            exact: list[int] = []
            while i < len(order) and self.token_bytes[order[i]].startswith(key):
                if self.token_bytes[order[i]] == key:
                    exact.append(i)
                i += 1
            # If a previous key is a prefix, this interval belongs to it.
            if groups and key.startswith(groups[-1].key):
                continue
            groups.append(StaticGroup(key=key, lo=lo, hi=i, exact_positions=tuple(exact)))
        return groups

    def token_ids_for_prefix(self, prefix: bytes) -> np.ndarray:
        """Return lexicographic positions whose bytes start with ``prefix``."""
        lo = bisect_left(self._lex_bytes, prefix)
        hi = lo
        while hi < len(self._lex_bytes) and self._lex_bytes[hi].startswith(prefix):
            hi += 1
        return self.lex_token_ids_cpu[lo:hi]

    def dynamic_groups(self, residuals: Iterable[bytes]) -> list[tuple[bytes, list[int], list[bytes]]]:
        """Merge sorted retained residuals with cached static group keys.

        The merge is linear in the number of static groups plus retained paths;
        no vocabulary pairwise prefix loop is used.
        """
        retained = sorted(set(bytes(x) for x in residuals))
        entries = [(key, i) for i, key in enumerate(self._group_keys)]
        merged: list[tuple[bytes, Optional[int]]] = []
        i = j = 0
        while i < len(entries) or j < len(retained):
            if j >= len(retained) or (i < len(entries) and entries[i][0] <= retained[j]):
                merged.append(entries[i]); i += 1
            else:
                merged.append((retained[j], None)); j += 1
        out: list[tuple[bytes, list[int], list[bytes]]] = []
        for key, static_idx in merged:
            if out and key.startswith(out[-1][0]):
                if static_idx is not None:
                    out[-1][1].append(static_idx)
                continue
            out.append((key, [] if static_idx is None else [static_idx], []))
        return out
