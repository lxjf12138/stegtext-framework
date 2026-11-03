"""Reviewed LOCAL window iMEC: exact rational marginals and strict confirmation.

The greedy coupling is an entropy heuristic, not an exact MEC optimizer. This
adapter preserves the frozen local block/window protocol, not the upstream
complete-message protocol. Fractions encode the supplied machine probabilities
exactly; no positive posterior atom is removed by a numerical tolerance.
"""
from __future__ import annotations

from fractions import Fraction as F
import heapq
import math

import torch


def normal(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().double().cpu().flatten().tolist()
    p = tuple(v if isinstance(v, F) else F(float(v)) for v in values)
    if not p or any(v < 0 for v in p) or sum(p) <= 0:
        raise ValueError('Invalid iMEC distribution')
    total = sum(p)
    return tuple(v/total for v in p)


def entropy(p):
    return -math.fsum(float(x)*(math.log2(x.numerator)-math.log2(x.denominator)) for x in p if x)


def coupling(p, q, strategy="greedy"):
    if strategy == "prefix":
        return ordered_coupling(p, q)
    if strategy != "greedy":
        raise ValueError("Unknown coupling strategy")
    p, q = normal(p), normal(q)
    rows, columns = [dict() for _ in p], [dict() for _ in q]
    a, b = [(-v, i) for i, v in enumerate(p) if v], [(-v, i) for i, v in enumerate(q) if v]
    heapq.heapify(a); heapq.heapify(b)
    while a and b:
        nr, i = heapq.heappop(a)
        nc, j = heapq.heappop(b)
        r, c = -nr, -nc
        mass = min(r, c)
        rows[i][j] = mass
        columns[j][i] = mass
        if r > mass:
            heapq.heappush(a, (-(r-mass), i))
        if c > mass:
            heapq.heappush(b, (-(c-mass), j))
    assert not a and not b
    assert all(sum(row.values()) == mass for row, mass in zip(rows, p))
    assert all(sum(col.values()) == mass for col, mass in zip(columns, q))
    return p, q, rows, columns


def ordered_coupling(p, q):
    """Northwest-corner transport in MSB row order; no optimality claim."""
    p, q = normal(p), normal(q)
    rows, columns = [dict() for _ in p], [dict() for _ in q]
    remaining = list(q)
    j = 0
    for i, mass in enumerate(p):
        while mass:
            while not remaining[j]:
                j += 1
            take = min(mass, remaining[j])
            rows[i][j] = take
            columns[j][i] = take
            mass -= take
            remaining[j] -= take
    assert not any(remaining)
    return p, q, rows, columns


def support_prefix(p, block_bits):
    """Exact LCP: every strictly positive atom counts, however small."""
    words = [format(i, f'0{block_bits}b') for i, mass in enumerate(p) if mass > 0]
    if not words:
        raise ValueError('Empty posterior support')
    a, b = min(words), max(words)
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), block_bits)
    return a[:n]


def sample(values, rng):
    """Refine a binary uniform interval until it lies inside one positive cell.

    Uses iid uniform 53-bit blocks in the ideal model, and the existing shared
    CounterRNG in experiments. Unlike a single fixed-grid CDF lookup, boundary
    overlap requests another block, so there is no deliberate probability tail
    cutoff. PRF security is separate from this exact-sampling construction.
    """
    p = normal(values)
    edges, total = [], F(0)
    for i, mass in enumerate(p):
        if mass:
            edges.append((i, total, total+mass))
            total += mass
    if len(edges) == 1:
        return edges[0][0]
    numerator, denominator = 0, 1
    while True:
        value = float(rng.random())
        bits = value*(1 << 53)
        if not 0 <= value < 1 or not bits.is_integer():
            raise ValueError('iMEC sampler requires uniform 53-bit-grid random blocks')
        numerator = (numerator << 53)+int(bits)
        denominator <<= 53
        lo, hi = F(numerator, denominator), F(numerator+1, denominator)
        for i, left, right in edges:
            if left <= lo and hi <= right:
                return i


def sparse(p):
    return [[i, str(v)] for i, v in enumerate(p) if v]


class IMECSession:
    def __init__(self, block_size=6, window_size=3, mode='encode', *,
                 coupling_strategy='greedy', schedule='max_entropy', drain_all=False):
        if block_size <= 0 or window_size <= 0 or mode not in ('encode', 'decode'):
            raise ValueError('Invalid local iMEC parameters')
        if coupling_strategy not in ('greedy', 'prefix') or schedule not in ('max_entropy', 'head_first'):
            raise ValueError('Invalid iMEC strategy')
        self.coupling_strategy, self.schedule, self.drain_all = coupling_strategy, schedule, drain_all
        self.block_size, self.window_size, self.mode = block_size, window_size, mode
        self.uniform = tuple([F(1, 1 << block_size)]*(1 << block_size))
        self.beliefs = [self.uniform for _ in range(window_size)]
        self.head, self.last_trace = 0, None
        # Diagnostic policy only. Do not serialize a large rational coupling
        # matrix at every step. Full details remain opt-in for small audits.
        # 256 keeps ordinary controlled runs compact while preserving the
        # complete trace for the small finite-source contracts.
        self.trace_action_limit, self.trace_preview = 256, 8

    def set_trace_policy(self, *, max_full_actions=256, preview=8):
        if (max_full_actions is not None and max_full_actions < 0) or preview < 0:
            raise ValueError('Invalid diagnostic trace policy')
        self.trace_action_limit, self.trace_preview = max_full_actions, preview

    def state(self):
        return dict(head_block=self.head, block_bits=self.block_size, window_blocks=self.window_size,
                    beliefs=[sparse(p) for p in self.beliefs],
                    entropies=[entropy(p) for p in self.beliefs])

    def _prepare(self, probs):
        before = [entropy(p) for p in self.beliefs]
        index = 0 if self.schedule == 'head_first' else max(range(self.window_size), key=lambda i: before[i])
        p, q, rows, columns = coupling(self.beliefs[index], probs, self.coupling_strategy)
        joint_entropy = entropy(v for row in rows for v in row.values())
        prior_entropy, group_entropy = entropy(p), entropy(q)
        row_error = max((abs(sum(row.values()) - p[i]) for i, row in enumerate(rows)), default=F(0))
        column_error = max((abs(sum(column.values()) - q[j]) for j, column in enumerate(columns)), default=F(0))
        joint_mass = sum(value for row in rows for value in row.values())
        self.last_trace = dict(selected_window_index=index, selected_block=self.head+index,
            head_before=self.head, window_entropy_before=before, posterior_before=sparse(p),
            group_probabilities=None, group_entropy_bits=group_entropy,
            coupling_entropy_bits=joint_entropy, expected_information_bits=prior_entropy+group_entropy-joint_entropy,
            coupling_edges=None, exact_marginals=True,
            row_normalization_error=float(row_error), column_normalization_error=float(column_error),
            coupling_mass_error=float(abs(joint_mass - 1)), coupling_support_rows=sum(bool(row) for row in rows),
            coupling_support_columns=sum(bool(column) for column in columns))
        if self.trace_action_limit is None or len(q) <= self.trace_action_limit:
            self.last_trace.update(group_probabilities=[str(v) for v in q],
                coupling_edges=[[i,j,str(v)] for i,row in enumerate(rows) for j,v in sorted(row.items())])
        else:
            self.last_trace.pop('group_probabilities')
            self.last_trace.pop('coupling_edges')
            # Construct only the requested prefix, not a full list to slice.
            self.last_trace.update(storage=dict(version='summary-imec-trace-v2',
                group_count=len(q), coupling_edge_count=sum(map(len, rows)),
                probability_head=[[j, str(q[j])] for j in range(min(self.trace_preview, len(q)))],
                detail='full rational probability vector and coupling edges were not materialized'))
        return index, p, q, rows, columns

    def _update(self, index, q, columns, action, target=None):
        if not 0 <= action < len(q) or not q[action]:
            raise ValueError('iMEC selected a zero-probability action')
        column = columns[action]
        self.beliefs[index] = tuple(column.get(i, F(0))/q[action] for i in range(len(self.uniform)))
        if target is not None and not self.beliefs[index][target]:
            raise ValueError('iMEC lost the true message atom')
        self.last_trace.update(action=action, posterior_after=sparse(self.beliefs[index]),
                               window_entropy_after_update=[entropy(p) for p in self.beliefs])
        self.last_trace.update(selected_group_probability=str(q[action]),
                               posterior_entropy_before_update=self.last_trace['window_entropy_before'][index],
                               posterior_entropy_after_update=entropy(self.beliefs[index]),
                               posterior_information_revealed_bits=(
                                   self.last_trace['window_entropy_before'][index] - entropy(self.beliefs[index])))
        if self.last_trace.get('storage', {}).get('version') == 'summary-imec-trace-v2':
            self.last_trace['storage']['selected_group_probability'] = str(q[action])
        out = ''
        while True:
            possible = [i for i, mass in enumerate(self.beliefs[0]) if mass > 0]
            if len(possible) != 1:
                break
            out += format(possible[0], f'0{self.block_size}b')
            self.beliefs.pop(0)
            self.beliefs.append(self.uniform)
            self.head += 1
            if not self.drain_all:
                break
        self.last_trace.update(committed_bits=out, head_after=self.head,
                               window_entropy_after_roll=[entropy(p) for p in self.beliefs])
        return out

    def finalize_confirmed_prefix(self):
        """Terminal-only view, idempotent: never advance the resumable window."""
        pieces = []
        for belief in self.beliefs:
            prefix = support_prefix(belief, self.block_size)
            pieces.append(prefix)
            if len(prefix) < self.block_size:
                break
        return ''.join(pieces)

    def prefix_audit(self):
        tail = self.finalize_confirmed_prefix()
        block, bit = divmod(len(tail), self.block_size)
        support = self.beliefs[block] if block < self.window_size else self.uniform
        known = sum(sum(len({(i >> k) & 1 for i, v in enumerate(p) if v > 0}) == 1
                        for k in range(self.block_size)) for p in self.beliefs)
        return dict(message_bit_order='MSB-first', terminal_extra_prefix_bits=len(tail),
                    first_unconfirmed_bit=self.head*self.block_size+len(tail),
                    first_unconfirmed_values=sorted({(i >> (self.block_size-1-bit)) & 1
                                                    for i, v in enumerate(support) if v > 0}),
                    unresolved_head_support=sparse(support), pending_blocks=self.state(),
                    known_noncontiguous_bits=known-len(tail),
                    zero_payload_reason='head_first_bit_ambiguous' if not tail else None)

    def encode(self, group_probs, bit_stream, *, rng):
        required = self.block_size*self.window_size
        if len(bit_stream) < required or set(bit_stream[:required]) - {'0', '1'}:
            raise ValueError('iMEC requires a full unexhausted lookahead window')
        index, p, q, rows, columns = self._prepare(group_probs)
        begin = index*self.block_size
        target = int(bit_stream[begin:begin+self.block_size], 2)
        if not p[target]:
            raise ValueError('True message has zero prior posterior mass')
        row = rows[target]
        action = sample(tuple(row.get(j,F(0))/p[target] for j in range(len(q))), rng)
        out = self._update(index, q, columns, action, target)
        if out != bit_stream[:len(out)]:
            raise ValueError('iMEC confirmed the wrong message prefix')
        return out, action

    def decode(self, group_probs, action, *, rng):
        index, p, q, rows, columns = self._prepare(group_probs)
        return self._update(index, q, columns, int(action))
