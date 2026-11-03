"""Exact-support planner and lazy SOA frontier; frozen reference remains untouched.

CDF subtraction is NOT used as the authoritative mass estimator: it can erase
positive tail intervals. Stable contiguous reductions compute group metadata.
GPU CDF winner selection has a local-interval fallback for cancellation.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from collections.abc import Sequence
import math

import numpy as np
import torch

from round3_common import add_v2_to_path


def prefix_upper(key):
    for i in range(len(key) - 1, -1, -1):
        if key[i] < 255:
            return key[:i] + bytes([key[i] + 1])
    return None


def interval(values, key):
    upper = prefix_upper(key)
    return bisect_left(values, key), len(values) if upper is None else bisect_left(values, upper)


def minimal_keys(sorted_values):
    out = []
    for key in sorted_values:
        if key and (not out or not key.startswith(out[-1])):
            out.append(key)
    return out


def merge_keys(left, right):
    out = []; i = j = 0
    while i < len(left) or j < len(right):
        if j == len(right) or (i < len(left) and left[i] <= right[j]):
            key = left[i]; i += 1
        else:
            key = right[j]; j += 1
        if not out or not key.startswith(out[-1]):
            out.append(key)
    return out


def xlog(values):
    out = np.zeros_like(values, dtype=np.float64)
    positive = values > 0
    out[positive] = values[positive] * np.log2(values[positive])
    return out


class LazyFrontier(Sequence):
    """Sorted contiguous arrays, deferring PathRef.append until a path wins.

    Iteration materializes diagnostic views; production accesses arrays only.
    No positive mass is dropped and no approximate path merging is performed.
    """
    def __init__(self, parents=(), tokens=(), residuals=(), masses=(), terminals=(), *, ordered=False):
        n = len(masses)
        order = np.arange(n) if ordered else np.asarray(sorted(range(n), key=lambda i: residuals[i]), dtype=np.int64)
        self.parents = tuple(parents[int(i)] for i in order)
        self.tokens = np.asarray(tokens, dtype=np.int64)[order]
        self.residuals = tuple(residuals[int(i)] for i in order)
        self.masses = np.asarray(masses, dtype=np.float64)[order]
        self.terminals = np.asarray(terminals, dtype=bool)[order]
        self.path_materializations = 0

    @classmethod
    def from_reference(cls, rows):
        if isinstance(rows, (cls, BlockFrontier)):
            return rows
        return cls([r.path for r in rows], [-1] * len(rows), [r.residual for r in rows],
                   [r.mass for r in rows], [r.terminal for r in rows])

    def __len__(self):
        return len(self.masses)

    def path_at(self, i):
        token = int(self.tokens[i]); parent = self.parents[i]
        if token < 0:
            return parent
        self.path_materializations += 1
        return parent.append(token)

    def __getitem__(self, i):
        from stegtext.v2.methods.las import RetainedPath
        if isinstance(i, slice):
            return tuple(self[j] for j in range(*i.indices(len(self))))
        return RetainedPath(self.path_at(i), self.residuals[i], float(self.masses[i]), bool(self.terminals[i]))

    @property
    def soa_bytes(self):
        return self.tokens.nbytes + self.masses.nbytes + self.terminals.nbytes + 16 * len(self)


@dataclass(frozen=True)
class PathBlock:
    """One parent plus an exact lex-position view, with a consumed prefix."""
    parent: object
    token_ids: object
    visible: object
    cut: int = 0


class BlockFrontier(Sequence):
    """C: exact shared interval descriptors, no per-candidate parent/path tuple.

    Positive support may contain holes: token_ids is then an exact index view,
    not a contiguous-interval approximation. The logical residual order is an
    integer merge map. Masses stay explicit float64, never quantized/pruned.
    Residual byte keys are cached on first planner access, not latent paths.
    """
    def __init__(self, blocks=(), block_ids=(), positions=(), masses=()):
        self.blocks = tuple(blocks)
        self.block_ids = np.asarray(block_ids, dtype=np.int32)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.masses = np.asarray(masses, dtype=np.float64)
        self.terminals = np.zeros(len(self.masses), dtype=bool)
        self.path_materializations = 0
        self._residuals = None

    def __len__(self):
        return len(self.masses)

    @classmethod
    def from_reference(cls, rows):
        if isinstance(rows, cls):
            return rows
        groups = {}; block_ids = []; positions = []; masses = []
        for row in rows:
            parent = row.path.parent if row.path.parent is not None else row.path
            token = row.path.token_id if row.path.parent is not None else -1
            identity = parent.token_ids()
            if identity not in groups:
                groups[identity] = [len(groups),parent,[],[]]
            bid, path, tokens, visible = groups[identity]
            block_ids.append(bid); positions.append(len(tokens)); masses.append(row.mass)
            tokens.append(token); visible.append(row.residual)
            if row.terminal:
                raise AssertionError("terminal atom cannot be a strict data frontier")
        blocks = [PathBlock(path,np.asarray(tokens,dtype=np.int64),tuple(visible)) for bid,path,tokens,visible in groups.values()]
        return cls(blocks,block_ids,positions,masses)

    @property
    def residuals(self):
        if self._residuals is None:
            self._residuals = tuple(self.blocks[int(b)].visible[int(j)][self.blocks[int(b)].cut:]
                                    for b, j in zip(self.block_ids, self.positions))
        return self._residuals

    def path_at(self, i):
        block = self.blocks[int(self.block_ids[i])]
        token = int(block.token_ids[int(self.positions[i])])
        self.path_materializations += 1
        return block.parent if token < 0 else block.parent.append(token)

    def __getitem__(self, i):
        from stegtext.v2.methods.las import RetainedPath
        if isinstance(i, slice):
            return tuple(self[j] for j in range(*i.indices(len(self))))
        return RetainedPath(self.path_at(i), self.residuals[i], float(self.masses[i]), False)

    @property
    def soa_bytes(self):
        return self.block_ids.nbytes + self.positions.nbytes + self.masses.nbytes + self.terminals.nbytes + 64 * len(self.blocks)


def make_block_frontier(old, fresh_positions, raw, lex_ids, lex_bytes, parent, rex, rhi, key, total):
    """Exact two-stream merge sharing immutable path descriptors."""
    if not isinstance(old, BlockFrontier):
        # Frozen real snapshots have concrete paths. Only the selected retained
        # subset is imported; subsequent C updates share these descriptors.
        blocks = [PathBlock(old.path_at(j), np.array([-1]), (old.residuals[j],), len(key))
                  for j in range(rex, rhi)]
        rblocks = np.arange(len(blocks), dtype=np.int32)
        rpositions = np.zeros(len(blocks), dtype=np.int64)
    else:
        used = np.unique(old.block_ids[rex:rhi])
        remap = {int(b): i for i, b in enumerate(used)}
        blocks = [PathBlock(old.blocks[int(b)].parent, old.blocks[int(b)].token_ids,
                            old.blocks[int(b)].visible, old.blocks[int(b)].cut + len(key)) for b in used]
        rblocks = np.array([remap[int(b)] for b in old.block_ids[rex:rhi]], dtype=np.int32)
        rpositions = old.positions[rex:rhi]
    fresh_block = len(blocks)
    if len(fresh_positions):
        # One immutable parent and one token interval/index view for all fresh
        # candidates; no PathRef.append and no per-candidate parent references.
        blocks.append(PathBlock(parent, lex_ids, lex_bytes, len(key)))
    rf = [lex_bytes[int(i)][len(key):] for i in fresh_positions]
    rr = [v[len(key):] for v in old.residuals[rex:rhi]]
    bids = []; positions = []; masses = []; a = b = 0
    while a < len(rf) or b < len(rr):
        if b == len(rr) or (a < len(rf) and rf[a] <= rr[b]):
            pos = int(fresh_positions[a]); bids.append(fresh_block); positions.append(pos)
            masses.append(raw[pos] / total); a += 1
        else:
            bids.append(int(rblocks[b])); positions.append(int(rpositions[b]))
            masses.append(old.masses[rex + b] / total); b += 1
    return BlockFrontier(blocks, bids, positions, masses)


def stable_interval_sums(values, lows, highs):
    """One reduceat pass, no subtraction of large cumulative prefix sums."""
    if not len(lows):
        return np.empty(0, dtype=np.float64)
    edges = np.unique(np.concatenate((lows, highs)))
    edges = edges[edges < len(values)]
    chunks = np.add.reduceat(values, edges) if len(edges) else np.empty(0)
    result = np.zeros(len(lows), dtype=np.float64)
    nonempty = highs > lows
    positions = np.searchsorted(edges, lows[nonempty])
    next_edges = np.append(edges[1:], len(values))
    if not len(positions) or np.array_equal(next_edges[positions], highs[nonempty]):
        result[nonempty] = chunks[positions]
        return result
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        if hi > lo:
            a = np.searchsorted(edges, lo); b = np.searchsorted(edges, hi)
            result[i] = chunks[a:b].sum(dtype=np.float64)
    return result


def make_exact_planner(index, representation="soa"):
    if representation not in ("soa", "block"):
        raise ValueError("representation must be soa or block")
    add_v2_to_path()
    from stegtext.v2.core.types import DecisionPlan
    from stegtext.v2.methods.las import LASMethod, LASState

    class ExactPlanner(LASMethod):
        def __init__(self):
            super().__init__(index, optimized=True)
            self.lex_bytes = tuple(index._lex_bytes)
            self.lex_ids = index.lex_token_ids_cpu
            self.lengths = np.asarray([len(x) for x in self.lex_bytes], dtype=np.int64)
            self.empty_positions = np.flatnonzero(self.lengths == 0)
            self.geometry = OrderedDict()
            self.suffix_cache = OrderedDict()
            self.last_metrics = {}

        def _fresh_keys(self, positive):
            signature = positive.tobytes()
            if signature not in self.geometry:
                self.geometry[signature] = minimal_keys(self.lex_bytes[int(i)] for i in positive)
                if len(self.geometry) > 32:
                    self.geometry.popitem(last=False)
            return self.geometry[signature]

        def plan(self, state, distributions, ctx):
            import time
            start = time.perf_counter()
            if len(distributions) != int(state.expansion_mass > 0):
                raise ValueError("wrong number of fresh laws")
            dist = distributions[0] if distributions else None
            frontier = LazyFrontier.from_reference(state.retained)
            transfer = 0
            if dist is None:
                raw = np.zeros(index.vocab_size, dtype=np.float64)
                device_cdf = None; lex_tensor = None
            else:
                ids = index.lex_token_ids_gpu.to(dist.probs.device)
                lex_tensor = dist.probs.index_select(0, ids).to(torch.float64) * float(state.expansion_mass)
                device_cdf = torch.cat((lex_tensor.new_zeros(1), lex_tensor.cumsum(0)))
                raw = lex_tensor.detach().cpu().numpy()
                transfer = int(dist.probs.device.type == "cuda")
                if (raw[self.empty_positions] > 0).any():
                    raise ValueError("positive invisible token")
            positive = np.flatnonzero(raw > 0)
            fresh_start = time.perf_counter()
            fresh_keys = self._fresh_keys(positive)
            fresh_ms = (time.perf_counter() - fresh_start) * 1000
            retained_start = time.perf_counter()
            # Sorted state is maintained by advance: no sorting in planner.
            keys = merge_keys(fresh_keys, minimal_keys(frontier.residuals))
            r_ranges = [interval(frontier.residuals, key) for key in keys]
            retained_ms = (time.perf_counter() - retained_start) * 1000
            f_ranges = [interval(self.lex_bytes, key) for key in keys]
            lows = np.asarray([r[0] for r in f_ranges]); highs = np.asarray([r[1] for r in f_ranges])
            f_mass = stable_interval_sums(raw, lows, highs)
            f_count = stable_interval_sums((raw > 0).astype(np.float64), lows, highs).astype(np.int64)
            elo = np.asarray([bisect_left(self.lex_bytes, key, lo, hi) for key, (lo, hi) in zip(keys, f_ranges)])
            ehi = np.asarray([bisect_right(self.lex_bytes, key, lo, hi) for key, (lo, hi) in zip(keys, f_ranges)])
            rlo = np.asarray([pair[0] for pair in r_ranges]); rhi = np.asarray([pair[1] for pair in r_ranges])
            rex = np.asarray([bisect_right(frontier.residuals, key, lo, hi) for key, (lo, hi) in zip(keys, r_ranges)])
            f_exact_mass = stable_interval_sums(raw, elo, ehi)
            r_exact_mass = stable_interval_sums(frontier.masses, rlo, rex)
            r_mass = stable_interval_sums(frontier.masses, rlo, rhi)
            exact_masses = f_exact_mass + r_exact_mass
            group_masses = f_mass + r_mass
            strict_masses = stable_interval_sums(raw, ehi, highs) + stable_interval_sums(frontier.masses, rex, rhi)
            strict_xlogs = stable_interval_sums(xlog(raw), ehi, highs) + stable_interval_sums(xlog(frontier.masses), rex, rhi)
            exact_counts = stable_interval_sums((raw > 0).astype(np.float64), elo, ehi).astype(np.int64) + (rex - rlo)
            strict_counts = f_count + (rhi - rlo) - exact_counts
            records = []; masses = []
            for j, (key, (lo, hi), (rl, rh)) in enumerate(zip(keys, f_ranges, r_ranges)):
                exact = float(exact_masses[j]); total = float(group_masses[j])
                if total <= 0:
                    continue
                exact_count = int(exact_counts[j])
                if not exact_count:
                    raise AssertionError("positive group has no exact representative")
                strict_xlog = float(strict_xlogs[j]); strict_count = int(strict_counts[j])
                u = max(0., math.log2(total) - ((exact * math.log2(exact) if exact > 0 else 0.) + strict_xlog) / total)
                records.append({"key": key, "fresh_interval": (lo, hi), "fresh_exact_interval": (int(elo[j]), int(ehi[j])),
                                "retained_interval": (rl, rh), "retained_exact_end": int(rex[j]),
                                "group_mass": total, "exact_mass": exact, "strict_mass": float(strict_masses[j]),
                                "exact_count": exact_count, "strict_count": strict_count,
                                "strict_xlogx_sum": strict_xlog, "las_posterior_entropy_bits": u})
                masses.append(total)
            if not masses:
                raise AssertionError("empty positive belief")
            probs = torch.as_tensor(masses, dtype=dist.probs.dtype if dist else torch.float64,
                                    device=dist.probs.device if dist else "cpu")
            probs = probs / probs.sum()
            self.last_metrics = {"fresh_grouping_ms": fresh_ms, "retained_grouping_ms": retained_ms,
                                 "gpu_to_cpu_transfers": transfer, "groups": len(records),
                                 "planner_ms": (time.perf_counter() - start) * 1000,
                                 "retained_sort_calls": 0}
            return DecisionPlan(probs, {"records": records, "masses": masses, "lex_masses": raw,
                                       "device_cdf": device_cdf, "lex_tensor": lex_tensor, "frontier": frontier})

        def _sample_fresh(self, plan, lo, hi, draw):
            weights = plan.opaque["lex_masses"][lo:hi]
            total = float(weights.sum())
            cdf = plan.opaque["device_cdf"]
            if cdf is not None and cdf.device.type == "cuda":
                # The global CDF may lose an arbitrarily tiny positive tail.
                # Compare both endpoints in ONE scalar transfer; use local CDF
                # if subtraction changes interval mass beyond float tolerance.
                bounds = cdf[[lo, hi]].detach().cpu().numpy()
                span = float(bounds[1] - bounds[0])
                if span > 0 and abs(span - total) <= 1e-12 * total:
                    value = cdf.new_tensor(float(bounds[0]) + float(draw))
                    chosen = int(torch.searchsorted(cdf[lo + 1:hi + 1], value, right=True).item())
                    if chosen < hi - lo and weights[chosen] > 0:
                        return lo + chosen
                local = plan.opaque["lex_tensor"][lo:hi].cumsum(0)
                chosen = int(torch.searchsorted(local, local.new_tensor(float(draw)), right=True).item())
            else:
                chosen = int(np.searchsorted(np.cumsum(weights), draw, side="right"))
            if chosen >= len(weights):
                chosen = int(np.flatnonzero(weights > 0)[-1])
            return lo + chosen

        def advance_fullsync(self, state, plan, action, rng, ctx):
            record = plan.opaque["records"][int(action)]
            lo, hi = record["fresh_interval"]; rlo, rhi = record["retained_interval"]
            frontier = plan.opaque["frontier"]; raw = plan.opaque["lex_masses"]
            fresh_mass = float(raw[lo:hi].sum()); total = fresh_mass + float(frontier.masses[rlo:rhi].sum())
            draw = float(rng.random()) * total
            if draw < fresh_mass:
                pos = self._sample_fresh(plan, lo, hi, draw); token = int(self.lex_ids[pos])
                path = state.base_path.append(token); visible = self.lex_bytes[pos]; terminal = ctx.codec.is_eos(token)
            else:
                i = rlo + min(int(np.searchsorted(np.cumsum(frontier.masses[rlo:rhi]), draw - fresh_mass, side="right")), rhi - rlo - 1)
                path = frontier.path_at(i); visible = frontier.residuals[i]; terminal = bool(frontier.terminals[i])
            if ctx.mode == "decode":
                ctx.target.consume(visible[len(record["key"]):])
            return LASState(path, 0. if terminal else 1., LazyFrontier(), state.resolved_bytes + visible,
                            bool(terminal), max(0, path.depth - ctx.prompt.path.depth))

        def advance(self, state, plan, action, rng, ctx):
            record = plan.opaque["records"][int(action)]; key = record["key"]; total = record["group_mass"]
            lo, hi = record["fresh_interval"]; elo, ehi = record["fresh_exact_interval"]
            rlo, rhi = record["retained_interval"]; rex = record["retained_exact_end"]
            old = plan.opaque["frontier"]; raw = plan.opaque["lex_masses"]
            exact_f = raw[elo:ehi]; exact_r = old.masses[rlo:rex]
            exact_mass = float(exact_f.sum() + exact_r.sum()); draw = float(rng.random()) * exact_mass
            if draw < exact_f.sum():
                pos = elo + min(int(np.searchsorted(np.cumsum(exact_f), draw, side="right")), len(exact_f) - 1)
                token = int(self.lex_ids[pos]); path = state.base_path.append(token); terminal = ctx.codec.is_eos(token)
            else:
                i = rlo + min(int(np.searchsorted(np.cumsum(exact_r), draw - exact_f.sum(), side="right")), len(exact_r) - 1)
                path = old.path_at(i); terminal = bool(old.terminals[i])
            fresh_positions = np.flatnonzero(raw[ehi:hi] > 0) + ehi
            eos_ids = getattr(ctx.codec, "eos_ids", ())
            strict_eos = (np.isin(self.lex_ids[fresh_positions], list(eos_ids)).any() if eos_ids
                          else any(ctx.codec.is_eos(int(self.lex_ids[i])) for i in fresh_positions))
            if strict_eos:
                raise AssertionError("terminal token cannot have a strict data continuation")
            if terminal and (len(fresh_positions) or rhi > rex):
                raise AssertionError("terminal-visible conflict")
            if representation == "block":
                frontier = make_block_frontier(old, fresh_positions, raw, self.lex_ids,
                                               self.lex_bytes, state.base_path, rex, rhi, key, total)
                return LASState(path, 0. if terminal else exact_mass / total, frontier,
                                state.resolved_bytes + key, bool(terminal), max(0, path.depth - ctx.prompt.path.depth))
            if key not in self.suffix_cache:
                self.suffix_cache[key] = {i: self.lex_bytes[i][len(key):] for i in range(ehi, hi)}
                if len(self.suffix_cache) > 16:
                    self.suffix_cache.popitem(last=False)
            suffix = self.suffix_cache[key]
            residual_f = [suffix[int(i)] for i in fresh_positions]
            residual_r = [value[len(key):] for value in old.residuals[rex:rhi]]
            # Two already-sorted streams, linear merge; no R log R sort.
            parents = []; tokens = []; residuals = []; masses = []; terminals = []
            a = b = 0
            while a < len(fresh_positions) or b < len(residual_r):
                if b == len(residual_r) or (a < len(fresh_positions) and residual_f[a] <= residual_r[b]):
                    pos = int(fresh_positions[a]); parents.append(state.base_path); tokens.append(int(self.lex_ids[pos]))
                    residuals.append(residual_f[a]); masses.append(float(raw[pos] / total)); terminals.append(False); a += 1
                else:
                    j = rex + b; parents.append(old.parents[j]); tokens.append(int(old.tokens[j]))
                    residuals.append(residual_r[b]); masses.append(float(old.masses[j] / total)); terminals.append(bool(old.terminals[j])); b += 1
            if terminal and masses:
                raise AssertionError("terminal-visible conflict")
            frontier = LazyFrontier(parents, tokens, residuals, masses, terminals, ordered=True)
            return LASState(path, 0. if terminal else exact_mass / total, frontier,
                            state.resolved_bytes + key, bool(terminal), max(0, path.depth - ctx.prompt.path.depth))

    return ExactPlanner()
