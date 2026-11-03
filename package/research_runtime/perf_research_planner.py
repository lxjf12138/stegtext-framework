"""Exact block-frontier implementation with bulk stable merging.

No support pruning, approximate merging, mass quantization, or policy changes.
The separate frozen reference planner is used for matched parity checks.
"""
from collections import OrderedDict
from bisect import bisect_left, bisect_right
import math
import time

import numpy as np
import torch

from round5_exact import (BlockFrontier, PathBlock, make_exact_planner,
                         prefix_upper, interval, merge_keys, stable_interval_sums, xlog)


_INTERVAL_CACHE_MIN_ATOMS = 4096


class IntervalSumCache:
    """Cache reduction geometry, preserving the original float64 sum order."""
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.geometry = OrderedDict()

    def __call__(self, values, lows, highs):
        if not len(lows):
            return np.empty(0, dtype=np.float64)
        signature = (len(values), np.asarray(lows, dtype=np.int64).tobytes(),
                     np.asarray(highs, dtype=np.int64).tobytes())
        geometry = self.geometry.get(signature)
        if geometry is None:
            edges = np.unique(np.concatenate((lows, highs)))
            edges = edges[edges < len(values)]
            nonempty = highs > lows
            positions = np.searchsorted(edges, lows[nonempty])
            next_edges = np.append(edges[1:], len(values))
            direct = not len(positions) or np.array_equal(next_edges[positions], highs[nonempty])
            slices = None if direct else [
                (i, np.searchsorted(edges, lo), np.searchsorted(edges, hi))
                for i, (lo, hi) in enumerate(zip(lows, highs)) if hi > lo]
            geometry = edges, nonempty, positions, slices
            self.geometry[signature] = geometry
            if len(self.geometry) > self.capacity:
                self.geometry.popitem(last=False)
        else:
            self.geometry.move_to_end(signature)
        edges, nonempty, positions, slices = geometry
        chunks = np.add.reduceat(values, edges) if len(edges) else np.empty(0)
        result = np.zeros(len(lows), dtype=np.float64)
        if slices is None:
            result[nonempty] = chunks[positions]
        else:
            for i, a, b in slices:
                result[i] = chunks[a:b].sum(dtype=np.float64)
        return result


def jumping_minimal_keys(values):
    """Skip each sorted prefix interval by bisection, including duplicates."""
    out = []; i = 0
    while i < len(values):
        key = values[i]
        if not key:
            i += 1
            continue
        out.append(key)
        upper = prefix_upper(key)
        i = len(values) if upper is None else bisect_left(values, upper, i+1)
    return out


class IndexedBytes:
    def __init__(self, values, positions):
        self.values = values; self.positions = positions
    def __len__(self):
        return len(self.positions)
    def __getitem__(self, i):
        return self.values[int(self.positions[i])]


class SuffixTables:
    def __init__(self, capacity=12):
        self.capacity = capacity
        self.tables = OrderedDict()

    def get(self, values, cut):
        key = (id(values), cut)
        entry = self.tables.get(key)
        if entry is None or entry[0] is not values:
            array = np.fromiter((value[cut:] for value in values), dtype=object, count=len(values))
            self.tables[key] = (values, array)
            while len(self.tables) > self.capacity:
                self.tables.popitem(last=False)
        else:
            array = entry[1]
            self.tables.move_to_end(key)
        return array


class BulkBlockFrontier(BlockFrontier):
    def __init__(self, blocks=(), block_ids=(), positions=(), masses=(), *, tables):
        super().__init__(blocks, block_ids, positions, masses)
        self.tables = tables
        self._residual_array = None

    @property
    def residual_array(self):
        if self._residual_array is None:
            out = np.empty(len(self), dtype=object)
            for bid in np.unique(self.block_ids):
                selected = np.flatnonzero(self.block_ids == bid)
                block = self.blocks[int(bid)]
                suffixes = self.tables.get(block.visible, block.cut)
                out[selected] = suffixes[self.positions[selected]]
            self._residual_array = out
        return self._residual_array

    @property
    def residuals(self):
        if self._residuals is None:
            self._residuals = tuple(self.residual_array)
        return self._residuals


def bulk_frontier(old, fresh_positions, raw, lex_ids, lex_bytes, parent, rex, rhi, key, total, tables):
    if isinstance(old, BlockFrontier):
        used = np.unique(old.block_ids[rex:rhi])
        blocks = [PathBlock(old.blocks[int(b)].parent, old.blocks[int(b)].token_ids,
                            old.blocks[int(b)].visible, old.blocks[int(b)].cut+len(key)) for b in used]
        # searchsorted replaces a Python dictionary lookup per retained atom.
        rblocks = np.searchsorted(used, old.block_ids[rex:rhi]).astype(np.int32)
        rpositions = old.positions[rex:rhi]
    else:
        blocks = [PathBlock(old.path_at(j), np.array([-1]), (old.residuals[j],), len(key)) for j in range(rex, rhi)]
        rblocks = np.arange(len(blocks), dtype=np.int32)
        rpositions = np.zeros(len(blocks), dtype=np.int64)
    fresh_bid = len(blocks)
    if len(fresh_positions):
        blocks.append(PathBlock(parent, lex_ids, lex_bytes, len(key)))
    nf = len(fresh_positions); nr = rhi-rex
    bids = np.empty(nf+nr, dtype=np.int32)
    positions = np.empty(nf+nr, dtype=np.int64)
    masses = np.empty(nf+nr, dtype=np.float64)
    if not nr:
        bids[:] = fresh_bid; positions[:] = fresh_positions; masses[:] = raw[fresh_positions]/total
    elif not nf:
        bids[:] = rblocks; positions[:] = rpositions; masses[:] = old.masses[rex:rhi]/total
    else:
        rf = tables.get(lex_bytes, len(key))[fresh_positions]
        rr = np.fromiter((value[len(key):] for value in old.residuals[rex:rhi]), dtype=object, count=nr)
        # Equal suffixes retain the reference order: all fresh atoms first.
        ftargets = np.arange(nf)+np.searchsorted(rr, rf, side="left")
        rtargets = np.arange(nr)+np.searchsorted(rf, rr, side="right")
        bids[ftargets] = fresh_bid; positions[ftargets] = fresh_positions
        masses[ftargets] = raw[fresh_positions]/total
        bids[rtargets] = rblocks; positions[rtargets] = rpositions
        masses[rtargets] = old.masses[rex:rhi]/total
    return BulkBlockFrontier(blocks, bids, positions, masses, tables=tables)


def make_bulk_planner(index):
    reference = make_exact_planner(index, "block")
    from stegtext.v2.methods.las import LASState

    class BulkPlanner(type(reference)):
        def __init__(self):
            super().__init__()
            self.tables = SuffixTables()

        def advance(self, state, plan, action, rng, ctx):
            record = plan.opaque["records"][int(action)]; key = record["key"]; total = record["group_mass"]
            _, hi = record["fresh_interval"]; elo, ehi = record["fresh_exact_interval"]
            rlo, rhi = record["retained_interval"]; rex = record["retained_exact_end"]
            old = plan.opaque["frontier"]; raw = plan.opaque["lex_masses"]
            exact_f = raw[elo:ehi]; exact_r = old.masses[rlo:rex]
            exact_mass = float(exact_f.sum()+exact_r.sum()); draw = float(rng.random())*exact_mass
            if draw < exact_f.sum():
                pos = elo+min(int(np.searchsorted(np.cumsum(exact_f), draw, side="right")), len(exact_f)-1)
                token = int(self.lex_ids[pos]); path = state.base_path.append(token); terminal = ctx.codec.is_eos(token)
            else:
                i = rlo+min(int(np.searchsorted(np.cumsum(exact_r), draw-exact_f.sum(), side="right")), len(exact_r)-1)
                path = old.path_at(i); terminal = bool(old.terminals[i])
            fresh_positions = np.flatnonzero(raw[ehi:hi] > 0)+ehi
            eos_ids = getattr(ctx.codec, "eos_ids", ())
            if eos_ids and np.isin(self.lex_ids[fresh_positions], list(eos_ids)).any():
                raise AssertionError("terminal token cannot have a strict data continuation")
            if not eos_ids and any(ctx.codec.is_eos(int(self.lex_ids[i])) for i in fresh_positions):
                raise AssertionError("terminal token cannot have a strict data continuation")
            if terminal and (len(fresh_positions) or rhi > rex):
                raise AssertionError("terminal-visible conflict")
            frontier = bulk_frontier(old, fresh_positions, raw, self.lex_ids, self.lex_bytes,
                                     state.base_path, rex, rhi, key, total, self.tables)
            return LASState(path, 0. if terminal else exact_mass/total, frontier, state.resolved_bytes+key,
                            bool(terminal), max(0, path.depth-ctx.prompt.path.depth))

    return BulkPlanner()


def make_jump_planner(index):
    bulk = make_bulk_planner(index)
    from stegtext.v2.core.types import DecisionPlan

    class JumpPlanner(type(bulk)):
        def __init__(self):
            super().__init__()
            self.interval_sums = IntervalSumCache()

        def _fresh_keys(self, positive):
            signature = positive.tobytes()
            if signature not in self.geometry:
                self.geometry[signature] = jumping_minimal_keys(IndexedBytes(self.lex_bytes, positive))
                if len(self.geometry) > 32:
                    self.geometry.popitem(last=False)
            return self.geometry[signature]

        def plan(self, state, distributions, ctx):
            # Keep group masses in float64 through coder normalization: products
            # of positive float32 source atoms can fall below float32 support.
            start = time.perf_counter()
            if len(distributions) != int(state.expansion_mass > 0):
                raise ValueError("wrong number of fresh laws")
            dist = distributions[0] if distributions else None
            from round5_exact import LazyFrontier
            frontier = LazyFrontier.from_reference(state.retained)
            transfer = 0
            if dist is None:
                raw = np.zeros(index.vocab_size, dtype=np.float64)
                device_cdf = None; lex_tensor = None
            else:
                ids = index.lex_token_ids_gpu.to(dist.probs.device)
                lex_tensor = dist.probs.index_select(0, ids).to(torch.float64)*float(state.expansion_mass)
                device_cdf = torch.cat((lex_tensor.new_zeros(1), lex_tensor.cumsum(0)))
                raw = lex_tensor.detach().cpu().numpy()
                transfer = int(dist.probs.device.type == "cuda")
                if (raw[self.empty_positions] > 0).any():
                    raise ValueError("positive invisible token")
            positive = np.flatnonzero(raw > 0)
            # Small-support workloads did not consistently benefit in the
            # microbenchmark. This selects a storage policy, never a cutoff.
            reduce_intervals = (self.interval_sums
                                if len(positive) + len(frontier) >= _INTERVAL_CACHE_MIN_ATOMS
                                else stable_interval_sums)
            fresh_start = time.perf_counter(); fresh_keys = self._fresh_keys(positive)
            fresh_ms = (time.perf_counter()-fresh_start)*1000
            retained_start = time.perf_counter()
            keys = merge_keys(fresh_keys, jumping_minimal_keys(frontier.residuals))
            r_ranges = [interval(frontier.residuals, key) for key in keys]
            retained_ms = (time.perf_counter()-retained_start)*1000
            f_ranges = [interval(self.lex_bytes, key) for key in keys]
            lows = np.asarray([r[0] for r in f_ranges]); highs = np.asarray([r[1] for r in f_ranges])
            f_mass = reduce_intervals(raw, lows, highs)
            f_count = reduce_intervals((raw > 0).astype(np.float64), lows, highs).astype(np.int64)
            elo = np.asarray([bisect_left(self.lex_bytes, key, lo, hi) for key, (lo, hi) in zip(keys, f_ranges)])
            ehi = np.asarray([bisect_right(self.lex_bytes, key, lo, hi) for key, (lo, hi) in zip(keys, f_ranges)])
            rlo = np.asarray([pair[0] for pair in r_ranges]); rhi = np.asarray([pair[1] for pair in r_ranges])
            rex = np.asarray([bisect_right(frontier.residuals, key, lo, hi) for key, (lo, hi) in zip(keys, r_ranges)])
            exact_masses = reduce_intervals(raw, elo, ehi)+reduce_intervals(frontier.masses, rlo, rex)
            group_masses = f_mass+reduce_intervals(frontier.masses, rlo, rhi)
            strict_masses = reduce_intervals(raw, ehi, highs)+reduce_intervals(frontier.masses, rex, rhi)
            strict_xlogs = reduce_intervals(xlog(raw), ehi, highs)+reduce_intervals(xlog(frontier.masses), rex, rhi)
            exact_counts = reduce_intervals((raw > 0).astype(np.float64), elo, ehi).astype(np.int64)+(rex-rlo)
            strict_counts = f_count+(rhi-rlo)-exact_counts
            records = []; masses = []
            for j, (key, (lo, hi), (rl, rh)) in enumerate(zip(keys, f_ranges, r_ranges)):
                exact = float(exact_masses[j]); total = float(group_masses[j])
                if total <= 0:
                    continue
                exact_count = int(exact_counts[j])
                if not exact_count:
                    raise AssertionError("positive group has no exact representative")
                strict_xlog = float(strict_xlogs[j]); strict_count = int(strict_counts[j])
                u = max(0., math.log2(total)-((exact*math.log2(exact) if exact > 0 else 0.)+strict_xlog)/total)
                records.append({"key": key, "fresh_interval": (lo, hi), "fresh_exact_interval": (int(elo[j]), int(ehi[j])),
                                "retained_interval": (rl, rh), "retained_exact_end": int(rex[j]),
                                "group_mass": total, "exact_mass": exact, "strict_mass": float(strict_masses[j]),
                                "exact_count": exact_count, "strict_count": strict_count,
                                "strict_xlogx_sum": strict_xlog, "las_posterior_entropy_bits": u})
                masses.append(total)
            if not masses:
                raise AssertionError("empty positive belief")
            probs = torch.as_tensor(masses, dtype=torch.float64,
                                    device=dist.probs.device if dist else "cpu")
            probs = probs/probs.sum()
            self.last_metrics = {"fresh_grouping_ms": fresh_ms, "retained_grouping_ms": retained_ms,
                                 "gpu_to_cpu_transfers": transfer, "groups": len(records),
                                 "planner_ms": (time.perf_counter()-start)*1000, "retained_sort_calls": 0}
            return DecisionPlan(probs, {"records": records, "masses": masses, "lex_masses": raw,
                                       "device_cdf": device_cdf, "lex_tensor": lex_tensor, "frontier": frontier})
    return JumpPlanner()
