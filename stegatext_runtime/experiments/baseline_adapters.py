"""Versioned adapters for the common lossless-byte and fixed-chat channel.

The upstream files and the LAS/coder kernels stay unchanged. Filtering still
changes the sampler law: retained mass is measured explicitly, never hidden.
"""
from __future__ import annotations

from bisect import bisect_left
import math

import torch

from stegtext.v2.core.errors import InvariantError, ReceiverMismatchError
from stegtext.v2.core.types import DecisionPlan
from stegtext.v2.methods.direct import DirectMethod, exact_realization_plan


def byte_classes(distribution, codec):
    """Common exact-realization classes, including one augmented EOS class.

    Classes use summed original masses; candidates within an equal-byte class
    are sampled with the same shared representative rule as native Direct.
    """
    count = int(torch.count_nonzero(distribution.probs).item())
    values, ids = torch.topk(distribution.probs, count)
    keys, members, weights, positions = [], [], [], {}
    for token, weight in zip(ids.cpu().tolist(), values.double().cpu().tolist()):
        key = codec.token_bytes(token)
        if not key:
            raise InvariantError("Positive candidate has no byte realization")
        if key not in positions:
            positions[key] = len(keys)
            keys.append(key)
            members.append([])
            weights.append([])
        i = positions[key]
        members[i].append(token)
        weights[i].append(weight)
    masses = [math.fsum(row) for row in weights]
    order = sorted(range(len(keys)), key=lambda i: (-masses[i], i))
    return ([keys[i] for i in order], [members[i] for i in order],
            [masses[i] for i in order])


def basic_indices(keys):
    """Keep the maximal elements of the strict byte-prefix relation.

    Keys are distinct exact-realization classes. A key has a strict descendant
    iff its immediate lexicographic successor starts with it.
    """
    order = sorted(range(len(keys)), key=keys.__getitem__)
    removed = {a for a, b in zip(order, order[1:]) if keys[b].startswith(keys[a])}
    return [i for i in range(len(keys)) if i not in removed]


def _upper_prefix(key):
    for i in range(len(key) - 1, -1, -1):
        if key[i] < 255:
            return key[:i] + bytes([key[i] + 1])
    return None


class _Counts:
    def __init__(self, n):
        self.tree = [0] * (n + 1)

    def add(self, i):
        i += 1
        while i < len(self.tree):
            self.tree[i] += 1
            i += i & -i

    def prefix(self, i):
        result = 0
        while i:
            result += self.tree[i]
            i -= i & -i
        return result


def mwis_indices(keys, weights):
    """Execute upstream's forest/level-DP approximation on byte classes.

    Prefix intervals count prior neighbours without constructing a quadratic
    graph. Only the induced working forest is materialized. Upstream BFS and
    level DP are called unchanged; parity is tested against its full graph.
    This is not a claim to solve arbitrary MWIS optimally.
    """
    from research_notes.full_validation.suite import load_pools
    load_pools()
    from tokenization_consistency.pools import _bfs_levels, _weighted_level_dp

    ordered = sorted(keys)
    positions = {key: i for i, key in enumerate(ordered)}
    seen, forest, counts = set(), [], _Counts(len(keys))
    for i, key in enumerate(keys):
        lo = positions[key]
        upper = _upper_prefix(key)
        hi = len(keys) if upper is None else bisect_left(ordered, upper)
        earlier_descendants = counts.prefix(hi) - counts.prefix(lo)
        earlier_ancestors = sum(key[:j] in seen for j in range(1, len(key)))
        if earlier_descendants + earlier_ancestors <= 1:
            forest.append(i)
        counts.add(lo)
        seen.add(key)
    adjacency = [[] for _ in keys]
    stack = []
    for i in sorted(forest, key=keys.__getitem__):
        while stack and not keys[i].startswith(keys[stack[-1]]):
            stack.pop()
        for ancestor in stack:
            adjacency[i].append(ancestor)
            adjacency[ancestor].append(i)
        stack.append(i)
    for i in forest:
        adjacency[i].sort()
    remaining, selected = set(forest), []
    for start in forest:
        if start not in remaining:
            continue
        levels, visited = _bfs_levels(adjacency, remaining, start)
        selected.extend(_weighted_level_dp(levels, weights))
        remaining -= visited
    return sorted(selected)


class BytePrefixBaseline(DirectMethod):
    def __init__(self, strategy):
        if strategy not in ("basic", "mwis"):
            raise ValueError("Unknown byte-prefix baseline")
        self.strategy = strategy

    def plan(self, state, distributions, ctx):
        distribution = distributions[0]
        keys, members, masses = byte_classes(distribution, ctx.codec)
        keep = basic_indices(keys) if self.strategy == "basic" else mwis_indices(keys, masses)
        chosen = [keys[i] for i in keep]
        ordered = sorted(chosen)
        if any(b.startswith(a) for a, b in zip(ordered, ordered[1:])):
            raise InvariantError("Byte-prefix baseline failed independence")
        total, retained = math.fsum(masses), math.fsum(masses[i] for i in keep)
        contract = dict(version="lossless-byte-classes-v1", strategy=self.strategy,
            terminal="common_augmented_eos_class", exact_duplicates="common_shared_representative",
            retained_mass=retained / total, source_classes=len(keys), retained_classes=len(keep),
            selector="strict_prefix_maximal" if self.strategy == "basic" else "upstream_forest_level_dp")
        return DecisionPlan(torch.tensor([masses[i] for i in keep], dtype=torch.float64),
            dict(keys=chosen, members=[members[i] for i in keep], dist=distribution,
                 byte_lookup={key: i for i, key in enumerate(chosen)},
                 byte_lengths=sorted(set(map(len, chosen))), baseline_contract=contract))

    def observe(self, state, plan, target, ctx):
        # The action alphabet is prefix-free in bytes, so decoding never needs
        # to interpret a UTF-8 suffix that may start inside a codepoint.
        remaining = target.remaining
        matches = [plan.opaque["byte_lookup"][remaining[:n]]
                   for n in plan.opaque["byte_lengths"]
                   if n <= len(remaining) and remaining[:n] in plan.opaque["byte_lookup"]]
        if len(matches) != 1:
            raise ReceiverMismatchError(f"Byte-prefix receiver found {len(matches)} actions")
        action = matches[0]
        target.consume(plan.opaque["keys"][action])
        return action


class _FixedChatTokenizer:
    """Explicit text channel around upstream's unchanged candidate predicate."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def batch_decode(self, sequences, **kwargs):
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)

    def __call__(self, texts, **kwargs):
        return self.tokenizer(texts, add_special_tokens=False, **kwargs)


class ChatStepwise(DirectMethod):
    """Algorithm 1 on the fixed rendered chat prefix and all sampler support.

    The model still conditions on its original chat IDs. Only the verification
    tokenizer view changes. No fallback samples outside the controlled source.
    EOS is observed through the common augmented delimiter, not deleted text.
    """
    def __init__(self, source):
        from research_notes.full_validation.suite import load_pools
        load_pools()
        from tokenization_consistency.tokenization import candidate_consistency_mask
        self.predicate = candidate_consistency_mask
        self.tokenizer = source.tokenizer
        self.view = _FixedChatTokenizer(source.tokenizer)

    def start(self, prompt_state):
        ids = list(prompt_state.token_ids)
        text = self.tokenizer.decode(ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        if self.tokenizer(text, add_special_tokens=False)["input_ids"] != ids:
            raise InvariantError("Fixed chat prompt is not canonical under the declared text channel")
        return super().start(prompt_state)

    def plan(self, state, distributions, ctx):
        distribution = distributions[0]
        ids, _ = distribution.compact()
        candidates = ids.cpu().tolist()
        data_ids = [i for i in candidates if not ctx.codec.is_eos(i)]
        mask = self.predicate(self.view, list(state.path.token_ids()), data_ids, batch_size=512)
        retained = [i for i, accepted in zip(data_ids, mask.tolist()) if accepted]
        retained += [i for i in candidates if ctx.codec.is_eos(i)]
        if not retained:
            raise InvariantError("Stepwise has no canonical candidate in the controlled sampler support")
        selected = torch.tensor(sorted(retained), dtype=torch.long, device=ids.device)
        plan = exact_realization_plan(selected, distribution, ctx)
        mass = float(distribution.probs.index_select(0, selected).double().sum().item()
                     / distribution.probs.double().sum().item())
        plan.opaque["baseline_contract"] = dict(version="fixed-chat-stepwise-v1",
            predicate="upstream_algorithm_1", verification_skip_special_tokens=False,
            verification_add_special_tokens=False, cleanup_spaces=False,
            candidate_support="all_positive_sampler_tokens", expands_sampler_support=False,
            terminal="common_augmented_eos_class", tested_data_candidates=len(data_ids),
            retained_mass=mass, retained_tokens=len(retained))
        return plan
