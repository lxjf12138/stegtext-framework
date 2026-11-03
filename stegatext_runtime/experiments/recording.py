"""Diagnostic storage only: canonical hashes and reconstructible large traces.

No action, posterior, RNG or sampler decision depends on this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from array import array
import sys


TRACE_LIMIT = 4096
TRACE_VERSION = "reconstructible-imec-trace-v1"
SUMMARY_VERSION = "summary-group-distribution-v2"
SINGLE_GENERATION_VERSION = "single-generation-recording-v2"


def _flat_probability_values(values):
    """Return a small, detached view for scalar probability diagnostics.

    The experiment already materializes a probability vector for the sampler.
    Recording uses that vector read-only and never serializes it.  Keeping the
    conversion here makes the recorder usable with both torch tensors and the
    finite-source Python lists used by the CPU contracts.
    """
    if hasattr(values, "detach"):
        return values.detach().double().flatten()
    return [float(value) for value in values]


def _scalar_sum(values):
    if hasattr(values, "sum"):
        return float(values.sum().item())
    return math.fsum(values)


def probability_summary(values, *, selected=None, prefix="distribution", preview=8):
    """Record bounded scalar facts about one probability vector.

    ``selected`` is an index in the vector, when available.  The returned
    preview is at most ``preview`` entries and is useful for diagnosing a
    changed top-k or temperature contract without writing a vocabulary-sized
    JSON record.
    """
    vector = _flat_probability_values(values)
    size = int(vector.numel()) if hasattr(vector, "numel") else len(vector)
    total = _scalar_sum(vector)
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{prefix} probability vector has invalid mass {total!r}")
    if hasattr(vector, "__getitem__"):
        positive = vector[vector > 0] if hasattr(vector, "numel") else [x for x in vector if x > 0]
    else:
        positive = [x for x in vector if x > 0]
    positive_count = int(positive.numel()) if hasattr(positive, "numel") else len(positive)
    if positive_count:
        if not hasattr(positive, "numel"):
            entropy = -math.fsum(float(x) * math.log2(float(x)) for x in positive)
        else:
            entropy = 0.0
    else:
        entropy = 0.0
    # Avoid importing torch at module import time.  Tensor entropy is kept in
    # this small local branch and has no effect on the sampling path.
    if hasattr(positive, "numel"):
        import torch
        entropy = float((-positive * torch.log2(positive)).sum().item())
        max_probability = float(positive.max().item()) if positive_count else 0.0
        min_probability = float(positive.min().item()) if positive_count else None
        if positive_count:
            k = min(int(preview), positive_count)
            top_values, top_indices = torch.topk(vector, k=k, largest=True, sorted=True)
            top_preview = [[int(i), float(v)] for v, i in zip(top_values.tolist(), top_indices.tolist()) if v > 0]
        else:
            top_preview = []
    else:
        max_probability = max(float(x) for x in positive) if positive_count else 0.0
        min_probability = min(float(x) for x in positive) if positive_count else None
        top_preview = [[i, float(x)] for i, x in sorted(
            ((i, float(x)) for i, x in enumerate(vector) if x > 0),
            key=lambda item: (-item[1], item[0]))[:int(preview)]]
    result = {
        f"{prefix}_vector_size": size,
        f"{prefix}_probability_sum": total,
        f"{prefix}_normalization_error": abs(total - 1.0),
        f"{prefix}_support_count": positive_count,
        f"{prefix}_entropy_bits": entropy / total + math.log2(total),
        f"{prefix}_max_probability": max_probability / total,
        f"{prefix}_min_positive_probability": (min_probability / total if min_probability is not None else None),
        f"{prefix}_top_preview": top_preview,
    }
    if selected is not None:
        index = int(selected)
        if index < 0 or index >= size:
            raise IndexError(f"{prefix} selected index outside probability vector")
        value = float(vector[index].item()) if hasattr(vector[index], "item") else float(vector[index])
        normalized = value / total
        if normalized > 0:
            if hasattr(vector, "numel"):
                greater = int((vector > value).sum().item())
                equal_before = int((vector[:index] == value).sum().item())
            else:
                greater = sum(x > value for x in vector)
                equal_before = sum(x == value for x in vector[:index])
            rank = greater + equal_before + 1
        else:
            rank = None
        result.update({f"{prefix}_selected_index": index,
                       f"{prefix}_selected_probability": normalized,
                       f"{prefix}_selected_information_bits": -math.log2(normalized) if normalized > 0 else None,
                       f"{prefix}_selected_rank": rank,
                       f"{prefix}_selected_is_argmax": bool(rank == 1) if rank is not None else False})
    return result


def _mass_entropy(masses):
    values = [float(m) for m in masses if float(m) > 0]
    total = math.fsum(values)
    if not values or total <= 0:
        return 0.0
    return -math.fsum((m / total) * math.log2(m / total) for m in values)


def state_summary(state):
    """Bounded posterior/frontier summary for one runtime state."""
    frontier = getattr(state, "retained", ())
    masses = getattr(frontier, "masses", None)
    if masses is None:
        masses = [float(item.mass) for item in frontier]
    else:
        masses = [float(item) for item in masses]
    active = float(getattr(state, "expansion_mass", 0.0))
    retained = math.fsum(masses)
    return {
        "active_mass": active,
        "retained_frontier_count": len(masses),
        "retained_frontier_mass": retained,
        "posterior_total_mass": active + retained,
        "posterior_mass_error": abs(active + retained - (0.0 if getattr(state, "terminal", False) else 1.0)),
        "retained_identity_entropy_bits": _mass_entropy(masses),
        "resolved_bytes": len(getattr(state, "resolved_bytes", b"")),
        "tokens_used": int(getattr(state, "tokens_used", 0)),
        "terminal": bool(getattr(state, "terminal", False)),
    }


def scalar_mapping(value, *, limit=32, exclude_timing=False):
    """Keep only JSON-safe scalar diagnostics from a stats mapping."""
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, item in value.items():
        if len(out) >= limit:
            break
        if exclude_timing and (str(key).endswith("_ms") or str(key).endswith("_seconds")):
            continue
        if isinstance(item, (bool, int, float, str)) or item is None:
            out[str(key)] = item
    return out


def canonical_digest(value):
    """Same bytes as config.canonical(), without building a whole JSON string."""
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256()
    for part in encoder.iterencode(value):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def coder_trace(trace):
    """Keep small traces byte-equivalent; large matrices are rebuilt for audit.

    The experiment trace separately retains every original float group weight.
    The sparse prior and the declared coupling strategy reconstruct the exact
    Fraction marginals/edges; their canonical hashes detect any difference.
    """
    if "coupling_edges" not in trace or len(trace.get("group_probabilities", ())) <= TRACE_LIMIT:
        return trace
    large = ("group_probabilities", "coupling_edges")
    result = {k: v for k, v in trace.items() if k not in large}
    result["storage"] = dict(version=TRACE_VERSION,
        group_probabilities_count=len(trace["group_probabilities"]),
        group_probabilities_sha256=canonical_digest(trace["group_probabilities"]),
        coupling_edges_count=len(trace["coupling_edges"]),
        coupling_edges_sha256=canonical_digest(trace["coupling_edges"]),
        reconstruction="exact normal(saved group weights), sparse posterior_before, declared coupling strategy")
    return result


def distribution_trace(probabilities, action, *, full_limit=TRACE_LIMIT, preview=8, raw_entropy=None):
    """O(preview) stored data; the full vector is used only by the algorithm.

    A binary64 hash identifies the coder input without storing or formatting
    all its entries. Exact CPU replay of a summary requires the model/source;
    it is deliberately not advertised as a self-contained probability trace.
    """
    if len(probabilities) <= full_limit:
        return dict(group_probabilities=probabilities)
    total = math.fsum(probabilities)
    if raw_entropy is None:
        raw_entropy = -math.fsum(x*math.log2(x) for x in probabilities if x > 0)
    values = array('d', probabilities)
    if sys.byteorder != 'little':
        values.byteswap()
    return dict(group_distribution=dict(version=SUMMARY_VERSION,
        count=len(probabilities), positive_count=sum(p > 0 for p in probabilities),
        weights_sum=total, entropy_bits_normalized=raw_entropy/total+math.log2(total),
        selected_probability_normalized=probabilities[int(action)]/total,
        weights_binary64le_sha256=hashlib.sha256(values).hexdigest(),
        head=[[j, probabilities[j]] for j in range(min(preview, len(probabilities)))],
        full_vector_recorded=False, replay='rerun frozen source and random streams'))


def ends_with_object(handle):
    """Reject a truncated diagnostic before allocating its possibly huge body."""
    handle.seek(0, 2)
    position = handle.tell()
    while position:
        begin = max(0, position - 8192)
        handle.seek(begin)
        tail = handle.read(position - begin).rstrip()
        if tail:
            return tail.endswith(b"}")
        position = begin
    return False
