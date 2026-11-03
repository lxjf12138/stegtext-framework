"""KL/TV diagnostics for a fixed conditional sampler step.

These helpers describe the probability space explicitly.  They are intended
for experiment instrumentation and offline audits; they do not affect an
action, a random stream, or a model probability.

For a filter that keeps source mass ``m`` and renormalizes the kept support,
q(x) = p(x) / m on A and q(x) = 0 outside A.  Consequently

    D_KL(q || p) = -log(m),
    D_KL(p || q) = +infinity when m < 1,
    TV(p, q) = 1 - m.

The reverse divergence is deliberately represented as ``None`` in JSON-facing
records rather than as a non-standard NaN/Infinity value.
"""
from __future__ import annotations

import math
from typing import Optional


def restricted_distribution_metrics(retained_mass: float, *, tolerance: float = 1e-12) -> dict:
    """Return exact closed-form diagnostics for support filtering.

    ``retained_mass`` is the source probability mass retained before
    renormalization.  The function rejects invalid masses instead of silently
    clipping them, because an invalid mass usually indicates a planner or
    trace-contract bug.
    """
    mass = float(retained_mass)
    if not math.isfinite(mass) or mass < 0.0 or mass > 1.0 + tolerance:
        raise ValueError(f"retained_mass must lie in [0,1], got {retained_mass!r}")
    if mass > 1.0:
        mass = 1.0
    if mass <= 0.0:
        return {
            "source_probability_mass_retained": mass,
            "kl_method_to_source_bits": None,
            "kl_source_to_method_bits": None,
            "tv_distance": 1.0,
            "support_mismatch": True,
            "finite_forward_kl": False,
            "reverse_kl_infinite": True,
        }
    forward = 0.0 if abs(mass - 1.0) <= tolerance else -math.log2(mass)
    return {
        "source_probability_mass_retained": mass,
        "kl_method_to_source_bits": forward,
        # Any deleted positive-mass source event has q=0, hence D(p||q)=∞.
        "kl_source_to_method_bits": 0.0 if abs(mass - 1.0) <= tolerance else None,
        "tv_distance": 0.0 if abs(mass - 1.0) <= tolerance else 1.0 - mass,
        "support_mismatch": mass < 1.0 - tolerance,
        "finite_forward_kl": True,
        "reverse_kl_infinite": mass < 1.0 - tolerance,
    }


def discrete_metrics(source: list[float], method: list[float], *, tolerance: float = 1e-12) -> dict:
    """Compute both KL orientations and TV on one common finite action space.

    This function is only for cases where both vectors are explicitly defined
    on the same actions.  Zero-probability support mismatches are handled with
    the usual extended-real convention and returned as ``None`` for JSON.
    """
    if len(source) != len(method) or not source:
        raise ValueError("source and method must be nonempty vectors of equal length")
    ps, qs = [float(x) for x in source], [float(x) for x in method]
    if any((not math.isfinite(x) or x < 0.0) for x in ps + qs):
        raise ValueError("probability vectors must be finite and nonnegative")
    sp, sq = math.fsum(ps), math.fsum(qs)
    if sp <= 0.0 or sq <= 0.0:
        raise ValueError("probability vectors must have positive mass")
    ps = [x / sp for x in ps]
    qs = [x / sq for x in qs]
    p_to_q = 0.0
    q_to_p = 0.0
    reverse_infinite = False
    forward_infinite = False
    tv = 0.0
    for p, q in zip(ps, qs):
        tv += abs(p - q)
        if p > 0.0:
            if q <= 0.0:
                reverse_infinite = True
            else:
                p_to_q += p * math.log2(p / q)
        if q > 0.0:
            if p <= 0.0:
                forward_infinite = True
            else:
                q_to_p += q * math.log2(q / p)
    return {
        "kl_method_to_source_bits": None if forward_infinite else max(0.0, q_to_p),
        "kl_source_to_method_bits": None if reverse_infinite else max(0.0, p_to_q),
        "tv_distance": 0.5 * tv,
        "support_mismatch": forward_infinite or reverse_infinite,
        "finite_forward_kl": not forward_infinite,
        "reverse_kl_infinite": reverse_infinite,
    }

