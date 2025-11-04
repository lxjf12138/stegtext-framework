
from __future__ import annotations
from typing import List, Dict, Any
import math

def entropy(probs: List[float])->float:
    s=0.0
    for p in probs:
        if p>0: s += -p*math.log2(p)
    return s

def compute_metrics_from_trace(trace: List[Dict[str, Any]])->Dict[str, Any]:
    # Aggregates step-level stats
    bits_total = sum(len(e.get("bits", "")) for e in trace)
    steps = len(trace)
    groups = [len(e.get("group_probs", [])) for e in trace]
    ent = [entropy(e.get("group_probs", [])) for e in trace]
    bpt = []
    toks = []
    for e in trace:
        toks.append(e.get("tokens_step", 0))
        bpt.append( (len(e.get("bits","")))/(e.get("tokens_step", 1) or 1) )
    return {
        "steps": steps,
        "bits_total": bits_total,
        "avg_groups": sum(groups)/len(groups) if groups else 0.0,
        "avg_entropy": sum(ent)/len(ent) if ent else 0.0,
        "avg_bits_per_token": sum(bpt)/len(bpt) if bpt else 0.0,
    }
