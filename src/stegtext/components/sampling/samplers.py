
from __future__ import annotations
from typing import Tuple, Optional
import torch
from .config import SamplingConfig

def softmax(x: torch.Tensor, t: float)->torch.Tensor:
    if t<=0: t=1e-6
    x = x / t
    x = x - x.max()
    ex = torch.exp(x)
    return ex / ex.sum()

def apply_top_k(probs: torch.Tensor, k: int)->torch.Tensor:
    if k is None or k<=0 or k>=probs.numel(): return probs
    v, idx = torch.topk(probs, k)
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[idx] = True
    out = torch.where(mask, probs, torch.zeros_like(probs))
    s = out.sum()
    return out / (s + 1e-12)

def apply_top_p(probs: torch.Tensor, p: float)->torch.Tensor:
    if p is None or p<=0 or p>=1: return probs
    val, idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(val, dim=0)
    mask = cumsum <= p
    # always include the first token
    mask[0] = True
    keep = idx[mask]
    out = torch.zeros_like(probs)
    out[keep] = probs[keep]
    s = out.sum()
    return out / (s + 1e-12)

def apply_min_p(probs: torch.Tensor, mp: float)->torch.Tensor:
    if mp is None or mp<=0 or mp>=1: return probs
    mask = probs >= mp
    out = torch.where(mask, probs, torch.zeros_like(probs))
    s = out.sum()
    if s.item()==0:  # keep argmax
        i = int(torch.argmax(probs))
        out[i] = 1.0
        s = out.sum()
    return out / (s + 1e-12)

def apply_typical(probs: torch.Tensor, typical_p: float)->torch.Tensor:
    if typical_p is None or typical_p<=0 or typical_p>=1: return probs
    # deviation from entropy
    p = probs[probs>0]
    H = float(-(p*torch.log(p)).sum().item())
    neglog = -torch.log(probs + 1e-12)
    dev = torch.abs(neglog - H)
    val, idx = torch.sort(dev, descending=False)
    c = torch.cumsum(probs[idx], dim=0)
    mask = c <= typical_p
    mask[0] = True
    keep = idx[mask]
    out = torch.zeros_like(probs)
    out[keep] = probs[keep]
    s = out.sum()
    return out / (s + 1e-12)

def sample_filter(logits: torch.Tensor, cfg: SamplingConfig)->torch.Tensor:
    probs = softmax(logits, cfg.temperature)
    if cfg.top_k: probs = apply_top_k(probs, cfg.top_k)
    if cfg.top_p: probs = apply_top_p(probs, cfg.top_p)
    if cfg.typical_p: probs = apply_typical(probs, cfg.typical_p)
    if cfg.min_p: probs = apply_min_p(probs, cfg.min_p)
    return probs
