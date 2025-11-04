from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
import torch

@dataclass
class Candidate:
    # 归一化概率（float64，数值稳健后）
    p: float
    # 完整序列 token（[1, L]）
    tokens: torch.LongTensor
    # “整句”的字节可见层（用于分组/匹配/前缀判断；必填）
    vb: bytes
    # 是否是 EOS 分支（仅由 token 决定）
    is_eos: bool
    # 可选：便于调试/展示的字符串（若无法解码则 None）
    v: Optional[str] = None
    # prompt 部分 token 的长度（用于定位 payload 起点）
    prompt_len: int = 0
    # prompt 部分可见字节长度（仅对需要还原 payload 字节的逻辑有用）
    prompt_byte_len: int = 0

class Group:
    def __init__(self,candidate:Candidate):
            # 组键可以是字节或字符串（取决于具体分组策略）
        self.key: bytes = candidate.vb
        self.members: List[Candidate] = [candidate]
        self.total_prob = candidate.p

    def add(self,candidate:Candidate):
        self.members.append(candidate)
        self.total_prob = self.total_prob + candidate.p

@dataclass
class GroupingResult:
    groups: List[Group]
    group_probs: torch.Tensor
    stats: Dict[str, object] = field(default_factory=dict)


@dataclass
class SentenceStat:
    text: str
    tokens: int
    chars: int
    bits_emitted: int
    groups: int
    vb: Optional[bytes] = None


@dataclass
class EncodeResult:
    finished: bool
    sentences: List[SentenceStat]
    stop_reason: str


@dataclass
class DecodeResult:
    status: str
    payload: Optional[str]
    consumed_bits: int
    total_bits: int
