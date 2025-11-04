from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


def pack_bits(payload_bits: str, len_bits: int = 32) -> str:
    """
    头部固定 len_bits 位表示正文位长，后接 payload_bits。
    """
    n = len(payload_bits)
    if n >= (1 << len_bits):
        raise ValueError("payload too long for header width")
    header = format(n, f"0{len_bits}b")
    return header + payload_bits


def parse_framed_bits(bits: str, len_bits: int = 32) -> Tuple[str, str, int]:
    """
    一次性解析：
      返回 (status, payload, consumed_bits)
      - status: "done" / "need_more"
      - payload: 完整负载（仅 done 时非空）
      - consumed_bits: 头+负载的真实比特长度（仅 done > 0；need_more 时为 0）
    """
    if len(bits) < len_bits:
        return "need_more", "", 0
    n = int(bits[:len_bits], 2)
    need = len_bits + n
    if len(bits) < need:
        return "need_more", "", 0
    return "done", bits[len_bits:need], need


@dataclass
class FrameParse:
    """
    增量帧解析结果：
      - status: "need_header" | "need_payload" | "done"
      - payload: 仅在 done 时给出
      - consumed_bits: 已知应消耗的位数（done=头+正文；need_payload=头部位数；need_header=0）
      - need_more: 距离“下一个状态”还差多少位
    """
    status: str
    payload: Optional[str]
    consumed_bits: int
    need_more: int


def parse_framed_progress(bits: str, len_bits: int = 32) -> FrameParse:
    # 头不完整
    if len(bits) < len_bits:
        return FrameParse(
            status="need_header",
            payload=None,
            consumed_bits=0,
            need_more=len_bits - len(bits),
        )

    # 头完整，读长度
    n = int(bits[:len_bits], 2)
    total_needed = len_bits + n

    # 正文不完整
    if len(bits) < total_needed:
        return FrameParse(
            status="need_payload",
            payload=None,
            consumed_bits=len_bits,
            need_more=total_needed - len(bits),
        )

    # 完整
    payload = bits[len_bits:total_needed]
    return FrameParse(
        status="done",
        payload=payload,
        consumed_bits=total_needed,
        need_more=0,
    )
