"""Utility helpers for converting between text and bit streams."""

from __future__ import annotations

import secrets

__all__ = ["random_bitstring", "text_to_bits", "bits_to_text"]


def random_bitstring(count: int) -> str:
    """Return a pseudo-random bitstring of length ``count``."""
    if count <= 0:
        return ""
    data = secrets.token_bytes((count + 7) // 8)
    return "".join(f"{byte:08b}" for byte in data)[:count]


def text_to_bits(text: str) -> str:
    """Encode UTF-8 text to a bitstring."""
    if not text:
        return ""
    data = text.encode("utf-8")
    return "".join(f"{byte:08b}" for byte in data)


def bits_to_text(bits: str) -> str:
    """Decode a bitstring back to UTF-8 text. Truncates incomplete bytes."""
    if not bits:
        return ""
    usable = len(bits) - (len(bits) % 8)
    if usable <= 0:
        return ""
    payload = bytearray(int(bits[i : i + 8], 2) for i in range(0, usable, 8))
    return payload.decode("utf-8", errors="ignore")

