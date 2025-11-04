from __future__ import annotations
import hmac
import hashlib
import struct
from dataclasses import dataclass


@dataclass
class CSPRNG:
    """
    简单计数器型 CSPRNG：HMAC-SHA256(key, counter) → 取高 53 位映射到 [0,1)。
    """
    key: bytes
    counter: int = 0

    def random(self) -> float:
        self.counter += 1
        msg = struct.pack(">Q", self.counter)
        d = hmac.new(self.key, msg, hashlib.sha256).digest()
        x = int.from_bytes(d[:8], "big")
        return (x & ((1 << 53) - 1)) / float(1 << 53)


def session_rng(master_key: bytes, nonce: bytes) -> CSPRNG:
    key = hmac.new(master_key, b"session:" + nonce, hashlib.sha256).digest()
    return CSPRNG(key=key)


def derive_rng(rng: CSPRNG, scope: str) -> CSPRNG:
    key = hmac.new(rng.key, scope.encode("utf-8"), hashlib.sha256).digest()
    return CSPRNG(key=key)
