from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass


@dataclass
class CSPRNG:
    key: bytes
    counter: int = 0

    def random(self) -> float:
        self.counter += 1
        digest = hmac.new(self.key, struct.pack(">Q", self.counter), hashlib.sha256).digest()
        return (int.from_bytes(digest[:8], "big") & ((1 << 53) - 1)) / float(1 << 53)


def session_rng(master_key: bytes, nonce: bytes = b"default") -> CSPRNG:
    return CSPRNG(hmac.new(master_key, b"session:" + nonce, hashlib.sha256).digest())


def derive_rng(rng: CSPRNG, scope: str) -> CSPRNG:
    return CSPRNG(hmac.new(rng.key, scope.encode("utf-8"), hashlib.sha256).digest())
