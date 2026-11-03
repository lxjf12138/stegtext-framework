from __future__ import annotations
import hashlib, hmac, json, struct
from pathlib import Path


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()


def stable_json_sha256(value) -> str:
    raw=json.dumps(value, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def derive_domain_seed(master_key: bytes, domain: str, counter: int = 0) -> bytes:
    if counter < 0: raise ValueError("counter must be >=0")
    return hmac.new(master_key, domain.encode('utf-8') + b':' + struct.pack('>Q', counter), hashlib.sha256).digest()
