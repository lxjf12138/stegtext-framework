from __future__ import annotations


def message_metrics(payload_bits: str, decoded_bits: str, *, elapsed_s: float | None = None, tokens: int | None = None) -> dict[str, float | int | bool]:
    n = max(len(payload_bits), len(decoded_bits))
    errors = sum(a != b for a, b in zip(payload_bits, decoded_bits)) + abs(len(payload_bits) - len(decoded_bits))
    out: dict[str, float | int | bool] = {
        "payload_bits": len(payload_bits),
        "decoded_bits": len(decoded_bits),
        "ber": float(errors / n) if n else 0.0,
        "exact_message": payload_bits == decoded_bits,
    }
    if tokens is not None:
        out["bpt"] = float(len(decoded_bits) / max(int(tokens), 1))
    if elapsed_s is not None:
        out["elapsed_s"] = float(elapsed_s)
        if tokens:
            out["bps"] = float(len(decoded_bits) / max(float(elapsed_s), 1e-12))
    return out
