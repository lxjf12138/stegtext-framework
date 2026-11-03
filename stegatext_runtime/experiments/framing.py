"""Optional in-band finite frame: magic16, length32, payload, CRC32.

CRC detects corruption; it is not authentication. Post-frame zero padding lets
the carrier continue to natural EOS, without declaring padding user payload.
"""
import zlib

MAGIC = f"{0x5354:016b}"
HEADER_BITS = 48
OVERHEAD_BITS = 80


def crc(length, bits):
    packed = int(bits.ljust((len(bits)+7)//8*8, "0") or "0", 2).to_bytes((len(bits)+7)//8, "big")
    return f"{zlib.crc32(length.to_bytes(4, 'big')+packed):032b}"


def encode(payload):
    if set(payload)-{"0", "1"} or len(payload) >= 2**32:
        raise ValueError("Invalid finite frame payload")
    return MAGIC+f"{len(payload):032b}"+payload+crc(len(payload), payload)


def inspect(bits):
    if len(bits) < HEADER_BITS or bits[:16] != MAGIC:
        return None
    n = int(bits[16:HEADER_BITS], 2)
    if len(bits) < n+OVERHEAD_BITS:
        return None
    payload = bits[HEADER_BITS:HEADER_BITS+n]
    if bits[HEADER_BITS+n:OVERHEAD_BITS+n] != crc(n, payload):
        return None
    return payload
