from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch

from ..core.errors import TokenizerContractError

# HF data bytes are escaped as ``0x01 || byte``.  This augmented terminal
# symbol is consequently prefix-disjoint from every escaped data token.
EOS_STEGA = b"\x00\x00"
_DATA_TAG = 1


def _byte_level_decoder() -> dict[str, int]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _byte_level_decoder()


class TokenizerCodec:
    """Canonical token/byte adapter shared by all methods."""

    def __init__(self, tokenizer: Any, eos_ids: Sequence[int] = ()) -> None:
        self.tokenizer = tokenizer
        self.eos_bytes = EOS_STEGA
        self.eos_ids = frozenset(int(x) for x in eos_ids)
        vocab = tokenizer.get_vocab()
        if not vocab:
            raise TokenizerContractError("tokenizer vocabulary is empty")
        self.vocab_size = max(int(x) for x in vocab.values()) + 1
        values = [self._escape_bytes(self._decode_token_bytes(i)) for i in range(self.vocab_size)]
        for i in self.eos_ids:
            if i >= len(values):
                values.extend([b""] * (i + 1 - len(values)))
            values[i] = EOS_STEGA
        self._token_bytes = tuple(values)

    def token_bytes(self, token_id: int) -> bytes:
        i = int(token_id)
        if i < 0 or i >= len(self._token_bytes):
            raise IndexError(f"token id {i} is outside vocabulary")
        return self._token_bytes[i]

    @property
    def token_bytes_table(self) -> tuple[bytes, ...]:
        return self._token_bytes

    def tokens_to_bytes(self, token_ids: Iterable[int]) -> bytes:
        return b"".join(self.token_bytes(i) for i in token_ids)

    def text_to_bytes(self, text: str) -> bytes:
        return self._escape_bytes(str(text).encode("utf-8", errors="strict"))

    def bytes_to_text(self, data: bytes, *, strip_eos: bool = False) -> str:
        if strip_eos:
            while data.endswith(EOS_STEGA):
                data = data[: -len(EOS_STEGA)]
        if EOS_STEGA in data:
            raise TokenizerContractError("terminal marker appears inside visible data")
        if len(data) % 2:
            raise TokenizerContractError("internal byte realization ends inside an escaped byte")
        raw = bytearray()
        for offset in range(0, len(data), 2):
            if data[offset] != _DATA_TAG:
                raise TokenizerContractError(
                    f"invalid internal byte tag {data[offset]} at offset {offset}"
                )
            raw.append(data[offset + 1])
        try:
            return bytes(raw).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise TokenizerContractError("canonical bytes are not valid UTF-8") from exc

    def is_eos(self, token_id: int) -> bool:
        return int(token_id) in self.eos_ids

    def tokenize_text(self, text: str) -> tuple[int, ...]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors=None)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return tuple(int(x) for x in ids)

    def next_token(self, history: Sequence[int], visible: bytes) -> int:
        """Return the next canonical token for a visible continuation."""
        if visible.startswith(EOS_STEGA) and self.eos_ids:
            return min(self.eos_ids)
        marker_at = visible.find(EOS_STEGA)
        suffix_bytes = visible if marker_at < 0 else visible[:marker_at]
        suffix = self.bytes_to_text(suffix_bytes, strip_eos=False)
        history_ids = tuple(int(x) for x in history)
        try:
            history_text = self.tokenizer.decode(
                list(history_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        except TypeError:
            history_text = self.tokenizer.decode(list(history_ids), skip_special_tokens=False)
        ids = self.tokenize_text(history_text + suffix)
        if len(ids) <= len(history_ids):
            raise TokenizerContractError("visible target has no token")
        return int(ids[len(history_ids)])

    def validate(self, sequences: Sequence[Sequence[int]]) -> None:
        for seq in sequences:
            ids = tuple(int(x) for x in seq)
            if any(self.is_eos(x) for x in ids):
                continue
            expected = self.tokenizer.decode(
                list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            try:
                composed = self.bytes_to_text(self.tokens_to_bytes(ids))
            except TokenizerContractError:
                # A sequence ending in a partial UTF-8 fragment is not
                # independently renderable; the next token completes it.
                continue
            if composed != expected:
                raise TokenizerContractError(
                    f"tokenizer byte realization disagrees for sequence {ids}: {composed!r} != {expected!r}"
                )

    def _decode_token_bytes(self, token_id: int) -> bytes:
        piece = None
        try:
            values = self.tokenizer.convert_ids_to_tokens([int(token_id)], skip_special_tokens=False)
            if values:
                piece = values[0]
        except (AttributeError, TypeError, ValueError):
            pass
        if piece is not None:
            try:
                return bytes(_BYTE_DECODER[c] for c in str(piece))
            except (KeyError, TypeError):
                pass
        try:
            text = self.tokenizer.decode(
                [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        except TypeError:
            text = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        if isinstance(text, bytes):
            return text
        return str(text).encode("utf-8", errors="strict")

    @staticmethod
    def _escape_bytes(data: bytes) -> bytes:
        out = bytearray(len(data) * 2)
        for index, value in enumerate(data):
            out[index * 2] = _DATA_TAG
            out[index * 2 + 1] = value
        return bytes(out)
