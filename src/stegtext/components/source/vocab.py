from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Iterable, Optional

import torch

from .base import EOS_STEGA


@dataclass
class TokenByteVocab:
    """Mapping from token ids to byte sequences with unified EOS handling."""

    mapping: Dict[int, bytes]
    vocab_size: int
    _byte_data: torch.Tensor = field(init=False, repr=False)
    _offsets: torch.Tensor = field(init=False, repr=False)
    _dirty: bool = field(init=False, default=True, repr=False)

    def __post_init__(self) -> None:
        self._byte_data = torch.empty(0, dtype=torch.uint8)
        self._offsets = torch.zeros(self.vocab_size + 1, dtype=torch.int32)
        self._dirty = True
        self._ensure_views()

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        *,
        end_token_ids: Sequence[int],
    ) -> "TokenByteVocab":
        vocab_map = tokenizer.get_vocab()
        if not vocab_map:
            raise ValueError("tokenizer.get_vocab() returned empty mapping")
        mapping: Dict[int, bytes] = {}
        max_id = -1
        for tok, tid in vocab_map.items():
            tid = int(tid)
            if tid > max_id:
                max_id = tid
            mapping[tid] = cls._decode_token_bytes(tokenizer, tid)
        if max_id < 0:
            raise ValueError("tokenizer vocab produced negative max id")
        vocab_size = max_id + 1

        for tid in end_token_ids:
            mapping[int(tid)] = EOS_STEGA
        return cls(mapping=mapping, vocab_size=vocab_size)

    def set_token_bytes(self, token_id: int, data: bytes) -> None:
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError("token_id must be non-negative")
        if token_id >= self.vocab_size:
            self.vocab_size = token_id + 1
        self.mapping[token_id] = data
        self._dirty = True

    def token_bytes(self, token_id: int) -> bytes:
        self._ensure_views()
        tid = int(token_id)
        if tid < 0 or tid >= self.vocab_size:
            return b""
        start = int(self._offsets[tid].item())
        end = int(self._offsets[tid + 1].item())
        if end <= start:
            return b""
        return self._byte_data[start:end].cpu().numpy().tobytes()

    def tokens_to_bytes(self, tokens: Sequence[int] | torch.Tensor, *, start: int = 0) -> bytes:
        self._ensure_views()
        if isinstance(tokens, torch.Tensor):
            token_ids = tokens.detach()
            if token_ids.device.type != "cpu":
                token_ids = token_ids.to("cpu")
            token_ids = token_ids.to(torch.long)
        else:
            token_ids = torch.as_tensor(list(tokens), dtype=torch.long)

        if start < 0:
            start = 0
        if start > 0:
            token_ids = token_ids[start:]
        if token_ids.numel() == 0:
            return b""

        starts = self._offsets.index_select(0, token_ids)
        ends = self._offsets.index_select(0, token_ids + 1)
        lengths = ends - starts
        total = int(lengths.sum().item())
        if total <= 0:
            return b""

        out = torch.empty(total, dtype=torch.uint8)
        dest = 0
        starts_list = starts.tolist()
        lengths_list = lengths.tolist()
        for s, length in zip(starts_list, lengths_list):
            if length <= 0:
                continue
            e = s + length
            out[dest : dest + length] = self._byte_data[s:e]
            dest += length
        return out.numpy().tobytes()

    def bytes_to_text(self, data: bytes, *, strip_eos: bool = False) -> str:
        if strip_eos and data.endswith(EOS_STEGA):
            eos_len = len(EOS_STEGA)
            while data.endswith(EOS_STEGA):
                data = data[:-eos_len]
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def _decode_token_bytes(tokenizer, token_id: int) -> bytes:
        # Prefer byte-level reconstruction so we keep exact UTF-8 bytes even for
        # partial characters emitted by ByteLevel BPE tokenizers.
        token_str: Optional[str] = None
        try:
            pieces = tokenizer.convert_ids_to_tokens(
                [int(token_id)], skip_special_tokens=False
            )
            if pieces:
                token_str = pieces[0]
        except Exception:
            token_str = None

        if token_str is None:
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is not None:
                try:
                    token_str = backend.id_to_token(int(token_id)) or ""
                except Exception:
                    token_str = None

        if token_str:
            try:
                return bytes(_BYTE_LEVEL_DECODER[ch] for ch in token_str)
            except KeyError:
                pass

        text = ""
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            try:
                token = backend.id_to_token(int(token_id))
                if token is None:
                    token = ""
                decoder = getattr(backend, "decoder", None)
                if decoder is not None:
                    try:
                        text = decoder.decode([token])
                    except Exception:
                        text = token
                else:
                    text = token
            except Exception:
                text = ""

        if text == "":
            piece: Optional[str] = None
            try:
                piece = tokenizer._convert_id_to_token(int(token_id))
            except Exception:
                piece = None
            if piece is None:
                try:
                    text = tokenizer.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    if isinstance(text, bytes):
                        piece = text.decode("utf-8", errors="ignore")
                    else:
                        piece = str(text)
                except TypeError:
                    try:
                        text = tokenizer.decode(
                            [int(token_id)],
                            skip_special_tokens=False,
                        )
                        piece = str(text)
                    except Exception:
                        piece = None
                except Exception:
                    piece = None

            if piece is None:
                try:
                    piece_list = tokenizer.convert_ids_to_tokens(
                        [int(token_id)], skip_special_tokens=False
                    )
                    if piece_list:
                        piece = piece_list[0]
                except Exception:
                    piece = None

            if piece is None:
                piece = ""

            text = piece

        if isinstance(text, bytes):
            return text
        return str(text).encode("utf-8", errors="ignore")

    def _ensure_views(self) -> None:
        if not self._dirty:
            return
        chunks: list[torch.Tensor] = []
        offsets = [0]
        for tid in range(self.vocab_size):
            payload = self.mapping.get(tid, b"")
            length = len(payload)
            if length:
                chunks.append(torch.tensor(list(payload), dtype=torch.uint8))
            offsets.append(offsets[-1] + length)
        if chunks:
            if len(chunks) == 1:
                self._byte_data = chunks[0]
            else:
                self._byte_data = torch.cat(chunks)
        else:
            self._byte_data = torch.empty(0, dtype=torch.uint8)
        self._offsets = torch.tensor(offsets, dtype=torch.int32)
        self._dirty = False


def _build_byte_level_decoder() -> Dict[str, int]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    byte_encoder = dict(zip(bs, cs))
    return {v: k for k, v in byte_encoder.items()}


_BYTE_LEVEL_DECODER = _build_byte_level_decoder()
