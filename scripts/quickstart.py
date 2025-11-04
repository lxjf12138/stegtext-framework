#!/usr/bin/env python3
"""
Quick Start: Single‑sentence encode→decode demo for StegText.

After `pip install -e .`, run one of:

  # 1) Toy source (no extra deps, runs on CPU)
  python scripts/quickstart.py --source toy --coder imec --dis lookahead \
         --prompt "Say hi" --message "secret"

  # 2) HF sources (requires torch+transformers and a local model path)
  CUDA_VISIBLE_DEVICES=0 \
  python scripts/quickstart.py --source qwen3 --model "~/path/Qwen3-4B" \
         --prompt "Say hi" --message "secret" --dis syncpool --coder rangeac

The script prints visible text, step count, and verifies round‑trip bits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Callable, Tuple

import torch

from stegtext.core.engine import StegoEngine
from stegtext.core.framing import pack_bits, parse_framed_bits
from stegtext.core.rng import CSPRNG
from stegtext.core.prob import sanitize1d  # noqa: F401 (useful for tinkering)

from stegtext.components.coding.range_ac import RangeAC
from stegtext.components.coding.discop import DisCop
from stegtext.components.coding.imec import IMEC

from stegtext.components.disambiguation.lookahead import LookAhead
from stegtext.components.disambiguation.syncpool import SyncPool
from stegtext.components.disambiguation.mwis import MWIS

# Toy source
from stegtext.components.source.base import EOS_STEGA
from stegtext.components.source.vocab import TokenByteVocab
from stegtext.components.source.toy import ToySource, ToySourceConfig


# ---------------------- helpers ----------------------

def _expand(path: str) -> str:
    return str(Path(path).expanduser())


def text_to_bits(s: str) -> str:
    b = s.encode("utf-8")
    return "".join(f"{byte:08b}" for byte in b)


def bits_to_text(bits: str) -> str:
    if len(bits) % 8 != 0:
        return ""  # only full bytes
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i : i + 8], 2))
    try:
        return out.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ---------------------- sources ----------------------

def _build_toy_source() -> ToySource:
    # Build a byte‑level vocab: ids 0..255 map to single bytes; one EOS id uses EOS_STEGA
    mapping: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    EOS_ID = 256
    mapping[EOS_ID] = EOS_STEGA
    vocab = TokenByteVocab(mapping=mapping, vocab_size=EOS_ID + 1)

    # Provide a minimal transition spec: from any prefix, allow a few bytes + EOS
    # This keeps the demo simple; real LMs will produce rich distributions.
    def spec_for_all() -> Dict[Tuple[int, ...], Tuple[Tuple[int, float], ...]]:
        space, a, b, dot, excl = 32, 97, 98, 46, 33
        # Unnormalized probs are fine; downstream sanitizes as needed
        base = (
            (space, 0.40),
            (a, 0.20),
            (b, 0.20),
            (dot, 0.10),
            (excl, 0.08),
            (EOS_ID, 0.02),
        )
        return {(): base}

    transitions = spec_for_all()
    cfg = ToySourceConfig(
        vocab=vocab,
        transitions=transitions,
        end_token_ids=(256,),
        prompt_to_tokens=None,  # default: byte‑level prompt
    )
    return ToySource(cfg)


def _build_hf_source(name: str, model_path: str, *, device: str) -> object:
    try:
        # When executed as `python scripts/quickstart.py`, the module below is importable as a sibling
        from hf_source_recipes import (
            make_qwen25_source,
            make_qwen3_source,
            make_deepseek_source,
            make_glm4_source,
            make_llama3_8b_source,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "HF recipes not found. Please run from repo root and ensure transformers is installed."
        ) from exc

    mp = _expand(model_path)
    if name == "qwen25":
        return make_qwen25_source(mp, device=device, dtype="float16", chat=True)
    if name == "qwen3":
        return make_qwen3_source(mp, device=device, dtype="float16", chat=True)
    if name == "deepseek":
        return make_deepseek_source(mp, device=device, dtype="float16", chat=True)
    if name == "glm4":
        return make_glm4_source(mp, device=device, dtype="float16", chat=True)
    if name == "llama3":
        return make_llama3_8b_source(mp, device=device, dtype="bfloat16", chat=False)
    raise KeyError(f"unknown hf source '{name}'")


# ----------------------- main ------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="StegText Quick Start: encode/decode one sentence")
    parser.add_argument("--source", default="toy", choices=[
        "toy", "qwen25", "qwen3", "deepseek", "glm4", "llama3",
    ])
    parser.add_argument("--model", default=None, help="HF model path or id (for non-toy sources)")
    parser.add_argument("--device", default="cuda:0", help="torch device for HF models (e.g., cuda:0 or cpu)")

    parser.add_argument("--dis", default="lookahead", choices=["lookahead", "syncpool", "mwis"], help="disambiguator")
    parser.add_argument("--coder", default="imec", choices=["imec", "discop", "rangeac"], help="coding strategy")

    parser.add_argument("--prompt", default="Say hi", help="user prompt")
    parser.add_argument("--message", default="secret", help="payload string to embed (UTF-8 → bits)")
    parser.add_argument("--len-bits", type=int, default=16, help="framing header bits for payload length")
    parser.add_argument("--max-steps", type=int, default=64, help="generation budget (planning steps)")

    args = parser.parse_args()

    # Build source
    if args.source == "toy":
        source = _build_toy_source()
    else:
        if not args.model:
            raise SystemExit("--model is required for HF sources. Try --source toy for a no-deps demo.")
        source = _build_hf_source(args.source, args.model, device=args.device)

    # Build disambiguator
    if args.dis == "lookahead":
        dis = LookAhead()
    elif args.dis == "syncpool":
        dis = SyncPool()
    else:
        dis = MWIS()

    # Build coder
    if args.coder == "imec":
        coder = IMEC(block_size=6)
    elif args.coder == "discop":
        coder = DisCop()
    else:
        coder = RangeAC(precision=40)

    # Prepare payload bits (framed)
    payload_bits = text_to_bits(args.message)
    framed = pack_bits(payload_bits, len_bits=int(args.len_bits))

    # Engine + init modules (coder/dis may keep state between runs)
    engine = StegoEngine(source, dis, coder)
    if hasattr(dis, "init"):
        dis.init()
    if hasattr(coder, "init"):
        coder.init()
    rng = CSPRNG(key=b"stegtext-quickstart")

    enc = engine.encode(args.prompt, framed, rng=rng, max_steps=int(args.max_steps))

    # Decode back from visible payload text
    # Fresh init for decode to mirror encode conditions
    if hasattr(dis, "init"):
        dis.init()
    if hasattr(coder, "init"):
        coder.init()
    dec = engine.decode(args.prompt, enc.text, rng=rng, max_steps=int(args.max_steps))

    # Parse framing to recover the message
    status, recovered_bits, consumed = parse_framed_bits(dec.bits, len_bits=int(args.len_bits))
    recovered = bits_to_text(recovered_bits) if status == "done" else ""

    print("\n=== Quick Start Result ===")
    print(f"Source     : {args.source}")
    print(f"Disambig   : {args.dis}")
    print(f"Coder      : {args.coder}")
    print(f"Prompt     : {args.prompt}")
    print(f"Message    : {args.message!r}")
    print(f"Visible    : {enc.text!r}")
    print(f"Steps      : {enc.tokens} tokens, stop={enc.stop_reason}")
    ok = (enc.emitted_bits == dec.bits)
    print(f"Round‑trip : {'OK' if ok else 'MISMATCH'} (decoded {len(dec.bits)} bits, consumed={consumed}, status={status})")
    if status == "done":
        print(f"Recovered  : {recovered!r}")


if __name__ == "__main__":  # pragma: no cover
    main()
