#!/usr/bin/env python3
"""Minimal StegText quickstart showing single- and multi-sentence flows."""

from __future__ import annotations

from stegtext.components.coding.range_ac import RangeAC
from stegtext.components.disambiguation.lookahead import LookAhead
from stegtext.components.disambiguation.syncpool import SyncPool
from stegtext.components.disambiguation.mwis import MWIS
from stegtext.components.coding.imec import IMEC
from stegtext.components.coding.discop import DisCop
from stegtext.core.engine import StegoEngine
from stegtext.core.framing import pack_bits, parse_framed_bits
from stegtext.core.payload import bits_to_text, random_bitstring, text_to_bits
from stegtext.core.rng import CSPRNG
from stegtext.integrations.toy import make_toy_source
from stegtext.integrations.hf_recipes import make_qwen25_source, make_llama3_8b_source




def main() -> None:
    # [Source]
    # make_toy_source() or make_qwen25_source() or make_llama3_8b_source()
    source = make_qwen25_source(
        "Qwen/Qwen2.5-3B-Instruct",
        device="cuda:0",
        dtype="float16",
        chat=True,
        temperature=0.9,
        top_k=16,
        top_p=0.95,
    )

    # [Disambiguator]
    # LookAhead() or SyncPool() or MWIS()
    dis = LookAhead()

    # [Coder]
    # RangeAC(...) or IMEC(...) or DisCop()
    coder = DisCop()

    engine = StegoEngine(source, dis, coder)

    prompt_single = "Explain quantum computing in one short sentence."
    payload_bits = random_bitstring(64)
    rng_single = CSPRNG(key=b"quickstart-single")

    engine.init()
    enc_single = engine.encode(prompt_single, payload_bits, rng=rng_single, max_steps=128)

    engine.init()
    dec_single = engine.decode(prompt_single, enc_single.text, rng=rng_single, max_steps=128)

    prefix_ok = payload_bits.startswith(dec_single.bits)

    print("=== Single sentence ===")
    print(f"Prompt       : {prompt_single}")
    print(f"Visible text : {enc_single.text!r}")
    print(f"Tokens used  : {enc_single.tokens} (stop={enc_single.stop_reason})")
    print(f"Bits emitted : {len(enc_single.emitted_bits)}")
    print(f"Prefix match : {prefix_ok}")

    prompt_multi = "Write a short beautiful poem."
    secret_text = "This is a secret message hidden within the text."
    framed_bits = pack_bits(text_to_bits(secret_text), len_bits=16)
    rng_multi = CSPRNG(key=b"quickstart-multi")

    engine.init()
    seq_enc = engine.encode_sequence(
        prompt_multi,
        framed_bits,
        rng=rng_multi,
        max_steps=128,
        max_sentences=32,
    )
    sentences = [s.text for s in seq_enc.sentences]

    print("\n=== Multi sentence ===")
    for idx, text in enumerate(sentences, 1):
        print(f"{idx}. {text}")

    total_bits = sum(len(s.emitted_bits) for s in seq_enc.sentences)
    total_tokens = sum(s.tokens for s in seq_enc.sentences) or 1
    avg_bpt = total_bits / total_tokens
    print(f"Average bits/token: {avg_bpt:.3f}")

    engine.init()
    seq_dec = engine.decode_sequence(
        prompt_multi,
        sentences,
        rng=rng_multi,
        max_steps=128,
    )
    status, recovered_bits, _ = parse_framed_bits(seq_dec.bits, len_bits=16)
    recovered_text = bits_to_text(recovered_bits)
    secret_ok = status == "done" and recovered_text == secret_text

    print(f"Recovered text   : {recovered_text!r}")
    print(f"Secret matches   : {secret_ok}")


if __name__ == "__main__":  # pragma: no cover
    main()
