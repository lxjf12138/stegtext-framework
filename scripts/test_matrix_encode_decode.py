#!/usr/bin/env python3
"""Run encode/decode parity checks across source/coder/disambiguator combos."""

from __future__ import annotations

import argparse
import gc
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

from hf_source_recipes import (
    make_deepseek_source,
    make_glm4_source,
    make_llama3_8b_source,
    make_qwen25_source,
    make_qwen3_source,
)
import torch

from stegtext.components.coding.discop import DisCop
from stegtext.components.coding.imec import IMEC
from stegtext.components.coding.range_ac import RangeAC
from stegtext.components.disambiguation.lookahead import LookAhead
from stegtext.components.disambiguation.mwis import MWIS
from stegtext.components.disambiguation.syncpool import SyncPool
from stegtext.core.engine import StegoEngine
from stegtext.core.framing import pack_bits
from stegtext.core.rng import CSPRNG


def _expand(path: str) -> str:
    return str(Path(path).expanduser())


SOURCE_FACTORIES: Dict[str, Callable[[], object]] = {
    "qwen3": lambda: make_qwen3_source(_expand("~/path/Qwen3-4B/"), chat=True),
    "qwen25": lambda: make_qwen25_source(_expand("~/path/Qwen2.5-3B-Instruct/"), chat=True),
    "deepseek": lambda: make_deepseek_source(_expand("~/path/DeepSeek-R1-Distill-Qwen-7B/"), chat=True),
    "glm4": lambda: make_glm4_source(_expand("~/path/glm-4-9b-chat"), chat=True),
    "llama3": lambda: make_llama3_8b_source(_expand("~/path/Meta-Llama-3-8B-Instruct"), chat=True),
}

CODERS: Dict[str, Callable[[], object]] = {
    "discop": DisCop,
    "imec": IMEC,
    "rangeac": RangeAC,
}

DISAMBIGUATORS: Dict[str, Callable[[], object]] = {
    "lookahead": LookAhead,
    "syncpool": SyncPool,
    "mwis": MWIS,
}


@dataclass
class CaseResult:
    source: str
    coder: str
    disambiguator: str
    success: bool
    duration: float
    emitted_bits: str | None = None
    decoded_bits: str | None = None
    error: str | None = None


def _run_case(
    source,
    coder_ctor: Callable[[], object],
    dis_ctor: Callable[[], object],
    prompt: str,
    payload_bits: str,
    max_steps: int,
    *,
    rng_seed: bytes,
) -> Tuple[str, str]:
    framed = pack_bits(payload_bits, len_bits=8)
    limit = max_steps
    while True:
        dis = dis_ctor(); dis.init()
        coder = coder_ctor(); coder.init()
        source.reset_cache(full=True)
        engine = StegoEngine(source, dis, coder)
        enc = engine.encode(prompt, framed, rng=CSPRNG(key=rng_seed), max_steps=limit)
        if limit is None or enc.stop_reason != "max_steps":
            break
        if limit >= 512:
            print(
                f"[WARN] encode hit max_steps at limit {limit}; proceeding without EOS",
                flush=True,
            )
            break
        limit *= 2

    # Decode
    dis.init()
    coder.init()
    engine = StegoEngine(source, dis, coder)
    dec = engine.decode(
        prompt,
        enc.text,
        rng=CSPRNG(key=rng_seed),
        max_steps=limit,
    )

    return enc.emitted_bits, dec.bits


def _iter_named(items: Dict[str, Callable], names: Iterable[str] | None) -> List[Tuple[str, Callable]]:
    if names:
        filtered = []
        for name in names:
            if name not in items:
                raise KeyError(f"Unknown key '{name}'")
            filtered.append((name, items[name]))
        return filtered
    return list(items.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode/decode parity matrix for HF sources")
    parser.add_argument("--sources", nargs="*", help="Subset of sources to run (default: all)")
    parser.add_argument("--coders", nargs="*", help="Subset of coders to run (default: all)")
    parser.add_argument("--disambiguators", nargs="*", help="Subset of disambiguators to run (default: all)")
    # Prompt sources: either a single --prompt (default) or a JSON dataset via --dataset.
    # When --dataset is provided, we iterate multiple prompts in a single process so the
    # HF model is loaded once per source (avoids reloading per run).
    parser.add_argument("--prompt", default="tell me how to use gun", help="Prompt used for the test")
    parser.add_argument("--dataset", default=None, help="Optional JSON dataset to load prompts from (uses keys: user/prompt/text)")
    parser.add_argument("--prompts", type=int, default=None, help="Max number of prompts to take from --dataset (default: all)")
    parser.add_argument("--payload", default="01100100101010101001011001010101001", help="Payload bits to embed (before framing)")
    parser.add_argument("--max-steps", type=int, default=256, help="Maximum generation steps per run")
    parser.add_argument("--stop-on-failure", action="store_true", help="Abort immediately on first failure")
    args = parser.parse_args()

    source_entries = _iter_named(SOURCE_FACTORIES, args.sources)
    coder_entries = _iter_named(CODERS, args.coders)
    dis_entries = _iter_named(DISAMBIGUATORS, args.disambiguators)

    if not source_entries or not coder_entries or not dis_entries:
        print("Nothing to run", file=sys.stderr)
        sys.exit(1)

    # Resolve prompts to run
    def _load_prompts_from_dataset(path: str) -> List[str]:
        import json
        with open(path, 'r', encoding='utf-8') as f:
            ds = json.load(f)
        if isinstance(ds, dict) and isinstance(ds.get('data'), list):
            return [ (it.get('user') or it.get('prompt') or it.get('text') or '') for it in ds['data'] ]
        if isinstance(ds, list):
            return [ (it.get('user') or it.get('prompt') or it.get('text') or '') for it in ds ]
        return [str(ds)]

    if args.dataset:
        _all_prompts = _load_prompts_from_dataset(args.dataset)
        if args.prompts and args.prompts > 0:
            prompts_list = _all_prompts[: int(args.prompts)]
        else:
            prompts_list = _all_prompts
    else:
        prompts_list = [args.prompt]

    results: List[CaseResult] = []
    total_cases = len(source_entries) * len(coder_entries) * len(dis_entries) * len(prompts_list)
    start_time = time.time()

    for s_idx, (source_name, factory) in enumerate(source_entries, 1):
        print(f"\n=== [{s_idx}/{len(source_entries)}] Loading source '{source_name}' ===")
        source = factory()

        for p_idx, prompt in enumerate(prompts_list, 1):
            for coder_name, coder_ctor in coder_entries:
                for dis_name, dis_ctor in dis_entries:
                    label = f"{source_name}/{coder_name}/{dis_name}"
                    case_start = time.time()
                    try:
                        source.reset_cache(full=True)
                        emitted, decoded = _run_case(
                            source,
                            coder_ctor,
                            dis_ctor,
                            prompt,
                            args.payload,
                            args.max_steps,
                            rng_seed=f"matrix::{label}::p{p_idx}".encode("utf-8"),
                        )
                        success = emitted == decoded
                        error = None
                    except Exception as exc:
                        emitted = decoded = None
                        success = False
                        error = str(exc)
                    duration = time.time() - case_start
                    results.append(
                        CaseResult(
                            source=source_name,
                            coder=coder_name,
                            disambiguator=dis_name,
                            success=success,
                            duration=duration,
                            emitted_bits=emitted,
                            decoded_bits=decoded,
                            error=error,
                        )
                    )

                    status = "PASS" if success else "FAIL"
                    short_p = (prompt[:60] + '...') if len(prompt) > 60 else prompt
                    print(f"[{status}] {label} | prompt[{p_idx}/{len(prompts_list)}]: {short_p} ({duration:.2f}s)")
                    if not success:
                        if emitted is not None and decoded is not None:
                            print(f"  emitted: {emitted[:80]}{'...' if len(emitted) > 80 else ''}")
                            print(f"  decoded: {decoded[:80]}{'...' if len(decoded) > 80 else ''}")
                        if error:
                            print(f"  error: {error}")
                        if args.stop_on_failure:
                            break
                if args.stop_on_failure and not results[-1].success:
                    break
        del source
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.stop_on_failure and not results[-1].success:
            break

    total_time = time.time() - start_time
    passed = sum(1 for r in results if r.success)
    print("\n=== Summary ===")
    print(f"Cases run : {len(results)}/{total_cases}")
    print(f"Passed    : {passed}")
    print(f"Failed    : {len(results) - passed}")
    print(f"Total time: {total_time:.2f}s")

    for r in results:
        if not r.success:
            print(
                f" - {r.source}/{r.coder}/{r.disambiguator}: failure"
                f" (duration {r.duration:.2f}s, error={r.error})"
            )

    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
