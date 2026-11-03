"""python -m stegatext_runtime {generate,encode,decode}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .source import MODEL_IDS, Source


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "experiment":
        from .experiments.cli import main as experiments
        return experiments(sys.argv[2:])
    parser = argparse.ArgumentParser(description="Shared official HF source for Qwen and Llama")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "encode", "decode"):
        p = subs.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name != "decode":
            p.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
            p.add_argument("--profile", choices=("checkpoint", "stego"), default="checkpoint")
            p.add_argument("--prompt-file", type=Path, required=True)
        if name == "generate":
            p.add_argument("--max-new-tokens", type=int, default=128)
            p.add_argument("--seed", type=int, default=42)
        if name == "encode":
            p.add_argument("--bits-file", type=Path, required=True)
            p.add_argument("--private-output", type=Path, required=True)
            p.add_argument("--policy", choices=("las", "fullsync", "adaptive"), default="las")
            p.add_argument("--tau", type=float)
            p.add_argument("--max-decisions", type=int, default=64)
        if name in ("encode", "decode"):
            p.add_argument("--key-hex", required=True)
        if name == "decode":
            p.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "decode":
        from .protocol import decode
        packet = json.loads(args.packet.read_text())
        source = Source.load(packet["source"]["model"], packet["source"]["profile"])
        write(args.output, decode(source, packet, key=bytes.fromhex(args.key_hex)))
    else:
        source = Source.load(args.model, args.profile)
        prompt = args.prompt_file.read_text()
        if args.command == "generate":
            write(args.output, source.generate(prompt, max_new_tokens=args.max_new_tokens, seed=args.seed))
        else:
            from .protocol import encode
            if args.policy == "adaptive" and args.tau is None:
                parser.error("adaptive requires an explicit --tau")
            packet, result = encode(source, prompt, args.bits_file.read_text().strip(),
                key=bytes.fromhex(args.key_hex), policy=args.policy, tau=args.tau,
                max_decisions=args.max_decisions)
            write(args.output, packet)
            write(args.private_output, result)


if __name__ == "__main__":
    main()
