from __future__ import annotations

import argparse
import json

from ..core.rng import session_rng
from .config import load_config
from .runner import build_runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="StegText v2 experiment runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--bits", default="")
    parser.add_argument("--text", help="decode visible text instead of encoding")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    runner = build_runner(config)
    rng = session_rng(str(config.execution.seed).encode("ascii"))
    if args.text is None:
        result = runner.encode(
            args.prompt,
            args.bits,
            rng=rng,
            max_steps=config.execution.max_steps,
            max_tokens=config.execution.max_tokens,
        )
    else:
        result = runner.decode(
            args.prompt,
            args.text,
            rng=rng,
            max_steps=config.execution.max_steps,
            max_tokens=config.execution.max_tokens,
        )
    print(json.dumps(result.__dict__, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
