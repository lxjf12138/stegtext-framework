"""The experiment subcommands of python -m stegatext_runtime."""
import argparse
import json
from pathlib import Path

from .config import resolve
from stegatext_runtime.models import MODEL_IDS


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified controlled experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Validate/expand a TOML configuration without loading model weights")
    plan.add_argument("--config", type=Path, required=True)
    run = sub.add_parser("run", help="Create an attempt; run independent sender/receiver workers sequentially")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=Path("runs/experiments"))
    run.add_argument("--resume", type=Path)
    compare = sub.add_parser("compare", help="Compare variants or two attempts and list changed controls")
    compare.add_argument("runs", nargs="+", type=Path)
    compare.add_argument("--output", type=Path)
    ppl = sub.add_parser("score", help="Score recorded native token paths (legacy text evaluator is explicit opt-in)")
    ppl.add_argument("--run-dir", action="append", type=Path, required=True)
    ppl.add_argument("--output-dir", type=Path, required=True)
    ppl.add_argument("--metric", choices=['native', 'legacy-text'], default='native')
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    worker.add_argument("--stage", choices=("sender", "receiver"), required=True)
    args = parser.parse_args(argv)
    from . import runner
    if args.command == "plan":
        print(json.dumps(runner.planning(resolve(args.config)), ensure_ascii=False, indent=2))
    elif args.command == "run":
        _, rows = runner.run(args.config, args.output_root, args.resume)
        if any(r["sender_status"] != "ok" or r["exact"] is False for r in rows): raise SystemExit(1)
    elif args.command == "compare":
        value = runner.compare(args.runs)
        if args.output: runner.write(args.output, value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.command == "score":
        if args.metric == 'native':
            from .native_ppl import evaluate
        else:
            from .ppl import evaluate
        evaluate(args.run_dir, args.output_dir)
    else:
        runner.worker(args.run_dir.resolve(), args.model, args.stage)
