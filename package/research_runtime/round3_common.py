"""Shared constants and provenance helpers for the Round-3 survey."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND3_DIR = REPO_ROOT / "engineering_survey" / "round3"
DEFAULT_V2_REPO = REPO_ROOT
MODEL_SPECS = {
    "qwen25": {
        "path": REPO_ROOT.parent / "models" / "Qwen2.5-3B-Instruct",
        "physical_gpu": 0,
        "trust_remote_code": True,
    },
    "llama3": {
        "path": REPO_ROOT.parent / "models" / "Meta-Llama-3-8B-Instruct",
        "physical_gpu": 1,
        "trust_remote_code": False,
    },
}


def add_v2_to_path(v2_repo: Path | None = None) -> Path:
    repo = (v2_repo or Path(os.environ.get("STEGTEXT_V2_REPO", DEFAULT_V2_REPO))).resolve()
    source = repo / "src"
    if not (source / "stegtext" / "v2").is_dir():
        raise FileNotFoundError(f"v2 source is missing: {source / 'stegtext' / 'v2'}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return repo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_output(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=not binary, stderr=subprocess.STDOUT
    )


def git_snapshot(repo: Path) -> dict[str, Any]:
    commit = str(git_output(repo, "rev-parse", "HEAD")).strip()
    status = str(git_output(repo, "status", "--short"))
    diff = bytes(git_output(repo, "diff", "--binary", "HEAD", binary=True))
    return {
        "repository": str(repo.resolve()),
        "base_commit": commit,
        "dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_status": status.splitlines(),
    }


def source_manifest(root: Path, patterns: Iterable[str] = ("*.py",)) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
        if path.is_file() and "__pycache__" not in path.parts
    ]


def model_file_manifest(model_path: Path) -> list[dict[str, Any]]:
    names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }
    paths = sorted(path for path in model_path.iterdir() if path.is_file() and path.name in names)
    return [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def load_prompts(path: Path | None = None) -> list[dict[str, str]]:
    target = path or ROUND3_DIR / "prompts.json"
    rows = json.loads(target.read_text(encoding="utf-8"))
    required = {"prompt_id", "language", "domain", "prompt"}
    if not rows or any(set(row) != required for row in rows):
        raise ValueError("Round-3 prompt set has an invalid schema")
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("Round-3 prompt ids are not unique")
    return rows


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

