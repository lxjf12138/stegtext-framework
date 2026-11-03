"""Download exactly pinned checkpoints, with mirrors and byte verification.

Uses project model lock after first creation. Large files use resumable aria2;
weight SHA-256 and small-file Git blob IDs are checked against HF metadata.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
LOCK = HERE / "models.lock.json"
REPORT = HERE / "model-downloads.json"
LOG_LOCK = threading.Lock()


def event(data):
    with LOG_LOCK:
        print(json.dumps(data, ensure_ascii=False), flush=True)


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def prepare_lock():
    if LOCK.exists():
        return json.loads(LOCK.read_text())
    specs = []
    for alias, repo, mirror, dirname in [
        ("qwen", "Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B-Base", "Qwen3-4B-Base"),
        ("llama", "meta-llama/Llama-3.2-3B", "LLM-Research/Llama-3.2-3B", "Llama-3.2-3B"),
    ]:
        official = json.loads((ROOT / f".cache/setup/{alias}_hf.json").read_text())
        ms = json.loads((ROOT / f".cache/setup/{alias}_ms.json").read_text())
        mirror_files = {f["Path"]: f for f in ms["Data"]["Files"]}
        files = []
        for f in official["siblings"]:
            name = f["rfilename"]
            if "/" in name or name == ".gitattributes":
                continue
            if not (name.endswith((".safetensors", ".json", ".txt", ".md")) or name == "LICENSE"):
                continue
            canonical_url = f"https://huggingface.co/{repo}/resolve/{official['sha']}/{name}?download=true"
            hf_mirror_url = canonical_url.replace("https://huggingface.co", "https://hf-mirror.com")
            urls = [canonical_url, hf_mirror_url] if alias == "qwen" else []
            m = mirror_files.get(name)
            if m:
                # Match the public official LFS SHA before accepting a mirror.
                if f.get("lfs"):
                    assert m["Sha256"] == f["lfs"]["sha256"] and m["Size"] == f["size"]
                ms_url = f"https://modelscope.cn/models/{mirror}/resolve/{m['Revision']}/{name}?download=true"
                # Mainland mirror first for weights; small public Qwen files
                # use the official endpoint. All bytes get the same checks.
                if f.get("lfs"):
                    urls.insert(0, ms_url)
                else:
                    urls.append(ms_url)
            assert urls, (repo, name)
            files.append({"name": name, "size": f["size"], "git_blob_id": f["blobId"],
                          "lfs_sha256": f.get("lfs", {}).get("sha256"), "urls": urls})
        specs.append({"repo_id": repo, "revision": official["sha"],
                      "tokenizer_revision": official["sha"], "local_path": f"models/{dirname}",
                      "dtype": "bfloat16", "quantization": "none", "trust_remote_code": False,
                      "mirror_repo_id": mirror, "files": files})
    data = {"created_utc": datetime.now(timezone.utc).isoformat(), "models": specs,
            "verification": "HF official LFS SHA256 for weights, official Git blob SHA1 for non-LFS files; per-file resolved revisions."}
    atomic_json(LOCK, data)
    return data


def verify(path, f):
    if not path.exists() or path.stat().st_size != f["size"]:
        return None
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {f['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
            blob.update(block)
    digest = sha.hexdigest()
    if f["lfs_sha256"]:
        if digest != f["lfs_sha256"]:
            return None
    elif blob.hexdigest() != f["git_blob_id"]:
        return None
    return digest


def download(spec, f):
    path = ROOT / spec["local_path"] / f["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = verify(path, f)
    if digest:
        return {"model": spec["repo_id"], "file": f["name"], "bytes": f["size"],
                "sha256": digest, "status": "already_verified"}
    started = time.monotonic()
    errors = []
    # Retry transient CDN failures as well as switching source.
    for url in f["urls"] * 3:
        event({"event": "start", "model": spec["repo_id"], "file": f["name"],
               "host": urlparse(url).hostname})
        try:
            if f["lfs_sha256"]:
                env = os.environ.copy()
                for name in ("http_proxy", "https_proxy"):
                    if env.get(name.upper()):
                        env.setdefault(name, env[name.upper()])
                if urlparse(url).hostname in {"modelscope.cn", "hf-mirror.com"}:
                    for name in list(env):
                        if name.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                            env.pop(name)
                cmd = ["aria2c", "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=true",
                       "--file-allocation=none", "--split=8", "--max-connection-per-server=8",
                       "--min-split-size=16M", "--connect-timeout=20", "--timeout=45",
                       "--max-tries=4", "--retry-wait=3", "--lowest-speed-limit=512K",
                       "--summary-interval=0", "--console-log-level=warn", "--download-result=hide",
                       "--enable-color=false", "--show-console-readout=false",
                       f"--checksum=sha-256={f['lfs_sha256']}", f"--dir={path.parent}",
                       f"--out={path.name}", url]
                subprocess.run(cmd, check=True, env=env)
            else:
                partial = path.with_name(path.name + ".partial")
                with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                if not verify(partial, f):
                    raise ValueError("File differs from official revision (size or Git blob ID)")
                partial.replace(path)
            digest = verify(path, f)
            if not digest:
                raise ValueError("Downloaded file failed canonical size/hash verification")
            record = {"model": spec["repo_id"], "file": f["name"], "bytes": f["size"],
                      "sha256": digest, "git_blob_verified": not bool(f["lfs_sha256"]),
                      "source_url": url, "status": "downloaded_verified",
                      "seconds": round(time.monotonic() - started, 2)}
            event({"event": "verified", "model": spec["repo_id"], "file": f["name"],
                   "seconds": record["seconds"]})
            return record
        except Exception as error:
            errors.append({"url": url, "error": type(error).__name__ + ": " + str(error)})
            event({"event": "retry_next_source", "file": f["name"],
                   "error_type": type(error).__name__})
    raise RuntimeError(json.dumps({"model": spec["repo_id"], "file": f["name"], "errors": errors}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Select locked repo ID (repeatable)")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    lock = prepare_lock()
    specs = lock["models"]
    if args.model:
        unknown = set(args.model) - {s["repo_id"] for s in specs}
        if unknown:
            parser.error(f"Unknown locked model IDs: {sorted(unknown)}")
        specs = [s for s in specs if s["repo_id"] in args.model]
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prior_files = json.loads(report_path.read_text()).get("files", []) if report_path.exists() else []
    previous = {(f["model"], f["file"], f["sha256"]): f for f in prior_files}
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "files": [], "complete": False}
    atomic_json(report_path, report)
    small = [(s, f) for s in specs for f in s["files"] if not f["lfs_sha256"]]
    weights_by_model = [[(s, f) for f in s["files"] if f["lfs_sha256"]] for s in specs]
    weights = [rows[i] for i in range(max(map(len, weights_by_model))) for rows in weights_by_model if i < len(rows)]
    for jobs, workers in ((small, 4), (weights, 2)):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download, *job) for job in jobs]
            for future in as_completed(futures):
                item = future.result()
                prior = previous.get((item["model"], item["file"], item["sha256"]), {})
                report["files"].append(prior | item)
                atomic_json(report_path, report)
    report["complete"] = True
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["total_verified_bytes"] = sum(f["bytes"] for f in report["files"])
    atomic_json(report_path, report)
    event({"event": "all_models_verified", "bytes": report["total_verified_bytes"]})


if __name__ == "__main__":
    main()
