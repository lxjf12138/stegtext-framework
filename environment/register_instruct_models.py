"""Append official Instruct snapshots without changing existing model locks.

Small files retain the official Git blob ID; mirrored LFS files must match the
official SHA256 before their URL is accepted. All downloads verify bytes again.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import urllib.request

from environment.download_models import ROOT, LOCK, atomic_json


MODELS = (
    ("qwen_instruct", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507"),
    ("llama_instruct", "meta-llama/Llama-3.2-3B-Instruct", "LLM-Research/Llama-3.2-3B-Instruct"),
)


def metadata(alias, kind, url):
    path = ROOT / f".cache/setup/{alias}_{kind}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, json.load(urllib.request.urlopen(url, timeout=60)))
    return json.loads(path.read_text())


def main():
    lock = json.loads(LOCK.read_text())
    for alias, repo, mirror in MODELS:
        if any(s["repo_id"] == repo for s in lock["models"]):
            if alias == "llama_instruct":
                spec = next(s for s in lock["models"] if s["repo_id"] == repo)
                # The official template otherwise inserts a time-varying date.
                spec["template_options"].setdefault("date_string", datetime.now(timezone.utc).strftime("%d %b %Y"))
            continue
        hf = metadata(alias, "hf", f"https://huggingface.co/api/models/{repo}?blobs=true")
        ms = metadata(alias, "ms", f"https://modelscope.cn/api/v1/models/{mirror}/repo/files?Revision=master&Recursive=true")
        mirror_files = {f["Path"]: f for f in ms["Data"]["Files"]}
        files = []
        for f in hf["siblings"]:
            name = f["rfilename"]
            if "/" in name or name == ".gitattributes":
                continue
            if not (name.endswith((".safetensors", ".json", ".txt", ".md")) or name == "LICENSE"):
                continue
            canonical = f"https://huggingface.co/{repo}/resolve/{hf['sha']}/{name}?download=true"
            urls = [canonical, canonical.replace("huggingface.co", "hf-mirror.com")]
            m = mirror_files.get(name)
            if m:
                if f.get("lfs") and (m["Sha256"] != f["lfs"]["sha256"] or m["Size"] != f["size"]):
                    raise ValueError(f"Mirror weights differ from official snapshot: {repo}/{name}")
                urls.insert(0, f"https://modelscope.cn/models/{mirror}/resolve/{m['Revision']}/{name}?download=true")
            files.append(dict(name=name, size=f["size"], git_blob_id=f["blobId"],
                              lfs_sha256=f.get("lfs", {}).get("sha256"), urls=urls))
        lock["models"].append(dict(repo_id=repo, revision=hf["sha"], tokenizer_revision=hf["sha"],
            local_path=f"models/{repo.split('/')[-1]}", dtype="bfloat16", quantization="none",
            trust_remote_code=False, mirror_repo_id=mirror, prompt_mode="chat",
            template_options={"date_string": datetime.now(timezone.utc).strftime("%d %b %Y")} if alias == "llama_instruct" else {}, files=files))
        print(f"Locked {alias}: {repo}@{hf['sha']}")
    lock["instruct_added_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(LOCK, lock)


if __name__ == "__main__":
    main()
