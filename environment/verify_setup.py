"""Validate isolation, frozen core selftests, and offline BF16 CUDA forwards."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import site
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def main():
    assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), sys.prefix
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    downloads = json.loads((HERE / "model-downloads.json").read_text())
    assert downloads["complete"], "Complete verified model downloads first"
    lock = json.loads((HERE / "models.lock.json").read_text())
    versions = {n: importlib.metadata.version(n) for n in (
        "torch", "transformers", "accelerate", "numpy", "pandas", "pyarrow",
        "huggingface-hub", "safetensors", "sentencepiece", "pytest", "pypdf")}
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
              "base_prefix": sys.base_prefix, "user_site_enabled": site.ENABLE_USER_SITE,
              "versions": versions,
              "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
              "model_lock_sha256": hashlib.sha256((HERE / "models.lock.json").read_bytes()).hexdigest()}
    with (HERE / "selftest.log").open("w") as out:
        subprocess.run([sys.executable, "-B", str(ROOT / "package/scripts/selftest.py")],
                       cwd=ROOT, env=os.environ.copy(), stdout=out, stderr=subprocess.STDOUT, check=True)
    report["frozen_core_and_starter_selftests"] = "passed"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicLayer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    report["cuda"] = {"torch_cuda_version": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(0), "logical_device": "cuda:0",
                      "bf16_supported": True, "DynamicLayer_import": True,
                      "qwen3_and_llama_classes_import": True}
    report["models"] = []
    for spec in lock["models"]:
        path = ROOT / spec["local_path"]
        print("Loading local BF16 checkpoint:", spec["repo_id"], flush=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
        assert all(p.device.type == "cuda" and p.dtype == torch.bfloat16 for p in model.parameters())
        prompt = "隐写研究需要严谨的数学基础。" if spec["repo_id"].startswith("Qwen/") else "Steganography requires careful mathematical reasoning."
        encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            first = model(**encoded, use_cache=True, logits_to_keep=1)
            assert torch.isfinite(first.logits).all()
            token = first.logits[:, -1].argmax(dim=-1, keepdim=True)
            second = model(input_ids=token, past_key_values=first.past_key_values,
                           attention_mask=torch.cat((encoded.attention_mask,
                               torch.ones((1, 1), dtype=encoded.attention_mask.dtype, device="cuda:0")), dim=1),
                           use_cache=True, logits_to_keep=1)
            assert torch.isfinite(second.logits).all()
        torch.cuda.synchronize(0)
        layers = getattr(second.past_key_values, "layers", ())
        record = {"repo_id": spec["repo_id"], "revision": spec["revision"],
                  "local_path": spec["local_path"], "model_class": type(model).__name__,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "parameter_dtype": "bfloat16", "quantization": "none",
                  "prompt_mode": "plain_completion", "local_files_only": True,
                  "tokenizer_size": len(tokenizer), "output_head_size": int(first.logits.shape[-1]),
                  "prompt_tokens": int(encoded.input_ids.shape[1]),
                  "full_forward_finite": True, "incremental_forward_finite": True,
                  "logits_to_keep_supported": True, "cache_type": type(second.past_key_values).__name__,
                  "cache_layer_types": sorted({type(x).__name__ for x in layers}),
                  "all_standard_DynamicLayer": bool(layers) and all(type(x) is DynamicLayer for x in layers),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        report["models"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        del second, first, token, encoded, model, tokenizer, layers
        gc.collect()
        torch.cuda.empty_cache()
    report["scope"] = "Environment setup only: original CPU selftests and two short offline BF16 CUDA forwards per model; no steganography experiment or threshold calibration."
    (HERE / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Environment and both local checkpoints verified.", flush=True)


if __name__ == "__main__":
    main()
