from __future__ import annotations

import gc
import json
import re
import traceback
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from ..coders.registry import CODER_TYPES
from ..core.rng import session_rng
from ..core.runner import StegoRunner
from ..methods.registry import method_factories
from ..model.hf_backend import HFBackend
from ..model.prompt import DefaultPromptPolicy
from ..sampling.policies import FullSoftmax, MinP, TopK, TopP


DEFAULT_SENTENCE_PROMPT = (
    "Write exactly one grammatical English sentence of 8 to 16 words about water. "
    "Output only the sentence, with terminal punctuation, then end your response immediately."
)


@dataclass(frozen=True)
class MatrixModel:
    name: str
    path: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    template_options: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MatrixOptions:
    methods: tuple[str, ...] = ("direct", "consistency", "las", "mwis", "syncpool")
    coders: tuple[str, ...] = ("discop", "range_ac", "imec")
    prompt: str = DEFAULT_SENTENCE_PROMPT
    payload_bits: str = "010101"
    seed: int = 42
    sampling: str = "top_k"
    top_k: int = 32
    top_p: float = 0.95
    min_p: float = 0.05
    temperature: float = 1.0
    max_tokens: int = 96
    physical_gpu: int = 1


def assess_sentence(text: str, *, latent_text: str | None = None) -> dict[str, Any]:
    stripped = str(text).strip()
    words = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)?", stripped)
    sentence_marks = re.findall(r"[.!?]+(?:[\"']|$)", stripped)
    lower_words = [word.lower() for word in words]
    suspicious_fragments = [
        word for word in words if len(word) == 1 and word.lower() not in {"a", "i"}
    ]
    trigrams = [tuple(lower_words[i : i + 3]) for i in range(max(0, len(lower_words) - 2))]
    repeated_trigram = len(set(trigrams)) != len(trigrams)
    leaked_markers = [
        marker
        for marker in ("<|im_start|>", "<|im_end|>", "<|eot_id|>", "<think>", "</think>")
        if marker in stripped
    ]
    issues: list[str] = []
    instruction_issues: list[str] = []
    latent_similarity = None
    if latent_text is not None:
        latent_similarity = SequenceMatcher(None, stripped, str(latent_text).strip()).ratio()
    if not stripped:
        issues.append("empty text")
    if stripped and stripped[-1:] not in ".!?。！？":
        issues.append("missing terminal punctuation")
    if len(sentence_marks) != 1:
        issues.append(f"expected one sentence terminator, found {len(sentence_marks)}")
    if not 8 <= len(words) <= 16:
        instruction_issues.append(f"expected 8-16 English words, found {len(words)}")
    if leaked_markers:
        issues.append("chat/reasoning template marker leaked")
    if repeated_trigram:
        issues.append("repeated trigram")
    if suspicious_fragments:
        issues.append("suspicious one-letter word fragments")
    if latent_similarity is not None and latent_similarity < 1.0:
        issues.append("visible text differs from latent model text")
    return {
        "passed": not issues,
        "issues": issues,
        "instruction_passed": not instruction_issues,
        "instruction_issues": instruction_issues,
        "word_count": len(words),
        "sentence_terminators": len(sentence_marks),
        "repeated_trigram": repeated_trigram,
        "suspicious_fragments": suspicious_fragments,
        "leaked_markers": leaked_markers,
        "latent_text": latent_text,
        "latent_similarity": latent_similarity,
    }


def eos_diagnosis(encoded: Mapping[str, Any]) -> dict[str, Any]:
    stats = dict(encoded.get("stats", {}))
    trace = list(stats.get("trace", ()))
    eos_steps = [step for step in trace if float(step.get("eos_probability", 0.0)) > 0.0]
    ranks = [int(step["eos_rank"]) for step in eos_steps if step.get("eos_rank") is not None]
    probabilities = [float(step["eos_probability"]) for step in trace]
    diagnosis = "natural EOS selected"
    if encoded.get("stop_reason") != "eos":
        if not eos_steps:
            diagnosis = "EOS never entered sampler support; inspect prompt/model readiness or truncation policy"
        else:
            diagnosis = "EOS entered sampler support but the deterministic coder path did not select it"
    return {
        "summary": diagnosis,
        "eos_in_support_steps": len(eos_steps),
        "max_eos_probability": max(probabilities, default=0.0),
        "best_eos_rank": min(ranks, default=None),
        "last_step": trace[-1] if trace else None,
    }


def _sampler(options: MatrixOptions):
    if options.sampling == "full_softmax":
        return FullSoftmax(options.temperature)
    if options.sampling == "top_k":
        return TopK(options.top_k, options.temperature)
    if options.sampling == "top_p":
        return TopP(options.top_p, options.temperature)
    if options.sampling == "min_p":
        return MinP(options.min_p, options.temperature)
    raise ValueError(f"unknown sampling policy {options.sampling!r}")


def _json_result(result: Any) -> dict[str, Any]:
    return asdict(result)


def run_case(
    backend: HFBackend,
    model: MatrixModel,
    method_name: str,
    coder_name: str,
    options: MatrixOptions,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model": model.name,
        "model_path": model.path,
        "method": method_name,
        "coder": coder_name,
        "sampling": options.sampling,
        "sampling_options": {
            "top_k": options.top_k,
            "top_p": options.top_p,
            "min_p": options.min_p,
            "temperature": options.temperature,
        },
        "prompt": options.prompt,
        "prompt_template_options": dict(model.template_options or {}),
        "payload_bits": options.payload_bits,
        "seed": options.seed,
        "max_tokens_guard": options.max_tokens,
    }
    try:
        factory = method_factories(
            backend,
            {"initial_k": options.top_k, "max_k": max(options.top_k, 128)},
        ).get(method_name)
        coder_type = CODER_TYPES.get(coder_name)
        if factory is None:
            raise ValueError(f"unknown method {method_name!r}")
        if coder_type is None:
            raise ValueError(f"unknown coder {coder_name!r}")
        coder_options = {"precision": 40} if coder_name == "range_ac" else {}
        if coder_name == "imec":
            coder_options = {"block_size": 6, "window_size": 3}
        runner = StegoRunner(backend, _sampler(options), factory(), coder_type(**coder_options))
        rng_key = f"eos-matrix:{options.seed}:{model.name}:{method_name}:{coder_name}".encode("utf-8")
        encoded = runner.encode(
            options.prompt,
            options.payload_bits,
            rng=session_rng(rng_key),
            max_tokens=options.max_tokens,
            collect_trace=True,
        )
        record["encode"] = _json_result(encoded)
        record["sentence"] = assess_sentence(
            encoded.text,
            latent_text=encoded.stats.get("latent_text"),
        )
        record["eos_diagnosis"] = eos_diagnosis(record["encode"])
        decoded = runner.decode(
            options.prompt,
            encoded.text,
            rng=session_rng(rng_key),
            max_tokens=options.max_tokens,
            collect_trace=encoded.finished,
        )
        record["decode"] = _json_result(decoded)
        protocol_checks = {
            "encode_natural_eos": encoded.finished and encoded.stop_reason == "eos",
            "decode_natural_eos": decoded.finished and decoded.stop_reason == "eos",
            "text_roundtrip": decoded.text == encoded.text,
            "bit_roundtrip": decoded.bits == encoded.emitted_bits,
            "target_fully_consumed": decoded.stats.get("target_remaining") == 0,
            "terminal_token_is_eos": encoded.stats.get("terminal_token_id")
            in encoded.stats.get("eos_token_ids", ()),
            "payload_complete": encoded.emitted_bits.startswith(options.payload_bits),
        }
        record["protocol_checks"] = protocol_checks
        required = {key: value for key, value in protocol_checks.items() if key != "payload_complete"}
        record["protocol_passed"] = all(required.values())
        record["passed"] = record["protocol_passed"] and record["sentence"]["passed"]
    except Exception as exc:
        record["passed"] = False
        record["protocol_passed"] = False
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    return record


def run_matrix(
    models: Sequence[MatrixModel],
    options: MatrixOptions,
    *,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with destination.open("w", encoding="utf-8") as handle:
        for model in models:
            backend: HFBackend | None = None
            try:
                backend = HFBackend.from_pretrained(
                    model.path,
                    physical_gpu=options.physical_gpu,
                    dtype=model.dtype,
                    trust_remote_code=model.trust_remote_code,
                    prompt_policy=DefaultPromptPolicy(template_options=model.template_options),
                )
                backend.validate_tokenizer_contract()
                for method_name in options.methods:
                    for coder_name in options.coders:
                        record = run_case(backend, model, method_name, coder_name, options)
                        results.append(record)
                        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                        handle.flush()
            except Exception as exc:
                record = {
                    "model": model.name,
                    "model_path": model.path,
                    "passed": False,
                    "protocol_passed": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
                results.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
            finally:
                del backend
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return results


def summarize_matrix(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    case_rows = [row for row in rows if "method" in row and "coder" in row]
    return {
        "rows": len(case_rows),
        "models": sorted({str(row.get("model")) for row in rows}),
        "natural_eos": sum(
            row.get("encode", {}).get("finished") is True
            and row.get("encode", {}).get("stop_reason") == "eos"
            for row in case_rows
        ),
        "roundtrip_passed": sum(bool(row.get("protocol_passed")) for row in case_rows),
        "sentence_passed": sum(bool(row.get("sentence", {}).get("passed")) for row in case_rows),
        "payload_complete": sum(
            bool(row.get("protocol_checks", {}).get("payload_complete")) for row in case_rows
        ),
        "passed": sum(bool(row.get("passed")) for row in case_rows),
        "failed_rows": [
            {
                "model": row.get("model"),
                "method": row.get("method"),
                "coder": row.get("coder"),
                "stop_reason": row.get("encode", {}).get("stop_reason"),
                "text": row.get("encode", {}).get("text"),
                "sentence_issues": row.get("sentence", {}).get("issues"),
                "instruction_issues": row.get("sentence", {}).get("instruction_issues"),
                "error": row.get("error"),
                "eos_diagnosis": row.get("eos_diagnosis"),
            }
            for row in case_rows
            if not row.get("passed")
        ],
    }
