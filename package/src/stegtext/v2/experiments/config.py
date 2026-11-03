from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    path: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class ExecutionConfig:
    physical_gpu: int = 1
    seed: int = 42
    max_steps: int | None = None
    max_tokens: int | None = 128


@dataclass(frozen=True)
class SamplingConfig:
    kind: str = "full_softmax"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    execution: ExecutionConfig = ExecutionConfig()
    sampling: SamplingConfig = SamplingConfig()
    method: str = "direct"
    coder: str = "discop"
    method_options: dict[str, Any] = field(default_factory=dict)
    coder_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ExperimentConfig":
        model = ModelConfig(**dict(data.get("model", {})))
        execution = ExecutionConfig(**dict(data.get("execution", {})))
        sampling = SamplingConfig(**dict(data.get("sampling", {})))
        method_section = data.get("method", "direct")
        coder_section = data.get("coder", "discop")
        method_options = (
            {str(key): value for key, value in method_section.items() if key != "name"}
            if isinstance(method_section, dict)
            else {}
        )
        method = method_section.get("name", "direct") if isinstance(method_section, dict) else str(method_section)
        coder_options = (
            {str(key): value for key, value in coder_section.items() if key != "name"}
            if isinstance(coder_section, dict)
            else {}
        )
        coder = coder_section.get("name", "discop") if isinstance(coder_section, dict) else str(coder_section)
        return cls(
            model=model,
            execution=execution,
            sampling=sampling,
            method=str(method),
            coder=str(coder),
            method_options=method_options,
            coder_options=coder_options,
        )


def load_config(path: str | Path) -> ExperimentConfig:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.9/3.10
        import tomli as tomllib  # type: ignore[no-redef]

    with Path(path).open("rb") as handle:
        return ExperimentConfig.from_mapping(tomllib.load(handle))
