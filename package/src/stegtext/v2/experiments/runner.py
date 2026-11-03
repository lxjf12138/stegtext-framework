from __future__ import annotations

from ..coders.registry import CODER_TYPES
from ..methods.registry import method_factories
from ..model.hf_backend import HFBackend
from ..sampling.policies import FullSoftmax, MinP, TopK, TopP
from ..core.runner import StegoRunner
from .config import ExperimentConfig


def build_runner(config: ExperimentConfig):
    backend = HFBackend.from_pretrained(
        config.model.path,
        physical_gpu=config.execution.physical_gpu,
        dtype=config.model.dtype,
        trust_remote_code=config.model.trust_remote_code,
    )
    backend.validate_tokenizer_contract()
    sampling = config.sampling
    if sampling.kind == "full_softmax":
        sampler = FullSoftmax(sampling.temperature)
    elif sampling.kind == "top_k":
        sampler = TopK(int(sampling.top_k or 1), sampling.temperature)
    elif sampling.kind == "top_p":
        sampler = TopP(float(sampling.top_p or 1.0), sampling.temperature)
    elif sampling.kind == "min_p":
        sampler = MinP(float(sampling.min_p or 0.0), sampling.temperature)
    else:
        raise ValueError(f"unknown sampling policy {sampling.kind!r}")
    method_name = config.method.lower()
    factory = method_factories(backend, config.method_options).get(method_name)
    if factory is None:
        raise ValueError(f"unknown method {config.method!r}")
    method = factory()
    coder_type = CODER_TYPES.get(config.coder.lower())
    if coder_type is None:
        raise ValueError(f"unknown coder {config.coder!r}")
    return StegoRunner(backend, sampler, method, coder_type(**config.coder_options))
