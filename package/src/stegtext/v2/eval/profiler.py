from __future__ import annotations

import time


class RunProfiler:
    """Small explicit profiler; method/backend counters are merged at finish."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.counters: dict[str, int | float] = {}

    def add(self, name: str, value: int | float = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def finish(self) -> dict[str, int | float]:
        out: dict[str, int | float] = {**self.counters, "total_ms": (time.perf_counter() - self.started) * 1000.0}
        try:
            import torch

            if torch.cuda.is_available():
                out["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        except (ImportError, RuntimeError):
            # Profiling must never hide the experiment's own exception path.
            pass
        return out
