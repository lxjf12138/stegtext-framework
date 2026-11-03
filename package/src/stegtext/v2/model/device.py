from __future__ import annotations

import os
import torch


def resolve_physical_device(physical_index: int = 1) -> str:
    """Resolve a physical GPU index without silently falling back to GPU0."""
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable; a real-model v2 run requires physical GPU{int(physical_index)}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        values = [x.strip() for x in visible.split(",") if x.strip()]
        wanted = str(int(physical_index))
        if wanted not in values:
            raise RuntimeError(f"physical GPU {physical_index} is hidden by CUDA_VISIBLE_DEVICES={visible!r}")
        return f"cuda:{values.index(wanted)}"
    count = torch.cuda.device_count()
    if int(physical_index) < 0 or int(physical_index) >= count:
        raise RuntimeError(f"physical GPU {physical_index} is unavailable (count={count})")
    return f"cuda:{int(physical_index)}"
