from __future__ import annotations
from dataclasses import dataclass
import math


def select_kernel(u_post: float, tau: float) -> str:
    u = float(u_post); t = float(tau)
    if not math.isfinite(u) or u < 0:
        raise ValueError("u_post must be finite and >= 0")
    if math.isnan(t) or t < 0:
        raise ValueError("tau must be in [0, +inf]")
    return "las" if u >= t else "fullsync"


@dataclass(frozen=True)
class AdaptiveLASController:
    """The paper method's sole gate. It does not own the planner or coder."""
    tau: float

    def __post_init__(self):
        t=float(self.tau)
        if math.isnan(t) or t < 0:
            raise ValueError("tau must be in [0, +inf]")

    def select(self, u_post: float) -> str:
        return select_kernel(u_post, self.tau)
