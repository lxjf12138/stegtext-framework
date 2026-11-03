from __future__ import annotations

from .consistency import ConsistencyMethod
from .direct import DirectMethod
from .las import LASMethod, NaiveLASMethod
from .mwis import MWISMethod
from .syncpool import SyncPoolMethod


def method_factories(backend, options=None):
    options = dict(options or {})
    return {
        "direct": lambda: DirectMethod(),
        "none": lambda: DirectMethod(),
        "consistency": lambda: ConsistencyMethod(
            backend.oracle,
            initial_k=int(options.get("initial_k", 32)),
            max_k=options.get("max_k"),
        ),
        "stepwise": lambda: ConsistencyMethod(
            backend.oracle,
            initial_k=int(options.get("initial_k", 32)),
            max_k=options.get("max_k"),
        ),
        "las": lambda: LASMethod(backend.codec_index, optimized=bool(options.get("optimized", True))),
        "naive_las": lambda: NaiveLASMethod(backend.codec_index),
        "mwis": lambda: MWISMethod(),
        "syncpool": lambda: SyncPoolMethod(),
    }
