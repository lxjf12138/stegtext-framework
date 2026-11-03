"""Small model override registry.

Most causal LMs use the generic HF backend.  Overrides are data, not new
backend classes, and are intentionally kept empty until a model requires one.
"""

MODEL_OVERRIDES: dict[str, dict[str, object]] = {}
