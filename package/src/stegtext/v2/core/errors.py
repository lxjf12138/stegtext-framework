class InvariantError(RuntimeError):
    """Raised when a scientific or protocol invariant is violated."""


class ProbabilityError(InvariantError):
    """Raised for NaN, negative or non-normalizable probabilities."""


class TokenizerContractError(InvariantError):
    """Raised when a tokenizer cannot provide canonical byte semantics."""


class ReceiverMismatchError(InvariantError):
    """Raised when visible text identifies zero or multiple actions."""
