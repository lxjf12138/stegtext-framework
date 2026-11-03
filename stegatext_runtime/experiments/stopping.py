"""Version-two stopping semantics; no probability kernel is changed here."""
from __future__ import annotations


def normalize(value):
    stop = dict(value)
    allowed = {"mode", "max_steps", "utf8_extension", "sentences", "paragraphs", "min_bits",
               "watchdog_generated_tokens", "watchdog_protocol_decisions", "law_policy", "max_new_tokens"}
    if set(stop) - allowed:
        raise ValueError(f"Unknown stopping fields: {sorted(set(stop)-allowed)}")
    if stop.get("mode") not in ("natural_eos_only", "bounded_operational", "sentence_boundary", "token_budget"):
        raise ValueError("v2 stopping.mode must be natural_eos_only, bounded_operational, sentence_boundary or token_budget")
    stop.setdefault("watchdog_generated_tokens", 512)
    stop.setdefault("watchdog_protocol_decisions", 1024)
    for name in ("watchdog_generated_tokens", "watchdog_protocol_decisions"):
        if type(stop[name]) is not int or stop[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if stop['mode'] == 'token_budget':
        if set(stop) - {'mode','max_new_tokens','watchdog_generated_tokens','watchdog_protocol_decisions'}:
            raise ValueError('token_budget accepts only max_new_tokens and resource watchdogs')
        if type(stop.get('max_new_tokens')) is not int or stop['max_new_tokens'] <= 0:
            raise ValueError('max_new_tokens must be a positive integer')
    elif stop['mode'] == 'sentence_boundary':
        if set(stop) - {'mode','sentences','watchdog_generated_tokens','watchdog_protocol_decisions','law_policy'}:
            raise ValueError('sentence_boundary accepts only one-sentence goal and watchdogs')
        if stop.get('law_policy', 'atomic_operational_v1') != 'atomic_operational_v1':
            raise ValueError('Unknown sentence law policy')
        stop.setdefault('sentences', 1)
        if type(stop['sentences']) is not int or stop['sentences'] != 1:
            raise ValueError('sentence_boundary currently requires sentences=1')
    elif stop["mode"] == "natural_eos_only":
        if any(k in stop for k in ("sentences", "paragraphs", "min_bits")):
            raise ValueError("Text/bit goals are only allowed in bounded_operational")
        # Accept legacy fallback fields from earlier configurations, but remove them from the
        # effective natural-EOS contract. They cannot terminate this mode.
        for name in ("max_steps", "utf8_extension"):
            if name in stop and (type(stop[name]) is not int or stop[name] < (name == "max_steps")):
                raise ValueError(f"Invalid inactive stopping field {name}")
            stop.pop(name, None)
    else:
        stop.setdefault("max_steps", 192)
        stop.setdefault("utf8_extension", 32)
        for name in ("max_steps", "utf8_extension", "sentences", "paragraphs", "min_bits"):
            if name in stop and (type(stop[name]) is not int or stop[name] < (name != "utf8_extension")):
                raise ValueError(f"Invalid stopping field {name}")
        if "sentences" in stop and "paragraphs" in stop:
            raise ValueError("Choose sentences or paragraphs")
    if 'law_policy' in stop and stop['mode'] != 'sentence_boundary':
        raise ValueError('law_policy is only supported for sentence_boundary')
    if 'max_new_tokens' in stop and stop['mode'] != 'token_budget':
        raise ValueError('max_new_tokens is only supported for token_budget')
    return stop


def contract(variant, stop):
    ordinary = variant["kind"] == "ordinary"
    unit = "generated_tokens" if ordinary else "protocol_decisions"
    watchdog = stop["watchdog_generated_tokens" if ordinary else "watchdog_protocol_decisions"]
    result = dict(kind="ordinary-v2" if ordinary else "planner-v2", mode=stop["mode"],
                unit=unit, watchdog_limit=watchdog, eos_is_normal_completion=True,
                watchdog_is_censoring=True, bounded_unit=unit,
                **{k: v for k, v in stop.items() if k != "mode"})
    if stop['mode'] == 'sentence_boundary':
        result.update(sentence_rule='text_rules_v2_first_at_valid_atomic_boundary',
                      last_action='preserved_no_text_clipping', budget='watchdog_only')
    if stop.get('law_policy'):
        result.update(visible_law='operational_method_dependent', strict_cover_law_claim=False,
                      prefix_delivery='terminal-lcp-v1' if variant.get('coder') == 'imec_prefix_exact' else 'online-only-v1')
    if stop['mode'] == 'token_budget':
        result.update(bounded_unit='source_generated_tokens',
                      visible_law='source_capped_at_max_new_tokens',
                      token_budget_rule='per_history_source_cap_v1',
                      sentence_stopping=False, token_limit_is_censoring=True,
                      internal_boundary_eos='delimiter_only_not_natural_eos_or_generated_token')
    return result


def render(codec, data):
    """Retain exact bytes on invalid UTF-8; never repair or resample text."""
    from stegtext.v2.core.errors import TokenizerContractError
    try:
        return codec.bytes_to_text(data), "valid_utf8"
    except (TokenizerContractError, UnicodeDecodeError):
        return None, "invalid_utf8"


def termination(reason, count, limit, unit, framing):
    return dict(terminal_reason=reason, censored=reason in ("resource_watchdog", "token_limit"),
                framing_status=framing,
                watchdog=dict(unit=unit, limit=limit, observed=count,
                              triggered=reason == "resource_watchdog"))
