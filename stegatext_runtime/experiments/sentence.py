"""Shared visible sentence predicate; preserve the complete final action."""
from research_notes.problem_resolution.text_rules import complete_sentences


def observation(text, language):
    units = complete_sentences(text, language) if text is not None else []
    boundary = None
    if units:
        first = units[0].rstrip()
        begin = text.find(first)
        if begin >= 0:
            boundary = len(text[:begin+len(first)].encode('utf-8'))
    size = len(text.encode('utf-8')) if text is not None else None
    return dict(sentence_goal_reached=boundary is not None, first_sentence_utf8_bytes=boundary,
                overshoot_utf8_bytes=size-boundary if boundary is not None else None,
                completed_sentences=len(units), sentence_boundary_rule='text_rules_v2_first_at_valid_atomic_boundary',
                last_action_preserved=True)
