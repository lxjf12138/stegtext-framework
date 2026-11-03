"""Versioned user messages for dataset-conditioned Instruct continuation."""
from __future__ import annotations

import html
import re

VERSION = 'review-continuation-instruct-v1'
PREFILL_VERSION = 'review-continuation-assistant-prefill-v1'
PREFILL_INSTRUCTIONS = {
    'en': (
        'Continue the movie review that has been started in your response. '
        'Write in English, keeping the viewpoint and tone consistent. '
        'Continue directly from where the opening stops. Output only the '
        'continuation, without repeating the opening or adding a title or explanation.'
    ),
    'zh': (
        '请接着你当前回答中已有的电影评论开头，用中文自然续写影评，保持叙述视角和语气连贯。'
        '从开头的末尾直接接着写，只输出新续写的正文，不重复已有内容，不添加标题或说明。'
    ),
}
TEMPLATES = {
    'en': (
        'Continue the movie review below from where it stops. Write in English, '
        'keeping the viewpoint and tone consistent with the opening. '
        'Output only the new review text, without repeating the opening or '
        'adding a title or explanation.\n\nReview opening:\n{prefix}'
    ),
    'zh': (
        '请从下面的电影评论开头接着往下写，用中文自然续写影评，保持叙述视角和语气连贯。'
        '只输出新续写的正文，不重复开头，不添加标题或说明。\n\n影评开头：\n{prefix}'
    ),
}


def clean_review(text):
    """Decode HTML entities/br tags and normalize whitespace, without paraphrase."""
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<br\s*/?>', ' ', text, flags=re.I))).strip()


def review_prefix(text, language):
    """Use <=256 English codepoints at a word boundary, or 64 Chinese codepoints."""
    limit = {'en': 256, 'zh': 64}[language]
    cleaned = clean_review(text)
    if len(cleaned) < limit + 16:
        raise ValueError('Review too short for the fixed prefix and held-out suffix')
    prefix = cleaned[:limit]
    if language == 'en' and not cleaned[limit].isspace() and not prefix[-1].isspace():
        prefix = prefix.rsplit(' ', 1)[0]
    return prefix.rstrip()


def user_message(prefix, language):
    if not prefix:
        raise ValueError('Review opening must be nonempty')
    return TEMPLATES[language].format(prefix=prefix)
