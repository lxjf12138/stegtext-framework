"""Conservative text-shape screening; no claim of grammatical correctness."""
import re

import pysbd


_EN = pysbd.Segmenter(language="en", clean=False)


def complete_sentences(text, language="en"):
    if language == "zh":
        units = re.findall(r"[^。！？]*[。！？]+[\"”’']?", text)
    else:
        units = [u for u in _EN.segment(text+" zxqcontinuation")
                 if "zxqcontinuation" not in u and re.search(r'[.!?][\"”’\']?\s*$', u)]
    return [u for u in units if any(c.isalpha() for c in u)]


def sentence_end(text, language="en"):
    units = complete_sentences(text, language)
    ending = r'[。！？][\"”’\']?\s*$' if language == "zh" else r'[.!?][\"”’\']?\s*$'
    return bool(units) and text.rstrip().endswith(units[-1].rstrip()) and bool(re.search(ending, text))
