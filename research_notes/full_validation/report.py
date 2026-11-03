"""Text-quality markers recorded for generated texts."""
from __future__ import annotations
import json
import re

import pysbd


def read(path): return json.loads(path.read_text())


def quality(text):
    sentences=pysbd.Segmenter(language="en",clean=False).segment(text+" zxqcontinuation")
    complete=[s for s in sentences if "zxqcontinuation" not in s and re.search(r'[.!?。！？][\"”’\']?\s*$',s)]
    words=re.findall(r"\w+",text.lower())
    ngrams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    return dict(complete_sentences=len(complete),replacement_characters=text.count("\ufffd"),
                alphanumeric_characters=sum(c.isalnum() for c in text),punctuation_only=bool(text) and not any(c.isalnum() for c in text),empty_text=not text,
                repeated_4gram_fraction=1-len(set(ngrams))/len(ngrams) if ngrams else 0.,
                ends_at_sentence_boundary=bool(complete) and sentences[-1].strip()=="zxqcontinuation",
                blank_line_paragraphs=len([p for p in re.split(r"\n\s*\n",text.strip()) if p.strip()]))
