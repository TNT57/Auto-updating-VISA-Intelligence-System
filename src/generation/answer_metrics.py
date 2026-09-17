"""
Deterministic measures of answer *shape*.

Retrieval quality has been measured since `scripts/eval_retrieval.py`; answer
quality never was, which is how three identical IELTS blocks shipped in one
reply. These functions are the offline half of that: pure string checks, no
model, no network, so they can be unit-tested and asserted on.

They measure shape, not truth. Whether an answer is *correct* is checked by
marker presence in the eval set and by the grounding call in the live path.
"""

import re

# Matches money, band scores, ages and years: AUD5,750.00 / 6.5 / 35 / 2026.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def word_count(answer: str) -> int:
    """Words in the answer. A proxy for length, not for readability."""
    return len(answer.split())


def streams_named(answer: str) -> dict[str, int]:
    """
    How many times each stream is named.

    Counts the name with or without a trailing "stream", so
    "Post-Vocational Education Work" and "Post-Vocational Education Work
    stream" are the same mention.
    """
    from src.retrieval.retriever import RetrievalResult

    low = answer.lower()
    counts: dict[str, int] = {}
    for name in RetrievalResult.STREAM_NAMES.values():
        bare = name.lower().removesuffix(" stream")
        n = low.count(bare)
        if n:
            counts[name] = n
    return counts


def distinct_numbers(answer: str) -> set[str]:
    """Distinct numeric tokens, normalised so 5,750 and 5750 are one value."""
    return {m.replace(",", "").rstrip(".") for m in _NUMBER.findall(answer)}


def leads_with_answer(answer: str, max_words: int = 40) -> bool:
    """
    Whether the reply opens with prose rather than a heading or a list.

    "Answer first, then detail" means the reader gets the figure before the
    breakdown. An answer that opens with a bullet has buried it.
    """
    for line in answer.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in "#-*|>" or stripped.startswith("1."):
            return False
        return len(stripped.split()) <= max_words
    return False


def has_blanket_disclaimer(answer: str) -> bool:
    """Whether the model appended a generic not-legal-advice notice."""
    low = answer.lower()
    return "not legal advice" in low or "registered migration agent" in low


def _stream_segments(answer: str) -> list[set[str]]:
    """
    Split the answer at each stream mention and collect each segment's numbers.

    A redundant answer repeats the same figures under one heading per stream,
    so comparing the numbers *per segment* is what distinguishes it from a
    genuine per-stream breakdown, where the segments differ.
    """
    from src.retrieval.retriever import RetrievalResult

    low = answer.lower()
    cuts: list[int] = []
    for name in RetrievalResult.STREAM_NAMES.values():
        bare = name.lower().removesuffix(" stream")
        start = low.find(bare)
        while start != -1:
            cuts.append(start)
            start = low.find(bare, start + 1)

    if len(cuts) < 2:
        return []

    cuts.sort()
    bounds = cuts + [len(answer)]
    return [
        distinct_numbers(answer[bounds[i]:bounds[i + 1]])
        for i in range(len(cuts))
    ]


def redundant_enumeration(answer: str) -> bool:
    """
    Whether the answer repeats the same figures once per stream.

    This is the bug the user reported: three streams, one IELTS score, three
    identical blocks. Detected by segmenting the answer at each stream mention
    and comparing the numbers in each segment — two segments carrying the same
    non-empty set of figures means the same value was stated twice under two
    headings.

    Deliberately narrow. A genuine breakdown (AUD5,750 for one stream,
    AUD2,265 for another) has different numbers per segment and does not trip
    it. That asymmetry matters: a false positive here would push us to collapse
    two values that really do differ.
    """
    segments = [s for s in _stream_segments(answer) if s]
    if len(segments) < 2:
        return False

    return any(
        segments[i] == segments[j]
        for i in range(len(segments))
        for j in range(i + 1, len(segments))
    )
