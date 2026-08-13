"""
Query expansion for the vocabulary gap between questions and source text.

Applicants ask in migration-agent shorthand — "onshore", "offshore", "PR",
"work rights". Home Affairs pages never use those words; they say "be in
Australia when you apply", "outside Australia", "permanent residence". Dense
retrieval scores the right passage highly when the question happens to share
the site's phrasing and poorly when it doesn't:

    "Do I need to be in Australia when I apply?"    rank 1, 62%
    "Does it matter, onshore or offshore?"          rank 3, 25%

Same chunk, same model. So the fix is not a bigger embedding model — it is
asking the question in more than one vocabulary and fusing the results.

Two expanders:
  * `expand_with_synonyms` — deterministic, no network, no cost. Covers the
    known jargon of this domain.
  * `expand_with_llm` — paraphrases via the configured LLM. Broader coverage
    for phrasings nobody anticipated, at the cost of a call per query.
"""

import re

from loguru import logger

# Migration shorthand -> the wording Home Affairs actually publishes.
# Keys are matched case-insensitively on word boundaries.
DOMAIN_SYNONYMS: dict[str, list[str]] = {
    "onshore": ["in Australia when you apply", "be in Australia"],
    "offshore": ["outside Australia", "not in Australia"],
    "pr": ["permanent residence", "permanent visa"],
    "permanent residency": ["permanent residence"],
    "work rights": ["work in Australia", "conditions"],
    "english test": ["English language requirements", "acceptable minimum scores"],
    "english requirement": ["English language requirements"],
    "cost": ["visa application charge", "the visa costs"],
    "price": ["visa application charge", "the visa costs"],
    "fee": ["visa application charge", "the visa costs"],
    "how long": ["stay period", "length of stay", "processing times"],
    "age limit": ["aged 35 years or under", "age requirement"],
    "insurance": ["adequate health insurance"],
    "partner": ["family members", "subsequent entrant"],
    "spouse": ["family members", "subsequent entrant"],
    "kids": ["family members", "dependent children"],
    "children": ["family members", "dependent children"],
    "police check": ["Australian Federal Police check", "character requirements"],
    "extend": ["second Temporary Graduate visa", "further stay"],
    "apply from": ["be in Australia when you apply", "at the time of decision"],
}

MAX_VARIANTS = 4

LLM_EXPANSION_PROMPT = """Rewrite this question about the Australian Temporary \
Graduate (subclass 485) visa into {n} alternative phrasings that would match \
the wording used on official Department of Home Affairs pages.

Rules:
- Use plain official terminology, not migration-agent slang.
- Keep each rewrite a single line, no numbering, no commentary.
- Preserve the meaning exactly. Do not answer the question.

Question: {question}"""


def expand_with_synonyms(query: str) -> list[str]:
    """
    Return `query` plus variants where known jargon is swapped for official
    wording. The original is always first and always kept.
    """
    variants: list[str] = []
    lowered = query.lower()

    for term, replacements in DOMAIN_SYNONYMS.items():
        if not re.search(rf"\b{re.escape(term)}\b", lowered):
            continue
        for replacement in replacements:
            variant = re.sub(
                rf"\b{re.escape(term)}\b", replacement, query, flags=re.IGNORECASE
            )
            if variant.lower() != lowered and variant not in variants:
                variants.append(variant)

    return [query, *variants[:MAX_VARIANTS]]


def expand_with_llm(query: str, llm=None, n: int = 3) -> list[str]:
    """
    Paraphrase `query` into official wording via the LLM.

    Falls back to synonym expansion if the LLM is unavailable or errors —
    retrieval must never fail because a rewrite call did.
    """
    try:
        if llm is None:
            from src.generation.llm_client import LLMClient

            llm = LLMClient()

        raw = llm.generate(
            prompt=LLM_EXPANSION_PROMPT.format(question=query, n=n),
            system_prompt="You rewrite search queries. Output only the rewrites.",
            max_tokens=200,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("LLM query expansion failed ({}), using synonyms", exc)
        return expand_with_synonyms(query)

    variants = []
    for line in (raw or "").splitlines():
        cleaned = line.strip().lstrip("-•*0123456789. ").strip()
        if cleaned and cleaned.lower() != query.lower():
            variants.append(cleaned)

    if not variants:
        return expand_with_synonyms(query)

    return [query, *variants[:MAX_VARIANTS]]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> dict[str, float]:
    """
    Fuse several ranked ID lists into one score per ID.

    RRF sums 1/(k + rank) across lists, so a chunk that appears mid-table for
    several phrasings beats one that tops a single phrasing. That is exactly
    the behaviour wanted here: agreement across vocabularies is the signal.
    `k` damps the influence of the very top ranks; 60 is the value from the
    original RRF paper and is not sensitive.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
