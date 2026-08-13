"""
Semantic retrieval engine for RAG queries.

Handles query embedding, similarity search, and result formatting.
Returns structured results ready for LLM generation.
"""

from dataclasses import dataclass, field

from loguru import logger

from src.ingestion.vectorstore_manager import VectorStoreManager


@dataclass
class RetrievalResult:
    """A single retrieved chunk with relevance scoring."""
    content: str
    source: str
    page_number: int
    chunk_index: int
    distance: float            # ChromaDB distance (lower = more similar)
    doc_type: str
    metadata: dict = field(default_factory=dict)

    @property
    def relevance_score(self) -> float:
        """
        Convert distance to a 0-1 relevance score (1 = most relevant).

        Assumes the collection uses cosine distance (set in
        VectorStoreManager.collection), which ranges 0-2. The clamp keeps
        the score sane if a collection built with another distance metric
        is ever queried.
        """
        return max(0.0, min(1.0, 1.0 - self.distance))


@dataclass
class QueryResults:
    """Complete retrieval results for a query."""
    query: str
    results: list[RetrievalResult]
    total_found: int

    def format_for_llm(self) -> str:
        """Format results as context text for LLM prompt."""
        if not self.results:
            return "No relevant documents found."

        context_parts = []
        for i, r in enumerate(self.results, 1):
            source_info = f"Source: {r.source}, Page {r.page_number}"
            context_parts.append(
                f"[{i}] {source_info}\n{r.content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_sources(self) -> list[dict]:
        """Get unique source citations with excerpt text."""
        seen = set()
        sources = []
        for r in self.results:
            key = f"{r.source}_{r.page_number}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": r.source,
                    "page": r.page_number,
                    "relevance": f"{r.relevance_score:.1%}",
                    "excerpt": r.content[:500].strip() + ("..." if len(r.content) > 500 else ""),
                })
        return sources


class Retriever:
    """
    Semantic retriever over ChromaDB.

    Queries are expanded into several phrasings before searching and the
    ranked lists fused, because applicants and Home Affairs use different
    vocabularies for the same thing. See query_expansion for the evidence.
    """

    # Each variant needs a deeper slice than the caller asked for, or fusion
    # has nothing to disagree about.
    OVERFETCH = 4

    def __init__(
        self,
        vectorstore: VectorStoreManager | None = None,
        expansion: str | None = None,
    ):
        """
        `expansion` is one of:
          "llm"      — paraphrase via the LLM (best measured; costs a call)
          "synonyms" — deterministic jargon mapping (free, instant)
          "none"     — single query, original behaviour

        Defaults to settings.query_expansion (QUERY_EXPANSION in .env).
        """
        from src.utils.config import settings

        self.vectorstore = vectorstore or VectorStoreManager()
        self.expansion = expansion or settings.query_expansion or "synonyms"

    def _variants(self, query: str) -> list[str]:
        if self.expansion == "none":
            return [query]
        if self.expansion == "llm":
            from src.retrieval.query_expansion import expand_with_llm

            return expand_with_llm(query)
        from src.retrieval.query_expansion import expand_with_synonyms

        return expand_with_synonyms(query)

    def _parse(self, raw: dict) -> list[tuple[str, RetrievalResult]]:
        """Turn one Chroma response into (chunk_id, RetrievalResult) pairs."""
        parsed: list[tuple[str, RetrievalResult]] = []
        if not raw or not raw.get("documents"):
            return parsed

        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]

        # Chroma always returns ids, but callers and fakes may not. Fusion
        # only needs a stable key per chunk, so fall back to position.
        raw_ids = (raw.get("ids") or [[]])[0]
        ids = list(raw_ids) if len(raw_ids) == len(docs) else [
            f"__pos_{i}" for i in range(len(docs))
        ]

        # Chroma returns these lists at equal length; strict=False keeps a
        # malformed response from crashing the chat UI.
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists,
                                             strict=False):
            parsed.append((chunk_id, RetrievalResult(
                content=doc,
                source=meta.get("source", "unknown"),
                page_number=meta.get("page_number", 0),
                chunk_index=meta.get("chunk_index", 0),
                distance=dist,
                doc_type=meta.get("doc_type", "unknown"),
                metadata=meta,
            )))
        return parsed

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
    ) -> QueryResults:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: The user's question
            n_results: Number of results to return

        Returns:
            QueryResults with ranked retrieval results
        """
        logger.info("Retrieving for query: '{}'", query[:100])

        variants = self._variants(query)
        if len(variants) > 1:
            logger.info("Expanded into {} phrasings", len(variants))

        ranked_lists: list[list[str]] = []
        by_id: dict[str, RetrievalResult] = {}

        for variant in variants:
            raw = self.vectorstore.query(
                query_text=variant,
                n_results=n_results * self.OVERFETCH,
            )
            parsed = self._parse(raw)
            ranked_lists.append([chunk_id for chunk_id, _ in parsed])
            for chunk_id, result in parsed:
                # Keep the best distance seen for a chunk across variants, so
                # the reported relevance reflects its strongest phrasing.
                existing = by_id.get(chunk_id)
                if existing is None or result.distance < existing.distance:
                    by_id[chunk_id] = result

        if len(ranked_lists) == 1:
            ordered_ids = ranked_lists[0]
        else:
            from src.retrieval.query_expansion import reciprocal_rank_fusion

            fused = reciprocal_rank_fusion(ranked_lists)
            ordered_ids = sorted(fused, key=fused.get, reverse=True)

        results = [by_id[cid] for cid in ordered_ids[:n_results] if cid in by_id]

        logger.info(
            "Retrieved {} results (top relevance: {})",
            len(results),
            results[0].relevance_score if results else "N/A",
        )

        return QueryResults(
            query=query,
            results=results,
            total_found=len(results),
        )