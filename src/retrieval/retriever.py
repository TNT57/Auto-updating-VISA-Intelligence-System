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
        """Convert distance to a 0-1 relevance score (1 = most relevant)."""
        return max(0.0, 1.0 - self.distance)


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
    """Semantic retriever using ChromaDB vector search."""

    def __init__(self, vectorstore: VectorStoreManager | None = None):
        self.vectorstore = vectorstore or VectorStoreManager()

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

        raw_results = self.vectorstore.query(
            query_text=query,
            n_results=n_results,
        )

        # Parse results into structured objects
        results = []
        if raw_results and raw_results["documents"]:
            docs = raw_results["documents"][0]
            metas = raw_results["metadatas"][0]
            dists = raw_results["distances"][0]

            for doc, meta, dist in zip(docs, metas, dists):
                result = RetrievalResult(
                    content=doc,
                    source=meta.get("source", "unknown"),
                    page_number=meta.get("page_number", 0),
                    chunk_index=meta.get("chunk_index", 0),
                    distance=dist,
                    doc_type=meta.get("doc_type", "unknown"),
                    metadata=meta,
                )
                results.append(result)

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