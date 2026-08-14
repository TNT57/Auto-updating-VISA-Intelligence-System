"""
Vectorstore management using ChromaDB.

Handles creation, persistence, and querying of the vector database.
Supports incremental updates — add new documents without rebuilding.
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from transformers import logging as tf_logging

# models, not pdf_loader: importing the dataclass must not require pdfplumber,
# which the deployed app does not install.
from src.ingestion.models import DocumentChunk
from src.utils.config import settings


class VectorStoreManager:
    """Manages ChromaDB vector store for document embeddings."""

    COLLECTION_NAME = "visa_485_documents"

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self._client = None
        self._collection = None
        self._embedding_fn = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Lazy-initialize ChromaDB client with persistence."""
        if self._client is None:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            logger.info("ChromaDB client initialized at {}", self.persist_dir)
        return self._client

    @property
    def embedding_function(self):
        """Lazy-initialize the embedding function (runs locally)."""
        if self._embedding_fn is None:
            logger.info(
                "Loading embedding model: {} ...", settings.embedding_model
            )
            # Suppress noisy transformers v5 loading report warnings
            # (e.g. "UNEXPECTED embeddings.position_ids" which is benign
            # for sentence-transformer models loaded cross-task/architecture)
            _prev_verbosity = tf_logging.get_verbosity()
            tf_logging.set_verbosity_error()
            try:
                self._embedding_fn = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=settings.embedding_model
                    )
                )
            finally:
                tf_logging.set_verbosity(_prev_verbosity)
            logger.info("Embedding model loaded successfully")
        return self._embedding_fn

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the main document collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "485 Visa policy documents",
                    # Chroma defaults to squared L2, which ranges 0-4 for
                    # normalized embeddings and breaks the 1-distance
                    # relevance score in retriever.py. Cosine keeps distances
                    # in 0-2 and makes the score meaningful.
                    # NOTE: changing this requires rebuilding the collection —
                    # run scripts/initial_setup.py --rebuild.
                    "hnsw:space": "cosine",
                },
            )
            logger.info(
                "Collection '{}' loaded — {} documents",
                self.COLLECTION_NAME,
                self._collection.count(),
            )
        return self._collection

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Add document chunks to the vector store.

        Returns the number of chunks added.
        """
        if not chunks:
            logger.warning("No chunks to add to vectorstore")
            return 0

        # Prepare batch data
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # Create a unique ID based on source, page, and chunk index
            chunk_id = (
                f"{chunk.source}_p{chunk.page_number}_c{chunk.chunk_index}_{i}"
            )
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append({
                "source": chunk.source,
                "page_number": chunk.page_number or 0,
                "chunk_index": chunk.chunk_index,
                "doc_type": chunk.doc_type,
                **{
                    k: str(v)
                    for k, v in chunk.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                },
            })

        # Add to collection (upsert — updates if ID exists)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            "Added {} chunks to vectorstore (total: {})",
            len(ids),
            self.collection.count(),
        )
        return len(ids)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> dict:
        """
        Query the vector store for relevant chunks.

        Returns a dict with documents, metadatas, distances, and ids.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        logger.debug(
            "Query returned {} results for: '{}'",
            len(results["documents"][0]) if results["documents"] else 0,
            query_text[:80],
        )
        return results

    def get_collection_stats(self) -> dict:
        """Get statistics about the current collection."""
        count = self.collection.count()
        return {
            "collection_name": self.COLLECTION_NAME,
            "total_chunks": count,
            "persist_dir": self.persist_dir,
            "embedding_model": settings.embedding_model,
        }

    def reset_collection(self) -> None:
        """Delete and recreate the collection (use with caution)."""
        logger.warning("Resetting collection: {}", self.COLLECTION_NAME)
        self.client.delete_collection(self.COLLECTION_NAME)
        self._collection = None  # Force re-creation
        _ = self.collection  # Trigger re-creation
        logger.info("Collection reset complete")

    def delete_document(self, source: str) -> int:
        """Remove all chunks belonging to a specific source file."""
        # Query for chunks from this source
        results = self.collection.get(
            where={"source": source},
        )
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(
                "Deleted {} chunks for source: {}", len(results["ids"]), source
            )
            return len(results["ids"])
        return 0