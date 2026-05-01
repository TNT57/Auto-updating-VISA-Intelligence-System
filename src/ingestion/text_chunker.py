"""
Text chunking strategy with source tracking.

Splits extracted document text into overlapping chunks
suitable for embedding and retrieval. Preserves metadata
for source citation in RAG responses.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.ingestion.pdf_loader import DocumentChunk
from src.utils.config import settings


class TextChunker:
    """Splits document text into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(
            "TextChunker initialized — size: {}, overlap: {}",
            self.chunk_size,
            self.chunk_overlap,
        )

    def chunk_document(self, pages: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        Split a document's pages into smaller chunks.

        Each page is split independently to maintain page-level
        metadata for citations.
        """
        all_chunks = []

        for page in pages:
            # Split the page text into smaller pieces
            text_pieces = self.splitter.split_text(page.content)

            for idx, piece in enumerate(text_pieces):
                if not piece.strip():
                    continue

                chunk = DocumentChunk(
                    content=piece.strip(),
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=idx,
                    doc_type=page.doc_type,
                    metadata={
                        **page.metadata,
                        "chunk_size": len(piece.strip()),
                        "total_chunks_on_page": len(text_pieces),
                    },
                )
                all_chunks.append(chunk)

        logger.info(
            "Chunked {} pages into {} chunks (size={}, overlap={})",
            len(pages),
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

    def chunk_documents(
        self, documents: list[list[DocumentChunk]]
    ) -> list[DocumentChunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc_pages in documents:
            chunks = self.chunk_document(doc_pages)
            all_chunks.extend(chunks)

        logger.info("Total chunks across all documents: {}", len(all_chunks))
        return all_chunks