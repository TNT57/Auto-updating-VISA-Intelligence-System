"""
Data structures shared by ingestion and retrieval.

Kept free of third-party imports on purpose. DocumentChunk used to live in
pdf_loader, which imports pdfplumber at module level, so the retrieval path —
which only needs the dataclass — pulled a PDF parsing library in with it. The
deployed app, whose requirements deliberately exclude ingestion dependencies,
then failed to start with "No module named 'pdfplumber'".
"""

from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """A chunk of text extracted from a document with metadata."""
    content: str
    source: str               # File name or URL
    page_number: int | None   # Page number (for PDFs)
    chunk_index: int          # Index within the document
    doc_type: str             # "pdf", "webpage", "text"
    metadata: dict = field(default_factory=dict)
