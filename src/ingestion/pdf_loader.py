"""
PDF document loader with page-level metadata extraction.

Supports both pypdf and pdfplumber for robust text extraction.
Preserves source information for citation in RAG responses.
"""

import hashlib
from pathlib import Path

import pdfplumber
from loguru import logger

# Re-exported for existing callers. DocumentChunk itself lives in models.py so
# that importing the dataclass does not require pdfplumber — the retrieval
# path needs the former and not the latter.
from src.ingestion.models import DocumentChunk
from src.utils.config import settings

__all__ = ["DocumentChunk", "PDFLoader"]


class PDFLoader:
    """Loads PDF files and extracts text with page metadata."""

    def __init__(self, pdf_dir: Path | None = None):
        self.pdf_dir = pdf_dir or settings.raw_pdf_dir

    def list_pdfs(self) -> list[Path]:
        """List all PDF files in the configured directory."""
        if not self.pdf_dir.exists():
            logger.warning("PDF directory does not exist: {}", self.pdf_dir)
            return []
        pdfs = list(self.pdf_dir.glob("*.pdf"))
        logger.info("Found {} PDF files in {}", len(pdfs), self.pdf_dir)
        return pdfs

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of a file for change detection."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def load_pdf(self, file_path: Path) -> list[DocumentChunk]:
        """
        Extract text and tables from a PDF file, page by page.

        Tables are extracted via pdfplumber and formatted as markdown
        rows appended to the page text, preserving structured data
        (fees, processing times, eligibility criteria, etc.).

        Returns a list of DocumentChunk objects, one per page.
        """
        chunks = []
        source_name = file_path.name

        try:
            with pdfplumber.open(file_path) as pdf:
                logger.info(
                    "Loading PDF: {} ({} pages)", source_name, len(pdf.pages)
                )
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""

                    # --- Table-aware extraction (Improvement #4) ---
                    tables = page.extract_tables()
                    if tables:
                        table_parts = []
                        for t_idx, table in enumerate(tables):
                            if not table or len(table) < 2:
                                continue
                            # First row as header
                            header = table[0]
                            rows = table[1:]
                            # Build markdown table
                            md_lines = []
                            clean_header = [
                                (c or "").strip().replace("\n", " ")
                                for c in header
                            ]
                            md_lines.append("| " + " | ".join(clean_header) + " |")
                            md_lines.append(
                                "| " + " | ".join("---" for _ in clean_header) + " |"
                            )
                            for row in rows:
                                clean_row = [
                                    (c or "").strip().replace("\n", " ")
                                    for c in row
                                ]
                                md_lines.append(
                                    "| " + " | ".join(clean_row) + " |"
                                )
                            table_parts.append(
                                f"\n\n[Table {t_idx + 1}]\n"
                                + "\n".join(md_lines)
                            )
                        if table_parts:
                            text += "".join(table_parts)

                    if text.strip():
                        chunk = DocumentChunk(
                            content=text.strip(),
                            source=source_name,
                            page_number=page_num,
                            chunk_index=0,  # Will be set by chunker
                            doc_type="pdf",
                            metadata={
                                "total_pages": len(pdf.pages),
                                "file_hash": self.compute_file_hash(file_path),
                                "has_tables": bool(tables),
                            },
                        )
                        chunks.append(chunk)

                logger.info(
                    "Extracted {} pages with content from {}",
                    len(chunks),
                    source_name,
                )

        except Exception as e:
            logger.error("Failed to load PDF {}: {}", source_name, e)

        return chunks

    def load_all_pdfs(self) -> list[DocumentChunk]:
        """Load and extract text from all PDFs in the configured directory."""
        all_chunks = []
        pdfs = self.list_pdfs()

        if not pdfs:
            logger.warning(
                "No PDFs found in {}. Add PDF files and re-run.",
                self.pdf_dir,
            )
            return all_chunks

        for pdf_path in pdfs:
            chunks = self.load_pdf(pdf_path)
            all_chunks.extend(chunks)

        logger.info(
            "Total: {} pages extracted from {} PDF files",
            len(all_chunks),
            len(pdfs),
        )
        return all_chunks