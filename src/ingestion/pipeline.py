"""
Shared ingestion steps: turn fetched pages and downloaded PDFs into vectors.

Both entry points use these — `scripts/fetch_and_index.py` for a one-off build
of the knowledge base, and `scripts/daily_update.py` for the scheduled refresh.
Keeping them here means the two scripts cannot drift apart.
"""

from pathlib import Path

from loguru import logger

from src.utils.config import settings
from src.utils.db_manager import DatabaseManager


def is_relevant_pdf(pages: list, keywords: list[str] | None = None) -> bool:
    """
    Whether an extracted PDF is about the visa rather than generic paperwork.

    Home Affairs pages link their universal application forms (80, 1221, 956,
    47a...) from every visa page. Those are form-filling instructions, not
    policy, and indexing them contributed over half the corpus while mentioning
    the visa zero times. Judged on extracted text rather than a filename
    denylist, so a renamed or newly-linked form is handled too.
    """
    keywords = (
        settings.pdf_relevance_keywords if keywords is None else keywords
    )
    if not keywords:
        return True

    blob = " ".join(p.content for p in pages).lower()
    return any(kw.lower() in blob for kw in keywords)


def ingest_pdfs(db: DatabaseManager, pdf_dir: Path | None = None) -> int:
    """
    Ingest new or changed PDFs into ChromaDB.

    A PDF whose hash matches the recorded one is skipped, so re-running this is
    cheap. PDFs that fail the relevance check are skipped entirely. Returns the
    number of chunks added.
    """
    from src.ingestion.pdf_loader import PDFLoader
    from src.ingestion.text_chunker import TextChunker
    from src.ingestion.vectorstore_manager import VectorStoreManager

    pdf_dir = pdf_dir or settings.raw_pdf_dir
    if not pdf_dir.exists():
        return 0

    loader = PDFLoader(pdf_dir)
    pdfs = loader.list_pdfs()
    if not pdfs:
        return 0

    total_chunks = 0
    chunker = TextChunker()
    vs = VectorStoreManager()

    for pdf_path in pdfs:
        file_hash = loader.compute_file_hash(pdf_path)

        with db.get_session() as session:
            from src.utils.db_manager import Document

            existing = (
                session.query(Document)
                .filter(Document.file_path == str(pdf_path))
                .first()
            )
            if existing and existing.file_hash == file_hash:
                logger.debug("PDF unchanged, skipping: {}", pdf_path.name)
                continue

        logger.info("Ingesting PDF: {}", pdf_path.name)
        pages = loader.load_pdf(pdf_path)
        if not pages:
            logger.warning("No text extracted from {} — scanned image?",
                           pdf_path.name)
            continue

        if not is_relevant_pdf(pages):
            logger.info(
                "Skipping {} — no mention of {}; generic form, not policy",
                pdf_path.name, settings.pdf_relevance_keywords,
            )
            # Drop any chunks indexed before the filter existed.
            vs.delete_document(pdf_path.name)
            continue

        chunks = chunker.chunk_document(pages)
        # Replace this file's previous chunks so an updated edition doesn't
        # leave the old text behind to be retrieved later.
        vs.delete_document(pdf_path.name)
        vs.add_chunks(chunks)

        db.add_document(
            file_path=str(pdf_path),
            file_hash=file_hash,
            source_url=None,
        )
        total_chunks += len(chunks)
        logger.info("Ingested {} chunks from {}", len(chunks), pdf_path.name)

    return total_chunks


def ingest_pages(results: list[dict]) -> int:
    """
    Embed scraped page text into ChromaDB.

    Without this the chat can only answer from PDFs, leaving the monitored
    pages — processing times and fees among them — invisible to it. Each URL's
    chunks are deleted before re-adding so stale page content doesn't
    accumulate across runs.

    Returns the number of chunks added.
    """
    from src.ingestion.pdf_loader import DocumentChunk
    from src.ingestion.text_chunker import TextChunker
    from src.ingestion.vectorstore_manager import VectorStoreManager

    pages = [
        r for r in results
        if not r.get("error") and (r.get("content") or "").strip()
    ]
    if not pages:
        return 0

    chunker = TextChunker()
    vs = VectorStoreManager()
    total_chunks = 0

    for result in pages:
        url = result["url"]
        # `source` doubles as the citation label and the delete key, so it has
        # to be stable across runs — the page title is not.
        page = DocumentChunk(
            content=result["content"],
            source=url,
            page_number=1,
            chunk_index=0,
            doc_type="webpage",
            metadata={
                "title": result.get("title") or "",
                "content_hash": result.get("content_hash") or "",
                "scraped_at": result.get("timestamp") or "",
            },
        )

        chunks = chunker.chunk_document([page])
        if not chunks:
            continue

        vs.delete_document(url)
        vs.add_chunks(chunks)

        total_chunks += len(chunks)
        logger.info("Ingested {} chunks from page {}", len(chunks), url)

    return total_chunks
