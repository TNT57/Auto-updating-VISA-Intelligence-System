"""
End-to-end integration tests using a real ChromaDB collection.

Everything else in the suite mocks the vector store, which means retrieval
quality is never actually exercised — the mocks return fixed-order results
regardless of the query. These tests build a real collection from a generated
PDF and assert that the right chunk ranks first for a real query.

They are slow (the embedding model is downloaded on first run) and are
deselected by default. Run them with:

    pytest -m integration
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.integration


PAGES = {
    "English language requirements": (
        "To satisfy the English language requirement for the Temporary "
        "Graduate visa you must provide a test score of IELTS 6.0 overall, "
        "with a minimum of 5.0 in each of the four components. PTE Academic "
        "and TOEFL iBT are also accepted."
    ),
    "Application charge": (
        "The base application charge for the Subclass 485 visa is AUD 2,235. "
        "Additional applicant charges apply for family members included in "
        "the application. Payment is made at the time of lodgement."
    ),
    "Visa validity period": (
        "The Post-Higher Education Work stream allows a stay of up to two "
        "years for bachelor degree holders and up to three years for masters "
        "by research graduates. The period begins on the date the visa is "
        "granted."
    ),
}


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    """Generate a small multi-page PDF covering three distinct topics."""
    fpdf = pytest.importorskip("fpdf", reason="fpdf2 is required for this test")

    pdf_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = pdf_dir / "485_test_doc.pdf"

    doc = fpdf.FPDF()
    for heading, body in PAGES.items():
        doc.add_page()
        # new_x/new_y reset the cursor to the left margin on the next line;
        # the default (XPos.RIGHT) would leave no width for the next cell.
        doc.set_font("Helvetica", size=14)
        doc.multi_cell(0, 10, heading, new_x="LMARGIN", new_y="NEXT")
        doc.set_font("Helvetica", size=11)
        doc.multi_cell(0, 8, body, new_x="LMARGIN", new_y="NEXT")
    doc.output(str(pdf_path))

    return pdf_path


@pytest.fixture(scope="module")
def pdf_chunks(sample_pdf):
    """Extract and chunk the generated PDF. No network needed."""
    from src.ingestion.pdf_loader import PDFLoader
    from src.ingestion.text_chunker import TextChunker

    pages = PDFLoader(pdf_dir=sample_pdf.parent).load_all_pdfs()
    assert pages, "PDF extraction produced no pages"

    chunks = TextChunker().chunk_document(pages)
    assert chunks, "Chunking produced no chunks"
    return chunks


@pytest.fixture(scope="module")
def populated_store(pdf_chunks, tmp_path_factory):
    """
    Build a real ChromaDB collection from the generated PDF.

    Skips rather than fails where the embedding model cannot be fetched —
    sandboxed environments often block huggingface.co. CI has network access,
    so this runs for real there.
    """
    from src.ingestion.vectorstore_manager import VectorStoreManager

    persist_dir = tmp_path_factory.mktemp("vectorstore")
    store = VectorStoreManager(persist_dir=str(persist_dir))

    try:
        store.add_chunks(pdf_chunks)
    except Exception as exc:  # noqa: BLE001 — any transport/model failure
        pytest.skip(f"Embedding model unavailable ({type(exc).__name__}: {exc})")

    return store


class TestRealIngestion:
    """PDF → pages → chunks against a real file. No network required."""

    def test_every_page_is_extracted(self, pdf_chunks):
        sources = {c.source for c in pdf_chunks}
        pages = {c.page_number for c in pdf_chunks}

        assert sources == {"485_test_doc.pdf"}
        assert pages == {1, 2, 3}

    def test_chunk_content_survives_extraction(self, pdf_chunks):
        """The facts we wrote into the PDF must come back out of it."""
        combined = " ".join(c.content for c in pdf_chunks)

        assert "IELTS" in combined
        assert "2,235" in combined
        assert "two years" in combined

    def test_chunks_carry_citation_metadata(self, pdf_chunks):
        for chunk in pdf_chunks:
            assert chunk.doc_type == "pdf"
            assert chunk.page_number >= 1
            assert chunk.metadata["total_pages"] == 3
            assert chunk.metadata["file_hash"]


class TestRealRetrieval:
    """Retrieval against real embeddings, not mocks."""

    def test_collection_is_populated(self, populated_store):
        stats = populated_store.get_collection_stats()
        assert stats["total_chunks"] > 0

    @pytest.mark.parametrize(
        "query,expected_phrase",
        [
            ("What are the English language requirements?", "IELTS"),
            ("How much does the visa cost?", "2,235"),
            ("How long can I stay on this visa?", "two years"),
        ],
    )
    def test_query_ranks_the_right_chunk_first(
        self, populated_store, query, expected_phrase
    ):
        """
        The top hit must actually be about the thing asked. This is what the
        mock-based tests cannot check, since their fixture ignores the query.
        """
        from src.retrieval.retriever import Retriever

        results = Retriever(vectorstore=populated_store).retrieve(query, n_results=3)

        assert results.total_found > 0
        assert expected_phrase in results.results[0].content, (
            f"Top result for {query!r} was: {results.results[0].content[:200]!r}"
        )

    def test_relevance_score_is_meaningful(self, populated_store):
        """
        Regression: the collection previously used Chroma's default squared-L2
        space, where distances routinely exceed 1.0 and every relevance score
        displayed as 0.0%.
        """
        from src.retrieval.retriever import Retriever

        results = Retriever(vectorstore=populated_store).retrieve(
            "What are the English language requirements?", n_results=3
        )

        top = results.results[0]
        assert 0.0 < top.relevance_score <= 1.0, (
            f"distance={top.distance} produced relevance={top.relevance_score}"
        )
        # Results come back ordered, so relevance must be non-increasing.
        scores = [r.relevance_score for r in results.results]
        assert scores == sorted(scores, reverse=True)

    def test_sources_carry_citation_metadata(self, populated_store):
        from src.retrieval.retriever import Retriever

        results = Retriever(vectorstore=populated_store).retrieve(
            "What is the application charge?", n_results=2
        )
        sources = results.get_sources()

        assert sources
        for src in sources:
            assert src["source"] == "485_test_doc.pdf"
            assert src["page"] >= 1
            assert src["excerpt"]
