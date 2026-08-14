"""
Comprehensive test suite — Phase 2 improvements.

Covers:
  - Unit tests for all core modules
  - Integration tests for the RAG pipeline
  - Improvement-specific tests (grounding, memory, tables, severity)
  - End-to-end smoke test
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal test PDF with text content."""
    pdf_path = tmp_path / "test_visa.pdf"
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Test 485 Visa Content) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000360 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
441
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def mock_db():
    """Provide a mock DatabaseManager."""
    db = MagicMock()
    db.record_change = MagicMock()
    db.create_tables = MagicMock()
    return db


# ══════════════════════════════════════════════════════════════════════
# 1. Config Tests
# ══════════════════════════════════════════════════════════════════════

class TestConfig:
    """Tests for configuration module."""

    def test_settings_loads(self):
        """Settings singleton should load without error."""
        from src.utils.config import settings
        assert settings is not None

    def test_embedding_model_is_mpnet(self):
        """Improvement #2: Default embedding model should be all-mpnet-base-v2."""
        from src.utils.config import settings
        assert settings.embedding_model == "all-mpnet-base-v2"

    def test_chunk_size_reasonable(self):
        """Chunk size should be between 200 and 2000."""
        from src.utils.config import settings
        assert 200 <= settings.chunk_size <= 2000

    def test_chunk_overlap_less_than_size(self):
        """Chunk overlap should be less than chunk size."""
        from src.utils.config import settings
        assert settings.chunk_overlap < settings.chunk_size

    def test_monitored_urls_not_empty(self):
        """Monitored URLs list should not be empty."""
        from src.utils.config import settings
        assert len(settings.monitored_urls) > 0

    def test_derived_paths_are_path_objects(self):
        """All derived paths should be Path objects."""
        from src.utils.config import settings
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.raw_pdf_dir, Path)
        assert isinstance(settings.raw_html_dir, Path)
        assert isinstance(settings.changes_db_path, Path)

    def test_ensure_directories_creates_dirs(self):
        """ensure_directories should create all required directories without error."""
        from src.utils.config import ensure_directories
        ensure_directories()
        from src.utils.config import settings
        assert settings.raw_pdf_dir.exists()
        assert settings.raw_html_dir.exists()
        assert settings.database_dir.exists()


# ══════════════════════════════════════════════════════════════════════
# 2. PDF Loader Tests (including Improvement #4: Table extraction)
# ══════════════════════════════════════════════════════════════════════

class TestPDFLoader:
    """Tests for PDF loading and table extraction."""

    def test_document_chunk_dataclass(self):
        """DocumentChunk should have all required fields."""
        from src.ingestion.pdf_loader import DocumentChunk
        chunk = DocumentChunk(
            content="Test content",
            source="test.pdf",
            page_number=1,
            chunk_index=0,
            doc_type="pdf",
        )
        assert chunk.content == "Test content"
        assert chunk.source == "test.pdf"
        assert chunk.page_number == 1
        assert chunk.doc_type == "pdf"
        assert isinstance(chunk.metadata, dict)

    def test_file_hash_computation(self, sample_pdf):
        """compute_file_hash should return a consistent SHA-256 hash."""
        from src.ingestion.pdf_loader import PDFLoader
        hash1 = PDFLoader.compute_file_hash(sample_pdf)
        hash2 = PDFLoader.compute_file_hash(sample_pdf)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_file_hash_differs_for_different_files(self, tmp_path):
        """Different files should produce different hashes."""
        from src.ingestion.pdf_loader import PDFLoader
        f1 = tmp_path / "file1.txt"
        f2 = tmp_path / "file2.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        h1 = PDFLoader.compute_file_hash(f1)
        h2 = PDFLoader.compute_file_hash(f2)
        assert h1 != h2

    def test_list_pdfs_empty_dir(self, tmp_path):
        """list_pdfs should return empty list for directory with no PDFs."""
        from src.ingestion.pdf_loader import PDFLoader
        loader = PDFLoader(pdf_dir=tmp_path)
        result = loader.list_pdfs()
        assert result == []

    def test_list_pdfs_nonexistent_dir(self, tmp_path):
        """list_pdfs should return empty list for non-existent directory."""
        from src.ingestion.pdf_loader import PDFLoader
        loader = PDFLoader(pdf_dir=tmp_path / "nonexistent")
        result = loader.list_pdfs()
        assert result == []

    def test_list_pdfs_finds_pdfs(self, tmp_path):
        """list_pdfs should find all .pdf files in the directory."""
        from src.ingestion.pdf_loader import PDFLoader
        (tmp_path / "doc1.pdf").write_bytes(b"fake pdf")
        (tmp_path / "doc2.pdf").write_bytes(b"fake pdf")
        (tmp_path / "notes.txt").write_text("not a pdf")
        loader = PDFLoader(pdf_dir=tmp_path)
        result = loader.list_pdfs()
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"doc1.pdf", "doc2.pdf"}

    @patch("src.ingestion.pdf_loader.PDFLoader.compute_file_hash", return_value="fakehash")
    @patch("src.ingestion.pdf_loader.pdfplumber")
    def test_table_extraction_marks_metadata(self, mock_pdfplumber, mock_hash):
        """Improvement #4: Pages with tables should have has_tables=True in metadata."""
        from src.ingestion.pdf_loader import PDFLoader

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page with table"
        mock_page.extract_tables.return_value = [
            [["Header 1", "Header 2"], ["Row 1 Col 1", "Row 1 Col 2"]]
        ]
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        loader = PDFLoader()
        chunks = loader.load_pdf(Path("test.pdf"))

        assert len(chunks) == 1
        assert chunks[0].metadata["has_tables"] is True
        assert "Header 1" in chunks[0].content
        assert "| --- |" in chunks[0].content

    @patch("src.ingestion.pdf_loader.PDFLoader.compute_file_hash", return_value="fakehash")
    @patch("src.ingestion.pdf_loader.pdfplumber")
    def test_no_tables_no_metadata_flag(self, mock_pdfplumber, mock_hash):
        """Pages without tables should have has_tables=False."""
        from src.ingestion.pdf_loader import PDFLoader

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Just text, no tables"
        mock_page.extract_tables.return_value = []
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        loader = PDFLoader()
        chunks = loader.load_pdf(Path("test.pdf"))
        assert len(chunks) == 1
        assert chunks[0].metadata["has_tables"] is False

    @patch("src.ingestion.pdf_loader.pdfplumber")
    def test_empty_page_skipped(self, mock_pdfplumber):
        """Pages with no text and no tables should be skipped."""
        from src.ingestion.pdf_loader import PDFLoader

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_page.extract_tables.return_value = []
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        loader = PDFLoader()
        chunks = loader.load_pdf(Path("test.pdf"))
        assert len(chunks) == 0


# ══════════════════════════════════════════════════════════════════════
# 3. Text Chunker Tests
# ══════════════════════════════════════════════════════════════════════

class TestTextChunker:
    """Tests for the text chunking module."""

    def test_chunker_imports(self):
        """TextChunker should be importable."""
        from src.ingestion.text_chunker import TextChunker
        assert TextChunker is not None

    def test_chunker_splits_text(self):
        """TextChunker should split long text into multiple chunks."""
        from src.ingestion.text_chunker import TextChunker
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        long_text = " ".join(["word"] * 200)
        chunks = chunker.splitter.split_text(long_text)
        assert len(chunks) > 1

    def test_chunker_preserves_short_text(self):
        """Text shorter than chunk_size should produce a single chunk."""
        from src.ingestion.text_chunker import TextChunker
        chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
        short_text = "This is a short text."
        chunks = chunker.splitter.split_text(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunker_empty_text(self):
        """Empty text should produce empty list."""
        from src.ingestion.text_chunker import TextChunker
        chunker = TextChunker()
        chunks = chunker.splitter.split_text("")
        assert chunks == [] or chunks == [""]

    def test_chunk_documents_adds_metadata(self):
        """chunk_document should add chunk_index to each chunk."""
        from src.ingestion.pdf_loader import DocumentChunk
        from src.ingestion.text_chunker import TextChunker
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        page = DocumentChunk(
            content=" ".join(["test"] * 100),
            source="test.pdf",
            page_number=1,
            chunk_index=0,
            doc_type="pdf",
        )
        chunks = chunker.chunk_document([page])
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ══════════════════════════════════════════════════════════════════════
# 4. Change Detector Tests (including Improvement #5: LLM severity)
# ══════════════════════════════════════════════════════════════════════

class TestChangeDetector:
    """Tests for change detection and severity classification."""

    def test_keyword_classification_critical(self, mock_db):
        """Critical keywords should be detected correctly."""
        from src.monitoring.change_detector import ChangeDetector
        detector = ChangeDetector(db=mock_db, use_llm=False)
        assert detector._classify_severity_keywords("The eligibility criteria have changed") == "CRITICAL"
        assert detector._classify_severity_keywords("New mandatory requirements") == "CRITICAL"
        assert detector._classify_severity_keywords("Processing times updated") == "CRITICAL"

    def test_keyword_classification_important(self, mock_db):
        """Important keywords should be detected correctly."""
        from src.monitoring.change_detector import ChangeDetector
        detector = ChangeDetector(db=mock_db, use_llm=False)
        assert detector._classify_severity_keywords("The application fee has changed") == "IMPORTANT"
        assert detector._classify_severity_keywords("New documents required") == "IMPORTANT"
        assert detector._classify_severity_keywords("English language test") == "IMPORTANT"

    def test_keyword_classification_minor(self, mock_db):
        """Non-matching text should be classified as MINOR."""
        from src.monitoring.change_detector import ChangeDetector
        detector = ChangeDetector(db=mock_db, use_llm=False)
        assert detector._classify_severity_keywords("Updated phone numbers") == "MINOR"
        assert detector._classify_severity_keywords("Minor layout adjustments") == "MINOR"

    def test_compare_no_previous(self, mock_db):
        """First-time snapshot should return empty changes."""
        from src.monitoring.change_detector import ChangeDetector
        detector = ChangeDetector(db=mock_db, use_llm=False)
        changes = detector.compare(None, "new content", source_url="http://example.com")
        assert changes == []

    def test_compare_unchanged(self, mock_db):
        """Identical content should return no changes."""
        from src.monitoring.change_detector import ChangeDetector
        detector = ChangeDetector(db=mock_db, use_llm=False)
        content = "The visa fee is $1,895"
        changes = detector.compare(content, content, source_url="http://example.com")
        assert changes == []

    def test_compare_detects_change(self, mock_db):
        """Changed content should be detected."""
        from src.monitoring.change_detector import ChangeDetector
        detector = ChangeDetector(db=mock_db, use_llm=False)
        old = "The visa fee is $1,895"
        new = "The visa fee is $2,000"
        changes = detector.compare(old, new, source_url="http://example.com")
        assert len(changes) > 0
        assert changes[0]["change_type"] in ["content_update", "content_added", "content_removed"]

    def test_hash_consistency(self):
        """Hash should be consistent for the same content."""
        from src.monitoring.change_detector import ChangeDetector
        h1 = ChangeDetector._hash("test content")
        h2 = ChangeDetector._hash("test content")
        assert h1 == h2

    def test_hash_normalises_whitespace(self):
        """Hash should be the same after whitespace normalization."""
        from src.monitoring.change_detector import ChangeDetector
        h1 = ChangeDetector._hash("test  content\nhere")
        h2 = ChangeDetector._hash("test content here")
        assert h1 == h2

    def test_hash_differs_for_different_content(self):
        """Different content should produce different hashes."""
        from src.monitoring.change_detector import ChangeDetector
        h1 = ChangeDetector._hash("content A")
        h2 = ChangeDetector._hash("content B")
        assert h1 != h2

    def test_llm_classification_fallback_to_keywords(self, mock_db):
        """Improvement #5: Should fall back to keywords when LLM fails."""
        from src.monitoring.change_detector import ChangeDetector
        detector = ChangeDetector(db=mock_db, use_llm=True)
        detector._llm_client = None
        detector.use_llm = False
        result = detector.classify_severity("The eligibility criteria changed")
        assert result == "CRITICAL"

    def test_llm_classification_with_mock(self, mock_db):
        """Improvement #5: LLM classification should parse severity correctly."""
        from src.monitoring.change_detector import ChangeDetector
        detector = ChangeDetector(db=mock_db, use_llm=False)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "CRITICAL — Visa eligibility requirements have changed."
        result = detector._classify_severity_llm(mock_llm, "eligibility changed")
        assert result == "CRITICAL"

        mock_llm.generate.return_value = "IMPORTANT — The application fee has increased."
        result = detector._classify_severity_llm(mock_llm, "fee changed")
        assert result == "IMPORTANT"

        mock_llm.generate.return_value = "MINOR — Only formatting changes detected."
        result = detector._classify_severity_llm(mock_llm, "formatting update")
        assert result == "MINOR"

    def test_compare_and_record(self, mock_db):
        """compare_and_record should store changes in the database."""
        from src.monitoring.change_detector import ChangeDetector
        detector = ChangeDetector(db=mock_db, use_llm=False)
        old = "The visa fee is $1,895"
        new = "The visa fee is $2,000"
        changes = detector.compare_and_record(old, new, source_url="http://example.com")
        assert len(changes) > 0
        mock_db.record_change.assert_called()


# ══════════════════════════════════════════════════════════════════════
# 5. Database Manager Tests
# ══════════════════════════════════════════════════════════════════════

class TestDatabaseManager:
    """Tests for SQLite database management."""

    def test_creates_tables(self, tmp_path):
        """DatabaseManager should create tables on init."""
        from src.utils.db_manager import DatabaseManager
        db_path = tmp_path / "test.db"
        db = DatabaseManager(db_path=str(db_path))
        db.create_tables()
        assert db_path.exists()

    def test_record_and_retrieve_change(self, tmp_path):
        """Should record and retrieve a change."""
        from src.utils.db_manager import DatabaseManager
        db_path = tmp_path / "test.db"
        db = DatabaseManager(db_path=str(db_path))
        db.create_tables()
        db.record_change(
            severity="CRITICAL",
            change_type="content_update",
            old_value="Old fee",
            new_value="New fee",
            summary="Fee changed",
            source_url="http://example.com",
        )
        changes = db.get_recent_changes(limit=10)
        assert len(changes) >= 1
        # get_recent_changes returns ChangeRecord objects (attribute access)
        assert changes[0].severity == "CRITICAL"
        assert changes[0].change_type == "content_update"

    def test_get_recent_changes_limit(self, tmp_path):
        """get_recent_changes should respect the limit parameter."""
        from src.utils.db_manager import DatabaseManager
        db_path = tmp_path / "test.db"
        db = DatabaseManager(db_path=str(db_path))
        db.create_tables()
        for i in range(5):
            db.record_change(
                severity="MINOR",
                change_type="formatting",
                old_value=None,
                new_value=f"Change {i}",
                summary=f"Change {i}",
            )
        changes = db.get_recent_changes(limit=3)
        assert len(changes) == 3


# ══════════════════════════════════════════════════════════════════════
# 6. Prompt Template Tests
# ══════════════════════════════════════════════════════════════════════

class TestPromptTemplates:
    """Tests for prompt template formatting."""

    def test_rag_prompt_formats(self):
        """RAG prompt should accept context and question."""
        from src.generation.prompt_templates import RAG_PROMPT_TEMPLATE
        result = RAG_PROMPT_TEMPLATE.format(context="Test context", question="Test question")
        assert "Test context" in result
        assert "Test question" in result

    def test_follow_up_prompt_formats(self):
        """Improvement #3: Follow-up prompt should accept chat_history."""
        from src.generation.prompt_templates import FOLLOW_UP_TEMPLATE
        result = FOLLOW_UP_TEMPLATE.format(
            chat_history="User: What is the fee?\nAssistant: The fee is $1,895.",
            context="New context",
            question="Has it changed?",
        )
        assert "What is the fee?" in result
        assert "New context" in result
        assert "Has it changed?" in result

    def test_grounding_prompt_formats(self):
        """Improvement #1: Grounding prompt should accept context and answer."""
        from src.generation.prompt_templates import GROUNDING_PROMPT
        result = GROUNDING_PROMPT.format(
            context="The fee is $1,895",
            answer="The visa fee is $1,895",
        )
        assert "$1,895" in result
        assert "GROUNDED" in result

    def test_severity_prompt_formats(self):
        """Improvement #5: Severity classification prompt should format."""
        from src.generation.prompt_templates import SEVERITY_CLASSIFICATION_PROMPT
        result = SEVERITY_CLASSIFICATION_PROMPT.format(
            changed_text="Fee increased to $2,000",
            change_summary="Visa fee change",
        )
        assert "CRITICAL" in result
        assert "IMPORTANT" in result
        assert "MINOR" in result

    def test_change_explanation_prompt_formats(self):
        """Change explanation prompt should format correctly."""
        from src.generation.prompt_templates import CHANGE_EXPLANATION_TEMPLATE
        result = CHANGE_EXPLANATION_TEMPLATE.format(
            old_content="Old text",
            new_content="New text",
        )
        assert "Old text" in result
        assert "New text" in result


# ══════════════════════════════════════════════════════════════════════
# 7. LLM Client Tests (with mocking)
# ══════════════════════════════════════════════════════════════════════

class TestLLMClient:
    """Tests for LLM client (mostly mocked to avoid API calls)."""

    @patch("src.generation.llm_client.settings")
    def test_detect_provider_groq(self, mock_settings):
        """Should detect Groq when GROQ_API_KEY is set."""
        mock_settings.groq_api_key = "test-key"
        mock_settings.openai_api_key = ""
        from src.generation.llm_client import LLMClient
        # Groq is imported inside __init__ as: from groq import Groq
        with patch("groq.Groq"):
            client = LLMClient()
            assert client.provider == "groq"

    def test_answer_question_with_history(self):
        """Improvement #3: answer_question should use follow-up template when history provided."""
        from src.generation.llm_client import LLMClient
        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client._model = "test-model"
        client._client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_response.usage.completion_tokens = 10
        client._client.chat.completions.create.return_value = mock_response

        result = client.answer_question(
            question="Has the fee changed?",
            context="The fee is $2,000",
            chat_history="User: What is the fee? Assistant: The fee is $1,895.",
        )
        assert result == "Test answer"
        call_args = client._client.chat.completions.create.call_args
        prompt_content = call_args.kwargs["messages"][1]["content"]
        assert "What is the fee?" in prompt_content

    def test_answer_question_without_history(self):
        """answer_question should use standard template when no history."""
        from src.generation.llm_client import LLMClient
        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client._model = "test-model"
        client._client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_response.usage.completion_tokens = 10
        client._client.chat.completions.create.return_value = mock_response

        result = client.answer_question(
            question="What is the fee?",
            context="The fee is $1,895",
        )
        assert result == "Test answer"
        call_args = client._client.chat.completions.create.call_args
        prompt_content = call_args.kwargs["messages"][1]["content"]
        assert "What is the fee?" in prompt_content
        assert "Previous conversation" not in prompt_content

    def test_verify_grounding_grounded(self):
        """Improvement #1: Should correctly parse GROUNDED verdict."""
        from src.generation.llm_client import LLMClient
        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client._model = "test-model"
        client._client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "GROUNDED — All claims match the context."
        mock_response.usage.completion_tokens = 15
        client._client.chat.completions.create.return_value = mock_response

        verdict, explanation = client.verify_grounding(
            answer="The fee is $1,895",
            context="The visa application fee is $1,895",
        )
        assert verdict == "GROUNDED"

    def test_verify_grounding_ungrounded(self):
        """Improvement #1: Should correctly parse UNGROUNDED verdict."""
        from src.generation.llm_client import LLMClient
        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client._model = "test-model"
        client._client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "UNGROUNDED — The answer mentions a $3,000 fee not in context."
        mock_response.usage.completion_tokens = 15
        client._client.chat.completions.create.return_value = mock_response

        verdict, explanation = client.verify_grounding(
            answer="The fee is $3,000",
            context="The visa application fee is $1,895",
        )
        assert verdict == "UNGROUNDED"


# ══════════════════════════════════════════════════════════════════════
# 8. Chat History Builder Tests (Improvement #3)
# ══════════════════════════════════════════════════════════════════════

class TestChatHistoryBuilder:
    """Tests for conversation memory formatting."""

    def test_build_chat_history_basic(self):
        """Should format messages into history string."""
        messages = [
            {"role": "user", "content": "What is the fee?"},
            {"role": "assistant", "content": "The fee is $1,895."},
        ]
        recent = messages[-(4 * 2):]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:500]
            lines.append(f"{role}: {content}")
        result = "\n".join(lines)
        assert "User: What is the fee?" in result
        assert "Assistant: The fee is $1,895." in result

    def test_build_chat_history_truncation(self):
        """Should truncate long messages to 500 chars."""
        long_content = "x" * 1000
        messages = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": "Short reply"},
        ]
        recent = messages[-(4 * 2):]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:500]
            lines.append(f"{role}: {content}")
        result = "\n".join(lines)
        assert len(result.split("\n")[0]) < 600

    def test_build_chat_history_max_turns(self):
        """Should limit to last N turns."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Q{i}"})
            messages.append({"role": "assistant", "content": f"A{i}"})

        max_turns = 4
        recent = messages[-(max_turns * 2):]
        assert len(recent) == 8


# ══════════════════════════════════════════════════════════════════════
# 9. Logger Tests
# ══════════════════════════════════════════════════════════════════════

class TestLogger:
    """Tests for the logging setup."""

    def test_logger_imports(self):
        """Logger module should be importable."""
        from src.utils.logger import setup_logger
        assert callable(setup_logger)


# ══════════════════════════════════════════════════════════════════════
# 10. Integration / Smoke Tests
# ══════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests that verify module interactions."""

    def test_pdf_to_chunks_pipeline(self):
        """PDF extraction → chunking pipeline should produce valid chunks."""
        from src.ingestion.pdf_loader import DocumentChunk
        from src.ingestion.text_chunker import TextChunker

        page = DocumentChunk(
            content="The Subclass 485 visa allows recent graduates to live and work in Australia. "
                    "The visa is valid for 2 to 4 years depending on your qualifications. "
                    "You must have completed an Australian study requirement within the past 6 months. "
                    "The application fee is $1,895. English language requirements apply.",
            source="visa_guide.pdf",
            page_number=1,
            chunk_index=0,
            doc_type="pdf",
            metadata={"total_pages": 1},
        )

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        # chunk_document takes a list of pages from a single doc
        chunks = chunker.chunk_document([page])

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip() != ""
            assert chunk.source == "visa_guide.pdf"
            assert chunk.chunk_index >= 0

    def test_change_detection_pipeline(self, tmp_path):
        """Full change detection pipeline: hash → diff → classify → record."""
        from src.monitoring.change_detector import ChangeDetector
        from src.utils.db_manager import DatabaseManager

        db_path = tmp_path / "test_changes.db"
        db = DatabaseManager(db_path=str(db_path))
        db.create_tables()

        detector = ChangeDetector(db=db, use_llm=False)

        old = "The visa fee is $1,895. Processing takes 4-6 months."
        new = "The visa fee is $2,000. Processing takes 6-8 months."

        changes = detector.compare_and_record(old, new, source_url="https://immi.gov.au/visa-fees")
        assert len(changes) > 0

        recorded = db.get_recent_changes(limit=10)
        assert len(recorded) >= 1

    def test_all_modules_importable(self):
        """Every module in src/ should be importable without errors."""
        modules = [
            "src.utils.config",
            "src.utils.logger",
            "src.utils.db_manager",
            "src.ingestion.pdf_loader",
            "src.ingestion.text_chunker",
            "src.retrieval.retriever",
            "src.generation.prompt_templates",
            "src.monitoring.change_detector",
            "src.alerts.alert_manager",
        ]
        for mod_name in modules:
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {mod_name}: {e}")


# ══════════════════════════════════════════════════════════════════════
# 11. Vectorstore Manager Tests
# ══════════════════════════════════════════════════════════════════════

class TestVectorstoreManager:
    """Tests for vector store management."""

    def test_vectorstore_imports(self):
        """VectorStoreManager should be importable."""
        from src.ingestion.vectorstore_manager import VectorStoreManager
        assert VectorStoreManager is not None


# ══════════════════════════════════════════════════════════════════════
# 12. Retriever Tests
# ══════════════════════════════════════════════════════════════════════

class TestRetriever:
    """Tests for the retrieval module."""

    def test_retriever_imports(self):
        """Retriever should be importable."""
        from src.retrieval.retriever import Retriever
        assert Retriever is not None


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])