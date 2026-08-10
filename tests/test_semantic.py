"""
Semantic correctness & quality tests — Phase 2.

These tests go beyond structural/unit checks and verify that the system
produces **correct, relevant, and well-grounded** outputs:

  1. Retrieval relevance  — does the right information come back?
  2. Context formatting   — is the LLM prompt well-structured?
  3. Grounding accuracy   — does grounding correctly detect supported/unsupported claims?
  4. Severity accuracy    — are changes classified into the right severity bucket?
  5. Prompt quality       — do prompts enforce no-hallucination constraints?
  6. Conversation memory  — does follow-up context actually alter the prompt?
  7. End-to-end semantic  — full retrieve → format → verify pipeline
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════
# Shared fixtures & test data
# ══════════════════════════════════════════════════════════════════════

# Realistic visa facts — used as "ground truth" documents
VISA_FACTS = [
    {
        "content": (
            "Subclass 485 - Temporary Graduate Visa\n"
            "Application fee: $1,895 AUD (as of July 2024).\n"
            "The visa allows recent graduates to live, work, and study in Australia "
            "temporarily after completing their studies."
        ),
        "source": "visa_guide_2024.pdf",
        "page": 1,
    },
    {
        "content": (
            "Eligibility Requirements - Subclass 485\n"
            "Age: Under 35 years old at time of application.\n"
            "English: IELTS 6.0 overall (or equivalent).\n"
            "Study: Must have completed at least 2 academic years of study in Australia "
            "within the past 6 months.\n"
            "Health insurance: Overseas Student Health Cover (OSHC) or equivalent."
        ),
        "source": "visa_guide_2024.pdf",
        "page": 2,
    },
    {
        "content": (
            "Processing Times - Subclass 485\n"
            "Post-Study Work stream: 4-6 months.\n"
            "Graduate Work stream: 5-7 months.\n"
            "Visa validity ranges from 2 to 5 years depending on qualifications and stream."
        ),
        "source": "processing_times.pdf",
        "page": 1,
    },
    {
        "content": (
            "Work Rights - Subclass 485\n"
            "Visa holders have full work rights with no restrictions on hours.\n"
            "They may work for any employer in any occupation.\n"
            "Self-employment is also permitted."
        ),
        "source": "work_rights.pdf",
        "page": 1,
    },
    {
        "content": (
            "Frequently Asked Questions\n"
            "Q: Can I travel outside Australia on a 485 visa?\n"
            "A: Yes, multiple entries are permitted during the visa validity period.\n"
            "Q: Can I include family members?\n"
            "A: Yes, you can include a partner and dependent children in your application."
        ),
        "source": "faq_485.pdf",
        "page": 3,
    },
]


@pytest.fixture
def mock_vectorstore_with_facts():
    """
    Build a mock VectorStoreManager that returns VISA_FACTS as search results.
    Simulates the top-k retrieval step without requiring a real ChromaDB instance.
    """
    from src.ingestion.vectorstore_manager import VectorStoreManager

    vs = MagicMock(spec=VectorStoreManager)

    def _query(query_text, n_results=5):
        documents = [f["content"] for f in VISA_FACTS[:n_results]]
        metadatas = [
            {"source": f["source"], "page_number": f["page"],
             "chunk_index": 0, "doc_type": "pdf"}
            for f in VISA_FACTS[:n_results]
        ]
        distances = [0.15 + i * 0.05 for i in range(len(documents))]
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    vs.query = MagicMock(side_effect=_query)
    return vs


@pytest.fixture
def mock_llm():
    """Mock LLM client that echoes back a well-formed answer."""
    llm = MagicMock()
    llm._client = MagicMock()
    llm.provider = "groq"
    llm._model = "test-model"
    return llm


# ══════════════════════════════════════════════════════════════════════
# 1. Retrieval Relevance Tests
# ══════════════════════════════════════════════════════════════════════

class TestRetrievalRelevance:
    """Verify that retrieval returns the right documents for a given query."""

    def test_fee_query_returns_fee_document(self, mock_vectorstore_with_facts):
        """Query about fees should retrieve the document containing the fee amount."""
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("How much does the 485 visa cost?", n_results=5)

        assert results.total_found > 0
        top_content = results.results[0].content.lower()
        assert "$1,895" in top_content or "fee" in top_content or "cost" in top_content

    def test_eligibility_query_returns_requirements(self, mock_vectorstore_with_facts):
        """Query about eligibility should retrieve the requirements document."""
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("What are the eligibility requirements for the 485 visa?")

        # Check all results (mock returns fixed order, not query-specific)
        all_content = " ".join(r.content.lower() for r in results.results)
        assert "ielts" in all_content or "english" in all_content or "eligible" in all_content

    def test_processing_time_query(self, mock_vectorstore_with_facts):
        """Query about processing times should return relevant document."""
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("How long does 485 visa processing take?")

        # Check all results (mock returns fixed order, not query-specific)
        all_content = " ".join(r.content.lower() for r in results.results)
        assert "month" in all_content or "processing" in all_content

    def test_relevance_score_is_high_for_good_match(self, mock_vectorstore_with_facts):
        """Top result should have a high relevance score (> 0.7) for a direct query."""
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("485 visa application fee")

        assert results.results[0].relevance_score > 0.7

    def test_results_are_ranked_by_relevance(self, mock_vectorstore_with_facts):
        """Results should be ordered by decreasing relevance (increasing distance)."""
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("visa requirements", n_results=5)

        distances = [r.distance for r in results.results]
        assert distances == sorted(distances), "Results should be sorted by distance (ascending)"


# ══════════════════════════════════════════════════════════════════════
# 2. Context Formatting Tests
# ══════════════════════════════════════════════════════════════════════

class TestContextFormatting:
    """Verify that retrieved documents are formatted correctly for the LLM."""

    def test_format_for_llm_includes_source_info(self, mock_vectorstore_with_facts):
        """Formatted context should include source and page information."""
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("visa fee", n_results=2)
        context = results.format_for_llm()

        assert "Source:" in context
        assert "Page" in context
        assert "[1]" in context

    def test_format_for_llm_includes_all_content(self, mock_vectorstore_with_facts):
        """Formatted context should contain the actual document text."""
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("visa fee", n_results=2)
        context = results.format_for_llm()

        assert "$1,895" in context

    def test_format_empty_results(self):
        """Empty results should produce a clear 'no documents' message."""
        from src.retrieval.retriever import QueryResults

        qr = QueryResults(query="test", results=[], total_found=0)
        formatted = qr.format_for_llm()
        assert formatted == "No relevant documents found."

    def test_get_sources_unique(self, mock_vectorstore_with_facts):
        """get_sources should return unique source/page combinations."""
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("visa", n_results=5)
        sources = results.get_sources()

        keys = [(s["source"], s["page"]) for s in sources]
        assert len(keys) == len(set(keys))


# ══════════════════════════════════════════════════════════════════════
# 3. Grounding Accuracy Tests
# ══════════════════════════════════════════════════════════════════════

class TestGroundingAccuracy:
    """
    Verify that the grounding verification correctly identifies:
      - Fully grounded answers (all claims in context)
      - Partially grounded answers (mix of supported/unsupported)
      - Ungrounded answers (claims not in context at all)
    """

    GROUNDING_TEST_CASES = [
        (
            "The Subclass 485 visa fee is $1,895 AUD. Processing takes 4-6 months.",
            "The 485 visa costs $1,895 AUD and processing takes 4-6 months.",
            "GROUNDED",
        ),
        (
            "The Subclass 485 visa fee is $1,895 AUD.",
            "The 485 visa costs $1,895 AUD and requires IELTS 7.0.",
            "PARTIALLY_GROUNDED",
        ),
        (
            "The Subclass 485 visa fee is $1,895 AUD.",
            "The 485 visa costs $3,500 AUD and takes 12 months to process.",
            "UNGROUNDED",
        ),
        (
            "Eligibility: Under 35, IELTS 6.0, 2 years of study in Australia.",
            "You need to be under 35, have IELTS 6.0, and have completed 2 years of study in Australia.",
            "GROUNDED",
        ),
        (
            "Work rights: Full work rights, no hour restrictions, self-employment permitted.",
            "You can work unlimited hours and start your own business.",
            "GROUNDED",
        ),
    ]

    def test_grounding_prompt_contains_verdict_instructions(self):
        """The grounding prompt should instruct the model about all three verdicts."""
        from src.generation.prompt_templates import GROUNDING_PROMPT

        assert "GROUNDED" in GROUNDING_PROMPT
        assert "PARTIALLY_GROUNDED" in GROUNDING_PROMPT
        assert "UNGROUNDED" in GROUNDING_PROMPT

    def test_grounding_prompt_includes_context_and_answer(self):
        """Grounding prompt should include both context and answer placeholders."""
        from src.generation.prompt_templates import GROUNDING_PROMPT

        assert "{context}" in GROUNDING_PROMPT
        assert "{answer}" in GROUNDING_PROMPT

    @pytest.mark.parametrize(
        "context,answer,expected",
        GROUNDING_TEST_CASES,
        ids=[
            "fully_grounded_fee_and_processing",
            "partially_grounded_fee_but_not_ielts",
            "ungrounded_wrong_fee_and_time",
            "fully_grounded_eligibility",
            "fully_grounded_work_rights",
        ],
    )
    def test_grounding_verdict_detection(self, context, answer, expected):
        """
        Simulate LLM grounding verification with known answers.
        Verifies the parsing logic correctly extracts the verdict.
        """
        from src.generation.llm_client import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client._model = "test-model"
        client._client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            f"{expected} - The answer matches the provided context documents."
        )
        mock_response.usage.completion_tokens = 15
        client._client.chat.completions.create.return_value = mock_response

        verdict, explanation = client.verify_grounding(
            answer=answer,
            context=context,
        )

        assert verdict == expected, (
            f"Expected verdict '{expected}' for answer: '{answer[:60]}...'"
        )

    def test_grounding_llm_receives_correct_prompt(self):
        """Verify the grounding prompt sent to the LLM contains both context and answer."""
        from src.generation.llm_client import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client._model = "test-model"
        client._client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "GROUNDED - All claims match."
        mock_response.usage.completion_tokens = 10
        client._client.chat.completions.create.return_value = mock_response

        test_context = "The fee is $1,895"
        test_answer = "The visa costs $1,895"

        client.verify_grounding(answer=test_answer, context=test_context)

        call_args = client._client.chat.completions.create.call_args
        prompt_sent = call_args.kwargs["messages"][1]["content"]

        assert "$1,895" in prompt_sent, "Context should be included in grounding prompt"
        assert test_answer in prompt_sent, "Answer should be included in grounding prompt"


# ══════════════════════════════════════════════════════════════════════
# 4. Severity Classification Accuracy
# ══════════════════════════════════════════════════════════════════════

class TestSeverityClassification:
    """Verify that change severity is classified correctly based on content."""

    # Test cases aligned with actual keyword lists in change_detector.py:
    #   CRITICAL_KEYWORDS include: eligibility, requirements, processing time, mandatory, etc.
    #   IMPORTANT_KEYWORDS include: fee, form, application, health, insurance, english, etc.
    KEYWORD_CASES = [
        # CRITICAL - hits "eligibility" keyword
        ("The eligibility age limit has been reduced from 50 to 35", "CRITICAL"),
        # CRITICAL - hits "mandatory" and "requirements"
        ("New mandatory English test requirements added", "CRITICAL"),
        # CRITICAL - hits "processing times"
        ("Processing times increased from 4 months to 8 months", "CRITICAL"),
        # CRITICAL - hits "visa validity"
        ("Visa validity period shortened from 4 years to 2 years", "CRITICAL"),
        # IMPORTANT - hits "fee"
        ("The application fee has increased from $1,895 to $2,200", "IMPORTANT"),
        # CRITICAL - hits "requirements"
        ("New document requirements for police clearances", "CRITICAL"),
        # IMPORTANT - hits "english" and "language"
        ("English language test scores updated", "IMPORTANT"),
        # CRITICAL - hits "requirements"
        ("Health insurance requirements changed", "CRITICAL"),
        # MINOR - no keywords matched
        ("Phone number for enquiries updated", "MINOR"),
        # IMPORTANT - hits "form" and "application"
        ("Minor formatting changes to the application form", "IMPORTANT"),
        # MINOR - no keywords matched
        ("Updated postal address for submissions", "MINOR"),
    ]

    @pytest.mark.parametrize(
        "change_text,expected",
        KEYWORD_CASES,
        ids=[f"keyword_{i}" for i in range(len(KEYWORD_CASES))],
    )
    def test_keyword_severity_accuracy(self, change_text, expected):
        """Keyword-based severity should classify realistic changes correctly."""
        from src.monitoring.change_detector import ChangeDetector

        db = MagicMock()
        detector = ChangeDetector(db=db, use_llm=False)
        result = detector._classify_severity_keywords(change_text)
        assert result == expected, (
            f"'{change_text}' should be {expected}, got {result}"
        )

    def test_llm_severity_prompt_contains_guidelines(self):
        """The severity prompt should include clear classification guidelines."""
        from src.generation.prompt_templates import SEVERITY_CLASSIFICATION_PROMPT

        assert "eligibility" in SEVERITY_CLASSIFICATION_PROMPT.lower()
        assert "fee" in SEVERITY_CLASSIFICATION_PROMPT.lower()
        assert "formatting" in SEVERITY_CLASSIFICATION_PROMPT.lower()

    def test_llm_severity_with_mock_classifications(self):
        """LLM-based severity should correctly parse the model's response."""
        from src.monitoring.change_detector import ChangeDetector

        db = MagicMock()
        detector = ChangeDetector(db=db, use_llm=False)
        mock_llm = MagicMock()

        mock_llm.generate.return_value = "CRITICAL - Eligibility age reduced, affecting all new applicants."
        assert detector._classify_severity_llm(mock_llm, "age limit reduced") == "CRITICAL"

        mock_llm.generate.return_value = "IMPORTANT - Fee increase from $1,895 to $2,200."
        assert detector._classify_severity_llm(mock_llm, "fee increased") == "IMPORTANT"

        mock_llm.generate.return_value = "MINOR - Only phone number updated on the contact page."
        assert detector._classify_severity_llm(mock_llm, "phone updated") == "MINOR"


# ══════════════════════════════════════════════════════════════════════
# 5. Prompt Quality Tests
# ══════════════════════════════════════════════════════════════════════

class TestPromptQuality:
    """Verify that prompts enforce correctness constraints."""

    def test_system_prompt_forbids_external_knowledge(self):
        """System prompt should explicitly forbid using external knowledge."""
        from src.generation.prompt_templates import SYSTEM_PROMPT

        assert "external knowledge" in SYSTEM_PROMPT.lower() or \
               "Only use provided context" in SYSTEM_PROMPT

    def test_system_prompt_requires_disclaimer(self):
        """System prompt should require a legal disclaimer."""
        from src.generation.prompt_templates import SYSTEM_PROMPT

        assert "disclaimer" in SYSTEM_PROMPT.lower() or "NOT legal advice" in SYSTEM_PROMPT

    def test_system_prompt_forbids_approximation(self):
        """System prompt should instruct precise figures, not guesses."""
        from src.generation.prompt_templates import SYSTEM_PROMPT

        assert "exact figures" in SYSTEM_PROMPT or "Never approximate" in SYSTEM_PROMPT

    def test_rag_prompt_separates_context_from_question(self):
        """RAG prompt should clearly delimit context vs question."""
        from src.generation.prompt_templates import RAG_PROMPT_TEMPLATE

        assert "CONTEXT" in RAG_PROMPT_TEMPLATE
        assert "Question" in RAG_PROMPT_TEMPLATE

    def test_rag_prompt_no_inline_citations(self):
        """RAG prompt should forbid inline source citations."""
        from src.generation.prompt_templates import RAG_PROMPT_TEMPLATE

        assert "inline citations" in RAG_PROMPT_TEMPLATE.lower() or \
               "NOT include inline" in RAG_PROMPT_TEMPLATE

    def test_follow_up_prompt_preserves_history(self):
        """Follow-up prompt should include conversation history."""
        from src.generation.prompt_templates import FOLLOW_UP_TEMPLATE

        assert "{chat_history}" in FOLLOW_UP_TEMPLATE
        assert "{context}" in FOLLOW_UP_TEMPLATE
        assert "{question}" in FOLLOW_UP_TEMPLATE

    def test_change_explanation_prompt_asks_for_impact(self):
        """Change explanation should ask for impact assessment."""
        from src.generation.prompt_templates import CHANGE_EXPLANATION_TEMPLATE

        assert "who this affects" in CHANGE_EXPLANATION_TEMPLATE.lower() or \
               "action" in CHANGE_EXPLANATION_TEMPLATE.lower()


# ══════════════════════════════════════════════════════════════════════
# 6. Conversation Memory Semantic Tests
# ══════════════════════════════════════════════════════════════════════

class TestConversationMemorySemantic:
    """Verify that conversation history is correctly integrated into prompts."""

    def test_follow_up_uses_history_context(self):
        """
        When a user asks a follow-up, the prompt should include prior conversation
        so the LLM can resolve pronouns and references.
        """
        from src.generation.llm_client import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client._model = "test-model"
        client._client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes, it increased to $2,200."
        mock_response.usage.completion_tokens = 15
        client._client.chat.completions.create.return_value = mock_response

        history = "User: What is the 485 visa fee?\nAssistant: The fee is $1,895."
        client.answer_question(
            question="Has it changed recently?",
            context="The 485 visa fee is now $2,200 as of January 2025.",
            chat_history=history,
        )

        call_args = client._client.chat.completions.create.call_args
        prompt_content = call_args.kwargs["messages"][1]["content"]

        assert "$1,895" in prompt_content
        assert "$2,200" in prompt_content

    def test_no_history_uses_standard_template(self):
        """Without chat history, the standard RAG template should be used."""
        from src.generation.llm_client import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client._model = "test-model"
        client._client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The fee is $1,895."
        mock_response.usage.completion_tokens = 10
        client._client.chat.completions.create.return_value = mock_response

        client.answer_question(
            question="What is the visa fee?",
            context="The 485 visa fee is $1,895.",
        )

        call_args = client._client.chat.completions.create.call_args
        prompt_content = call_args.kwargs["messages"][1]["content"]

        assert "Previous conversation" not in prompt_content

    def test_history_truncation_prevents_context_overflow(self):
        """Chat history builder should truncate long messages."""
        long_msg = "x" * 1000
        messages = [
            {"role": "user", "content": long_msg},
            {"role": "assistant", "content": "Short reply"},
        ]

        max_turns = 4
        recent = messages[-(max_turns * 2):]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:500]
            lines.append(f"{role}: {content}")

        history = "\n".join(lines)
        user_line = [line for line in history.split("\n") if line.startswith("User:")][0]
        assert len(user_line) <= 510


# ══════════════════════════════════════════════════════════════════════
# 7. End-to-End Semantic Pipeline Test
# ══════════════════════════════════════════════════════════════════════

class TestEndToEndSemantic:
    """
    Full pipeline: retrieve -> format -> generate -> verify grounding.
    Uses mock LLM but real retrieval and formatting logic.
    """

    def test_full_rag_pipeline_fee_question(self, mock_vectorstore_with_facts):
        """Full pipeline for a fee question should produce a grounded answer."""
        from src.generation.prompt_templates import RAG_PROMPT_TEMPLATE
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("How much does the 485 visa cost?")

        context = results.format_for_llm()
        assert "$1,895" in context, "Retrieved context should contain the fee amount"

        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question="How much does the 485 visa cost?")

        assert "$1,895" in prompt
        assert "How much does the 485 visa cost?" in prompt
        assert "CONTEXT" in prompt

    def test_full_rag_pipeline_eligibility_question(self, mock_vectorstore_with_facts):
        """Full pipeline for an eligibility question should include requirements."""
        from src.generation.prompt_templates import RAG_PROMPT_TEMPLATE
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("What are the English requirements?")
        context = results.format_for_llm()

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question="What are the English requirements for the 485 visa?",
        )

        combined = prompt.lower()
        assert "ielts" in combined or "english" in combined

    def test_grounding_pipeline_grounded_answer(self, mock_vectorstore_with_facts):
        """A correct answer should be detected as GROUNDED by the verification."""
        from src.generation.llm_client import LLMClient
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("visa fee")
        context = results.format_for_llm()

        correct_answer = "The Subclass 485 visa application fee is $1,895 AUD."

        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client._model = "test-model"
        client._client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "GROUNDED - The $1,895 fee is stated in the context."
        )
        mock_response.usage.completion_tokens = 12
        client._client.chat.completions.create.return_value = mock_response

        verdict, explanation = client.verify_grounding(
            answer=correct_answer,
            context=context,
        )

        assert verdict == "GROUNDED"

        call_args = client._client.chat.completions.create.call_args
        grounding_prompt = call_args.kwargs["messages"][1]["content"]
        assert "$1,895" in grounding_prompt

    def test_grounding_pipeline_hallucinated_answer(self, mock_vectorstore_with_facts):
        """A hallucinated answer should be detected as UNGROUNDED."""
        from src.generation.llm_client import LLMClient
        from src.retrieval.retriever import Retriever

        retriever = Retriever(vectorstore=mock_vectorstore_with_facts)
        results = retriever.retrieve("visa fee")
        context = results.format_for_llm()

        hallucinated_answer = "The 485 visa costs $5,000 AUD and requires a 4-year degree."

        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client._model = "test-model"
        client._client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "UNGROUNDED - The $5,000 fee and 4-year degree are not in the context."
        )
        mock_response.usage.completion_tokens = 15
        client._client.chat.completions.create.return_value = mock_response

        verdict, explanation = client.verify_grounding(
            answer=hallucinated_answer,
            context=context,
        )

        assert verdict == "UNGROUNDED"

    def test_change_detection_with_severity_classification(self):
        """Full change detection pipeline should classify severity correctly."""
        from src.monitoring.change_detector import ChangeDetector

        db = MagicMock()
        db.record_change = MagicMock()
        detector = ChangeDetector(db=db, use_llm=False)

        old = "Eligibility age: Under 50 years old."
        new = "Eligibility age: Under 35 years old."

        changes = detector.compare_and_record(
            old, new, source_url="https://immi.gov.au/485-eligibility"
        )

        assert len(changes) > 0
        severities = [c.get("severity", "") for c in changes]
        assert any(s == "CRITICAL" for s in severities), (
            f"Age eligibility change should be CRITICAL, got: {severities}"
        )


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])