"""
Tests for stream labelling and the grounding-check context window.

The 485 visa has several streams and cost, stay length and eligibility differ
between them. Both figures live in the corpus, so an unlabelled context left
the model to pick one and present it as "the" answer.
"""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE = "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485"


def _result(source, content="text", distance=0.3):
    from src.retrieval.retriever import RetrievalResult

    return RetrievalResult(content=content, source=source, page_number=1,
                           chunk_index=0, distance=distance, doc_type="webpage")


class TestStreamDetection:
    def test_identifies_each_stream_from_its_url(self):
        cases = {
            f"{BASE}/post-higher-education-work":
                "Post-Higher Education Work stream",
            f"{BASE}/post-vocational-education-work":
                "Post-Vocational Education Work stream",
            f"{BASE}/graduate-work": "Graduate Work stream",
        }
        for url, expected in cases.items():
            assert _result(url).stream == expected

    def test_second_stream_is_not_shadowed_by_its_substring(self):
        """
        "second-post-higher-education-work" contains
        "post-higher-education-work", so shortest-match-first would mislabel
        the second visa — which is the one with the different fee.
        """
        assert _result(f"{BASE}/second-post-higher-education-work").stream == (
            "Second Post-Higher Education Work stream"
        )

    def test_stream_is_none_for_generic_sources(self):
        assert _result(BASE).stream is None
        assert _result("485_guide.pdf").stream is None


class TestContextLabelling:
    def _results(self):
        from src.retrieval.retriever import QueryResults

        return QueryResults(
            query="How much does it cost?",
            results=[
                _result(f"{BASE}/post-higher-education-work",
                        "The visa costs AUD5,750.00."),
                _result(f"{BASE}/second-post-higher-education-work",
                        "The visa costs AUD2,265.00."),
            ],
            total_found=2,
        )

    def test_context_names_the_stream_for_each_passage(self):
        context = self._results().format_for_llm()

        assert "Stream: Post-Higher Education Work stream" in context
        assert "Stream: Second Post-Higher Education Work stream" in context
        assert "AUD5,750.00" in context
        assert "AUD2,265.00" in context

    def test_streams_present_lists_them_in_rank_order(self):
        assert self._results().streams_present() == [
            "Post-Higher Education Work stream",
            "Second Post-Higher Education Work stream",
        ]

    def test_unlabelled_sources_are_left_alone(self):
        from src.retrieval.retriever import QueryResults

        context = QueryResults(
            query="q", results=[_result("guide.pdf", "text")], total_found=1
        ).format_for_llm()

        assert "Stream:" not in context


class TestSystemPromptRequiresBreakdown:
    def test_prompt_instructs_a_per_stream_answer(self):
        from src.generation.prompt_templates import SYSTEM_PROMPT

        # Collapse whitespace: the template uses backslash continuations, so
        # its rendered text carries the next line's indentation mid-sentence.
        low = " ".join(SYSTEM_PROMPT.split()).lower()
        assert "do not pick one and present it" in low
        assert "post-higher education work" in low
        assert "post-vocational education work" in low


class TestGroundingContextWindow:
    """
    Regression: the grounding check truncated context to 3,000 characters
    while a 5-chunk context runs to ~5,500. A correct answer citing the later
    chunks was reported PARTIALLY_GROUNDED because its evidence had been cut
    off — a false alarm on the one safety mechanism.
    """

    def _client(self):
        from src.generation.llm_client import LLMClient

        with patch.object(LLMClient, "_setup_client"):
            client = LLMClient(provider="groq")
        client._model = "test"
        return client

    def test_full_context_reaches_the_checker(self):
        client = self._client()
        marker = "AUD2,265.00 for the second visa"
        context = ("filler. " * 700) + marker  # ~5,600 chars, marker at end

        with patch.object(client, "generate", return_value="GROUNDED: ok") as gen:
            client.verify_grounding(answer="It costs AUD2,265.00.",
                                    context=context)

        assert marker in gen.call_args.kwargs["prompt"], (
            "evidence at the end of the context was truncated away"
        )

    def test_pathological_context_is_still_bounded(self):
        client = self._client()

        with patch.object(client, "generate", return_value="GROUNDED: ok") as gen:
            client.verify_grounding(answer="a", context="x" * 200_000)

        assert len(gen.call_args.kwargs["prompt"]) < 60_000

    def test_verdicts_parse(self):
        client = self._client()
        for raw, expected in [
            ("GROUNDED: fine", "GROUNDED"),
            ("PARTIALLY_GROUNDED: some", "PARTIALLY_GROUNDED"),
            ("UNGROUNDED: no", "UNGROUNDED"),
        ]:
            with patch.object(client, "generate", return_value=raw):
                verdict, _ = client.verify_grounding(answer="a", context="b")
            assert verdict == expected

    def test_checker_failure_does_not_raise(self):
        client = self._client()
        with patch.object(client, "generate", side_effect=RuntimeError("429")):
            verdict, explanation = client.verify_grounding(answer="a", context="b")

        assert verdict == "UNKNOWN"
        assert "429" in explanation
