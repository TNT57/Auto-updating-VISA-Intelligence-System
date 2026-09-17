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
    """
    Both halves of the stream rule must survive.

    Enumerate-when-different is what stops one stream's fee being presented as
    everyone's. Collapse-when-same is what stops three identical IELTS blocks.
    Dropping either one reintroduces a bug we have already shipped, so each is
    asserted on keywords rather than a quoted sentence — the wording should be
    free to change, the behaviour should not.
    """

    @staticmethod
    def _prompt() -> str:
        # Collapse whitespace: the template uses backslash continuations, so
        # its rendered text carries the next line's indentation mid-sentence.
        from src.generation.prompt_templates import SYSTEM_PROMPT

        return " ".join(SYSTEM_PROMPT.split()).lower()

    def test_prompt_names_the_streams(self):
        low = self._prompt()
        assert "post-higher education work" in low
        assert "post-vocational education work" in low

    def test_prompt_requires_a_breakdown_when_values_differ(self):
        low = self._prompt()
        assert "do not pick one" in low
        assert "differ" in low

    def test_prompt_requires_collapsing_when_values_are_the_same(self):
        """Guards the fix for three identical IELTS blocks in one answer."""
        low = self._prompt()
        assert "same across the streams" in low
        assert "never repeat an identical value" in low

    def test_prompt_puts_the_answer_before_the_detail(self):
        low = self._prompt()
        assert "answer first" in low


class TestFollowUpTemplateHonoursNarrowing:
    """A follow-up that narrows must not re-dump the broad answer."""

    @staticmethod
    def _template() -> str:
        from src.generation.prompt_templates import FOLLOW_UP_TEMPLATE

        return " ".join(FOLLOW_UP_TEMPLATE.split()).lower()

    def test_template_instructs_answering_only_the_narrowed_case(self):
        low = self._template()
        assert "narrows the question" in low
        assert "answer only that narrowed case" in low

    def test_template_forbids_restating_the_previous_answer(self):
        low = self._template()
        assert "do not restate" in low

    def test_template_still_has_its_placeholders(self):
        from src.generation.prompt_templates import FOLLOW_UP_TEMPLATE

        for slot in ("{chat_history}", "{context}", "{question}"):
            assert slot in FOLLOW_UP_TEMPLATE


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


class TestContextCollapsesDuplicates:
    """
    The corpus repeats whole blocks verbatim across stream pages — 91 of 226
    indexed chunks are exact duplicates, every group spanning different URLs.
    Sending the model the same paragraph once per stream is what made it
    answer with one heading per stream.
    """

    SHARED = (
        "Acceptable minimum scores: IELTS overall 6.5 with at least 5.5 "
        "in each component."
    )

    def _context(self, pairs):
        from src.retrieval.retriever import QueryResults

        return QueryResults(
            query="q",
            results=[_result(src, text) for src, text in pairs],
            total_found=len(pairs),
        ).format_for_llm()

    def test_identical_passages_appear_once(self):
        context = self._context([
            (f"{BASE}/post-higher-education-work", self.SHARED),
            (f"{BASE}/post-vocational-education-work", self.SHARED),
            (f"{BASE}/graduate-work", self.SHARED),
        ])

        assert context.count(self.SHARED) == 1
        assert "[2]" not in context, "duplicates were not collapsed"

    def test_collapsed_passage_names_every_stream_it_covers(self):
        context = self._context([
            (f"{BASE}/post-higher-education-work", self.SHARED),
            (f"{BASE}/post-vocational-education-work", self.SHARED),
        ])

        assert "Streams:" in context
        assert "Post-Higher Education Work stream" in context
        assert "Post-Vocational Education Work stream" in context

    def test_whitespace_differences_still_collapse(self):
        context = self._context([
            (f"{BASE}/post-higher-education-work", self.SHARED),
            (f"{BASE}/graduate-work", self.SHARED.replace(" ", "  ") + "\n"),
        ])

        assert "[2]" not in context

    def test_passages_differing_only_in_the_fee_are_kept_apart(self):
        """
        The regression guard on dedup itself.

        These two differ by one number and would score above 0.95 on any
        similarity ratio. Collapsing them is exactly the bug that stream
        labelling was added to fix, so exact matching must keep them apart.
        """
        context = self._context([
            (f"{BASE}/post-higher-education-work",
             "The visa application charge is AUD5,750.00."),
            (f"{BASE}/second-post-higher-education-work",
             "The visa application charge is AUD2,265.00."),
        ])

        assert "AUD5,750.00" in context
        assert "AUD2,265.00" in context
        assert "[2]" in context, "two genuinely different passages were merged"

    def test_single_stream_passage_keeps_the_singular_label(self):
        context = self._context([
            (f"{BASE}/post-higher-education-work", self.SHARED),
        ])

        assert "Stream: Post-Higher Education Work stream" in context
        assert "Streams:" not in context
