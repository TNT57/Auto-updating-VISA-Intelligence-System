"""
Tests for the answer-shape metrics.

These pin the measuring instruments against hand-written answers. They do not
test the model — that needs `scripts/eval_answers.py` and a live LLM. What they
guarantee is that when the eval says "redundant enumeration", it means it.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The shape the user actually got: three streams, one value, three blocks.
REDUNDANT = """English-language requirement for IELTS (test on or after 7 August 2025)

- Post-Higher Education Work stream
  - IELTS Academic - overall 6.5 and at least 5.5 in each component
- Post-Vocational Education Work stream
  - IELTS Academic - overall 6.5 and at least 5.5 in each component
- Graduate Work stream
  - IELTS Academic - overall 6.5 and at least 5.5 in each component
"""

# The shape wanted instead.
COLLAPSED = (
    "All three streams require IELTS 6.5 overall with at least 5.5 in each "
    "component, for tests taken on or after 7 August 2025."
)

# Genuinely different values — must never be flagged as redundant.
PER_STREAM = (
    "The charge depends on the stream.\n"
    "- Post-Higher Education Work stream: AUD5,750.00\n"
    "- Second Post-Higher Education Work stream: AUD2,265.00"
)


class TestWordCount:
    def test_counts_words(self):
        from src.generation.answer_metrics import word_count

        assert word_count("one two three") == 3
        assert word_count("") == 0

    def test_collapsed_answer_is_much_shorter(self):
        from src.generation.answer_metrics import word_count

        assert word_count(COLLAPSED) < word_count(REDUNDANT) / 2


class TestStreamsNamed:
    def test_counts_each_stream_once_per_mention(self):
        from src.generation.answer_metrics import streams_named

        named = streams_named(REDUNDANT)
        assert named["Post-Higher Education Work stream"] == 1
        assert named["Post-Vocational Education Work stream"] == 1
        assert named["Graduate Work stream"] == 1

    def test_matches_with_or_without_the_word_stream(self):
        from src.generation.answer_metrics import streams_named

        assert streams_named("the Graduate Work stream applies")
        assert streams_named("Graduate Work applies")

    def test_returns_nothing_when_no_stream_is_named(self):
        from src.generation.answer_metrics import streams_named

        assert streams_named(COLLAPSED) == {}


class TestDistinctNumbers:
    def test_normalises_thousands_separators(self):
        from src.generation.answer_metrics import distinct_numbers

        assert distinct_numbers("AUD5,750 and AUD5750") == {"5750"}

    def test_separates_genuinely_different_values(self):
        from src.generation.answer_metrics import distinct_numbers

        nums = distinct_numbers(PER_STREAM)
        assert "5750.00" in nums or "5750" in nums
        assert "2265.00" in nums or "2265" in nums


class TestRedundantEnumeration:
    def test_flags_several_streams_sharing_one_value(self):
        """The reported bug."""
        from src.generation.answer_metrics import redundant_enumeration

        assert redundant_enumeration(REDUNDANT) is True

    def test_does_not_flag_genuinely_different_values(self):
        """
        The safety property that matters most.

        If this ever returns True, the eval would push us to collapse two fees
        that really do differ — the exact bug stream labelling was added to
        fix.
        """
        from src.generation.answer_metrics import redundant_enumeration

        assert redundant_enumeration(PER_STREAM) is False

    def test_does_not_flag_a_collapsed_answer(self):
        from src.generation.answer_metrics import redundant_enumeration

        assert redundant_enumeration(COLLAPSED) is False

    def test_does_not_flag_a_single_stream_answer(self):
        from src.generation.answer_metrics import redundant_enumeration

        assert redundant_enumeration(
            "The Graduate Work stream requires IELTS 6.5."
        ) is False


class TestLeadsWithAnswer:
    def test_prose_opening_passes(self):
        from src.generation.answer_metrics import leads_with_answer

        assert leads_with_answer(COLLAPSED) is True

    def test_opening_with_a_bullet_fails(self):
        from src.generation.answer_metrics import leads_with_answer

        assert leads_with_answer("- first bullet\n- second") is False

    def test_opening_with_a_heading_fails(self):
        from src.generation.answer_metrics import leads_with_answer

        assert leads_with_answer("# English requirements\n\nIELTS 6.5.") is False

    def test_a_long_opening_paragraph_fails(self):
        from src.generation.answer_metrics import leads_with_answer

        assert leads_with_answer(" ".join(["word"] * 60)) is False


class TestBlanketDisclaimer:
    def test_detects_the_notice(self):
        from src.generation.answer_metrics import has_blanket_disclaimer

        assert has_blanket_disclaimer(
            "IELTS 6.5. This is not legal advice."
        ) is True

    def test_absent_from_a_clean_answer(self):
        from src.generation.answer_metrics import has_blanket_disclaimer

        assert has_blanket_disclaimer(COLLAPSED) is False
