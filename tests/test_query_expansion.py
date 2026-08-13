"""
Tests for query expansion and rank fusion.

The behaviour under test is the fix for a measured problem: the same gold
chunk ranked 1st for "Do I need to be in Australia when I apply?" and 3rd at
a quarter of the relevance for "onshore or offshore?". Expansion asks in both
vocabularies and fuses the results.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSynonymExpansion:
    def test_original_query_is_always_first(self):
        from src.retrieval.query_expansion import expand_with_synonyms

        q = "Does the applying location matter, onshore or offshore?"
        assert expand_with_synonyms(q)[0] == q

    def test_jargon_is_translated_to_official_wording(self):
        from src.retrieval.query_expansion import expand_with_synonyms

        variants = " ".join(expand_with_synonyms("Can I apply onshore?"))
        assert "in Australia" in variants

    def test_offshore_maps_to_outside_australia(self):
        from src.retrieval.query_expansion import expand_with_synonyms

        variants = " ".join(expand_with_synonyms("Can I lodge offshore?"))
        assert "outside Australia" in variants

    def test_query_without_jargon_is_unchanged(self):
        from src.retrieval.query_expansion import expand_with_synonyms

        q = "What documents must I provide?"
        assert expand_with_synonyms(q) == [q]

    def test_matching_is_case_insensitive(self):
        from src.retrieval.query_expansion import expand_with_synonyms

        assert len(expand_with_synonyms("Applying OFFSHORE?")) > 1

    def test_substrings_do_not_falsely_match(self):
        """'pr' must not fire inside 'proof' or 'processing'."""
        from src.retrieval.query_expansion import expand_with_synonyms

        assert expand_with_synonyms("What proof of processing is needed?") == [
            "What proof of processing is needed?"
        ]

    def test_variant_count_is_bounded(self):
        from src.retrieval.query_expansion import MAX_VARIANTS, expand_with_synonyms

        q = "onshore offshore cost how long age limit insurance partner"
        assert len(expand_with_synonyms(q)) <= MAX_VARIANTS + 1


class TestLLMExpansion:
    def test_parses_the_model_response(self):
        from src.retrieval.query_expansion import expand_with_llm

        llm = MagicMock()
        llm.generate.return_value = (
            "1. Do I need to be in Australia when I apply?\n"
            "- Can I apply from outside Australia?\n"
            "Where must I be at time of decision?"
        )
        out = expand_with_llm("onshore or offshore?", llm=llm)

        assert out[0] == "onshore or offshore?"
        assert "Do I need to be in Australia when I apply?" in out
        # numbering and bullets stripped
        assert not any(v.startswith(("1.", "-")) for v in out)

    def test_llm_failure_falls_back_to_synonyms(self):
        """Retrieval must never fail because a rewrite call did."""
        from src.retrieval.query_expansion import expand_with_llm

        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("rate limited")

        out = expand_with_llm("Can I apply onshore?", llm=llm)

        assert out[0] == "Can I apply onshore?"
        assert any("in Australia" in v for v in out)

    def test_empty_response_falls_back_to_synonyms(self):
        from src.retrieval.query_expansion import expand_with_llm

        llm = MagicMock()
        llm.generate.return_value = "   \n  \n"

        out = expand_with_llm("Can I apply offshore?", llm=llm)

        assert any("outside Australia" in v for v in out)


class TestReciprocalRankFusion:
    def test_agreement_across_lists_beats_a_single_top_hit(self):
        """The whole point: consistency across phrasings is the signal."""
        from src.retrieval.query_expansion import reciprocal_rank_fusion

        scores = reciprocal_rank_fusion([
            ["spike", "steady", "x"],
            ["y", "steady", "z"],
            ["w", "steady", "v"],
        ])
        assert scores["steady"] > scores["spike"]

    def test_rank_one_scores_above_rank_two(self):
        from src.retrieval.query_expansion import reciprocal_rank_fusion

        scores = reciprocal_rank_fusion([["a", "b"]])
        assert scores["a"] > scores["b"]

    def test_empty_input_is_safe(self):
        from src.retrieval.query_expansion import reciprocal_rank_fusion

        assert reciprocal_rank_fusion([]) == {}


class TestRetrieverExpansion:
    def _store(self):
        """A store returning a different top hit per phrasing."""
        store = MagicMock()

        def query(query_text, n_results):
            if "outside Australia" in query_text or "in Australia" in query_text:
                ids = ["gold", "noise1", "noise2"]
            else:
                ids = ["noise1", "noise2", "gold"]
            return {
                "ids": [ids],
                "documents": [[f"text of {i}" for i in ids]],
                "metadatas": [[{"source": "s", "doc_type": "webpage"} for _ in ids]],
                "distances": [[0.3, 0.4, 0.5]],
            }

        store.query.side_effect = query
        return store

    def test_expansion_promotes_the_chunk_the_jargon_missed(self):
        from src.retrieval.retriever import Retriever

        store = self._store()
        baseline = Retriever(vectorstore=store, expansion="none")
        assert baseline.retrieve("apply offshore?", n_results=3).results[0].content \
            == "text of noise1"

        expanded = Retriever(vectorstore=store, expansion="synonyms")
        assert expanded.retrieve("apply offshore?", n_results=3).results[0].content \
            == "text of gold"

    def test_none_strategy_issues_a_single_query(self):
        from src.retrieval.retriever import Retriever

        store = self._store()
        Retriever(vectorstore=store, expansion="none").retrieve("apply offshore?")
        assert store.query.call_count == 1

    def test_expansion_overfetches_per_variant(self):
        """Fusion needs depth or there is nothing to disagree about."""
        from src.retrieval.retriever import Retriever

        store = self._store()
        r = Retriever(vectorstore=store, expansion="none")
        r.retrieve("plain question", n_results=5)

        assert store.query.call_args.kwargs["n_results"] == 5 * Retriever.OVERFETCH

    def test_respects_configured_default(self):
        from src.retrieval.retriever import Retriever

        with patch("src.utils.config.settings.query_expansion", "none"):
            assert Retriever(vectorstore=MagicMock()).expansion == "none"

    def test_result_count_is_capped(self):
        from src.retrieval.retriever import Retriever

        out = Retriever(vectorstore=self._store(),
                        expansion="synonyms").retrieve("apply offshore?", n_results=2)
        assert len(out.results) <= 2
