"""
Smoke tests for the Streamlit pages.

The UI was the one part of the system never exercised after the retrieval,
ingestion and prompt changes. A page that raises on import fails silently in
a browser — Streamlit shows a traceback in the page body and the process
stays healthy — so an HTTP 200 proves nothing. AppTest runs each page's
script and surfaces exceptions.

Marked integration: these load the embedding model and touch the real index.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.integration

PAGES = [
    "app/streamlit_app.py",
    "app/pages/0_🏠_Home.py",
    "app/pages/1_💬_Chat.py",
    "app/pages/2_📊_Changes.py",
    "app/pages/3_🔔_Alerts.py",
]


def _run(page: str, timeout: int = 180):
    at = pytest.importorskip(
        "streamlit.testing.v1", reason="streamlit not installed"
    ).AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=timeout)
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_exception(page):
    at = _run(page)
    assert not at.exception, (
        f"{page} raised: {[str(e.value) for e in at.exception]}"
    )


def test_operator_home_reports_a_populated_vector_db():
    """
    Regression: this referenced settings.collection_name, which does not
    exist, and the AttributeError was swallowed — so it always claimed the
    vector DB was uninitialised even with a full index.
    """
    at = _run("app/pages/0_🏠_Home.py")
    metrics = {m.label: str(m.value) for m in at.metric}

    assert "Indexed chunks" in metrics
    assert metrics["Indexed chunks"] not in ("—", "0")


class TestPublicMode:
    """
    A visitor should see the chat and nothing else. Scrape history, webhook
    status and chunk counts are operator tooling.
    """

    def test_public_mode_exposes_only_the_chat_page(self, monkeypatch):
        from src.utils.config import settings

        monkeypatch.setattr(settings, "public_mode", True)
        at = _run("app/streamlit_app.py")

        assert not at.exception
        # Operator-only widgets must not render for a visitor.
        assert not at.sidebar.slider, "retrieval depth slider leaked to visitors"
        metric_labels = [m.label for m in at.metric]
        assert "Total Chunks" not in metric_labels

    def test_operator_mode_shows_the_controls(self, monkeypatch):
        from src.utils.config import settings

        monkeypatch.setattr(settings, "public_mode", False)
        at = _run("app/pages/1_💬_Chat.py")

        assert len(at.sidebar.slider) >= 1
        assert "sources" in at.sidebar.slider[0].label.lower()

    def test_example_questions_are_offered_in_both_modes(self, monkeypatch):
        from src.utils.config import settings

        for mode in (True, False):
            monkeypatch.setattr(settings, "public_mode", mode)
            at = _run("app/pages/1_💬_Chat.py")
            labels = [b.label for b in at.sidebar.button]
            assert any("English" in label for label in labels), (
                f"no example questions in public_mode={mode}"
            )

    def test_default_is_public(self):
        """A fresh deployment must not expose operator tooling by accident."""
        from src.utils.config import Settings

        assert Settings().public_mode is True


def test_alerts_page_shows_the_parked_state():
    """Alerts are off by default; the page must say so rather than imply live."""
    at = _run("app/pages/3_🔔_Alerts.py")
    body = " ".join(str(getattr(el, "value", "")) for el in at.info) + " ".join(
        str(getattr(el, "value", "")) for el in at.warning
    )
    assert "parked" in body.lower() or "disabled" in body.lower()
