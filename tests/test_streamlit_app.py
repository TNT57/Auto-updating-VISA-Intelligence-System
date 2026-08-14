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


def test_home_reports_a_populated_vector_db():
    """
    Regression: the sidebar referenced settings.collection_name, which does
    not exist, and the AttributeError was swallowed — so the home page always
    claimed the vector DB was uninitialised even with a full index.
    """
    at = _run("app/streamlit_app.py")
    sidebar_text = " ".join(
        str(getattr(el, "value", "")) for el in at.sidebar.markdown
    ) + " ".join(str(getattr(el, "value", "")) for el in at.sidebar.success)

    assert "not initialized" not in sidebar_text.lower()
    assert "chunks" in sidebar_text.lower()


def test_chat_page_exposes_the_sources_slider():
    at = _run("app/pages/1_💬_Chat.py")
    assert len(at.sidebar.slider) >= 1
    slider = at.sidebar.slider[0]
    assert "sources" in slider.label.lower()


def test_alerts_page_shows_the_parked_state():
    """Alerts are off by default; the page must say so rather than imply live."""
    at = _run("app/pages/3_🔔_Alerts.py")
    body = " ".join(str(getattr(el, "value", "")) for el in at.info) + " ".join(
        str(getattr(el, "value", "")) for el in at.warning
    )
    assert "parked" in body.lower() or "disabled" in body.lower()
