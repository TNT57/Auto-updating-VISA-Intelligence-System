"""
Shared test configuration.

The shipped default for query expansion is "llm", which is the best-scoring
strategy but issues a real API call per query. Unit tests must not depend on
network access or an API key, so expansion is forced off unless a test asks
for it explicitly.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _no_network_query_expansion(monkeypatch):
    """Default every Retriever in the suite to single-query retrieval."""
    from src.utils.config import settings

    monkeypatch.setattr(settings, "query_expansion", "none", raising=False)
