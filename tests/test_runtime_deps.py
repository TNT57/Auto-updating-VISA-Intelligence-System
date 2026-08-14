"""
The deployed app must import using only requirements.txt.

requirements.txt is the runtime; ingestion and scraping libraries live in
requirements-dev.txt. Nothing verified that split, so a transitive import
broke the deployment: the retriever pulled in VectorStoreManager, which
imported DocumentChunk from pdf_loader, which imports pdfplumber at module
level. The app died on boot with "No module named 'pdfplumber'".

These tests hide the dev-only packages and import the serving path in a
subprocess, so they fail here rather than in production.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Declared in requirements-dev.txt and therefore absent from the deployment.
DEV_ONLY = [
    "pdfplumber",
    "pypdf",
    "bs4",
    "lxml",
    "scrapling",
    "curl_cffi",
    "playwright",
    "langchain_text_splitters",
    "fpdf",
    "pytest",
]

# Every module the app touches to answer a question.
SERVING_IMPORTS = [
    "src.utils.config",
    "src.ingestion.models",
    "src.ingestion.vectorstore_manager",
    "src.retrieval.retriever",
    "src.retrieval.query_expansion",
    "src.generation.llm_client",
    "src.generation.prompt_templates",
    "src.utils.db_manager",
]

BLOCKER = """
import sys

BLOCKED = {blocked!r}


class _Blocker:
    \"\"\"Refuse dev-only packages, simulating the deployment environment.\"\"\"

    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in BLOCKED:
            raise ImportError(
                f"{{name}} is a dev-only dependency and is not installed "
                f"in the deployed app"
            )
        return None


sys.meta_path.insert(0, _Blocker())
sys.path.insert(0, {root!r})

{body}
print("OK")
"""


def _run_isolated(body: str) -> subprocess.CompletedProcess:
    script = BLOCKER.format(
        blocked=DEV_ONLY, root=str(PROJECT_ROOT), body=textwrap.dedent(body)
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT),
    )


@pytest.mark.integration
@pytest.mark.parametrize("module", SERVING_IMPORTS)
def test_serving_module_imports_without_dev_dependencies(module):
    result = _run_isolated(f"import {module}")
    assert result.returncode == 0, (
        f"{module} needs a dev-only dependency:\n{result.stderr[-1500:]}"
    )


@pytest.mark.integration
def test_the_whole_answer_path_imports():
    """The exact chain the chat page walks to answer a question."""
    result = _run_isolated(
        """
        from src.retrieval.retriever import Retriever, QueryResults
        from src.generation.llm_client import LLMClient
        from src.ingestion.vectorstore_manager import VectorStoreManager
        """
    )
    assert result.returncode == 0, result.stderr[-1500:]


@pytest.mark.integration
def test_the_blocker_actually_blocks():
    """Guard against the test passing because the mechanism silently broke."""
    result = _run_isolated("import pdfplumber")
    assert result.returncode != 0
    assert "dev-only" in result.stderr


class TestRequirementsSplit:
    def _names(self, filename):
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        names = set()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(
                line.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower()
            )
        return names

    def test_ingestion_libraries_are_not_in_the_runtime_file(self):
        runtime = self._names("requirements.txt")
        for pkg in ("pdfplumber", "scrapling", "playwright", "pytest", "ruff"):
            assert pkg not in runtime, (
                f"{pkg} belongs in requirements-dev.txt; the deployed app "
                f"does not build the index"
            )

    def test_retrieval_libraries_are_in_the_runtime_file(self):
        runtime = self._names("requirements.txt")
        for pkg in ("chromadb", "sentence-transformers", "groq", "streamlit"):
            assert pkg in runtime, f"{pkg} is needed to answer a question"

    def test_dev_file_carries_the_toolchain(self):
        dev = self._names("requirements-dev.txt")
        for pkg in ("pytest", "ruff", "pdfplumber", "scrapling"):
            assert pkg in dev
