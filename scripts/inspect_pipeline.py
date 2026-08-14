"""
Show the actual output of each RAG stage so it can be checked by hand.

Nothing here is asserted or summarised — it prints what the system really
holds and really returns, with source URLs, so every claim can be verified
against the live Home Affairs pages.

    python scripts/inspect_pipeline.py                  # all three stages
    python scripts/inspect_pipeline.py --stage ingest
    python scripts/inspect_pipeline.py --stage retrieve --ask "your question"
    python scripts/inspect_pipeline.py --stage generate --ask "your question"
    python scripts/inspect_pipeline.py --full-chunks     # untruncated text
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        # errors="replace" matters: scraped government pages carry zero-width
        # and smart-quote characters that a cp1252 console cannot encode, and
        # the default "strict" kills the script mid-report.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.vectorstore_manager import VectorStoreManager  # noqa: E402

DEFAULT_QUESTIONS = [
    "What is the English requirement?",
    "Does the applying location matter, whether I'm onshore or offshore?",
]

RULE = "=" * 78
THIN = "-" * 78


def _wrap(text: str, width: int = 76, indent: str = "    ") -> str:
    """Wrap without collapsing the blank lines that separate list items."""
    import textwrap

    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=width,
                                 initial_indent=indent,
                                 subsequent_indent=indent) or [indent])
    return "\n".join(out)


# ── Stage 1 ──────────────────────────────────────────────────────────────

def stage_ingest(vs: VectorStoreManager, full: bool) -> None:
    print(RULE)
    print("STAGE 1 — INGESTION: what is actually in the index")
    print(RULE)

    data = vs.collection.get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]

    if not docs:
        print("\nIndex is empty. Run: python scripts/fetch_and_index.py --crawl")
        return

    types = Counter(m.get("doc_type", "?") for m in metas)
    sources = Counter(m.get("source", "?") for m in metas)

    print(f"\nTotal chunks: {len(docs)}")
    print(f"By type:      {dict(types)}")
    print(f"Distinct sources: {len(sources)}\n")

    print("Every source, with chunk count:")
    print(THIN)
    for src, n in sources.most_common():
        kind = "PDF " if src.lower().endswith(".pdf") else "PAGE"
        print(f"  [{kind}] {n:4} chunks  {src}")

    print("\n" + THIN)
    print("Sample chunk from each web page — compare these against the live")
    print("page to confirm the text was extracted correctly.")
    print(THIN)

    shown = set()
    for doc, meta in zip(docs, metas, strict=False):
        src = meta.get("source", "?")
        if meta.get("doc_type") != "webpage" or src in shown:
            continue
        shown.add(src)
        print(f"\nSOURCE: {src}")
        print(f"TITLE : {meta.get('title', '')[:70]}")
        print(f"CHARS : {len(doc)}")
        print("TEXT:")
        print(_wrap(doc if full else doc[:700] + ("..." if len(doc) > 700 else "")))


# ── Stage 2 ──────────────────────────────────────────────────────────────

def stage_retrieve(vs: VectorStoreManager, questions: list[str],
                   k: int, full: bool) -> None:
    from src.retrieval.retriever import Retriever
    from src.utils.config import settings

    print("\n" + RULE)
    print("STAGE 2 — RETRIEVAL: which passages come back, and from where")
    print(RULE)
    print(f"\nExpansion strategy: {settings.query_expansion}")
    print("Relevance = 1 - cosine distance. Higher is closer.")

    retriever = Retriever(vectorstore=vs)

    for question in questions:
        print("\n" + RULE)
        print(f"Q: {question}")
        print(RULE)

        variants = retriever._variants(question)
        if len(variants) > 1:
            print("\nSearched as:")
            for v in variants:
                print(f"  - {v}")

        results = retriever.retrieve(question, n_results=k).results
        if not results:
            print("\nNo results.")
            continue

        for i, r in enumerate(results, 1):
            print(f"\n[{i}] relevance {r.relevance_score:.1%}   type={r.doc_type}")
            print(f"    SOURCE: {r.source}")
            body = r.content if full else r.content[:600] + (
                "..." if len(r.content) > 600 else "")
            print(_wrap(body))


# ── Stage 3 ──────────────────────────────────────────────────────────────

def stage_generate(vs: VectorStoreManager, questions: list[str], k: int) -> None:
    from src.generation.llm_client import LLMClient
    from src.retrieval.retriever import Retriever

    print("\n" + RULE)
    print("STAGE 3 — GENERATION: the answer, and whether it is grounded")
    print(RULE)

    retriever = Retriever(vectorstore=vs)
    try:
        llm = LLMClient()
    except Exception as exc:
        print(f"\nLLM unavailable ({exc}). Set GROQ_API_KEY in .env.")
        return
    print(f"\nModel: {llm._model}")

    for question in questions:
        print("\n" + RULE)
        print(f"Q: {question}")
        print(RULE)

        results = retriever.retrieve(question, n_results=k)
        context = results.format_for_llm()
        print(f"\nContext sent to the model: {len(context)} chars "
              f"from {len(results.results)} chunk(s)")

        answer = llm.answer_question(question=question, context=context)
        print("\nANSWER:")
        print(_wrap(answer.strip()))

        verdict, explanation = llm.verify_grounding(answer=answer, context=context)
        print(f"\nGROUNDING: {verdict}")
        print(_wrap(explanation.strip()))

        print("\nCITED SOURCES — open these to check the answer:")
        for s in results.get_sources():
            print(f"  {s['relevance']:>6}  {s['source']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the real output of each RAG stage.")
    parser.add_argument("--stage", choices=("ingest", "retrieve", "generate"),
                        help="Run one stage instead of all three.")
    parser.add_argument("--ask", action="append", metavar="QUESTION",
                        help="Question to test. Repeatable.")
    parser.add_argument("-k", type=int, default=3,
                        help="Passages to retrieve (default: 3).")
    parser.add_argument("--full-chunks", action="store_true",
                        help="Print chunk text in full, no truncation.")
    args = parser.parse_args()

    questions = args.ask or DEFAULT_QUESTIONS
    vs = VectorStoreManager()

    if vs.get_collection_stats()["total_chunks"] == 0:
        print("Index is empty. Run: python scripts/fetch_and_index.py --crawl")
        sys.exit(1)

    if args.stage in (None, "ingest"):
        stage_ingest(vs, full=args.full_chunks)
    if args.stage in (None, "retrieve"):
        stage_retrieve(vs, questions, k=args.k, full=args.full_chunks)
    if args.stage in (None, "generate"):
        stage_generate(vs, questions, k=args.k)


if __name__ == "__main__":
    main()
