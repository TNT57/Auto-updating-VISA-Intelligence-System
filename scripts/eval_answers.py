"""
Measure answer quality against a fixed question set.

`eval_retrieval.py` scores whether the right passage comes back. This scores
what the model then does with it — which is where the reported problems lived:
an answer that repeated one IELTS score under three stream headings was a
perfect retrieval and a poor answer.

    python scripts/eval_answers.py                  # score every case
    python scripts/eval_answers.py --verbose        # print each answer
    python scripts/eval_answers.py --runs 3         # average over 3 runs

Needs a built index and a GROQ_API_KEY. Runs at temperature 0 to keep variance
below the effect being measured, but it calls a real LLM, so it is a script and
not a test — it is non-deterministic and spends quota.

WHAT THIS CANNOT CHECK: whether a collapse is *true*. The metrics can see that
an answer said "the same for all three streams"; they cannot see whether the
streams really do agree. The guard is indirect — the per_stream cases carry
must_contain values that differ between streams, so an over-eager collapse
fails them. That is the most important safety property here.
"""

import argparse
import os
import statistics
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.answer_metrics import (  # noqa: E402
    has_blanket_disclaimer,
    leads_with_answer,
    redundant_enumeration,
    streams_named,
    word_count,
)
from src.generation.llm_client import LLMClient  # noqa: E402
from src.ingestion.vectorstore_manager import VectorStoreManager  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402

# kind="shared"     — the value is the same across streams; must collapse
# kind="per_stream" — the value genuinely differs; must stay broken down
# kind="narrowing"  — a follow-up that narrows; must not re-dump the breadth
EVAL_SET: list[dict] = [
    {
        "q": "What is the English requirement?",
        "kind": "shared",
        "must_contain": ["IELTS"],
        "max_words": 200,
    },
    {
        "q": "What are the English requirements if I take IELTS in 2026?",
        "kind": "shared",
        "must_contain": ["IELTS"],
        "max_words": 150,
    },
    {
        "q": "How much does it cost?",
        "kind": "per_stream",
        "must_contain": ["5,750", "2,265"],
        "max_words": 220,
    },
    {
        "q": "How long can I stay on this visa?",
        "kind": "per_stream",
        "must_contain": ["year"],
        "max_words": 220,
    },
    {
        "q": "What are the English requirements?",
        "follow_up": "What if I take IELTS in 2026?",
        "kind": "narrowing",
        "must_contain": ["IELTS"],
        "max_words": 150,
    },
]


def _answer(llm, retriever, question, k, chat_history=None) -> tuple[str, str]:
    results = retriever.retrieve(question, n_results=k)
    context = results.format_for_llm()
    return llm.answer_question(
        question=question, context=context, chat_history=chat_history
    ), context


def score_case(case: dict, llm, retriever, k: int, verbose: bool) -> dict:
    """Run one case and return its metrics. Never raises on a bad answer."""
    answer, context = _answer(llm, retriever, case["q"], k)
    turn1_words = word_count(answer)

    if case["kind"] == "narrowing":
        # Feed the real first answer back, the way the app does.
        history = f"User: {case['q']}\nAssistant: {answer[:500]}"
        answer, context = _answer(
            llm, retriever, case["follow_up"], k, chat_history=history
        )

    low = answer.lower()
    checks = {
        "grounded_markers": all(m.lower() in low for m in case["must_contain"]),
        "within_length": word_count(answer) <= case["max_words"],
        "leads_with_answer": leads_with_answer(answer),
        "no_blanket_disclaimer": not has_blanket_disclaimer(answer),
    }

    if case["kind"] == "shared":
        checks["collapsed"] = not redundant_enumeration(answer)
    elif case["kind"] == "per_stream":
        # An over-eager collapse would drop one of the must_contain values, so
        # grounded_markers already guards it; assert the breakdown survived.
        checks["kept_breakdown"] = len(streams_named(answer)) >= 2
    elif case["kind"] == "narrowing":
        checks["stayed_narrow"] = word_count(answer) < turn1_words

    if verbose:
        label = case.get("follow_up") or case["q"]
        print(f"\n  Q: {label}")
        print(f"     {len(context)} chars of context, "
              f"{word_count(answer)} words out")
        for name, ok in checks.items():
            print(f"     [{'ok ' if ok else 'FAIL'}] {name}")
        print("     " + "\n     ".join(answer.strip().splitlines()[:12]))

    return {
        "kind": case["kind"],
        "words": word_count(answer),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score answer quality against a fixed question set.")
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat the set N times and average (default 1).")
    parser.add_argument("-k", type=int, default=5,
                        help="Chunks retrieved per question (default 5).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each answer and its checks.")
    args = parser.parse_args()

    vs = VectorStoreManager()
    total = vs.get_collection_stats()["total_chunks"]
    if not total:
        print("Index is empty. Run scripts/fetch_and_index.py first.")
        sys.exit(1)

    try:
        llm = LLMClient()
    except Exception as exc:
        print(f"LLM unavailable ({exc}). Set GROQ_API_KEY in .env.")
        sys.exit(1)

    retriever = Retriever(vectorstore=vs)
    print(f"Model: {llm._model}   index: {total} chunks   "
          f"k={args.k}   runs={args.runs}\n")

    rows: list[dict] = []
    for run in range(args.runs):
        if args.runs > 1:
            print(f"--- run {run + 1}/{args.runs} ---")
        for case in EVAL_SET:
            rows.append(score_case(case, llm, retriever, args.k, args.verbose))

    print("\n" + "=" * 62)
    print(f"{'kind':12} {'pass':>7} {'median words':>14} {'failed checks'}")
    print("=" * 62)

    for kind in ("shared", "per_stream", "narrowing"):
        subset = [r for r in rows if r["kind"] == kind]
        if not subset:
            continue
        passed = sum(r["passed"] for r in subset)
        median = statistics.median(r["words"] for r in subset)
        failed = sorted({
            name for r in subset for name, ok in r["checks"].items() if not ok
        })
        print(f"{kind:12} {passed:>3}/{len(subset):<3} {median:>14.0f} "
              f"{', '.join(failed) or '—'}")

    overall = sum(r["passed"] for r in rows)
    print("=" * 62)
    print(f"{'TOTAL':12} {overall:>3}/{len(rows):<3}")

    if overall < len(rows):
        print("\nRe-run with --verbose to see the answers that failed.")


if __name__ == "__main__":
    main()
