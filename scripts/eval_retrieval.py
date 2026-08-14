"""
Measure retrieval quality against a fixed question set.

Retrieval changes are easy to talk yourself into and hard to judge by eye, so
every strategy is scored on the same questions. Each question carries markers —
strings that only appear in a genuinely correct passage — and a chunk counts as
a hit when it contains any of them.

    python scripts/eval_retrieval.py                     # compare all strategies
    python scripts/eval_retrieval.py --strategy synonyms
    python scripts/eval_retrieval.py --verify            # check the eval set only

Requires an index built by fetch_and_index.py.
"""

import argparse
import os
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

from src.ingestion.vectorstore_manager import VectorStoreManager  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402

# Questions phrased the way an applicant would ask, paired with wording that
# only a correct passage contains. Deliberately includes jargon the Home
# Affairs pages never use — that gap is the thing being measured.
EVAL_SET: list[dict] = [
    {
        "q": "What is the English requirement?",
        "markers": ["IELTS", "Acceptable minimum scores", "English Language"],
    },
    {
        "q": "Does the applying location matter, onshore or offshore?",
        "markers": ["immigration clearance", "be in Australia when you apply",
                    "in or outside Australia"],
    },
    {
        "q": "Can I apply for this visa from overseas?",
        "markers": ["immigration clearance", "be in Australia when you apply",
                    "in or outside Australia"],
    },
    {
        "q": "How much does it cost?",
        "markers": ["AUD", "visa costs", "application charge"],
    },
    {
        "q": "Is there an age limit?",
        "markers": ["35 years", "aged 35"],
    },
    {
        "q": "Do I need health insurance?",
        "markers": ["health insurance"],
    },
    {
        "q": "Can I bring my partner and kids?",
        "markers": ["family members", "subsequent entrant", "dependent"],
    },
    {
        "q": "What study do I need to have completed?",
        "markers": ["CRICOS", "Australian study requirement", "degree"],
    },
    {
        "q": "Do I need a police check?",
        "markers": ["Australian Federal Police", "character requirement"],
    },
    {
        "q": "How long can I stay on this visa?",
        "markers": ["stay period", "years", "length of stay"],
    },
]

STRATEGIES = ("none", "synonyms", "llm")


def is_hit(text: str, markers: list[str]) -> bool:
    low = text.lower()
    return any(m.lower() in low for m in markers)


def verify_eval_set(vs: VectorStoreManager) -> bool:
    """
    Confirm every question is actually answerable from the index.

    A question whose markers appear nowhere measures the corpus, not the
    retriever, and would quietly cap the score below 100%.
    """
    docs = vs.collection.get(include=["documents"])["documents"]
    print(f"Verifying {len(EVAL_SET)} questions against {len(docs)} chunks\n")

    ok = True
    for case in EVAL_SET:
        n = sum(1 for d in docs if is_hit(d, case["markers"]))
        flag = "ok " if n else "MISSING"
        if not n:
            ok = False
        print(f"  [{flag}] {n:4} chunk(s)  {case['q']}")
    return ok


def evaluate(strategy: str, vs: VectorStoreManager, k: int = 5,
             verbose: bool = False) -> dict:
    retriever = Retriever(vectorstore=vs, expansion=strategy)

    hit1 = hit3 = hitk = 0
    reciprocal = 0.0

    for case in EVAL_SET:
        results = retriever.retrieve(case["q"], n_results=k).results
        rank = next(
            (i for i, r in enumerate(results, 1) if is_hit(r.content, case["markers"])),
            None,
        )
        if rank:
            reciprocal += 1 / rank
            hitk += 1
            hit3 += rank <= 3
            hit1 += rank == 1

        if verbose:
            shown = f"rank {rank}" if rank else "MISS"
            top = results[0].relevance_score if results else 0
            print(f"    {shown:8} top={top:5.1%}  {case['q']}")

    n = len(EVAL_SET)
    return {
        "strategy": strategy,
        "hit@1": hit1 / n,
        "hit@3": hit3 / n,
        f"hit@{k}": hitk / n,
        "mrr": reciprocal / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval strategies.")
    parser.add_argument("--strategy", choices=STRATEGIES,
                        help="Evaluate one strategy instead of all.")
    parser.add_argument("--k", type=int, default=5, help="Results per query.")
    parser.add_argument("--verify", action="store_true",
                        help="Only check that the eval set is answerable.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show the rank of each question.")
    args = parser.parse_args()

    vs = VectorStoreManager()
    total = vs.get_collection_stats()["total_chunks"]
    if not total:
        print("Index is empty — run scripts/fetch_and_index.py first.")
        sys.exit(1)

    if args.verify:
        sys.exit(0 if verify_eval_set(vs) else 1)

    if not verify_eval_set(vs):
        print("\nSome questions are unanswerable from the current index; "
              "their scores measure corpus coverage, not retrieval.\n")

    strategies = [args.strategy] if args.strategy else list(STRATEGIES)
    rows = []
    for strategy in strategies:
        print(f"\n--- {strategy} ---")
        rows.append(evaluate(strategy, vs, k=args.k, verbose=args.verbose))

    print("\n" + "=" * 62)
    print(f"{'strategy':12} {'hit@1':>8} {'hit@3':>8} {f'hit@{args.k}':>8} {'MRR':>8}")
    print("=" * 62)
    for r in rows:
        print(f"{r['strategy']:12} {r['hit@1']:>7.0%} {r['hit@3']:>8.0%} "
              f"{r[f'hit@{args.k}']:>8.0%} {r['mrr']:>8.3f}")
    print("=" * 62)
    print(f"{len(EVAL_SET)} questions, {total} chunks indexed")


if __name__ == "__main__":
    main()
