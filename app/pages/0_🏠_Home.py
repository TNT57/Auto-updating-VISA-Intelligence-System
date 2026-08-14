"""
Operator home page — system status and quick links.

Only reachable when PUBLIC_MODE=false. A visitor asking a visa question has
no use for chunk counts and embedding model names, and publishing them
advertises how the system is wired.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

import app.bootstrap  # noqa: F401  — sets sys.path and bridges st.secrets
from src.utils.config import settings

st.title("🛂 Subclass 485 Visa Intelligence System")
st.caption("Operator view — set PUBLIC_MODE=true to serve visitors chat only.")

# ---- System status ----
col1, col2, col3 = st.columns(3)

try:
    import chromadb

    from src.ingestion.vectorstore_manager import VectorStoreManager

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_collection(VectorStoreManager.COLLECTION_NAME)
    col1.metric("Indexed chunks", collection.count())
except Exception:
    col1.metric("Indexed chunks", "—")
    st.warning(
        "Vector DB not initialised. Run `python scripts/fetch_and_index.py "
        "--crawl` to build it."
    )

col2.metric("Embedding model", settings.embedding_model)
col3.metric("Query expansion", settings.query_expansion)

st.divider()

st.subheader("Pipeline")
st.markdown(
    "1. **Ingestion** — `scripts/fetch_and_index.py` renders the Home Affairs "
    "pages in a browser, extracts the text, splits it into chunks and embeds "
    "them.\n"
    "2. **Retrieval** — the question is rewritten into several phrasings, each "
    "searched, and the rankings fused.\n"
    "3. **Generation** — the top passages go to the LLM, whose answer is then "
    "re-checked against those same passages."
)

st.subheader("Useful commands")
st.code(
    "python scripts/fetch_and_index.py --crawl     # rebuild the knowledge base\n"
    "python scripts/inspect_pipeline.py            # see each stage's output\n"
    "python scripts/eval_retrieval.py              # score retrieval quality\n"
    "python scripts/daily_update.py                # scrape + detect changes",
    language="bash",
)

st.divider()
st.caption(
    "⚠️ **Disclaimer**: This system provides information only. It is NOT legal "
    "advice. Always consult a registered migration agent for your specific "
    "situation."
)
