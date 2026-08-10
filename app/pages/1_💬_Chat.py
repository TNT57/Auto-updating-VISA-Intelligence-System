"""
Chat page — Interactive RAG Q&A interface.

Users ask questions about the 485 visa and get AI-powered answers
with source citations from official documents.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src.utils.config import settings

st.set_page_config(
    page_title="💬 Chat — 485 Visa Intelligence",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Ask About the 485 Visa")
st.markdown(
    "Ask any question about the Temporary Graduate (Subclass 485) visa. "
    "Answers are sourced from official documents with citations."
)

# ---- Initialize session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

def _build_chat_history(messages: list[dict], max_turns: int = 4) -> str:
    """Format recent chat history for the follow-up prompt (Improvement #3)."""
    # Take the last N turns (user + assistant pairs)
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long messages to keep context manageable
        content = msg["content"][:500]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

@st.cache_resource
def _init_llm_client():
    from src.generation.llm_client import LLMClient
    return LLMClient()

@st.cache_resource
def _init_retriever():
    from src.retrieval.retriever import Retriever
    return Retriever()

if "llm_client" not in st.session_state:
    try:
        st.session_state.llm_client = _init_llm_client()
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {e}")
        st.info("Make sure you have set your GROQ_API_KEY in the .env file.")
        st.stop()

if "retriever" not in st.session_state:
    try:
        st.session_state.retriever = _init_retriever()
    except Exception as e:
        st.error(f"Failed to initialize retriever: {e}")
        st.info("Make sure you have run `python scripts/initial_setup.py` first.")
        st.stop()

# ---- Disclaimer banner ----
st.warning(
    "⚠️ **Disclaimer**: This is NOT legal advice. Always consult a registered "
    "migration agent for your specific situation. Answers are generated from "
    "official documents but may not reflect the very latest changes."
)

# ---- Sidebar: Chat controls ----
# Rendered before the chat handler below so `n_results` is defined by the
# time a question is answered — Streamlit executes the script top to bottom.
with st.sidebar:
    st.markdown("### 💬 Chat Controls")

    n_results = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More sources = more context but slower responses",
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Show vectorstore stats
    st.markdown("### 📊 Vector Store")
    try:
        stats = st.session_state.retriever.vectorstore.get_collection_stats()
        st.metric("Total Chunks", stats["total_chunks"])
        st.caption(f"Model: {stats['embedding_model']}")
    except Exception:
        st.info("Run setup script first")

    st.divider()

    # Example questions — clicking one queues it as the next question.
    st.markdown("### 💡 Example Questions")
    examples = [
        "What are the English requirements?",
        "How long is the 485 visa valid?",
        "What documents do I need?",
        "What is the application fee?",
        "Can I include family members?",
        "What are the eligibility criteria?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()


def _render_sources(sources: list[dict], key_prefix: str = "src") -> None:
    """Render source citations with excerpts and PDF download links."""
    pdf_dir = settings.raw_pdf_dir
    with st.expander("📚 Sources", expanded=False):
        for i, src in enumerate(sources, 1):
            st.markdown(
                f"**Source {i}:** `{src['source']}` — Page {src['page']} "
                f"(Relevance: {src['relevance']})"
            )
            # Show excerpt from the retrieved chunk
            if "excerpt" in src and src["excerpt"]:
                st.caption(f"> {src['excerpt']}")
            # Provide PDF download link if file exists
            pdf_path = pdf_dir / src["source"]
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=f"📄 Download {src['source']}",
                        data=f.read(),
                        file_name=src["source"],
                        mime="application/pdf",
                        key=f"{key_prefix}_dl_{i}_{src['source']}_{src['page']}",
                    )
            if i < len(sources):
                st.divider()


# ---- Display chat history ----
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            _render_sources(message["sources"], key_prefix=f"hist_{msg_idx}")

# ---- Chat input ----
# A queued example question stands in for typed input on the rerun that
# follows the button click.
prompt = st.chat_input("Ask about the 485 visa...") or st.session_state.pop(
    "pending_question", None
)

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Step 1: Retrieve relevant chunks
                results = st.session_state.retriever.retrieve(
                    query=prompt,
                    n_results=n_results,
                )

                # Step 2: Format context for LLM
                context = results.format_for_llm()

                # Step 3: Build conversation history (Improvement #3)
                chat_history = None
                if st.session_state.messages:
                    chat_history = _build_chat_history(st.session_state.messages)

                # Step 4: Generate answer using LLM with streaming + memory
                response_placeholder = st.empty()
                full_response = ""

                for chunk in st.session_state.llm_client.stream_answer(
                    question=prompt,
                    context=context,
                    chat_history=chat_history,
                ):
                    full_response += chunk
                    response_placeholder.markdown(
                        full_response + "▌", unsafe_allow_html=True
                    )

                response_placeholder.markdown(full_response, unsafe_allow_html=True)

                # Step 5: Grounding verification (Improvement #1)
                grounding_verdict = None
                try:
                    grounding_verdict, grounding_explanation = (
                        st.session_state.llm_client.verify_grounding(
                            answer=full_response,
                            context=context,
                        )
                    )
                    if grounding_verdict == "GROUNDED":
                        st.success(f"✅ Answer verified: {grounding_explanation}")
                    elif grounding_verdict == "PARTIALLY_GROUNDED":
                        st.warning(f"⚠️ Partially grounded: {grounding_explanation}")
                    elif grounding_verdict == "UNGROUNDED":
                        st.error(f"❌ Ungrounded answer: {grounding_explanation}")
                except Exception:
                    pass  # Non-critical — don't block the answer

                # Show sources with excerpts and PDF downloads
                sources = results.get_sources()
                if sources:
                    n_msg = len(st.session_state.messages)
                    _render_sources(sources, key_prefix=f"live_{n_msg}")

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = (
                    f"Sorry, I encountered an error: {e}\n\n"
                    "Please check that:\n"
                    "1. Your GROQ_API_KEY is set in `.env`\n"
                    "2. You've run `python scripts/initial_setup.py`\n"
                    "3. There are PDF documents in `data/raw/pdfs/`"
                )
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

