"""
Chat page — Interactive RAG Q&A interface.

Users ask questions about the 485 visa and get AI-powered answers
with source citations from official documents.
"""

import streamlit as st

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

if "llm_client" not in st.session_state:
    try:
        from src.generation.llm_client import LLMClient
        st.session_state.llm_client = LLMClient()
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {e}")
        st.info("Make sure you have set your GROQ_API_KEY in the .env file.")
        st.stop()

if "retriever" not in st.session_state:
    try:
        from src.retrieval.retriever import Retriever
        st.session_state.retriever = Retriever()
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

# ---- Display chat history ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(
                        f"- **{src['source']}** — Page {src['page']} "
                        f"(Relevance: {src['relevance']})"
                    )

# ---- Chat input ----
if prompt := st.chat_input("Ask about the 485 visa..."):
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
                    n_results=5,
                )

                # Step 2: Format context for LLM
                context = results.format_for_llm()

                # Step 3: Generate answer using LLM with streaming
                response_placeholder = st.empty()
                full_response = ""

                for chunk in st.session_state.llm_client.stream_answer(
                    question=prompt,
                    context=context,
                ):
                    full_response += chunk
                    response_placeholder.markdown(
                        full_response + "▌", unsafe_allow_html=True
                    )

                response_placeholder.markdown(full_response, unsafe_allow_html=True)

                # Show sources
                sources = results.get_sources()
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for src in sources:
                            st.markdown(
                                f"- **{src['source']}** — Page {src['page']} "
                                f"(Relevance: {src['relevance']})"
                            )

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

# ---- Sidebar: Chat controls ----
with st.sidebar:
    st.markdown("### 💬 Chat Controls")

    # Number of results
    n_results = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More sources = more context but slower responses",
    )

    # Clear chat
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

    # Example questions
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
            st.chat_input("Ask about the 485 visa...", key=f"input_{ex}")
            # This sets the prompt — user clicks example then presses enter