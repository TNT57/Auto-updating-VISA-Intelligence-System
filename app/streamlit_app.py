"""
Main Streamlit application — Visa 485 Intelligence System.

Multi-page app with chat interface, change history, alerts, and dashboard.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="485 Visa Intelligence",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Sidebar ----
st.sidebar.title("🛂 485 Visa Intelligence")
st.sidebar.markdown(
    "Never miss a visa policy change again.\n\n"
    "This system monitors Australian immigration documents and answers "
    "your 485 visa questions using RAG (Retrieval-Augmented Generation)."
)

st.sidebar.divider()

# Navigation info
st.sidebar.markdown("### 📍 Pages")
st.sidebar.markdown("- 💬 **Chat** — Ask questions about 485 visa")
st.sidebar.markdown("- 📊 **Changes** — Policy change timeline")
st.sidebar.markdown("- 🔔 **Alerts** — Configure notifications *(Phase 3)*")
st.sidebar.markdown("- 📈 **Dashboard** — System health *(Phase 4)*")

st.sidebar.divider()

# System status — lightweight check (no model loading!)
st.sidebar.markdown("### ⚙️ System Status")

try:
    import chromadb
    from src.utils.config import settings

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_collection(settings.collection_name)
    count = collection.count()
    st.sidebar.success(f"✅ Vector DB: {count} chunks")
    st.sidebar.info(f"🤖 Model: {settings.embedding_model}")
except Exception as e:
    st.sidebar.warning("⚠️ Vector DB not initialized")
    st.sidebar.caption("Run `python scripts/initial_setup.py` first")

st.sidebar.divider()

# Disclaimer
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Disclaimer**: This system provides information only. "
    "It is NOT legal advice. Always consult a registered migration agent "
    "for your specific situation."
)

# ---- Main Page ----
st.title("🛂 Subclass 485 Visa Intelligence System")
st.markdown(
    "Welcome! Ask any question about the Temporary Graduate (Subclass 485) visa. "
    "Answers are generated from official Australian government documents with "
    "source citations."
)

# Quick start guide
with st.expander("🚀 Getting Started", expanded=False):
    st.markdown("""
    ### How to use this system:

    1. **Navigate to the Chat page** using the sidebar or click 💬 Chat below
    2. **Ask a question** about the 485 visa in natural language
    3. **Review the answer** with source citations from official documents
    4. **Check confidence** — each answer shows the relevance of retrieved sources

    ### Example questions:
    - "What are the English language requirements for the 485 visa?"
    - "How long is the 485 visa valid for?"
    - "What documents do I need to apply?"
    - "What is the application fee for the 485 visa?"
    - "Can I include family members in my application?"

    ### First time?
    Run the setup script to download documents and build the vector database:
    ```bash
    python scripts/initial_setup.py
    ```
    """)

# Show quick action buttons
col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/1_💬_Chat.py", label="💬 Ask a Question", use_container_width=True)

with col2:
    st.page_link("pages/2_📊_Changes.py", label="📊 View Changes", use_container_width=True)

with col3:
    st.page_link("pages/3_🔔_Alerts.py", label="🔔 Configure Alerts", use_container_width=True)

# Footer
st.divider()
st.caption(
    "Built with ❤️ for international students | "
    "Powered by RAG + Groq LLM + ChromaDB | "
    "Data source: immi.homeaffairs.gov.au"
)