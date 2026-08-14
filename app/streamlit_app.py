"""
Entry point for the 485 Visa Intelligence app.

Acts as a router. Which pages exist depends on PUBLIC_MODE:

  public (default)  Chat only. A visitor came to ask a visa question; scrape
                    history, webhook status and chunk counts are operator
                    tooling, and showing them both clutters the interface and
                    advertises how the system is wired.
  operator          Home, Chat, Changes and Alerts.

Streamlit's automatic pages/ discovery is bypassed by st.navigation, so a
page omitted here is genuinely unreachable, not merely hidden from the menu.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import app.bootstrap  # noqa: F401  — sets sys.path and bridges st.secrets
from src.utils.config import settings

st.set_page_config(
    page_title="485 Visa Intelligence",
    page_icon="🛂",
    layout="centered" if settings.public_mode else "wide",
    initial_sidebar_state="collapsed" if settings.public_mode else "expanded",
)

PAGES_DIR = Path(__file__).parent / "pages"

chat = st.Page(
    str(PAGES_DIR / "1_💬_Chat.py"),
    title="Ask a question",
    icon="💬",
    default=True,
)

if settings.public_mode:
    pages = [chat]
else:
    pages = [
        st.Page(str(PAGES_DIR / "0_🏠_Home.py"), title="Home", icon="🏠",
                default=True),
        st.Page(str(PAGES_DIR / "1_💬_Chat.py"), title="Chat", icon="💬"),
        st.Page(str(PAGES_DIR / "2_📊_Changes.py"), title="Changes", icon="📊"),
        st.Page(str(PAGES_DIR / "3_🔔_Alerts.py"), title="Alerts", icon="🔔"),
    ]

st.navigation(pages, position="sidebar" if not settings.public_mode else "hidden").run()
