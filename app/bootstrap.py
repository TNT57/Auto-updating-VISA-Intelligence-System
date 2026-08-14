"""
Environment bootstrap shared by every Streamlit page.

Configuration is read by pydantic-settings from environment variables and
.env (src/utils/config.py). Streamlit Cloud has no .env — it provides values
through st.secrets — so without a bridge the deployed app falls back to
defaults and has no API key.

Import this before anything that touches `settings`.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure the project root is importable regardless of how Streamlit was started.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_secrets_into_env() -> None:
    """
    Copy st.secrets into os.environ without clobbering real env vars.

    Existing environment variables win, so a local .env keeps working and
    only the gaps are filled from secrets. Accessing st.secrets raises when
    no secrets file exists, which is the normal local case.
    """
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        return

    try:
        items = list(secrets.items())
    except Exception:
        return

    for key, value in items:
        # Nested tables would need flattening; this app uses flat keys only.
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)


load_secrets_into_env()
