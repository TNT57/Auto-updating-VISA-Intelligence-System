"""
Alerts page — Discord notification status and delivery history.

Configuration lives in `.env` (DISCORD_WEBHOOK_URL, ALERT_MIN_SEVERITY) rather
than in this UI, because the daily pipeline runs headless in GitHub Actions and
reads the same settings.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src.utils.config import settings

# set_page_config lives in streamlit_app.py, the navigation entry point.

st.title("🔔 Alerts")
st.markdown("Discord notifications for detected 485 visa policy changes.")

# ---- Current status ----
webhook_configured = bool(settings.discord_webhook_url)

if not settings.alerts_enabled:
    st.info(
        "⏸️ **Change notification is currently parked.** Changes are still "
        "detected and recorded — see the Changes page — they just aren't "
        "pushed anywhere. The alerting code is intact and tested."
    )

col1, col2 = st.columns(2)
with col1:
    if not settings.alerts_enabled:
        st.warning("⏸️ Alerts disabled")
    elif webhook_configured:
        st.success("✅ Discord webhook configured")
    else:
        st.warning("⚠️ Enabled, but no webhook URL set")
with col2:
    st.info(f"📊 Minimum severity: **{settings.alert_min_severity.upper()}**")

if not settings.alerts_enabled or not webhook_configured:
    st.markdown(
        "To turn alerts back on, set both of these in your `.env` file:\n\n"
        "```bash\n"
        "ALERTS_ENABLED=true\n"
        "DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...\n"
        "ALERT_MIN_SEVERITY=IMPORTANT  # CRITICAL | IMPORTANT | MINOR\n"
        "```"
    )

st.divider()

# ---- Severity reference ----
st.subheader("Severity levels")
st.markdown("""
| Severity | Triggered by | Example |
|---|---|---|
| 🔴 **CRITICAL** | Eligibility, requirements, conditions, processing times | Stay period reduced from 3 years to 2 |
| 🟡 **IMPORTANT** | Fees, documents, evidence, English tests | Application charge increased |
| 🟢 **MINOR** | Formatting, contact details, wording | Footer link updated |

Severity is classified by the LLM where a Groq key is available, falling back
to keyword matching otherwise.
""")

st.divider()

# ---- Delivery history ----
st.subheader("📜 Delivery history")

try:
    from src.utils.db_manager import AlertLog, DatabaseManager

    db = DatabaseManager()
    db.create_tables()

    with db.get_session() as session:
        logs = (
            session.query(AlertLog)
            .order_by(AlertLog.sent_at.desc())
            .limit(25)
            .all()
        )

    if not logs:
        st.info(
            "No alerts sent yet. Alerts are dispatched by "
            "`scripts/daily_update.py` when it detects a qualifying change."
        )
    else:
        sent = sum(1 for entry in logs if entry.status == "sent")
        failed = len(logs) - sent

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Sent", sent)
        metric_col2.metric("Failed", failed)

        st.dataframe(
            [
                {
                    "When": entry.sent_at.strftime("%Y-%m-%d %H:%M")
                    if entry.sent_at else "—",
                    "Channel": entry.channel,
                    "Status": "✅ sent" if entry.status == "sent" else "❌ failed",
                    "Change ID": entry.change_id,
                    "Error": entry.error_message or "",
                }
                for entry in logs
            ],
            use_container_width=True,
            hide_index=True,
        )
except Exception as exc:
    st.warning(f"Could not read alert history: {exc}")
    st.caption("Run `python scripts/initial_setup.py` to create the database.")
