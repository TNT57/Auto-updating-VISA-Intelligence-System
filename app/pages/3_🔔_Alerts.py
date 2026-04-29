"""
Alerts Configuration page — Set up notifications for policy changes.

Phase 3: Will support Discord webhooks and email alerts.
"""

import streamlit as st

st.set_page_config(
    page_title="🔔 Alerts — 485 Visa Intelligence",
    page_icon="🔔",
    layout="wide",
)

st.title("🔔 Alert Configuration")
st.markdown("Get notified when 485 visa policies change.")

st.info("🚧 **This feature is coming in Phase 3** (Weeks 5-6)")

st.markdown("""
### Planned Alert Channels:

**🎮 Discord Webhook** (Recommended — Free & Instant)
- Get instant notifications in your Discord server
- Rich embeds with change details and severity
- Configure per-channel severity filters

**📧 Email Alerts** (Optional)
- Daily digest of all changes
- Instant alerts for critical changes only
- Uses Gmail SMTP (free)

### Alert Severity Filters:
- 🔴 **Critical** — Instant notification (processing times, eligibility changes)
- 🟡 **Important** — Daily digest (fees, document requirements)
- 🟢 **Minor** — Weekly summary (formatting, contact details)
""")

# Preview configuration form
with st.expander("👀 Preview — Alert Configuration", expanded=False):
    st.text_input("Discord Webhook URL", placeholder="https://discord.com/api/webhooks/...")
    st.selectbox("Minimum severity for instant alerts", ["🔴 Critical", "🟡 Important", "🟢 Minor"])
    st.checkbox("Enable daily email digest", value=False)
    st.text_input("Email for alerts", placeholder="your@email.com")
    st.button("💾 Save Configuration (Preview Only)", disabled=True)