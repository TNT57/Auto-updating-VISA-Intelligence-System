"""
Change Timeline page — View detected policy changes.

Phase 2: Will show a timeline of detected changes with severity,
diff viewer, and impact analysis.
"""

import streamlit as st

st.set_page_config(
    page_title="📊 Changes — 485 Visa Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Policy Change Timeline")
st.markdown("Track detected changes to 485 visa policy documents over time.")

# ---- Placeholder content (Phase 2) ----
st.info("🚧 **This feature is coming in Phase 2** (Weeks 3-4)")

st.markdown("""
### What's Coming:

- **Automatic change detection** — Daily comparison of policy documents
- **Severity classification** — 🔴 Critical / 🟡 Important / 🟢 Minor
- **Diff viewer** — See exactly what changed, side by side
- **Timeline visualization** — Track changes over weeks and months
- **Impact analysis** — AI-generated explanations of what each change means

### How It Works:
1. Every day at 2 AM, the system scrapes `immi.homeaffairs.gov.au`
2. New content is compared against the previous snapshot
3. Changes are classified by severity and stored in the database
4. You see a beautiful timeline of all changes with explanations
""")

# Preview of what the page will look like
with st.expander("👀 Preview — What this page will look like", expanded=False):
    st.markdown("---")
    
    # Mock change entries
    changes = [
        {
            "severity": "🔴 CRITICAL",
            "title": "Processing Time Update",
            "date": "2025-03-15",
            "description": "Processing times increased from 5-7 months to 7-9 months for 75% of applications.",
            "source": "immi.homeaffairs.gov.au/visa-processing-times",
        },
        {
            "severity": "🟡 IMPORTANT",
            "title": "Application Fee Change",
            "date": "2025-02-28",
            "description": "Application fee updated from $1,895 to $1,935.",
            "source": "immi.homeaffairs.gov.au/visa-fees",
        },
        {
            "severity": "🟢 MINOR",
            "title": "Contact Details Update",
            "date": "2025-02-10",
            "description": "Updated contact phone numbers for the Adelaide office.",
            "source": "immi.homeaffairs.gov.au/contact",
        },
    ]
    
    for change in changes:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"**{change['severity']}**")
            st.caption(change["date"])
        with col2:
            st.markdown(f"**{change['title']}**")
            st.markdown(change["description"])
            st.caption(f"Source: {change['source']}")
        st.divider()