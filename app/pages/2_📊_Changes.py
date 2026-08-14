"""
Change Timeline page — View detected policy changes.

Displays a timeline of detected changes with severity badges,
diff viewer, filtering, and Plotly visualisation.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.db_manager import DatabaseManager

st.set_page_config(
    page_title="📊 Changes — 485 Visa Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Policy Change Timeline")
st.markdown("Track detected changes to 485 visa policy documents over time.")

# ---- Initialise DB ----
db = DatabaseManager()
db.create_tables()

# ---- Severity helpers ----
SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "IMPORTANT": "🟡",
    "MINOR": "🟢",
}
SEVERITY_COLOR = {
    "CRITICAL": "red",
    "IMPORTANT": "orange",
    "MINOR": "green",
}

# ---- Sidebar filters ----
st.sidebar.markdown("### 🔍 Filters")

severity_filter = st.sidebar.selectbox(
    "Severity",
    options=["All", "CRITICAL", "IMPORTANT", "MINOR"],
    index=0,
    format_func=lambda s: f"{SEVERITY_EMOJI.get(s, '')} {s}" if s != "All" else "All Severities",
)

page_limit = st.sidebar.slider("Changes per page", min_value=10, max_value=100, value=25, step=5)

# Manual trigger button
st.sidebar.divider()
st.sidebar.markdown("### ⚡ Actions")
if st.sidebar.button("🔄 Run Manual Scrape", use_container_width=True):
    with st.sidebar.status("Scraping…", expanded=True) as status:
        from src.scraping.homeaffairs_scraper import HomeAffairsScraper
        scraper = HomeAffairsScraper()
        results = scraper.run_daily_scrape()
        status.update(label=f"Scraped {len(results)} page(s)", state="complete")
    st.toast(f"✅ Scraped {len(results)} page(s) — refresh to see new data")

# ---- Stats bar ----
total_changes = db.get_total_changes()
counts = db.get_change_counts_by_severity()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Changes", total_changes)
with col2:
    st.metric("🔴 Critical", counts.get("CRITICAL", 0))
with col3:
    st.metric("🟡 Important", counts.get("IMPORTANT", 0))
with col4:
    st.metric("🟢 Minor", counts.get("MINOR", 0))

st.divider()

# ---- Load changes ----
severity = None if severity_filter == "All" else severity_filter
changes = db.get_changes(severity=severity, limit=page_limit)

if not changes:
    st.info(
        "No changes detected yet. Changes will appear here after the daily "
        "scrape runs or when you trigger a manual scrape.\n\n"
        "💡 Click **🔄 Run Manual Scrape** in the sidebar to check for changes now."
    )

    # Show how to set up automated scraping
    with st.expander("📖 How automated scraping works"):
        st.markdown("""
        ### Daily Automated Scraping
        
        The system checks these pages every day at 2 AM (via GitHub Actions):
        
        1. **Main 485 visa page** — Overview and eligibility
        2. **Documents you need** — Required documentation
        3. **Processing times** — Current processing time estimates
        4. **Visa fees** — Application charges
        
        When content changes are detected, they are classified by severity:
        
        - 🔴 **Critical** — Changes to requirements, eligibility, processing times
        - 🟡 **Important** — Changes to fees, forms, document requirements
        - 🟢 **Minor** — Formatting, contact details, cosmetic changes
        
        You can also run manually:
        ```bash
        python scripts/daily_update.py
        ```
        """)
    st.stop()

# ---- Timeline visualisation ----
# Build a dataframe for the Plotly chart
rows = []
for c in changes:
    rows.append({
        "Date": c.detected_at,
        "Severity": c.severity,
        "Summary": c.summary or c.change_type,
        "Source": c.source_url or "Unknown",
    })
df = pd.DataFrame(rows)

if len(df) > 0:
    # Convert to datetime for proper sorting
    df["Date"] = pd.to_datetime(df["Date"])

    fig = px.scatter(
        df,
        x="Date",
        y="Severity",
        color="Severity",
        color_discrete_map={
            "CRITICAL": "#ff4b4b",
            "IMPORTANT": "#ffa421",
            "MINOR": "#21c354",
        },
        hover_data={"Summary": True, "Source": True},
        height=300,
        title="Change Timeline",
    )
    fig.update_layout(
        yaxis={
            "categoryorder": "array",
            "categoryarray": ["CRITICAL", "IMPORTANT", "MINOR"],
        },
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- Change cards ----
st.subheader(f"📋 Recent Changes ({len(changes)})")

for change in changes:
    emoji = SEVERITY_EMOJI.get(change.severity, "⚪")
    color = SEVERITY_COLOR.get(change.severity, "gray")

    # Build the card header
    date_str = change.detected_at.strftime("%d %b %Y, %H:%M") if change.detected_at else "Unknown"
    change_type_nice = change.change_type.replace("_", " ").title()

    with st.container():
        col1, col2 = st.columns([1, 4])

        with col1:
            st.markdown(f"### {emoji}")
            st.caption(date_str)
            st.caption(change_type_nice)

        with col2:
            st.markdown(f"**{change.summary or change.change_type}**")

            # Show source
            if change.source_url:
                # Shorten URL for display
                display_url = change.source_url
                if "immi.homeaffairs.gov.au" in display_url:
                    display_url = display_url.split("immi.homeaffairs.gov.au", 1)[1]
                st.caption(f"🔗 `{display_url}`")

            # Diff viewer in expandable
            if change.old_value or change.new_value:
                with st.expander("👁️ View Details", expanded=False):
                    diff_col1, diff_col2 = st.columns(2)
                    with diff_col1:
                        st.markdown("**❌ Previous:**")
                        st.code(
                            change.old_value or "(none)",
                            language="diff",
                        )
                    with diff_col2:
                        st.markdown("**✅ Current:**")
                        st.code(
                            change.new_value or "(none)",
                            language="diff",
                        )

        st.divider()

# ---- Pagination info ----
st.caption(f"Showing {len(changes)} of {total_changes} total changes • Filter: {severity_filter}")