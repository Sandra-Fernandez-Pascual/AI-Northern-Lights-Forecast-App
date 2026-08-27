"""Search activity page for anonymous aurora forecast searches.

Reads rows from Neon and shows simple charts. If the database is down,
this page shows a short message and does not affect the forecast.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from database import fetch_searches


st.set_page_config(
    page_title="Search activity",
    page_icon="🌌"
)

st.title("Search activity")
st.caption(
    "Anonymous Generate clicks stored in Neon. "
    "No names, emails, or IP addresses."
)


def get_database_url():
    """Read the Neon connection string from Streamlit secrets."""
    try:
        return st.secrets.get("NEON_DATABASE_URL")
    except Exception:
        return None


def load_searches():
    """Load recent search rows, or an empty table if Neon is unavailable."""
    return fetch_searches(get_database_url())


def count_by_column(data, column_name):
    """Count how often each value appears in one column."""
    if data.empty or column_name not in data.columns:
        return pd.DataFrame(columns=[column_name, "searches"])
    counts = (
        data[column_name]
        .fillna("unknown")
        .value_counts()
        .reset_index()
    )
    counts.columns = [column_name, "searches"]
    return counts


searches = load_searches()

if searches.empty:
    st.info("Activity data unavailable.")
    st.stop()

st.metric("Total searches", len(searches))

probability = pd.to_numeric(searches.get("aurora_probability"), errors="coerce")
if probability.notna().any():
    st.metric("Average aurora probability", f"{probability.mean():.1f}%")

if "sky_too_bright" in searches.columns and searches["sky_too_bright"].notna().any():
    bright_share = searches["sky_too_bright"].fillna(False).astype(bool).mean() * 100
    st.metric("Sky too bright", f"{bright_share:.0f}%")

destination_counts = count_by_column(searches, "destination")
if not destination_counts.empty:
    st.subheader("Searches by destination")
    st.plotly_chart(
        px.bar(
            destination_counts,
            x="destination",
            y="searches",
            title="Most searched destinations"
        ),
        use_container_width=True
    )

outcome_counts = count_by_column(searches, "viewing_outcome")
if not outcome_counts.empty:
    st.subheader("Viewing outcome")
    st.plotly_chart(
        px.bar(
            outcome_counts,
            x="viewing_outcome",
            y="searches",
            title="favourable, low_probability, sky_too_bright, api_failed"
        ),
        use_container_width=True
    )

error_rows = searches[searches["error_type"].notna()] if "error_type" in searches.columns else pd.DataFrame()
if not error_rows.empty:
    st.subheader("API / pipeline errors")
    error_counts = count_by_column(error_rows, "error_type")
    st.plotly_chart(
        px.bar(
            error_counts,
            x="error_type",
            y="searches",
            title="Why a probability could not be computed"
        ),
        use_container_width=True
    )

st.subheader("Recent searches")
recent_columns = [
    column
    for column in ["destination", "forecast_date", "aurora_probability", "viewing_outcome"]
    if column in searches.columns
]
st.dataframe(searches[recent_columns].head(25), use_container_width=True)
