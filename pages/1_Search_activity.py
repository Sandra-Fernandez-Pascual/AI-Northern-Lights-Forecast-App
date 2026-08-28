"""Search activity page for anonymous aurora forecast searches.

Reads rows from Neon and shows simple charts. If the database is down,
this page shows a short message and does not affect the forecast.
"""

import re
import unicodedata

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from database import fetch_searches


MINT = "#79d8c1"
PURPLE = "#5c6bc0"
ICE = "#e8f6ff"
VIOLET = "#c084fc"
DARK_BAD = "#2a2458"
CREAM = "#dfeae7"
MUTED = "#9ecdc2"
OK = "#6ba89c"
NEUTRAL = "#7d9a94"
PLOT_BG = "rgba(13, 28, 25, 0.65)"
OUTCOME_COLORS = {
    "favourable": MINT,
    "low_probability": PURPLE,
    "sky_too_bright": ICE,
    "api_failed": VIOLET
}
QUALITY_COLORS = {
    "Excellent": MINT,
    "Very Good": MUTED,
    "Good": OK,
    "Fair": PURPLE,
    "Poor": DARK_BAD,
    "Unavailable": VIOLET
}
CLOUD_BAND_COLORS = {
    "0–20% almost clear": MINT,
    "21–50% some clouds": "#3d9a86",
    "51–80% may hide aurora": PURPLE,
    "over 80% heavily overcast": DARK_BAD
}
VIS_BAND_COLORS = {
    "Excellent (≥30 km)": MINT,
    "Very Good (≥20 km)": MUTED,
    "Good (≥10 km)": "#d4b5f0",
    "Fair (≥5 km)": PURPLE,
    "Poor (<5 km)": DARK_BAD
}
GEO_COLORS = {
    "Very High": MINT,
    "High": MUTED,
    "Moderate": PURPLE,
    "Low": DARK_BAD
}
CHANCE_COLORS = {
    "20% or more": MINT,
    "under 20%": PURPLE
}
CHART_CONFIG = {"displayModeBar": False}


st.set_page_config(
    page_title="Search activity",
    page_icon="🌌"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    [data-testid="stMetric"] {
        background: #0d1c19;
        border: 1px solid rgba(121, 216, 193, 0.22);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: #9ecdc2 !important;
    }
    [data-testid="stMetricValue"] {
        color: #79d8c1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("← Forecast"):
    st.switch_page("aurora_forecast_app.py")

st.title("Search activity")
st.caption(
    "Anonymous Generate clicks. "
    "First we look at what people searched, then at how good those nights were."
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
    """Count how often each non-empty value appears in one column."""
    if data.empty or column_name not in data.columns:
        return pd.DataFrame(columns=[column_name, "searches"])
    series = data[column_name].dropna()
    series = series[series.astype(str).str.strip() != ""]
    if series.empty:
        return pd.DataFrame(columns=[column_name, "searches"])
    counts = series.value_counts().reset_index()
    counts.columns = [column_name, "searches"]
    return counts


def destination_summary(data):
    """Count searches per destination and the average aurora probability."""
    if data.empty or "destination" not in data.columns:
        return pd.DataFrame()
    summary = data.copy()
    if "aurora_probability" in summary.columns:
        summary["aurora_probability"] = pd.to_numeric(
            summary["aurora_probability"],
            errors="coerce"
        )
    grouped = (
        summary.groupby("destination", dropna=False)
        .agg(
            searches=("destination", "size"),
            avg_probability=("aurora_probability", "mean")
        )
        .reset_index()
        .sort_values("searches", ascending=False)
    )
    return grouped


def destination_group_key(name):
    """Turn typed variants of the same city into one key."""
    if pd.isna(name):
        return ""
    text = str(name).lower().replace("ß", "ss")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("(", " ").replace(")", " ").replace(",", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    parts = text.split()
    if not parts:
        return ""
    if parts[0] in {"kiev", "kyiv"}:
        return "kyiv"
    two_word = {"new", "san", "los", "las", "st", "saint", "fort", "port", "cape"}
    if parts[0] in two_word and len(parts) >= 2:
        return " ".join(parts[:2])
    return parts[0]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def official_place_label(query):
    """Return Open-Meteo's city, country name, or None if the place is not found."""
    if query is None:
        return None
    text = str(query).strip()
    if text == "":
        return None
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": text,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return None
        name = str(results[0].get("name") or "").strip()
        country = str(results[0].get("country") or "").strip()
        if name and country:
            return f"{name}, {country}"
        return name or country or None
    except Exception:
        return None


def grouped_destination_labels(series):
    """Show one official city label per group when the geocoder finds it."""
    keys = series.map(destination_group_key)
    lookup = {}
    frame = pd.DataFrame({"label": series.astype(str), "key": keys})
    for key, group in frame.groupby("key"):
        if key == "":
            continue
        examples = sorted(group["label"].unique(), key=len, reverse=True)
        official = None
        for query in examples + [key]:
            official = official_place_label(query)
            if official:
                break
        lookup[key] = official if official else examples[0]
    return keys.map(lambda key: lookup.get(key, key))


def cloud_band(cloud_cover):
    """Same cloud buckets as cloud_comment() on the forecast page."""
    if pd.isna(cloud_cover):
        return None
    if cloud_cover <= 20:
        return "0–20% almost clear"
    if cloud_cover <= 50:
        return "21–50% some clouds"
    if cloud_cover <= 80:
        return "51–80% may hide aurora"
    return "over 80% heavily overcast"


def visibility_band(visibility_m):
    """Same visibility labels as classify_visibility() on the forecast page."""
    if pd.isna(visibility_m):
        return None
    km = visibility_m / 1000
    if km >= 30:
        return "Excellent (≥30 km)"
    if km >= 20:
        return "Very Good (≥20 km)"
    if km >= 10:
        return "Good (≥10 km)"
    if km >= 5:
        return "Fair (≥5 km)"
    return "Poor (<5 km)"


def weather_window(days_ahead):
    """Same 15-day split as get_environment() on the forecast page."""
    if pd.isna(days_ahead):
        return None
    if days_ahead <= 15:
        return "Live weather (0–15 days)"
    return "Typical clouds (16+ days)"


def add_missing_legend(fig, color_map):
    """Show every colour in the legend, even if it has no data yet."""
    shown = {trace.name for trace in fig.data}
    is_bar = bool(fig.data) and fig.data[0].type == "bar"
    for name, color in color_map.items():
        if name in shown:
            continue
        if is_bar:
            fig.add_bar(
                x=[None],
                y=[None],
                name=name,
                marker_color=color,
                showlegend=True,
                hoverinfo="skip"
            )
        else:
            fig.add_scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=0.5, color="#07110f")
                ),
                name=name,
                showlegend=True,
                hoverinfo="skip"
            )
    return fig


def add_missing_outcome_legend(fig):
    """Show every viewing-outcome colour in the legend, even if it has no data yet."""
    return add_missing_legend(fig, OUTCOME_COLORS)


def stacked_counts(data, x_column, color_column):
    """Count searches for each pair of labels, for stacked bars."""
    if data.empty or x_column not in data.columns or color_column not in data.columns:
        return pd.DataFrame()
    pair = data[[x_column, color_column]].dropna()
    if pair.empty:
        return pd.DataFrame()
    counts = (
        pair.groupby([x_column, color_column])
        .size()
        .reset_index(name="searches")
    )
    if color_column != "viewing_outcome":
        return counts
    x_values = counts[x_column].unique()
    full = pd.MultiIndex.from_product(
        [x_values, list(OUTCOME_COLORS.keys())],
        names=[x_column, color_column]
    )
    return (
        counts.set_index([x_column, color_column])
        .reindex(full, fill_value=0)
        .reset_index()
    )


def style_chart(fig, show_legend=False, legend_title=""):
    """Apply the forecast app night-sky colours to a Plotly chart."""
    fig.update_layout(
        title={"text": "", "font": {"size": 1, "color": "rgba(0,0,0,0)"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=CREAM),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(7, 17, 15, 0.55)",
            font=dict(color=CREAM),
            title_text=legend_title
        ),
        margin=dict(t=16, b=40, l=40, r=20)
    )
    fig.update_xaxes(
        gridcolor="rgba(121, 216, 193, 0.12)",
        zeroline=False,
        color=MUTED
    )
    fig.update_yaxes(
        gridcolor="rgba(121, 216, 193, 0.12)",
        zeroline=False,
        color=MUTED
    )
    return fig


def draw_chart(fig, show_legend=True, legend_title=""):
    """Show a Plotly chart with the app theme, a legend, and no mode bar."""
    st.plotly_chart(
        style_chart(fig, show_legend=show_legend, legend_title=legend_title),
        width="stretch",
        config=CHART_CONFIG
    )


def show_count_bar(data, column_name, title, caption=None, color_map=None,
                   legend_title="", show_legend=True, category_order=None):
    """Draw a bar chart of value counts, if the column has data."""
    counts = count_by_column(data, column_name)
    if counts.empty:
        return
    st.subheader(title)
    if caption:
        st.caption(caption)
    if color_map == OUTCOME_COLORS:
        for name in OUTCOME_COLORS:
            if name not in set(counts[column_name]):
                counts = pd.concat(
                    [counts, pd.DataFrame({column_name: [name], "searches": [0]})],
                    ignore_index=True
                )
        fig = px.bar(
            counts,
            x=column_name,
            y="searches",
            color=column_name,
            color_discrete_map=color_map,
            category_orders={column_name: list(OUTCOME_COLORS.keys())}
        )
        add_missing_outcome_legend(fig)
        draw_chart(fig, show_legend=True, legend_title="Viewing outcome")
        return
    if color_map:
        fig = px.bar(
            counts,
            x=column_name,
            y="searches",
            color=column_name,
            color_discrete_map=color_map,
            category_orders={column_name: category_order or list(color_map.keys())}
        )
        draw_chart(fig, show_legend=show_legend, legend_title=legend_title)
        return
    fig = px.bar(
        counts,
        x=column_name,
        y="searches",
        color_discrete_sequence=[NEUTRAL]
    )
    draw_chart(fig, show_legend=False)


searches = load_searches()

if searches.empty:
    st.info("Activity data unavailable.")
    st.stop()

probability = pd.to_numeric(searches.get("aurora_probability"), errors="coerce")
cloud_cover = pd.to_numeric(searches.get("cloud_cover"), errors="coerce")
visibility = pd.to_numeric(searches.get("visibility"), errors="coerce")
outcomes = searches["viewing_outcome"] if "viewing_outcome" in searches.columns else pd.Series(dtype=str)

searches = searches.copy()
searches["aurora_probability"] = probability
searches["cloud_cover"] = cloud_cover
searches["visibility"] = visibility
searches["cloud_band"] = cloud_cover.map(cloud_band)
searches["visibility_band"] = visibility.map(visibility_band)
if "destination" in searches.columns:
    searches["destination"] = grouped_destination_labels(searches["destination"])

if "forecast_date" in searches.columns:
    forecast_day = pd.to_datetime(searches["forecast_date"], errors="coerce")
    searches["forecast_night_key"] = forecast_day.dt.strftime("%Y-%m-%d")
    searches["forecast_night"] = forecast_day.dt.strftime("%d %b %Y")
    if "searched_at" in searches.columns:
        searched_day = pd.to_datetime(searches["searched_at"], utc=True, errors="coerce")
        searched_day = pd.to_datetime(searched_day.dt.strftime("%Y-%m-%d"))
        forecast_only = pd.to_datetime(forecast_day.dt.strftime("%Y-%m-%d"))
        days_ahead = (forecast_only - searched_day).dt.days
        searches["weather_window"] = days_ahead.map(weather_window)

st.subheader("What people look up")
st.caption("Demand: where users clicked Generate. This is not a ranking of the best aurora spots.")

look_cols = st.columns(3)
look_cols[0].metric(
    "Total searches",
    len(searches),
    help="One row is stored each time someone clicks Generate Aurora Forecast."
)

if "destination" in searches.columns:
    unique_places = searches["destination"].nunique()
    top_place = searches["destination"].mode().iloc[0]
    look_cols[1].metric(
        "Places searched",
        unique_places,
        help="How many different destinations appear in the log."
    )
    look_cols[2].metric(
        "Most searched",
        top_place,
        help="The destination with the most Generate clicks, not the highest aurora %."
    )

st.subheader("How those nights actually were")
st.caption("Result of the forecast for those clicks. A popular place can still be a weak night.")

night_cols = st.columns(3)
if probability.notna().any():
    night_cols[0].metric(
        "Average aurora probability",
        f"{probability.mean():.1f}%",
        help="Mean observation chance across all saved searches, including very low values."
    )
if "sky_too_bright" in searches.columns and searches["sky_too_bright"].notna().any():
    bright_share = searches["sky_too_bright"].fillna(False).astype(bool).mean() * 100
    night_cols[1].metric(
        "Sky too bright",
        f"{bright_share:.0f}%",
        help="Share of searches where the sky stays too bright for a viewing window, such as polar day."
    )
if not outcomes.empty:
    favourable_share = (outcomes == "favourable").mean() * 100
    night_cols[2].metric(
        "Favourable nights",
        f"{favourable_share:.0f}%",
        help="Share of searches with at least 20% probability and a usable dark window."
    )

extra_cols = st.columns(2)
if cloud_cover.notna().any():
    extra_cols[0].metric(
        "Average cloud cover",
        f"{cloud_cover.mean():.0f}%",
        help="Mean cloud cover for the night hours used in the forecast. Higher means a more blocked sky."
    )
if visibility.notna().any():
    extra_cols[1].metric(
        "Average visibility",
        f"{visibility.mean() / 1000:.1f} km",
        help="Mean meteorological visibility. Higher is clearer air; the app treats about 20 km as a clear sky."
    )

destination_counts = count_by_column(searches, "destination")
if not destination_counts.empty:
    st.subheader("Searches by destination")
    st.caption(
        "Taller bars were searched more often. "
        "This is only popularity, not whether the aurora was likely."
    )
    fig = px.bar(
        destination_counts,
        x="destination",
        y="searches",
        color_discrete_sequence=[MINT]
    )
    draw_chart(fig, show_legend=False)

summary = destination_summary(searches)
summary = summary.dropna(subset=["avg_probability"])
summary = summary[summary["searches"] >= 2]
if not summary.empty:
    st.subheader("Which places had better nights?")
    st.caption(
        "Taller = better average forecast. "
        "Compare with Searches by destination above. "
        "One search is not enough, so those places are left off."
    )
    fig = px.bar(
        summary.sort_values("avg_probability", ascending=False),
        x="destination",
        y="avg_probability",
        color_discrete_sequence=[MINT]
    )
    fig.update_yaxes(title="Average aurora %")
    draw_chart(fig, show_legend=False)

destination_outcomes = stacked_counts(searches, "destination", "viewing_outcome")
if not destination_outcomes.empty:
    st.subheader("Destination × viewing outcome")
    st.caption(
        "Each bar is one place. The colours split those searches into what happened: "
        "mint = good enough night (20%+ and dark), "
        "purple = weak chance (under 20%), "
        "ice = sky too bright, "
        "violet = city not found or the forecast did not run. "
        "A tall bar that is almost all purple means a popular place with mostly weak nights."
    )
    fig = px.bar(
        destination_outcomes,
        x="destination",
        y="searches",
        color="viewing_outcome",
        color_discrete_map=OUTCOME_COLORS,
        barmode="stack",
        category_orders={"viewing_outcome": list(OUTCOME_COLORS.keys())}
    )
    add_missing_outcome_legend(fig)
    draw_chart(fig, show_legend=True, legend_title="Viewing outcome")

show_count_bar(
    searches,
    "viewing_outcome",
    "Viewing outcome",
    "One colour per result. "
    "Mint = worth going out (20%+ and dark). "
    "Purple = chance under 20%. "
    "Ice = the sky stays too bright. "
    "Violet = no forecast (usually the city was not found).",
    OUTCOME_COLORS
)

if probability.notna().any():
    st.subheader("How strong were the aurora chances?")
    st.caption(
        "Each bar counts searches in that probability range. "
        "Mint = 20% or more (same as a favourable night). "
        "Purple = under 20% (weak). "
        "A pile of purple on the left means most clicks were weak nights."
    )
    hist_values = probability.dropna()
    hist_bins = pd.cut(hist_values, bins=10)
    hist = hist_bins.value_counts(sort=False).reset_index()
    hist.columns = ["range", "searches"]
    hist["chance"] = hist["range"].map(
        lambda interval: "20% or more" if (interval.left + interval.right) / 2 >= 20 else "under 20%"
    )
    hist["mid"] = hist["range"].map(lambda interval: (interval.left + interval.right) / 2)
    fig = px.bar(
        hist,
        x="mid",
        y="searches",
        color="chance",
        color_discrete_map=CHANCE_COLORS,
        category_orders={"chance": ["20% or more", "under 20%"]}
    )
    fig.update_xaxes(title="Aurora probability %")
    fig.update_yaxes(title="Number of searches")
    fig.add_vline(
        x=20,
        line_dash="dash",
        line_color=MUTED,
        annotation_text="20% cutoff",
        annotation_font_color=MUTED
    )
    add_missing_legend(fig, CHANCE_COLORS)
    draw_chart(fig, legend_title="Aurora chance")

scatter_data = searches.copy()
if "cloud_cover" in scatter_data.columns and "aurora_probability" in scatter_data.columns:
    scatter_data["cloud_cover"] = pd.to_numeric(scatter_data["cloud_cover"], errors="coerce")
    scatter_data["aurora_probability"] = pd.to_numeric(
        scatter_data["aurora_probability"],
        errors="coerce"
    )
    scatter_data = scatter_data.dropna(subset=["cloud_cover", "aurora_probability"])
    if not scatter_data.empty:
        st.subheader("Clouds vs aurora probability")
        st.caption(
            "Each dot is one search. "
            "Right = more clouds. Up = higher aurora %. "
            "The best nights would sit up and left (strong aurora, clearer sky). "
            "Dots at the bottom are already weak nights, so clouds are a second problem. "
            "Colour is the viewing outcome. "
            "If almost every dot is purple, most nights were under 20%."
        )
        hover = [
            column
            for column in ["destination", "forecast_date", "visibility"]
            if column in scatter_data.columns
        ]
        color_column = "viewing_outcome" if "viewing_outcome" in scatter_data.columns else None
        fig = px.scatter(
            scatter_data,
            x="cloud_cover",
            y="aurora_probability",
            color=color_column,
            color_discrete_map=OUTCOME_COLORS if color_column else None,
            hover_data=hover
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="#07110f")))
        fig.update_xaxes(title="Cloud cover %")
        fig.update_yaxes(title="Aurora probability %")
        add_missing_outcome_legend(fig)
        draw_chart(fig, show_legend=True, legend_title="Viewing outcome")

night_outcomes = stacked_counts(searches, "forecast_night", "viewing_outcome")
if not night_outcomes.empty:
    st.subheader("Forecast night × viewing outcome")
    st.caption(
        "Each bar is the date they picked in the app, not when they clicked. "
        "Same viewing-outcome colours: purple = weak chance, violet = no forecast. "
        "Darkness and polar day in the app depend on that date."
    )
    night_order = (
        searches.dropna(subset=["forecast_night", "forecast_night_key"])
        .drop_duplicates("forecast_night")
        .sort_values("forecast_night_key")["forecast_night"]
        .tolist()
    )
    night_outcomes["forecast_night"] = pd.Categorical(
        night_outcomes["forecast_night"],
        categories=night_order,
        ordered=True
    )
    night_outcomes = night_outcomes.sort_values("forecast_night")
    fig = px.bar(
        night_outcomes,
        x="forecast_night",
        y="searches",
        color="viewing_outcome",
        color_discrete_map=OUTCOME_COLORS,
        barmode="stack",
        category_orders={
            "forecast_night": night_order,
            "viewing_outcome": list(OUTCOME_COLORS.keys())
        }
    )
    fig.update_xaxes(type="category", title="Forecast night")
    add_missing_outcome_legend(fig)
    draw_chart(fig, show_legend=True, legend_title="Viewing outcome")

cloud_counts = count_by_column(searches, "cloud_band")
if not cloud_counts.empty:
    st.subheader("Cloud cover bands")
    st.caption(
        "Same buckets as the forecast cloud comments. "
        "Mint = 0–20% almost clear. "
        "Darker mint = 21–50% some clouds. "
        "Purple = 51–80% may hide aurora. "
        "Deep navy-violet = over 80% heavily overcast."
    )
    fig = px.bar(
        cloud_counts,
        x="cloud_band",
        y="searches",
        color="cloud_band",
        color_discrete_map=CLOUD_BAND_COLORS,
        category_orders={"cloud_band": list(CLOUD_BAND_COLORS.keys())}
    )
    draw_chart(fig, legend_title="Cloud cover")

visibility_counts = count_by_column(searches, "visibility_band")
if not visibility_counts.empty:
    st.subheader("Visibility bands")
    st.caption(
        "Same km cutoffs as the forecast page. "
        "Mint = Excellent, pale mint = Very Good, lavender = Good, "
        "purple = Fair, deep navy-violet = Poor."
    )
    fig = px.bar(
        visibility_counts,
        x="visibility_band",
        y="searches",
        color="visibility_band",
        color_discrete_map=VIS_BAND_COLORS,
        category_orders={"visibility_band": list(VIS_BAND_COLORS.keys())}
    )
    draw_chart(fig, legend_title="Visibility")

weather_outcomes = stacked_counts(searches, "weather_window", "viewing_outcome")
if not weather_outcomes.empty:
    st.subheader("Live weather vs typical clouds")
    st.caption(
        "The app uses Open-Meteo live hourly clouds and visibility only when the night "
        "is 0–15 days from the search day. Beyond that it uses typical (historical) clouds. "
        "This split uses searched_at and forecast_date."
    )
    fig = px.bar(
        weather_outcomes,
        x="weather_window",
        y="searches",
        color="viewing_outcome",
        color_discrete_map=OUTCOME_COLORS,
        barmode="stack",
        category_orders={
            "weather_window": [
                "Live weather (0–15 days)",
                "Typical clouds (16+ days)"
            ],
            "viewing_outcome": list(OUTCOME_COLORS.keys())
        }
    )
    add_missing_outcome_legend(fig)
    draw_chart(fig, show_legend=True, legend_title="Viewing outcome")

show_count_bar(
    searches,
    "darkness",
    "Darkness",
    "How dark the sky is for that destination and date "
    "(Excellent / Good / Fair / Poor). "
    "Mint = Excellent, darker teal = Good, purple = Fair, deep navy-violet = Poor. "
    "Poor often means summer at high latitude. "
    "Older searches may not have this label yet.",
    QUALITY_COLORS,
    legend_title="Darkness",
    category_order=["Excellent", "Good", "Fair", "Poor"]
)
darkness_outcomes = stacked_counts(searches, "darkness", "viewing_outcome")
if not darkness_outcomes.empty:
    st.subheader("Darkness × viewing outcome")
    st.caption(
        "Only searches that already have a darkness label. "
        "The probability formula cuts Poor darkness down to 0.1. "
        "Poor or Fair stacked with low_probability or sky_too_bright matches that logic."
    )
    fig = px.bar(
        darkness_outcomes,
        x="darkness",
        y="searches",
        color="viewing_outcome",
        color_discrete_map=OUTCOME_COLORS,
        barmode="stack",
        category_orders={
            "darkness": ["Excellent", "Good", "Fair", "Poor"],
            "viewing_outcome": list(OUTCOME_COLORS.keys())
        }
    )
    add_missing_outcome_legend(fig)
    draw_chart(fig, show_legend=True, legend_title="Viewing outcome")
show_count_bar(
    searches,
    "sky_clarity",
    "Sky clarity",
    "How clear the air was, from visibility (and clouds). "
    "Mint = Excellent, pale mint = Very Good, darker teal = Good, "
    "purple = Fair, deep navy-violet = Poor. "
    "Older searches may not have this label yet.",
    QUALITY_COLORS,
    legend_title="Sky clarity",
    category_order=["Excellent", "Very Good", "Good", "Fair", "Poor", "Unavailable"]
)
show_count_bar(
    searches,
    "geomagnetic_activity",
    "Geomagnetic activity",
    "How disturbed Earth's magnetic field was (from the Ap index). "
    "Mint = Very High, pale mint = High, purple = Moderate, deep navy-violet = Low. "
    "Clouds and darkness still matter. "
    "Older searches may not have this label yet.",
    GEO_COLORS,
    legend_title="Activity",
    category_order=["Very High", "High", "Moderate", "Low"]
)

error_rows = searches[searches["error_type"].notna()] if "error_type" in searches.columns else pd.DataFrame()
if not error_rows.empty:
    error_labels = {
        "geocode_failed": "City not found",
        "forecast_unavailable": "Space-weather data missing",
        "environment_unavailable": "Weather data missing",
        "sun_unavailable": "Sun / darkness data missing",
        "estimate_error": "Could not compute chance"
    }
    error_rows = error_rows.copy()
    error_rows["error_type"] = error_rows["error_type"].map(
        lambda name: error_labels.get(name, name)
    )
    failed_n = len(error_rows)
    city_not_found_n = int((error_rows["error_type"] == "City not found").sum())
    st.subheader("Failed forecasts")
    st.caption(
        "Clicks that never got a forecast. "
        "City not found = the place name was not recognised."
    )
    fail_cols = st.columns(2)
    fail_cols[0].metric("No forecast", failed_n)
    fail_cols[1].metric("City not found", city_not_found_n)
    error_counts = count_by_column(error_rows, "error_type")
    if len(error_counts) > 1:
        error_counts = error_counts.rename(columns={
            "error_type": "What failed",
            "searches": "Searches"
        })
        st.dataframe(error_counts, width="stretch", hide_index=True)

st.subheader("Recent searches")
st.caption("The latest Generate clicks. Scroll sideways on a small screen to see extra columns.")
recent_columns = [
    column
    for column in [
        "destination",
        "forecast_date",
        "aurora_probability",
        "viewing_outcome",
        "darkness",
        "sky_clarity",
        "geomagnetic_activity"
    ]
    if column in searches.columns
]
st.dataframe(searches[recent_columns], width="stretch")
