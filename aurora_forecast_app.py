import streamlit as st
import pandas as pd
import numpy as np
import requests
import pydeck as pdk
import plotly.graph_objects as go
import streamlit.components.v1 as components
import joblib

from datetime import datetime, timezone, date, timedelta

# =====================================================
# Functions
# =====================================================

# -----------------------------
# Geocoding function
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_coordinates(location):

    url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={location}"
        "&count=1"
        "&language=en"
        "&format=json"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            return None

        return {
            "latitude": data["results"][0]["latitude"],
            "longitude": data["results"][0]["longitude"],
            "country": data["results"][0]["country"],
            "name": data["results"][0]["name"],
        }

    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None
        
# -----------------------------
# NOAA Aurora Oval
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_aurora_oval():

    url = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return None


def prepare_aurora_oval(data):

    aurora_df = pd.DataFrame(
        data["coordinates"],
        columns=["longitude", "latitude", "intensity"]
    )

    # NOAA uses longitudes from 0 to 360.
    # Convert them to the standard -180 to 180 map system.
    aurora_df["longitude"] = aurora_df["longitude"].apply(
        lambda x: x - 360 if x > 180 else x
    )

    # Northern Lights only
    aurora_df = aurora_df[
        (aurora_df["latitude"] >= 45) &
        (aurora_df["intensity"] >= 5)
    ]

    return aurora_df

# -----------------------------
# Solar Cycle Forecast
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_smoothed_ssn():

    url = "https://services.swpc.noaa.gov/json/solar-cycle/solar-cycle-25-predicted.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        latest = data[0]

        return {
            "smoothed_ssn": float(latest["smoothed_ssn"])
        }

    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None

# -----------------------------
# 45-Day Space Weather Forecast
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_space_weather_forecast(forecast_date):

    url = "https://services.swpc.noaa.gov/json/45-day-forecast.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        forecast = response.json()["data"]

        target_date = forecast_date.strftime("%Y-%m-%d")

        ap_today = None
        f107_today = None

        for item in forecast:

            item_date = item["time"][:10]

            if item_date == target_date:

                if item["metric"] == "ap":
                    ap_today = item["value"]

                elif item["metric"] == "f107":
                    f107_today = item["value"]

        if ap_today is None or f107_today is None:
            return None

        return {
            "ap_today": float(ap_today),
            "f107_today": float(f107_today)
        }

    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None
        
# -----------------------------
# Open-Meteo API
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_environment(latitude, longitude, forecast_date):

    days_ahead = (forecast_date - date.today()).days

    # ---------------------------------
    # 0–15 days: real weather forecast
    # ---------------------------------
    if days_ahead <= 15:

        start_date = forecast_date.isoformat()
        end_date = (forecast_date + timedelta(days=1)).isoformat()

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            f"&hourly=cloud_cover,visibility"
            f"&start_date={start_date}"
            f"&end_date={end_date}"
            f"&timezone=auto"
        )

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

        except (requests.RequestException, ValueError, KeyError, TypeError):
            return None

        if "hourly" not in data:
            return None

        weather_df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "cloud_cover": data["hourly"]["cloud_cover"],
            "visibility": data["hourly"]["visibility"]
        })

        weather_df = weather_df[
            (
                (weather_df["time"].dt.date == forecast_date) &
                (weather_df["time"].dt.hour >= 21)
            )
            |
            (
                (weather_df["time"].dt.date == forecast_date + timedelta(days=1)) &
                (weather_df["time"].dt.hour <= 3)
            )
        ].copy()

        weather_source = "forecast"

    # ---------------------------------
    # 16–44 days: typical historical conditions
    # ---------------------------------
    else:

        historical_rows = []

        for years_back in range(1, 6):

            historical_date = forecast_date.replace(
                year=forecast_date.year - years_back
            )

            start_date = historical_date.isoformat()
            end_date = (historical_date + timedelta(days=1)).isoformat()

            url = (
                "https://historical-forecast-api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}"
                f"&longitude={longitude}"
                f"&hourly=cloud_cover,visibility"
                f"&start_date={start_date}"
                f"&end_date={end_date}"
                f"&timezone=auto"
            )

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

            except (requests.RequestException, ValueError, KeyError, TypeError):
                continue

            if "hourly" not in data:
                continue

            df = pd.DataFrame({
                "time": pd.to_datetime(data["hourly"]["time"]),
                "cloud_cover": data["hourly"]["cloud_cover"],
                "visibility": data["hourly"]["visibility"]
            })

            df = df[
                (
                    (df["time"].dt.date == historical_date) &
                    (df["time"].dt.hour >= 21)
                )
                |
                (
                    (df["time"].dt.date == historical_date + timedelta(days=1)) &
                    (df["time"].dt.hour <= 3)
                )
            ].copy()

            if not df.empty:
                historical_rows.append(df)

        if historical_rows:

            weather_df = pd.concat(
                historical_rows,
                ignore_index=True
            )

            weather_source = "historical"

        else:

            weather_df = pd.DataFrame({
                "time": [],
                "cloud_cover": [],
                "visibility": []
            })

            weather_source = "historical"

    # ---------------------------------
    # Night hours only
    # ---------------------------------

    weather_df["time"] = pd.to_datetime(
        weather_df["time"],
        errors="coerce"
    )

    if weather_df.empty:
        return {
            "cloud_cover": np.nan,
            "visibility": np.nan,
            "night_weather": weather_df,
            "weather_source": weather_source
        }

    night_df = weather_df.copy()

    cloud_values = night_df["cloud_cover"].dropna()
    visibility_values = night_df["visibility"].dropna()

    cloud_cover = (
        float(cloud_values.mean())
        if not cloud_values.empty
        else np.nan
    )

    visibility = (
        float(visibility_values.mean())
        if not visibility_values.empty
        else np.nan
    )

    return {
        "cloud_cover": cloud_cover,
        "visibility": visibility,
        "night_weather": night_df,
        "weather_source": weather_source
    }
    
# -----------------------------
# Sunrise-Sunset API
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_sun_data(latitude, longitude, forecast_date):

    sun_url = (
        "https://api.sunrise-sunset.org/v2"
        f"?lat={latitude}"
        f"&lng={longitude}"
        f"&date={forecast_date.isoformat()}"
    )

    try:
        sun_response = requests.get(sun_url, timeout=10)
        sun_response.raise_for_status()
        return sun_response.json()

    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None

# =====================================================
# Helper Functions
# =====================================================

# -----------------------------------------------------
# Sky Darkness
# -----------------------------------------------------

def classify_darkness(sun_data):

    # Polar day: no real darkness
    if sun_data["sun_status"] == "midnight_sun":
        return "Poor"

    # Polar night: darkness all day
    if sun_data["sun_status"] == "polar_night":
        return "Excellent"

    astronomical_begin = sun_data.get("astronomical_twilight_begin")
    astronomical_end = sun_data.get("astronomical_twilight_end")

    # If astronomical twilight does not occur,
    # treat the day conservatively
    if astronomical_begin is None or astronomical_end is None:
        return "Poor"

    astronomical_begin = datetime.fromisoformat(astronomical_begin)
    astronomical_end = datetime.fromisoformat(astronomical_end)

    dark_hours = 24 - (
        astronomical_end - astronomical_begin
    ).total_seconds() / 3600

    if dark_hours >= 8:
        return "Excellent"
    elif dark_hours >= 5:
        return "Good"
    elif dark_hours >= 2:
        return "Fair"

    return "Poor"


# -----------------------------------------------------
# Best Viewing Time
# -----------------------------------------------------

def get_best_viewing_time(environment):

    night_df = environment["night_weather"].copy()

    night_df = night_df.dropna(
        subset=["cloud_cover", "visibility"]
    )

    if night_df.empty:
        return "Weather estimate unavailable"

    night_df["cloud_score"] = 1 - (
        night_df["cloud_cover"] / 100
    )

    max_visibility = night_df["visibility"].max()

    if max_visibility == 0:
        night_df["visibility_score"] = 0
    else:
        night_df["visibility_score"] = (
            night_df["visibility"] / max_visibility
        )

    night_df["viewing_score"] = (
        0.7 * night_df["cloud_score"] +
        0.3 * night_df["visibility_score"]
    )

    best_hour = night_df.loc[
        night_df["viewing_score"].idxmax(),
        "time"
    ]

    best_end = best_hour + timedelta(hours=1)

    return (
        f"{best_hour.strftime('%H:%M')} - "
        f"{best_end.strftime('%H:%M')}"
    )

# -----------------------------------------------------
# Geomagnetic Activity
# -----------------------------------------------------

def classify_geomagnetic_activity(ap):

    if ap >= 30:
        return "Very High"
    elif ap >= 15:
        return "High"
    elif ap >= 8:
        return "Moderate"
    return "Low"


# -----------------------------------------------------
# Solar Activity
# -----------------------------------------------------

def classify_solar_activity(f107):

    if f107 >= 150:
        return "Very High"
    elif f107 >= 100:
        return "High"
    elif f107 >= 80:
        return "Moderate"
    return "Low"

# -----------------------------------------------------
# Sky Clarity
# -----------------------------------------------------

def classify_visibility(visibility, cloud_cover=None):

    if pd.isna(visibility):
        if cloud_cover <= 20:
            return "Excellent"
        elif cloud_cover <= 50:
            return "Good"
        elif cloud_cover <= 80:
            return "Fair"
        else:
            return "Poor"

    km = visibility / 1000

    if km >= 30:
        return "Excellent"
    elif km >= 20:
        return "Very Good"
    elif km >= 10:
        return "Good"
    elif km >= 5:
        return "Fair"

    return "Poor"

# -----------------------------------------------------
# Geomagnetic Storm Strength
# -----------------------------------------------------

def classify_dst(dst):

    if dst <= -250:
        return "Extreme"

    elif dst <= -100:
        return "Strong"

    elif dst <= -50:
        return "Moderate"

    elif dst <= -30:
        return "Weak"

    return "Quiet"

# -----------------------------------------------------
# Cloud Comment
# -----------------------------------------------------

def cloud_comment(cloud_cover):

    if cloud_cover <= 20:
        return "Perfect conditions! Hardly any clouds to block the view."

    elif cloud_cover <= 50:
        return "A few clouds around, but you should still have a good chance."

    elif cloud_cover <= 80:
        return "Clouds may hide parts of the aurora."

    else:
        return "The sky is heavily overcast. Even a strong aurora may remain hidden."

# -----------------------------------------------------
# Aurora Observation Probability
# -----------------------------------------------------

def estimate_aurora_probability(
    forecast,
    environment,
    sun_data,
    latitude,
):

    darkness = classify_darkness(sun_data)

    # -----------------------------
    # 1. Geomagnetic potential
    # -----------------------------

    if forecast["ap_today"] >= 30:
        ap_factor = 1.0
    elif forecast["ap_today"] >= 15:
        ap_factor = 0.70
    elif forecast["ap_today"] >= 8:
        ap_factor = 0.40
    else:
        ap_factor = 0.15

    geomagnetic_factor = ap_factor

    # -----------------------------
    # 2. Latitude factor
    # -----------------------------

    abs_lat = abs(latitude)

    if abs_lat >= 65:
        latitude_factor = 1.0
    elif abs_lat >= 60:
        latitude_factor = 0.75
    elif abs_lat >= 55:
        latitude_factor = 0.45
    elif abs_lat >= 50:
        latitude_factor = 0.20
    elif abs_lat >= 45:
        latitude_factor = 0.08
    else:
        latitude_factor = 0.02

    # -----------------------------
    # 3. Observation conditions
    # -----------------------------

    if darkness == "Excellent":
        darkness_factor = 1.0
    elif darkness == "Good":
        darkness_factor = 0.8
    elif darkness == "Fair":
        darkness_factor = 0.5
    else:
        darkness_factor = 0.1

    cloud_factor = max(
        0.05,
        1 - environment["cloud_cover"] / 100
    )

    if pd.isna(environment["visibility"]):
        visibility_factor = 1.0

    else:
        visibility_km = environment["visibility"] / 1000

        if visibility_km >= 20:
            visibility_factor = 1.0
        elif visibility_km >= 10:
            visibility_factor = 0.8
        elif visibility_km >= 5:
            visibility_factor = 0.5
        else:
            visibility_factor = 0.25

    # -----------------------------
    # Final estimate
    # -----------------------------

    aurora_potential = (
        geomagnetic_factor *
        latitude_factor
    )

    observation_conditions = (
        darkness_factor *
        cloud_factor *
        visibility_factor
    )

    probability = (
        aurora_potential *
        observation_conditions *
        100
    )

    probability = round(min(max(probability, 0), 100))

    return {
        "probability": probability,
        "darkness": darkness,
        "best_time": get_best_viewing_time(environment),
        "geomagnetic_activity": classify_geomagnetic_activity(
            forecast["ap_today"]
        ),
        "solar_activity": classify_solar_activity(
            forecast["f107_today"]
        ),
        "sky_clarity": classify_visibility(
            environment["visibility"],
            environment["cloud_cover"]
        )
    }

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="AI-powered Northern Lights Predictor",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>

/* ---------- Global background ---------- */

.stApp {
    background:
        radial-gradient(circle at 20% 0%, rgba(52, 145, 130, 0.10), transparent 35%),
        radial-gradient(circle at 80% 10%, rgba(72, 180, 170, 0.06), transparent 30%),
        #07110f;
    color: #f4f7f6;
}

[data-testid="stHeader"] {
    background: #07110f;
}

[data-testid="stToolbar"] {
    background: transparent;
}

/* ---------- Main container ---------- */

.block-container {
    max-width: 1200px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* ---------- Typography ---------- */

h1, h2, h3 {
    color: #f4f7f6 !important;
    letter-spacing: -0.02em;
}

p, label {
    color: #f5f7f6 !important;
}

.stCaption {
    color: #dfeae7 !important;
    opacity: 1 !important;
}

/* ---------- Sidebar Title ---------- */

.sidebar-title {
    color: #2f7468 !important;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    margin-bottom: 1.2rem;
}

/* ---------- Sidebar Labels ---------- */

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p {
    color: #2f7468 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #2f7468 !important;
    opacity: 1 !important;
}

/* ---------- Destination ---------- */

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #10201d !important;
}

/* ---------- Custom City ---------- */

[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #10201d !important;
    -webkit-text-fill-color: #10201d !important;
    caret-color: #79d8c1 !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
    color: #79a99d !important;
    -webkit-text-fill-color: #79a99d !important;
    opacity: 1 !important;
}

/* ---------- Forecast Date ---------- */

[data-testid="stSidebar"] [data-testid="stDateInput"] input {
    background-color: #ffffff !important;
    color: #10201d !important;
    -webkit-text-fill-color: #10201d !important;
    border-radius: 10px !important;
}

/* ---------- Button ---------- */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(118, 210, 188, 0.25);
    background: linear-gradient(
        135deg,
        rgba(64, 157, 137, 0.95),
        rgba(47, 116, 104, 0.95)
    );
    color: white;
    font-weight: 600;
    padding: 0.7rem 1rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(150, 235, 210, 0.45);
}

/* ---------- Dividers ---------- */

hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
}

/* ---------- Metric Cards ---------- */

[data-testid="stMetric"] {
    background: rgba(103, 196, 177, 0.12);
    border: 1px solid rgba(137, 220, 202, 0.20);
    border-radius: 16px;
    padding: 1.3rem;
}

[data-testid="stMetricLabel"] {
    color: #d9f5ee !important;
}

[data-testid="stMetricValue"] {
    color: #f4fffc !important;
}

/* ---------- Location Card ---------- */

.location-card {
    background: rgba(103, 196, 177, 0.12);
    border: 1px solid rgba(137, 220, 202, 0.20);
    border-radius: 16px;
    padding: 1.3rem;
    color: #f4fffc;
}

/* ---------- Forecast Condition Cards ---------- */

.condition-card {
    background: rgba(103, 196, 177, 0.08);
    border: 1px solid rgba(137, 220, 202, 0.16);
    border-radius: 16px;
    padding: 1.25rem;
    min-height: 145px;
    margin-bottom: 1rem;
}

.condition-title {
    color: #d9f5ee;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.45rem;
}

.condition-value {
    color: #ffffff;
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 0.55rem;
}

.condition-text {
    color: #dfeae7;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ---------- AI Aurora Estimate ---------- */

.ai-card {
    background:
        linear-gradient(
            135deg,
            rgba(103, 196, 177, 0.10),
            rgba(82, 126, 145, 0.08)
        );
    border: 1px solid rgba(137, 220, 202, 0.18);
    border-radius: 18px;
    padding: 1.6rem;
    margin-top: 0.5rem;
}

.ai-label {
    color: #d9f5ee;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

.ai-status {
    color: #ffffff;
    font-size: 1.45rem;
    font-weight: 650;
    margin-bottom: 1rem;
}

.ai-dst {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 650;
    margin-bottom: 0.4rem;
}

.ai-caption {
    color: #dfeae7;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ---------- Forecast Eyebrow ---------- */

.forecast-eyebrow {
    color: #79d8c1;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
}

.mobile-start-hint {
    display: none;
}

/* ---------- Mobile ---------- */

@media (max-width: 768px) {

.mobile-start-hint {
    display: block;
    background: rgba(103, 196, 177, 0.10);
    border: 1px solid rgba(137, 220, 202, 0.18);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
    color: #ffffff;
    font-size: 0.9rem;
}

.mobile-start-hint strong {
    color: #ffffff !important;
    font-size: 1.2rem;
    font-weight: 900;
}
[data-testid="stIconMaterial"] {
    color: #ffffff !important;
}

    .block-container {
        padding: 1.5rem 1rem 3rem 1rem;
    }

    h1 {
        font-size: 2rem !important;
        line-height: 1.15 !important;
    }

    h2 {
        font-size: 1.5rem !important;
    }

    .condition-card {
        min-height: auto;
        padding: 1rem;
    }

    .location-card {
        padding: 1rem;
    }

    .ai-card {
        padding: 1.2rem;
    }

    [data-testid="stMetric"] {
        padding: 1rem;
    }

    [data-testid="stPlotlyChart"] {
        width: 100% !important;
    }

}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Machine Learning model
# -----------------------------

import os

MODEL_PATH = "random_forest_model_compressed.pkl"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        url = st.secrets["model_url"]

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load("model_columns.pkl")

    return model, model_columns


try:
    model, model_columns = load_model()

except (requests.RequestException, OSError, ValueError, EOFError):
    model = None
    model_columns = None
    
# -----------------------------
# Application Header
# -----------------------------
st.title("🌌 AI-powered Northern Lights Predictor")

st.markdown("""
<div class="aurora-header">
    <div class="aurora-eyebrow">AURORA INTELLIGENCE</div>
    <h1>Chase the Northern Lights</h1>
    <p>
        AI-powered aurora forecasts combining space weather,
        atmospheric conditions and real-time solar data.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="mobile-start-hint">📍 Start here: tap the <strong>»</strong> button in the top-left corner to choose your destination and forecast date.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar.container(border=True):

    st.markdown(
        '<div class="sidebar-title">Plan your next adventure</div>',
        unsafe_allow_html=True
    )

    destinations = {
        "Tromsø, Norway": "Tromsø, Norway",
        "Abisko, Sweden": "Abisko, Sweden",
        "Rovaniemi, Finland": "Rovaniemi, Finland",
        "Reykjavík, Iceland": "Reykjavík, Iceland",
        "Fairbanks, Alaska": "Fairbanks, Alaska",
        "Yellowknife, Canada": "Yellowknife, Canada",
        "Other...": None
    }

    selection = st.selectbox(
        "Destination",
        list(destinations.keys())
    )

    if selection == "Other...":
        location = st.text_input(
            "Enter a city",
            placeholder="e.g. Kiruna, Sweden"
        )
    else:
        location = destinations[selection]

    forecast_date = st.date_input(
        "Forecast date",
        value=date.today(),
        min_value=date.today(),
        max_value=date.today() + timedelta(days=44)
    )

    st.caption(
        "Forecasts are available up to 45 days ahead, as reliable forecast data is not available beyond this range."
    )


st.sidebar.markdown("---")

generate = st.sidebar.button("🌌 Generate Aurora Forecast")

if generate:
    components.html(
        """
        <script>
        if (window.parent.innerWidth <= 768) {
            const btn = window.parent.document.querySelector(
                '.stSidebar button[kind="headerNoPadding"]'
            );

            if (btn) {
                btn.click();

                setTimeout(() => {
                    const target =
                        window.parent.document.getElementById("forecast-results");

                    if (target) {
                        target.scrollIntoView({
                            behavior: "smooth",
                            block: "start"
                        });
                    }
                }, 700);
            }
        }
        </script>
        """,
        height=0
    )
    
if not generate:
    st.stop()

# -----------------------------
# Geolocation
# -----------------------------

coordinates = get_coordinates(location)

if coordinates is None:
    st.error("Destination not found. Please select one of the suggested destinations or enter a valid city.")

    st.stop()

latitude = coordinates["latitude"]
longitude = coordinates["longitude"]


st.markdown(
    f"""
    <div class="location-card">
        📍 &nbsp; <strong>{coordinates['name']}, {coordinates['country']}</strong>
    </div>
    """,
    unsafe_allow_html=True
)


aurora_data = get_aurora_oval()

if aurora_data is not None:
    aurora_df = prepare_aurora_oval(aurora_data)
else:
    aurora_df = None

# -----------------------------
# NOAA Solar Wind API
# -----------------------------

try:
    url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    latest_wind = data[0]

    solar_wind = {
        "speed": float(latest_wind["proton_speed"]),
        "density": float(latest_wind["proton_density"]),
        "temperature": float(latest_wind["proton_temperature"])
    }

except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
    solar_wind = None
# -----------------------------
# NOAA Magnetic Field API
# -----------------------------

try:
    url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    latest_mag = data[0]

    magnetic_field = {
        "bx_gse": float(latest_mag["bx_gse"]),
        "by_gse": float(latest_mag["by_gse"]),
        "bz_gse": float(latest_mag["bz_gse"]),
        "theta_gse": float(latest_mag["theta_gse"]),
        "phi_gse": float(latest_mag["phi_gse"]),
        "bx_gsm": float(latest_mag["bx_gsm"]),
        "by_gsm": float(latest_mag["by_gsm"]),
        "bz_gsm": float(latest_mag["bz_gsm"]),
        "theta_gsm": float(latest_mag["theta_gsm"]),
        "phi_gsm": float(latest_mag["phi_gsm"]),
        "bt": float(latest_mag["bt"])
    }

except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
    magnetic_field = None

# =====================================================
# Solar Cycle
# =====================================================

ssn = get_smoothed_ssn()

# =====================================================
# 45-Day Space Weather Forecast
# =====================================================

forecast = get_space_weather_forecast(forecast_date)

if forecast is None:
    st.warning("Forecast data unavailable for the selected date. Please try another date.")

# =====================================================
# Open-Meteo API
# =====================================================

environment = get_environment(
    latitude,
    longitude,
    forecast_date
)

# =====================================================
# Sunrise-sunset API
# =====================================================

sun_data = get_sun_data(
    latitude,
    longitude,
    forecast_date
)

# =====================================================
# Prediction Engine
# =====================================================

prediction = None

if (
    model is not None
    and model_columns is not None
    and solar_wind is not None
    and magnetic_field is not None
    and ssn is not None
):

    api_data = {}

    api_data.update(solar_wind)
    api_data.update(magnetic_field)
    api_data.update(ssn)

    model_input = pd.DataFrame([api_data])

    model_input = model_input.reindex(columns=model_columns)

    prediction = model.predict(model_input)[0]

# =====================================================
# Aurora Forecast
# =====================================================

result = None

if environment is None or sun_data is None:
    st.warning("Environmental forecast data temporarily unavailable.")

if forecast is not None and environment is not None and sun_data is not None:
    result = estimate_aurora_probability(
        forecast,
        environment,
        sun_data,
        latitude
    )

# =====================================================
# Results
# =====================================================

# -----------------------------------------------------
# Aurora Observation Probability
# -----------------------------------------------------

st.markdown('<div id="forecast-results"></div>', unsafe_allow_html=True)

if result is not None:

    st.markdown(
        f'<div class="forecast-eyebrow">YOUR FORECAST · {forecast_date.strftime("%d %B %Y").upper()}</div>',
        unsafe_allow_html=True
    )

    st.subheader("Estimated chance of observing the Northern Lights")

    st.metric(
        label="Estimated Observation Chance",
        value=f"{result['probability']}%"
    )

    st.caption(
        "Estimation based on forecast geomagnetic activity (Ap), "
        "location, sky darkness, cloud cover and visibility."
    )

    if result["best_time"] != "Weather estimate unavailable":
        st.metric(
            label="Best Viewing Time",
            value=result["best_time"]
        )

    st.markdown("---")

    st.subheader("Forecast conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="condition-card">
            <div class="condition-title">🌑 SKY DARKNESS</div>
            <div class="condition-value">{result['darkness']}</div>
            <div class="condition-text">
                Estimated natural sky darkness based on sunrise, sunset and twilight times.
            </div>
        </div>
        """, unsafe_allow_html=True)

    cloud_title = (
        "☁️ CLOUD COVER"
        if environment["weather_source"] == "forecast"
        else "☁️ TYPICAL CLOUD COVER"
    )

    with col2:
        st.markdown(f"""
        <div class="condition-card">
            <div class="condition-title">{cloud_title}</div>
            <div class="condition-value">{environment['cloud_cover']:.0f}%</div>
            <div class="condition-text">
                {cloud_comment(environment["cloud_cover"])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    clarity_text = (
        f"Meteorological visibility: {environment['visibility']/1000:.1f} km."
        if not pd.isna(environment["visibility"])
        else "Estimated from typical cloud cover conditions."
    )

    with col3:

        clarity_title = (
            "👁 SKY CLARITY"
            if environment["weather_source"] == "forecast"
            else "👁 TYPICAL SKY CLARITY"
        )

        st.markdown(f"""
        <div class="condition-card">
            <div class="condition-title">{clarity_title}</div>
            <div class="condition-value">{result['sky_clarity']}</div>
            <div class="condition-text">
                {clarity_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    col4, col5 = st.columns(2)

    with col4:
        st.markdown(f"""
        <div class="condition-card">
            <div class="condition-title">🧲 GEOMAGNETIC ACTIVITY</div>
            <div class="condition-value">{result['geomagnetic_activity']}</div>
            <div class="condition-text">
                Measures the expected disturbance of Earth's magnetic field.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="condition-card">
            <div class="condition-title">☀️ SOLAR ACTIVITY</div>
            <div class="condition-value">{result['solar_activity']}</div>
            <div class="condition-text">
                Represents the overall level of solar activity.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

# -----------------------------------------------------
# Today's AI Aurora Forecast
# -----------------------------------------------------

st.subheader("Today's AI Aurora Estimate")

if prediction is None:

    st.warning("Real-time AI estimate temporarily unavailable.")

else:

    if prediction > -30:
        aurora_status = "🥺 No significant aurora activity expected"

    elif prediction > -50:
        aurora_status = "🌙 Faint aurora possible"

    elif prediction > -100:
        aurora_status = "🌠 Moderate aurora expected"

    elif prediction > -200:
        aurora_status = "✨ Strong aurora expected"

    else:
        aurora_status = "🌌 Exceptional aurora expected"

    st.markdown(f"""
    <div class="ai-card">
    <div class="ai-label">REAL-TIME AI ESTIMATE</div>

    <div class="ai-status">{aurora_status}</div>

    <div class="ai-dst">{prediction:.1f} nT</div>

    <div class="ai-caption">
    Predicted Dst Index
    <br><br>
    This AI estimate is based on current real-time solar wind and magnetic field measurements.
    <br><br>
    More negative Dst values usually indicate stronger geomagnetic storms and better aurora potential.
    </div>
    </div>
    """, unsafe_allow_html=True)

st.subheader("Today's Auroral Activity · Next 30–40 min")

if aurora_df is None:
    st.warning("Auroral oval data temporarily unavailable.")

else:
    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lon=aurora_df["longitude"],
            lat=aurora_df["latitude"],
            mode="markers",
            marker=dict(
                size=4,
                color=aurora_df["intensity"],
                colorscale=[
                    [0.0, "#123c35"],
                    [0.4, "#3f8f7e"],
                    [0.7, "#79d8c1"],
                    [1.0, "#d7fff4"]
                ],
                cmin=5,
                cmax=max(aurora_df["intensity"].max(), 10),
                opacity=0.75,
                colorbar=dict(
                    title="Aurora<br>Intensity",
                    thickness=12
                )
            ),
            hovertemplate="Aurora intensity: %{marker.color}<extra></extra>",
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lon=[longitude],
            lat=[latitude],
            mode="markers+text",
            marker=dict(
                size=11,
                color="white",
                line=dict(
                    color="#79d8c1",
                    width=3
                )
            ),
            text=[coordinates["name"]],
            textposition="top center",
            hovertemplate=(
                f"<b>{coordinates['name']}, "
                f"{coordinates['country']}</b><extra></extra>"
            ),
            showlegend=False
        )
    )

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(
            lon=-longitude,
            lat=-latitude
        ),
        showland=True,
        landcolor="#101b19",
        showocean=True,
        oceancolor="#07110f",
        showlakes=True,
        lakecolor="#07110f",
        showcountries=True,
        countrycolor="#425854",
        coastlinecolor="#536965",
        bgcolor="#07110f"
    )

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#07110f",
        plot_bgcolor="#07110f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.caption(
        f"NOAA short-term aurora forecast centered on "
        f"{coordinates['name']}. Brighter areas indicate stronger "
        f"expected auroral activity. The white marker shows your destination."
    )


# =====================================================
# Testing
# =====================================================


