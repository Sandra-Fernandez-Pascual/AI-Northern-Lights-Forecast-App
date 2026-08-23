import streamlit as st
import pandas as pd
import numpy as np
import requests
import pydeck as pdk
import plotly.graph_objects as go
import streamlit.components.v1 as components
import joblib
import time
import os
import math

from datetime import datetime, timezone, date, timedelta

_HTTP_HEADERS = {
    "User-Agent": "NorthernLightsForecastApp/1.0"
}


def _get_json(url, timeout=60, retries=2):
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers=_HTTP_HEADERS
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


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
@st.cache_data(ttl=300, show_spinner=False)
def get_aurora_oval():

    data = _get_json(
        "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json",
        timeout=90,
        retries=2
    )
    if not data:
        raise RuntimeError("NOAA aurora oval unavailable")

    coords = data.get("coordinates") or data.get("Coordinates")
    if not coords:
        raise RuntimeError("NOAA aurora oval missing coordinates")

    data["coordinates"] = coords
    return data


def prepare_aurora_oval(data):

    coords = data["coordinates"]

    if coords and isinstance(coords[0], dict):
        aurora_df = pd.DataFrame(coords)
        rename = {}
        for column in aurora_df.columns:
            name = str(column).lower()
            if name in ("lon", "longitude", "long"):
                rename[column] = "longitude"
            elif name in ("lat", "latitude"):
                rename[column] = "latitude"
            elif name in ("aurora", "intensity", "value"):
                rename[column] = "intensity"
        aurora_df = aurora_df.rename(columns=rename)
        aurora_df = aurora_df[["longitude", "latitude", "intensity"]]
    else:
        aurora_df = pd.DataFrame(
            coords,
            columns=["longitude", "latitude", "intensity"]
        )

    aurora_df["longitude"] = pd.to_numeric(aurora_df["longitude"], errors="coerce")
    aurora_df["latitude"] = pd.to_numeric(aurora_df["latitude"], errors="coerce")
    aurora_df["intensity"] = pd.to_numeric(aurora_df["intensity"], errors="coerce")
    aurora_df = aurora_df.dropna()

    aurora_df.loc[aurora_df["longitude"] > 180, "longitude"] = (
        aurora_df.loc[aurora_df["longitude"] > 180, "longitude"] - 360
    )

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
    data = _get_json(url)

    if not data:
        return None

    try:
        return {"smoothed_ssn": float(data[-1]["smoothed_ssn"])}
    except (TypeError, ValueError, KeyError, IndexError):
        return None

# -----------------------------
# 45-Day Space Weather Forecast
# -----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def _space_weather_rows():

    url = "https://services.swpc.noaa.gov/json/45-day-forecast.json"
    payload = _get_json(url)

    if not payload or "data" not in payload:
        raise RuntimeError("NOAA 45-day forecast unavailable")

    return payload["data"]


def get_space_weather_forecast(forecast_date):

    try:
        rows = _space_weather_rows()
    except (RuntimeError, TypeError, ValueError, KeyError):
        return None

    if isinstance(forecast_date, datetime):
        target = forecast_date.date()
    else:
        target = forecast_date

    by_date = {}

    for item in rows:
        try:
            item_date = datetime.fromisoformat(
                str(item["time"]).replace("Z", "+00:00")
            ).date()
            metric = item.get("metric")
            value = item.get("value")
        except (TypeError, ValueError, KeyError):
            continue

        if metric not in ("ap", "f107") or value is None:
            continue

        by_date.setdefault(item_date, {})[metric] = value

    if not by_date:
        return None

    if target in by_date:
        chosen = target
    else:
        chosen = min(by_date, key=lambda day: abs((day - target).days))
        if abs((chosen - target).days) > 1:
            return None

    values = by_date[chosen]

    if "ap" not in values or "f107" not in values:
        return None

    try:
        return {
            "ap_today": float(values["ap"]),
            "f107_today": float(values["f107"])
        }
    except (TypeError, ValueError):
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

        data = _get_json(url)

        if not data or "hourly" not in data or "time" not in data["hourly"]:
            return None

        hourly = data["hourly"]
        times = hourly["time"]
        clouds = hourly.get("cloud_cover") or [np.nan] * len(times)
        visibility = hourly.get("visibility") or [np.nan] * len(times)

        weather_df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "cloud_cover": clouds,
            "visibility": visibility
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

        if weather_df.empty:
            weather_df = pd.DataFrame({
                "time": pd.to_datetime(times),
                "cloud_cover": clouds,
                "visibility": visibility
            })
            weather_df = weather_df[
                weather_df["time"].dt.date == forecast_date
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

            data = _get_json(url)

            if not data or "hourly" not in data or "time" not in data["hourly"]:
                continue

            hourly = data["hourly"]
            times = hourly["time"]
            clouds = hourly.get("cloud_cover") or [np.nan] * len(times)
            visibility = hourly.get("visibility") or [np.nan] * len(times)

            df = pd.DataFrame({
                "time": pd.to_datetime(times),
                "cloud_cover": clouds,
                "visibility": visibility
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
@st.cache_data(ttl=1800, show_spinner=False)
def get_sun_data(latitude, longitude, forecast_date):

    sun_url = (
        "https://api.sunrise-sunset.org/json"
        f"?lat={latitude}"
        f"&lng={longitude}"
        f"&date={forecast_date.isoformat()}"
        "&formatted=0"
    )

    payload = _get_json(sun_url)

    if not payload:
        return None

    inner = payload.get("data") or payload.get("results") or payload

    if not isinstance(inner, dict):
        return None

    begin = _parse_sun_timestamp(inner.get("astronomical_twilight_begin"))
    end = _parse_sun_timestamp(inner.get("astronomical_twilight_end"))
    sunrise = _parse_sun_timestamp(inner.get("sunrise"))
    sunset = _parse_sun_timestamp(inner.get("sunset"))

    try:
        day_length = float(inner.get("day_length"))
    except (TypeError, ValueError):
        day_length = None

    sun_status = inner.get("sun_status")
    if sun_status not in ("midnight_sun", "polar_night", "normal"):
        if day_length is not None and day_length >= 86400:
            sun_status = "midnight_sun"
        elif day_length is not None and day_length <= 0:
            sun_status = "polar_night"
        elif sunrise is None and sunset is None:
            # Same sentinel is used for polar day and polar night.
            # Decide from solar geometry instead of assuming midnight sun.
            dark_hours = _astronomical_dark_hours(latitude, forecast_date)
            sun_status = "polar_night" if dark_hours >= 12 else "midnight_sun"
        else:
            sun_status = "normal"

    return {
        "sun_status": sun_status,
        "astronomical_twilight_begin": begin,
        "astronomical_twilight_end": end,
        "day_length": day_length,
        "latitude": latitude,
        "forecast_date": forecast_date,
    }

# -----------------------------
# NOAA Solar Wind API
# -----------------------------

@st.cache_data(ttl=120, show_spinner=False)
def get_solar_wind():

    data = _get_json(
        "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
    )
    if not data:
        return None

    try:
        latest = data[0]
        return {
            "speed": float(latest["proton_speed"]),
            "density": float(latest["proton_density"]),
            "temperature": float(latest["proton_temperature"]),
        }
    except (TypeError, ValueError, KeyError, IndexError):
        return None


# -----------------------------
# NOAA Magnetic Field API
# -----------------------------

@st.cache_data(ttl=120, show_spinner=False)
def get_magnetic_field():

    data = _get_json("https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json")
    if not data:
        return None

    try:
        latest = data[0]
        return {
            "bx_gse": float(latest["bx_gse"]),
            "by_gse": float(latest["by_gse"]),
            "bz_gse": float(latest["bz_gse"]),
            "theta_gse": float(latest["theta_gse"]),
            "phi_gse": float(latest["phi_gse"]),
            "bx_gsm": float(latest["bx_gsm"]),
            "by_gsm": float(latest["by_gsm"]),
            "bz_gsm": float(latest["bz_gsm"]),
            "theta_gsm": float(latest["theta_gsm"]),
            "phi_gsm": float(latest["phi_gsm"]),
            "bt": float(latest["bt"]),
        }
    except (TypeError, ValueError, KeyError, IndexError):
        return None

# =====================================================
# Helper Functions
# =====================================================

# -----------------------------------------------------
# Sky Darkness
# -----------------------------------------------------

def _parse_sun_timestamp(value):
    if value in (None, "", "None"):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.year <= 1970:
        return None
    return parsed


def _solar_declination_deg(day_of_year):
    return 23.45 * math.sin(
        math.radians((360.0 / 365.0) * (284 + day_of_year))
    )


def _astronomical_dark_hours(latitude, forecast_date):
    """Hours when the Sun is at least 18° below the horizon."""
    day_of_year = forecast_date.timetuple().tm_yday
    decl = math.radians(_solar_declination_deg(day_of_year))
    lat = math.radians(float(latitude))
    sin_target = math.sin(math.radians(-18))
    denom = math.cos(lat) * math.cos(decl)

    if abs(denom) < 1e-12:
        noon_sin = (
            math.sin(lat) * math.sin(decl) +
            math.cos(lat) * math.cos(decl)
        )
        midnight_sin = (
            math.sin(lat) * math.sin(decl) -
            math.cos(lat) * math.cos(decl)
        )
        if noon_sin <= sin_target:
            return 24.0
        if midnight_sin > sin_target:
            return 0.0
        return 0.0

    cos_hour_angle = (
        (sin_target - math.sin(lat) * math.sin(decl)) / denom
    )

    # cos H > 1: even at noon the Sun stays below -18° → polar night
    if cos_hour_angle >= 1:
        return 24.0
    # cos H < -1: even at midnight the Sun stays above -18° → white nights / midnight sun
    if cos_hour_angle <= -1:
        return 0.0

    hour_angle = math.degrees(math.acos(cos_hour_angle))
    return (360.0 - 2.0 * hour_angle) / 15.0


def _sun_elevation_local_deg(latitude, local_time):
    """Approximate solar elevation using local clock time."""
    day_of_year = local_time.timetuple().tm_yday
    decl = math.radians(_solar_declination_deg(day_of_year))
    lat = math.radians(float(latitude))
    hour = local_time.hour + local_time.minute / 60.0
    hour_angle = math.radians(15.0 * (hour - 12.0))
    sine_elevation = (
        math.sin(lat) * math.sin(decl) +
        math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    )
    sine_elevation = max(-1.0, min(1.0, sine_elevation))
    return math.degrees(math.asin(sine_elevation))


def _nearby_aurora_points(aurora_df, latitude, longitude, max_degrees=40):
    """Keep oval points around the destination so the globe cannot fit the whole Earth."""
    if aurora_df is None or aurora_df.empty:
        return aurora_df

    lat1 = np.radians(latitude)
    lon1 = np.radians(longitude)
    lat2 = np.radians(aurora_df["latitude"].to_numpy())
    lon2 = np.radians(aurora_df["longitude"].to_numpy())
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    haversine = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    distance_deg = 2 * np.degrees(
        np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    )
    local = aurora_df.loc[distance_deg <= max_degrees]
    if len(local) < 20:
        return aurora_df
    return local


SKY_TOO_BRIGHT = "No viewing window — sky stays too bright"


def classify_darkness(sun_data, latitude=None, forecast_date=None):

    if latitude is None:
        latitude = sun_data.get("latitude")
    if forecast_date is None:
        forecast_date = sun_data.get("forecast_date")

    if latitude is not None and forecast_date is not None:
        dark_hours = _astronomical_dark_hours(latitude, forecast_date)
        if dark_hours >= 8:
            return "Excellent"
        if dark_hours >= 5:
            return "Good"
        if dark_hours >= 2:
            return "Fair"
        return "Poor"

    if sun_data.get("sun_status") == "midnight_sun":
        return "Poor"
    if sun_data.get("sun_status") == "polar_night":
        return "Excellent"

    return "Poor"


# -----------------------------------------------------
# Best Viewing Time
# -----------------------------------------------------

def get_best_viewing_time(environment, latitude):

    night_df = environment["night_weather"].copy()

    night_df = night_df.dropna(
        subset=["cloud_cover", "visibility"]
    )

    if night_df.empty:
        return "Weather estimate unavailable"

    night_df["sun_elevation"] = night_df["time"].apply(
        lambda moment: _sun_elevation_local_deg(latitude, moment.to_pydatetime())
    )

    # Nautical darkness: Sun at least 12° below the horizon.
    # Brighter than this, aurora is washed out.
    dark_df = night_df[night_df["sun_elevation"] <= -12].copy()

    if dark_df.empty:
        return SKY_TOO_BRIGHT

    dark_df["cloud_score"] = 1 - (
        dark_df["cloud_cover"] / 100
    )

    max_visibility = dark_df["visibility"].max()

    if max_visibility == 0:
        dark_df["visibility_score"] = 0
    else:
        dark_df["visibility_score"] = (
            dark_df["visibility"] / max_visibility
        )

    dark_df["viewing_score"] = (
        0.7 * dark_df["cloud_score"] +
        0.3 * dark_df["visibility_score"]
    )

    best_hour = dark_df.loc[
        dark_df["viewing_score"].idxmax(),
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

        if pd.isna(cloud_cover):
            return "Unavailable"

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

    if pd.isna(cloud_cover):
        return "Cloud cover estimate unavailable."

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
    forecast_date,
):

    darkness = classify_darkness(sun_data, latitude, forecast_date)

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

    if pd.isna(environment["cloud_cover"]):
        cloud_factor = 0.5
    else:
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
        "best_time": get_best_viewing_time(environment, latitude),
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

html, body, [data-testid="stAppViewContainer"] {
    color-scheme: dark;
    background-color: #07110f !important;
    color: #f4f7f6 !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    background: transparent;
}

[data-testid="stAppViewContainer"] > .main {
    position: relative;
    z-index: 1;
}

[data-testid="stSidebar"] {
    z-index: 2;
    background-color: #07110f !important;
}

[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] {
    background-color: #07110f !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    color: #e8eeea !important;
}

[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div,
.js-plotly-plot,
.plot-container,
.svg-container {
    background: transparent !important;
    touch-action: pan-y;
}

[data-testid="stPlotlyChart"] .draglayer,
[data-testid="stPlotlyChart"] .nsewdrag,
[data-testid="stPlotlyChart"] .zoombox {
    pointer-events: none !important;
}

/* ---------- Night sky: stars, shooting stars, aurora ---------- */

.night-sky {
    pointer-events: none;
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
}

.night-sky .star {
    position: absolute;
    width: 2px;
    height: 2px;
    border-radius: 50%;
    background: #e8fff8;
    opacity: 0.7;
    box-shadow: 0 0 5px rgba(215, 255, 244, 0.7);
    animation: twinkle 4.8s ease-in-out infinite;
}

.aurora-curtain {
    position: absolute;
    left: -25%;
    width: 150%;
    height: 70%;
    filter: blur(38px);
    mix-blend-mode: screen;
    opacity: 0.75;
}

.aurora-curtain-1 {
    top: -12%;
    background: linear-gradient(
        115deg,
        transparent 8%,
        rgba(72, 255, 168, 0.72) 32%,
        rgba(46, 210, 140, 0.35) 52%,
        transparent 74%
    );
    animation: aurora-drift 14s ease-in-out infinite;
}

.aurora-curtain-2 {
    top: -2%;
    height: 62%;
    background: linear-gradient(
        98deg,
        transparent 18%,
        rgba(168, 92, 255, 0.62) 38%,
        rgba(255, 86, 176, 0.5) 54%,
        transparent 78%
    );
    animation: aurora-drift 19s ease-in-out infinite reverse;
}

.aurora-curtain-3 {
    top: 6%;
    height: 55%;
    background: linear-gradient(
        78deg,
        transparent 6%,
        rgba(64, 170, 255, 0.58) 28%,
        rgba(90, 240, 220, 0.4) 48%,
        transparent 70%
    );
    animation: aurora-drift 17s ease-in-out infinite;
    animation-delay: -5s;
}

.aurora-curtain-4 {
    top: 10%;
    height: 48%;
    background: linear-gradient(
        130deg,
        transparent 22%,
        rgba(255, 120, 90, 0.32) 40%,
        rgba(255, 70, 140, 0.4) 55%,
        transparent 76%
    );
    animation: aurora-drift 21s ease-in-out infinite reverse;
    animation-delay: -9s;
}

.shooting-star {
    position: absolute;
    top: 8%;
    left: -12%;
    width: 90px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #d7fff4, transparent);
    opacity: 0;
    animation: shoot 11s ease-in-out infinite;
}

.shooting-star-2 {
    top: 24%;
    width: 70px;
    animation: shoot-2 14s ease-in-out infinite;
    animation-delay: 5.5s;
}

@keyframes twinkle {
    0%, 100% { opacity: 0.25; transform: scale(0.8); }
    50% { opacity: 0.95; transform: scale(1.2); }
}

@keyframes shoot {
    0% { transform: translate(0, 0) rotate(18deg); opacity: 0; }
    8% { opacity: 0.9; }
    28% { transform: translate(85vw, 28vh) rotate(18deg); opacity: 0; }
    100% { transform: translate(85vw, 28vh) rotate(18deg); opacity: 0; }
}

@keyframes shoot-2 {
    0% { transform: translate(0, 0) rotate(28deg); opacity: 0; }
    10% { opacity: 0.7; }
    32% { transform: translate(70vw, 38vh) rotate(28deg); opacity: 0; }
    100% { transform: translate(70vw, 38vh) rotate(28deg); opacity: 0; }
}

@keyframes aurora-drift {
    0%, 100% {
        transform: translateX(-4%) translateY(0) skewX(-12deg) scaleY(1);
        opacity: 0.55;
    }
    50% {
        transform: translateX(8%) translateY(12%) skewX(10deg) scaleY(1.15);
        opacity: 0.9;
    }
}

.aurora-header {
    margin: 0.4rem 0 1.6rem 0;
}

.aurora-eyebrow {
    display: inline-block;
    color: #f4fffc !important;
    background: rgba(7, 17, 15, 0.9);
    border: 1px solid rgba(137, 220, 202, 0.28);
    border-radius: 8px;
    padding: 0.4rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    margin-bottom: 0.45rem;
}

@media (prefers-reduced-motion: reduce) {
    .night-sky .star,
    .aurora-curtain,
    .shooting-star,
    .stButton > button {
        animation: none;
    }
    .shooting-star {
        display: none;
    }
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
    text-shadow: 0 1px 10px #07110f;
}

p, label {
    color: #f5f7f6 !important;
}

.stCaption {
    color: #dfeae7 !important;
    opacity: 1 !important;
    text-shadow: 0 1px 8px #07110f;
}

/* ---------- Sidebar Title ---------- */

.sidebar-title {
    color: #79d8c1 !important;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    margin-bottom: 1.2rem;
}

/* ---------- Sidebar Labels ---------- */

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p {
    color: #d9f5ee !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #9ecdc2 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] hr {
    border-top: 1px solid rgba(226, 232, 228, 0.16);
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(10, 22, 20, 0.92);
    border-color: rgba(121, 216, 193, 0.22) !important;
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
    border: 1px solid rgba(226, 232, 228, 0.45);
    background-image:
        linear-gradient(
            180deg,
            rgba(244, 247, 246, 0.42) 0%,
            rgba(244, 247, 246, 0.08) 38%,
            transparent 58%
        ),
        linear-gradient(
            135deg,
            rgba(64, 157, 137, 0.98),
            rgba(47, 116, 104, 0.98)
        );
    color: #f4f7f6;
    font-weight: 600;
    padding: 0.55rem 0.7rem !important;
    min-height: 0 !important;
    height: auto !important;
    line-height: 1.15 !important;
    font-size: 0.84rem !important;
    white-space: nowrap !important;
    overflow: hidden;
    box-shadow:
        0 0 12px rgba(226, 232, 228, 0.5),
        0 0 26px rgba(121, 216, 193, 0.32),
        inset 0 1px 0 rgba(255, 255, 255, 0.4);
    animation: button-glow 2.8s ease-in-out infinite;
    transition: all 0.2s ease;
}

.stButton > button p {
    white-space: nowrap !important;
    font-size: 0.84rem !important;
    color: #f4f7f6 !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(244, 247, 246, 0.7);
    box-shadow:
        0 0 18px rgba(226, 232, 228, 0.7),
        0 0 34px rgba(121, 216, 193, 0.45),
        inset 0 1px 0 rgba(255, 255, 255, 0.5);
}

@keyframes button-glow {
    0%, 100% {
        box-shadow:
            0 0 10px rgba(226, 232, 228, 0.4),
            0 0 22px rgba(121, 216, 193, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.35);
    }
    50% {
        box-shadow:
            0 0 18px rgba(244, 247, 246, 0.7),
            0 0 36px rgba(121, 216, 193, 0.48),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
    }
}

/* ---------- Dividers ---------- */

hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
}

/* ---------- Metric Cards ---------- */

[data-testid="stMetric"] {
    background:
        linear-gradient(rgba(103, 196, 177, 0.12), rgba(103, 196, 177, 0.12)),
        #07110f;
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
    background:
        linear-gradient(rgba(103, 196, 177, 0.12), rgba(103, 196, 177, 0.12)),
        #07110f;
    border: 1px solid rgba(137, 220, 202, 0.20);
    border-radius: 16px;
    padding: 1.3rem;
    color: #f4fffc;
}

/* ---------- Forecast Condition Cards ---------- */

.condition-card {
    background:
        linear-gradient(rgba(103, 196, 177, 0.08), rgba(103, 196, 177, 0.08)),
        #07110f;
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
        ),
        #07110f;
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
    display: inline-block;
    color: #f4fffc !important;
    background: rgba(7, 17, 15, 0.9);
    border: 1px solid rgba(137, 220, 202, 0.28);
    border-radius: 8px;
    padding: 0.4rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 700;
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
    background:
        linear-gradient(rgba(103, 196, 177, 0.10), rgba(103, 196, 177, 0.10)),
        #07110f;
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
        max-width: 100%;
        touch-action: pan-y;
    }

    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container,
    [data-testid="stPlotlyChart"] .svg-container {
        width: 100% !important;
        max-width: 100% !important;
        touch-action: pan-y;
    }

    [data-testid="stPlotlyChart"] .draglayer,
    [data-testid="stPlotlyChart"] .nsewdrag,
    [data-testid="stPlotlyChart"] .zoombox {
        pointer-events: none !important;
    }

    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    .aurora-header {
        margin-bottom: 1rem;
    }

    .aurora-header h1 {
        font-size: 1.85rem !important;
    }

    .stButton > button,
    .stButton > button p {
        padding: 0.55rem 0.7rem !important;
        min-height: 0 !important;
        height: auto !important;
        font-size: 0.84rem !important;
        white-space: nowrap !important;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] {
        background-color: #07110f !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p {
        color: #d9f5ee !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #9ecdc2 !important;
    }

    .aurora-eyebrow,
    .forecast-eyebrow {
        color: #f4fffc !important;
        background: rgba(7, 17, 15, 0.9);
        font-size: 0.75rem;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stPlotlyChart"] > div,
    .js-plotly-plot,
    .plot-container {
        background: transparent !important;
        touch-action: pan-y;
    }

    [data-testid="stPlotlyChart"] .draglayer,
    [data-testid="stPlotlyChart"] .nsewdrag,
    [data-testid="stPlotlyChart"] .zoombox {
        pointer-events: none !important;
    }

}

</style>
""", unsafe_allow_html=True)


def _night_sky_html():
    spots = [
        (8, 12, 0), (22, 28, 1.1), (41, 9, 0.4), (63, 18, 1.8),
        (81, 7, 0.7), (91, 32, 2.2), (14, 44, 1.4), (35, 61, 0.2),
        (58, 48, 2.6), (77, 70, 0.9), (5, 78, 1.7), (48, 22, 3.1),
        (70, 40, 0.3), (88, 58, 2.0), (28, 86, 1.2), (52, 80, 2.4),
        (16, 18, 0.6), (84, 14, 1.9),
    ]
    stars = "".join(
        f'<span class="star" style="left:{left}%;top:{top}%;animation-delay:{delay}s;"></span>'
        for left, top, delay in spots
    )
    return (
        '<div class="night-sky">'
        + stars
        + '<div class="aurora-curtain aurora-curtain-1"></div>'
        + '<div class="aurora-curtain aurora-curtain-2"></div>'
        + '<div class="aurora-curtain aurora-curtain-3"></div>'
        + '<div class="aurora-curtain aurora-curtain-4"></div>'
        + '<div class="shooting-star"></div>'
        + '<div class="shooting-star shooting-star-2"></div>'
        + "</div>"
    )

st.markdown(_night_sky_html(), unsafe_allow_html=True)

# -----------------------------
# Load Machine Learning model
# -----------------------------

MODEL_PATH = "random_forest_model_compressed.pkl"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        response = requests.get(st.secrets["model_url"], timeout=120)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load("model_columns.pkl")

    return model, model_columns


try:
    model, model_columns = load_model()

except Exception:
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
            if (btn) btn.click();
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


try:
    aurora_df = prepare_aurora_oval(get_aurora_oval())
except Exception:
    aurora_df = None

# -----------------------------
# Real-time NOAA Space Weather
# -----------------------------

solar_wind = get_solar_wind()
magnetic_field = get_magnetic_field()

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

    try:
        prediction = model.predict(model_input)[0]
    except (ValueError, TypeError):
        prediction = None

# =====================================================
# Aurora Forecast
# =====================================================

result = None

if environment is None or sun_data is None:
    st.warning("Environmental forecast data temporarily unavailable.")

if forecast is not None and environment is not None and sun_data is not None:
    try:
        result = estimate_aurora_probability(
            forecast,
            environment,
            sun_data,
            latitude,
            forecast_date
        )
    except (TypeError, ValueError, KeyError):
        result = None
        st.warning("Environmental forecast data temporarily unavailable.")

# =====================================================
# Results
# =====================================================

# -----------------------------------------------------
# Aurora Observation Probability
# -----------------------------------------------------

st.markdown('<div id="forecast-results"></div>', unsafe_allow_html=True)

components.html(
    """
    <script>
    if (window.parent.innerWidth <= 768) {

        const scrollCheck = setInterval(() => {

            const target =
                window.parent.document.getElementById("forecast-results");

            if (target) {
                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

                clearInterval(scrollCheck);
            }

        }, 100);
    }
    </script>
    """,
    height=0
)

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

    if result["best_time"] == SKY_TOO_BRIGHT:
        st.warning(
            "No useful viewing time on this date because of the current season: "
            "at this latitude the sky does not get dark enough "
            "(midnight sun or white nights). "
            "Aurora would be washed out even with clear skies."
        )
    elif result["best_time"] != "Weather estimate unavailable":
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
                Estimated natural sky darkness for this date and latitude, including polar day and white nights.
            </div>
        </div>
        """, unsafe_allow_html=True)

    cloud_title = (
        "☁️ CLOUD COVER"
        if environment["weather_source"] == "forecast"
        else "☁️ TYPICAL CLOUD COVER"
    )

    cloud_value = (
        "—"
        if pd.isna(environment["cloud_cover"])
        else f"{environment['cloud_cover']:.0f}%"
    )

    with col2:
        st.markdown(f"""
        <div class="condition-card">
            <div class="condition-title">{cloud_title}</div>
            <div class="condition-value">{cloud_value}</div>
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
    local_aurora = _nearby_aurora_points(aurora_df, latitude, longitude)

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lon=local_aurora["longitude"],
            lat=local_aurora["latitude"],
            mode="markers",
            marker=dict(
                size=4,
                color=local_aurora["intensity"],
                colorscale=[
                    [0.0, "#123c35"],
                    [0.4, "#3f8f7e"],
                    [0.7, "#79d8c1"],
                    [1.0, "#d7fff4"]
                ],
                cmin=5,
                cmax=float(max(local_aurora["intensity"].max(), 10))
                if not local_aurora.empty
                else 10,
                opacity=0.75,
                colorbar=dict(
                    title=dict(
                        text="Aurora<br>Intensity",
                        font=dict(color="#dfeae7")
                    ),
                    thickness=12,
                    bgcolor="rgba(0,0,0,0)",
                    tickfont=dict(color="#dfeae7")
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

    view_lat = float(latitude)
    view_lon = float(longitude)

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(
            lon=view_lon,
            lat=view_lat,
            roll=0
        ),
        projection_scale=2.4,
        fitbounds=False,
        showland=True,
        landcolor="#1a433b",
        showocean=True,
        oceancolor="#102e28",
        showlakes=True,
        lakecolor="#102e28",
        showcountries=True,
        countrycolor="#5d8f84",
        coastlinecolor="#79d8c1",
        bgcolor="rgba(0,0,0,0)",
        showframe=False
    )

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dfeae7"),
        uirevision=f"{view_lat:.4f},{view_lon:.4f}",
        autosize=True,
        dragmode=False,
        hovermode=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"aurora-globe-{view_lat:.4f}-{view_lon:.4f}",
        config={
            "displayModeBar": False,
            "responsive": True,
            "staticPlot": True,
            "scrollZoom": False,
            "doubleClick": False,
            "displaylogo": False
        }
    )

    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            const opts = { capture: true, passive: false };
            if (doc._globeZoomHandler) {
                doc.removeEventListener("wheel", doc._globeZoomHandler.wheel, opts);
                doc.removeEventListener("touchmove", doc._globeZoomHandler.touch, opts);
                doc._globeZoomHandler = null;
            }
        })();
        </script>
        """,
        height=0
    )

    st.caption(
        f"NOAA short-term aurora forecast centered on "
        f"{coordinates['name']}. Brighter areas indicate stronger "
        f"expected auroral activity. The white marker shows your destination. "
        f"The globe is locked on this location so it does not spin when you scroll."
    )


# =====================================================
# Testing
# =====================================================


