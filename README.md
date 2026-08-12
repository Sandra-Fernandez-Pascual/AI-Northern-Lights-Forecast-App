# 🌌 AI-Powered Northern Lights Forecast

An end-to-end data science project that combines Machine Learning, real-time space weather data and environmental forecasts to help users identify favourable conditions for observing the Northern Lights.

🔗 **Live App:** https://northern-lights-forecast.streamlit.app  
📋 **Project Planning:** https://trello.com/b/Q96iavWk/northern-lights-final-project-ironhack

---

## 🎯 Research Question

> **Can data help users identify the best conditions to observe the Northern Lights?**

People planning to see the Northern Lights often need to consult multiple sources: space weather, geomagnetic activity, weather forecasts and darkness conditions.

This project brings those factors together into a single application designed to make that information easier to understand and use.

---

## 🌞 From the Sun to the Northern Lights

The Sun constantly releases charged particles known as the **solar wind**.

When solar wind reaches Earth, it interacts with Earth's magnetic field. Strong disturbances can produce geomagnetic storms, whose intensity can be measured using the **Dst (Disturbance Storm Time) index**.

In simplified form:

**Solar activity → Solar wind → Geomagnetic disturbance (Dst) → Aurora potential**

However, an aurora being present does not necessarily mean that it can be observed from a particular location.

Visibility also depends on factors such as:

- 📍 Location
- 🌙 Sky darkness
- ☁️ Cloud cover
- 👁️ Atmospheric visibility

This distinction is the basis of the application.

---

## 🤖 Machine Learning Model

The Machine Learning component predicts the **Dst index**, a continuous numerical measure of geomagnetic storm intensity.

### Features

Historical solar wind measurements including:

- Solar wind speed
- Density
- Temperature
- Magnetic field components (Bx, By, Bz)
- Smoothed sunspot number

### Target

- **Dst (Disturbance Storm Time) Index**

Because Dst is a continuous numerical variable, this is a **regression problem**.

Several regression models were trained, evaluated and compared before selecting and tuning the final **Random Forest** model.

The final model is used in the application to generate **Today's AI Aurora Estimate** from current NOAA space weather measurements.

---

## 🔮 Future Observation Forecast

The Machine Learning model is used for **current conditions**, not future dates.

The original project idea was to combine the ML model directly with forecast APIs. However, the solar wind and magnetic field variables required by the trained model are difficult to obtain as forecast data, particularly across longer time horizons.

For this reason, future observation chances are calculated separately using:

- 🧲 NOAA forecast **Ap index**
- 📍 Latitude
- 🌙 Sky darkness
- ☁️ Cloud cover
- 👁️ Visibility

This allows the application to provide an **Estimated Observation Chance** for a selected location and date without presenting unavailable ML inputs as future predictions.

---

## 📱 Streamlit Application

The application contains three main components:

### 1. 🤖 Today's AI Aurora Estimate

Real-time NOAA solar wind and magnetic field measurements are passed to the trained Random Forest model.

**Real-time NOAA data → ML model → Predicted Dst → Current geomagnetic estimate**

### 2. 🌌 Estimated Observation Chance

Users select a destination and forecast date.

The application combines forecast geomagnetic activity with geographical and atmospheric conditions to estimate how favourable the conditions are for observing the Northern Lights.

**Location + Date → Ap + Darkness + Weather + Latitude → Estimated Observation Chance**

### 3. 🗺️ Interactive Auroral Oval

The application retrieves NOAA's auroral oval forecast and displays expected auroral activity for approximately the next **30–40 minutes** on an interactive map.

---

## 📊 Historical Data

The Machine Learning workflow uses three historical datasets:

- **Solar Wind** — minute-level solar wind and magnetic field measurements.
- **Labels** — hourly Dst values used as the ML target.
- **Sunspots** — monthly smoothed sunspot numbers providing longer-term solar activity context.

Solar wind measurements were aggregated to hourly resolution so they could be aligned with the hourly Dst target.

### Dataset Source

Historical data:

**NASA and NOAA Satellites Solar-Wind Dataset — Kaggle**  
https://www.kaggle.com/datasets/arashnic/soalr-wind/data

The large raw datasets are not stored directly in this repository and can instead be obtained from the original source above.

---

## 📡 APIs

The application integrates several external data sources:

- **NOAA Real-Time Space Weather Data** — solar wind and magnetic field measurements used by the ML model.
- **NOAA Solar Cycle Forecast** — smoothed sunspot number used by the ML model.
- **NOAA 45-Day Forecast** — Ap and F10.7 solar activity forecasts.
- **NOAA Aurora Forecast** — auroral oval forecast for the next ~30–40 minutes.
- **Open-Meteo Forecast API** — cloud cover and visibility.
- **Open-Meteo Historical Weather API** — typical atmospheric conditions for longer-range dates.
- **Open-Meteo Geocoding API** — converts destinations into geographical coordinates.
- **Sunrise-Sunset API** — astronomical twilight information used to estimate sky darkness.

---

## 📚 Scientific Background

The modelling approach was informed by:

**MagNet: Model the Geomagnetic Field — DrivenData**  
https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/page/280/

The challenge explores predicting the **Dst index from solar wind measurements**, providing scientific background for the relationship modelled in this project.

---

## 🔄 Project Workflow

### Machine Learning

**Historical Data → Cleaning → Integration → EDA → Preprocessing → Model Training → Model Comparison → Hyperparameter Tuning → Random Forest**

### Application

**Real-Time NOAA Data → Random Forest → Predicted Dst → Today's AI Aurora Estimate**

**Selected Location + Date → Space Weather Forecast + Weather + Darkness + Geolocation → Estimated Observation Chance**

**NOAA Auroral Oval → Interactive Map**

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Requests
- Joblib
- Plotly
- Pydeck
- NOAA APIs
- Open-Meteo APIs
- Sunrise-Sunset API
- GitHub

---

## 📁 Repository Structure

```text
AI-Northern-Lights-Forecast-App/
│
├── Aurora_Forecast.ipynb
│   └── Data preparation, EDA and Machine Learning
│
├── App_Development.ipynb
│   └── API exploration and application development
│
├── aurora_forecast_app.py
│   └── Final Streamlit application
│
├── model_columns.pkl
│   └── Feature structure required by the ML model
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
