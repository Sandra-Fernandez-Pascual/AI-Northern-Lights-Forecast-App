# 🌌 AI-Powered Northern Lights Forecast

**An end-to-end data science project that combines Machine Learning, real-time space weather data and environmental forecasts to help users identify favourable conditions for observing the Northern Lights.**

🔗 **Live App:** https://northern-lights-forecast.streamlit.app  
📋 **Project Planning:** https://trello.com/b/Q96iavWk/northern-lights-final-project-ironhack

---

# 🎯 Project Goal

The goal of this project was not simply to build a Machine Learning model.

It was to create a complete **end-to-end data science application** that transforms complex space-weather information into something useful and understandable for non-experts.

Research Question > **Can data help users identify favourable conditions for observing the Northern Lights?**

This project explores that question by connecting solar activity, Earth's geomagnetic response, environmental conditions and Machine Learning in one interactive application.

---

## 🌠 About the Project

Seeing the Northern Lights depends on much more than simply knowing whether solar activity is high.

Travelers often need to check several different sources:

- ☀️ Solar activity
- 🌍 Geomagnetic conditions
- ☁️ Weather
- 🌙 Darkness
- 📍 Geographic location

This project brings those factors together into a single Streamlit application designed to answer two different questions:

> **What are the geomagnetic conditions right now?**

and

> **What are my chances of observing the Northern Lights at a selected location and date?**

The application therefore combines **Machine Learning for current geomagnetic conditions** with a separate **rule-based observation forecast for future planning**.

---

# ☀️ How Does an Aurora Happen?

## 🌞 Step 1 — The Sun

The Sun constantly releases charged particles into space.

This is called the **solar wind**.

Normally, this activity is relatively calm. But sometimes the Sun releases stronger and faster streams of particles toward Earth.

To understand these conditions, we measure variables such as:

- Solar wind speed
- Density
- Temperature
- Magnetic field components (Bx, By, Bz)

Together, these variables describe **what the Sun is sending toward Earth**.

---

## 🌍 Step 2 — Earth Reacts

When those particles reach Earth, they interact with Earth's magnetic field.

If the solar wind is weak:

➡️ Earth's magnetic field remains relatively calm.

If the interaction becomes stronger:

➡️ Earth's magnetic field becomes disturbed.

One way scientists measure this disturbance is the **Dst (Disturbance Storm Time) Index**.

This gives us the simplified relationship:

**Solar wind → Earth's geomagnetic response (Dst)**

This is the relationship learned by the Machine Learning model.

---

## 🌌 Step 3 — Northern Lights

During geomagnetic disturbances, charged particles can travel along Earth's magnetic field and interact with gases in the upper atmosphere.

Those interactions produce light.

✨ **That's an aurora.**

In simplified form:

```text
☀️ Solar activity
        ↓
🌬️ Solar wind
        ↓
🌍 Geomagnetic disturbance (Dst)
        ↓
🌌 Aurora becomes possible
```

---

## 📱 Step 4 — Seeing an Aurora Is Another Story

Even when geomagnetic activity is strong enough for auroras to occur, that does **not** mean an observer will necessarily see them.

Good viewing conditions also require:

- 🌙 Sufficient darkness
- ☁️ Clear skies
- 👁️ Good visibility
- 📍 A favourable geographic location

For this reason, the application separates **geomagnetic activity** from **observation conditions**.

The Machine Learning model answers:

> **"How disturbed is Earth's magnetic field right now?"**

For future observation planning, the application separately asks:

> **"Given the expected geomagnetic activity, location, darkness and weather conditions, what is the estimated chance of observing the Northern Lights?"**

---

# 🤖 Machine Learning — Today's AI Aurora Estimate

The Machine Learning component estimates **current geomagnetic storm intensity**.

A **Random Forest regression model** predicts the current **Dst Index** using real-time space weather conditions.

### Model inputs include:

- Solar wind speed
- Solar wind density
- Solar wind temperature
- Magnetic field components
- Smoothed sunspot number

### Target

**Dst Index**

The predicted Dst value is translated into a simple aurora activity estimate so that users do not need specialist knowledge to interpret the result.

```text
Real-time NOAA data
        ↓
Random Forest model
        ↓
Predicted Dst Index
        ↓
Today's AI Aurora Estimate
```

---

# 🔭 Future Aurora Observation Forecast

The future observation forecast is **separate from the Machine Learning model**.

The ML model requires solar wind and magnetic-field variables that are available in real time but are difficult to obtain reliably as forecasts across longer time horizons.

For future dates, the application therefore uses:

- 🌍 NOAA forecast geomagnetic activity (Ap)
- 📍 Geographic latitude
- 🌙 Sky darkness
- ☁️ Cloud cover
- 👁️ Visibility

These factors are combined in a **rule-based calculation** to estimate how favourable the conditions are for observing the Northern Lights.

The calculation separates:

**Aurora potential**

```text
Geomagnetic activity × Geographic latitude
```

from:

**Observation conditions**

```text
Darkness × Cloud conditions × Visibility
```

The result is presented as an **Estimated Observation Chance from 0–100%**.

---

# 🗺️ Interactive Auroral Oval

The application also displays NOAA's auroral oval forecast on an interactive map.

This provides a visual representation of where auroral activity is expected over approximately the next **30–40 minutes**.

Together, the application provides three complementary perspectives:

```text
🤖 Today's AI Aurora Estimate
          +
🌌 Estimated Observation Chance
          +
🗺️ NOAA Auroral Oval
```

---

# ❤️ Why This Approach Is Scientifically Coherent

A tempting approach would be to train a model to predict:

> **Aurora: Yes / No**

However, that would require reliable historical labels indicating whether an aurora was actually visible from a specific place and time.

Instead, this project predicts a **real physical quantity: the Dst Index**, which describes geomagnetic storm intensity.

The Machine Learning model therefore focuses on the physical relationship present in the available historical data:

```text
☀️ What is the Sun doing?
        ↓
🌍 How is Earth responding?
        ↓
🤖 ML → Dst
```

The application then treats future observation planning as a separate problem:

```text
🌍 What geomagnetic activity is expected?
        ↓
NOAA forecast → Ap
        ↓
📍 Where is the observer?
🌙 Will it be dark?
☁️ Will the sky be clear?
        ↓
🌌 Estimated Observation Chance
```

This keeps the Machine Learning task aligned with the available scientific data while still producing a practical tool for aurora observation planning.

---

# 📊 Data

The Machine Learning workflow uses three historical datasets:

### 🌬️ Solar Wind
Minute-level measurements including:

- Speed
- Density
- Temperature
- Magnetic field components

### 🌍 Dst Labels
Hourly **Dst Index** measurements used as the Machine Learning target.

### ☀️ Sunspots
Monthly smoothed sunspot numbers providing longer-term solar activity context.

### Dataset Source

**NASA Space Weather Data via Kaggle:**  
https://www.kaggle.com/datasets/arashnic/soalr-wind/data

### Scientific Background

**NOAA Magnetic Forecasting Competition — DrivenData:**  
https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/page/280/

---

# 🌐 APIs

The deployed application integrates several external data sources:

| API | Purpose |
|---|---|
| NOAA Real-Time Space Weather | Solar wind and magnetic field measurements |
| NOAA Solar Cycle | Smoothed sunspot number |
| NOAA 45-Day Forecast | Forecast geomagnetic activity |
| NOAA Aurora Forecast | Auroral oval visualization |
| Open-Meteo Forecast | Cloud cover and visibility |
| Open-Meteo Historical Weather | Typical conditions for longer-range dates |
| Open-Meteo Geocoding | Destination → coordinates |
| Sunrise-Sunset API | Astronomical twilight and darkness |

---

# 🔬 Data Science Workflow

```text
Historical datasets
        ↓
Data Cleaning & Integration
        ↓
Exploratory Data Analysis
        ↓
Feature Selection & Preprocessing
        ↓
Machine Learning
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Random Forest
        ↓
Real-time API Integration
        ↓
Streamlit Application
```

The project includes:

- Data cleaning
- Exploratory Data Analysis
- Feature selection
- Train-test split
- Feature scaling
- Regression modelling
- Model comparison
- Feature importance
- Hyperparameter tuning
- Model evaluation
- Real-time API integration

---

# 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Plotly
- Pydeck
- Requests
- Joblib
- NOAA APIs
- Open-Meteo APIs
- Sunrise-Sunset API
- GitHub

---

# 🚀 Future Improvements

Potential future developments include:

- Hourly aurora intensity forecasts for future dates
- Push notifications for favourable aurora conditions
- Travel recommendations
- Explainable AI
- Extending the Machine Learning model from real-time estimation toward future geomagnetic forecasting

---

## 📋 Project Planning

The development process and project tasks were organized using Trello:

https://trello.com/b/Q96iavWk/northern-lights-final-project-ironhack

---

### 🎓 Ironhack Data Analytics Final Project

Developed as the final project for the **Ironhack Data Analytics Bootcamp**.
