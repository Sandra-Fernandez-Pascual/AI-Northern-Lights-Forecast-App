# 🌌 AI-powered Northern Lights Predictor

### Aurora Forecast: When Are the Best Conditions to See the Northern Lights?

**An end-to-end data science project that combines Machine Learning, real-time space weather data and environmental forecasts to help users identify favourable conditions for observing the Northern Lights.**

🔗 **Live App:** https://northern-lights-forecast.streamlit.app  
🎤 **Final Presentation:** https://www.canva.com/design/DAHSKV9_810/SHlQImzLKEY5xYw1t5Kd4A/edit?ui=eyJBIjp7fX0  
📋 **Project Planning:** https://trello.com/b/Q96iavWk/northern-lights-final-project-ironhack

---

# 🌠 Project Overview

Aurora Forecast is an **end-to-end decision support application** that integrates Machine Learning with multiple real-time and forecast APIs.

The application provides:

- 🤖 A real-time AI estimate of current geomagnetic conditions
- 🌌 A personalized Northern Lights observation forecast
- 🗺️ An interactive visualization of NOAA's auroral oval

The goal is to transform complex space weather and environmental information into something understandable and useful for people planning to observe the Northern Lights.

---

# ❓ Problem Statement

People who want to see the Northern Lights often need to consult multiple sources of information:

- ☀️ Solar and space weather activity
- 🌍 Geomagnetic conditions
- ☁️ Weather forecasts
- 🌙 Darkness conditions
- 📍 Geographic location

This process can be time-consuming and difficult for non-experts.

The goal of this project is to simplify that process by bringing these factors together into **one application**.

---

# 🎯 Project Goal

The goal is to develop an end-to-end Machine Learning project that predicts **current geomagnetic storm intensity** using real-time space weather data and integrates this prediction into a Streamlit application.

The application also provides personalized aurora forecasts for a selected location and date using forecast geomagnetic activity and environmental data.

This project builds on my previous Machine Learning project, which focused on predicting whether a **Coronal Mass Ejection (CME)** would become geoeffective.

Instead of modelling the CME itself, this project goes one step further:

> **How strongly is Earth actually responding to current space weather conditions?**

The Machine Learning model therefore predicts the resulting geomagnetic storm intensity using the **Dst Index**.

This prediction becomes:

### 🤖 Today's AI Aurora Estimate

Future forecasts are handled separately using forecast geomagnetic activity (`Ap`), geographic location, darkness and atmospheric conditions: 

### 🌌 Estimated chance of observing the Northern Lights

---

# 🔬 Main Research Question

> ## Can data help users identify the best conditions to observe the Northern Lights?

---

# 👥 Target Users

- ✈️ Travelers
- 🌌 Aurora hunters
- 🔭 Astronomy enthusiasts
- 📍 Anyone planning to observe the Northern Lights

---

# ☀️ The Science Behind the Project

## 🌞 Step 1 — The Sun

The Sun constantly releases charged particles into space.

This is the **solar wind**.

Normally, solar wind conditions are relatively calm. But sometimes stronger and faster streams of particles travel toward Earth.

Important measurements include:

- Solar wind speed
- Density
- Temperature
- Magnetic field components (Bx, By, Bz)

These variables describe:

> **What is the Sun sending toward Earth?**

---

## 🌍 Step 2 — Earth Reacts

When those particles reach Earth, they interact with Earth's magnetic field.

If the interaction is weak:

➡️ Earth's magnetic field remains relatively calm.

If it becomes stronger:

➡️ Earth's magnetic field becomes disturbed.

One way scientists measure this disturbance is the:

### Dst — Disturbance Storm Time Index

In simplified form:

```text
Solar Wind
     ↓
Earth's Geomagnetic Response
     ↓
Dst Index
```

This is the relationship the Machine Learning model learns.

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

Even if auroral activity exists, that does **not** mean an observer will necessarily see it.

Good viewing conditions also require:

- 🌙 Darkness
- ☁️ Clear skies
- 👁️ Good visibility
- 📍 A favourable geographic location

That's why the application separates **current geomagnetic activity** from **future observation conditions**.

---

# 🔗 The Logic That Connects Everything Together

There are **two complementary forecasting systems** inside the application.

## 🤖 1. Current Geomagnetic Conditions — Machine Learning

```text
☀️ Real-Time Solar Wind
        +
🧲 Real-Time Magnetic Field
        +
☀️ Solar Cycle
        ↓
🤖 Random Forest
        ↓
🌍 Predicted Current Dst
        ↓
🌌 Today's AI Aurora Estimate
```

The Machine Learning model answers:

> **"How disturbed is Earth's magnetic field right now?"**

---

## 🌌 2. Future Observation Conditions — Rule-Based Forecast

```text
🌍 NOAA Forecast Geomagnetic Activity (Ap)
        +
📍 Geographic Location
        +
🌙 Darkness
        +
☁️ Cloud Cover
        +
👁️ Visibility
        ↓
🌌 Estimated Observation Chance
```

This part answers:

> **"Given the expected geomagnetic activity and local conditions, what is the estimated chance of observing the Northern Lights?"**

### Important distinction

The Machine Learning model **does not predict the future observation probability**.

The model estimates **current Dst**.

Future observation chances are calculated separately using forecast and environmental data.

---

# 📊 Historical Data

The Machine Learning workflow uses three historical datasets from NASA's Space Weather collection.

## ☀️ Solar Wind — Features

Minute-level measurements of solar wind conditions.

Main variables include:

- Speed
- Density
- Temperature
- Bx
- By
- Bz
- Other magnetic field measurements

These become the main **features (`X`)** for the Machine Learning model.

---

## 🌍 Dst Labels — Target

Hourly measurements of the:

### `Dst — Disturbance Storm Time Index`

This represents Earth's geomagnetic response to solar activity.

It becomes the **target (`y`)**.

---

## ☀️ Sunspots — Feature

Monthly smoothed sunspot numbers provide longer-term solar activity context.

Main variable:

```text
smoothed_ssn
```

---

# 🔄 Data Integration

The three datasets have different time resolutions, so they must first be aligned.

## Step 1 — Solar Wind → Hourly

Solar wind observations originally have **minute-level resolution**.

They are converted to hourly averages because Dst is measured hourly.

```text
Minute-level Solar Wind
        ↓
Hourly Average
```

---

## Step 2 — Solar Wind + Dst

The hourly solar wind observations are merged with the corresponding Dst measurements.

Each observation now contains:

```text
Solar Conditions
        +
Earth's Geomagnetic Response (Dst)
```

---

## Step 3 — Add Sunspots

Sunspot measurements are monthly.

Forward fill (`ffill`) is used so that the latest available sunspot value is carried forward until a new monthly observation becomes available.

---

## Final Machine Learning Dataset

### Features (`X`)

```text
Solar Wind Variables
+
Magnetic Field Variables
+
Smoothed Sunspot Activity
```

### Target (`y`)

```text
Dst
```

---

# 🎯 Why Dst Was Selected as the Target

The **Dst Index** was selected because it represents Earth's actual geomagnetic response to incoming space weather conditions.

Solar wind measurements tell us:

> ☀️ **What is arriving from the Sun?**

Dst tells us:

> 🌍 **How strongly is Earth's magnetic field responding?**

From a Machine Learning perspective, Dst is also appropriate because it is a **continuous numerical variable** that can be predicted from the available solar wind features.

For example:

| Solar Conditions | Example Dst |
|---|---:|
| Weak activity | -8 |
| Moderate activity | -45 |
| Strong activity | -120 |

Therefore, this project is a:

# 📈 Regression Problem

The objective is:

> **Predict the Dst Index from historical space weather data.**

If we were predicting:

- Aurora → Yes / No
- Storm → Weak / Strong
- Storm category → G1 / G2 / G3

that would instead be a **classification problem**.

---

# 🔎 Exploratory Data Analysis

The EDA focuses on understanding both the target variable and its relationship with the space weather features.

Main objectives:

- Dataset overview
- Dst Index distribution
- Feature distributions
- Correlation analysis
- Feature relationships

---

# ⚙️ Data Preprocessing

Before training the models, the integrated dataset goes through:

- Missing value handling
- Feature selection
- Train-test split
- Feature scaling using `StandardScaler`

---

# 🤖 Machine Learning

The project follows a complete supervised regression workflow:

```text
Historical Data
        ↓
Data Cleaning & Integration
        ↓
EDA
        ↓
Preprocessing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Feature Importance
        ↓
Hyperparameter Tuning
        ↓
Final Model Selection
```

The Machine Learning stage includes:

- Model training
- Model evaluation
- Model comparison
- Feature importance
- Hyperparameter tuning
- Final model selection

### 🏆 Final Model

**Random Forest Regressor**

The final trained model is saved and later loaded by the Streamlit application.

---

# 🤖 Today's AI Aurora Estimate

> **Based on a Machine Learning prediction.**

The deployed application retrieves real-time NOAA data corresponding to the same type of variables used during model training.

Inputs include:

- Real-time solar wind
- Real-time magnetic field
- Solar-cycle information

The Random Forest predicts the current:

### `Dst Index`

The application then translates the numerical Dst prediction into an easy-to-understand estimate of current aurora activity.

```text
NOAA Real-Time Data
        ↓
Prepare ML Input
        ↓
Random Forest
        ↓
Predicted Current Dst
        ↓
🤖 Today's AI Aurora Estimate
```

---

# 🌌 Estimated Observation Chance

> **Based on a rule-based calculation, not a Machine Learning prediction.**

For a selected location and date, the application combines:

- 🌍 Forecast geomagnetic activity (`Ap`)
- 📍 Geographic latitude
- 🌙 Sky darkness
- ☁️ Cloud cover
- 👁️ Visibility

The calculation separates two concepts.

### Aurora Potential

```text
Geomagnetic Activity × Geographic Latitude
```

### Observation Conditions

```text
Darkness × Cloud Conditions × Visibility
```

These factors are combined to produce an:

### 🌌 Estimated Observation Chance from 0–100%

---

# 💡 Why ML Is Used for Current Conditions but Not Future Dates

The initial idea was to combine the Machine Learning model directly with forecast APIs to predict Dst for future dates.

However, the trained model requires specific solar wind and magnetic field variables.

These variables are available through **real-time NOAA measurements**, but equivalent forecast data is difficult to obtain reliably across longer time horizons.

For this reason, the final architecture separates the two tasks:

```text
CURRENT CONDITIONS
Real-Time NOAA Data
        ↓
Machine Learning
        ↓
Dst
```

versus:

```text
FUTURE PLANNING
NOAA 45-Day Ap Forecast
        +
Location
        +
Weather
        +
Darkness
        ↓
Estimated Observation Chance
```

This allows the Machine Learning model to remain aligned with the data on which it was trained while still providing useful future observation planning.

---

# 🗺️ Interactive Auroral Oval

The application also integrates NOAA's auroral oval forecast.

This provides an interactive visualization of where auroral activity is expected approximately over the next:

### ~30–40 minutes

The final application therefore contains three complementary components:

```text
🤖 Today's AI Aurora Estimate
              +
🌌 Estimated Observation Chance
              +
🗺️ Interactive NOAA Auroral Oval
```

---

# ❤️ Why This Approach Is Scientifically Coherent

A tempting approach would be to train a model that directly predicts:

> **Aurora: Yes / No**

However, that would require reliable historical labels indicating whether an aurora was actually visible from a specific location and time.

The historical datasets instead contain a scientifically measurable physical quantity:

### `Dst`

Therefore, the Machine Learning model predicts **geomagnetic storm intensity**, rather than claiming to directly predict whether somebody will see an aurora.

The complete logic becomes:

```text
☀️ What is the Sun doing?
        ↓
Solar Wind + Magnetic Field
        ↓
🌍 How is Earth responding?
        ↓
🤖 Machine Learning → Dst
        ↓
🌌 Current Aurora Activity
```

And separately:

```text
🌍 Expected Geomagnetic Activity (Ap)
        +
📍 Location
        +
🌙 Darkness
        +
☁️ Weather
        ↓
🌌 Estimated Observation Chance
```

This keeps the Machine Learning problem aligned with the available historical data while still producing a practical tool for Northern Lights observation planning.

---

# 📓 Project Workflow

## `Aurora_Forecast.ipynb`

```text
Load & Clean Historical NASA Datasets
│
├── Integrate Solar Wind + Dst + Sunspots
│
├── Exploratory Data Analysis
│
├── Preprocessing
│   ├── Missing Value Handling
│   ├── Feature Selection
│   ├── Train-Test Split
│   └── Feature Scaling
│
├── Train Regression Models
├── Model Evaluation
├── Model Comparison
├── Feature Importance
├── Hyperparameter Tuning
├── Select Final Random Forest
└── Save Trained Model
```

↓

## `App_Development.ipynb`

```text
Explore & Integrate External APIs
│
├── NOAA Real-Time Solar Wind
├── NOAA Real-Time Magnetic Field
├── NOAA Solar Cycle
├── NOAA 45-Day Forecast
├── Open-Meteo
├── Sunrise-Sunset
└── Geocoding
│
├── Prepare Real-Time ML Input
├── Load Random Forest
├── Predict Current Dst
├── Interpret ML Prediction
├── Develop Observation Forecast
└── Define Streamlit Logic
```

↓

## `aurora_forecast_app.py`

```text
Final Streamlit Application
│
├── 🤖 Today's AI Aurora Estimate
├── 🌌 Estimated Observation Chance
└── 🗺️ Interactive Auroral Oval
```

---

# 🌐 Real-Time & Forecast APIs

| API | Purpose |
|---|---|
| NOAA Real-Time Space Weather | Solar wind and magnetic field measurements used by ML |
| NOAA Solar Cycle Forecast | Smoothed sunspot number required by ML |
| NOAA 45-Day Forecast | Ap and F10.7 forecasts for selected future dates |
| NOAA Aurora Forecast | Auroral oval forecast |
| Open-Meteo Forecast | Cloud cover and visibility |
| Open-Meteo Historical Weather | Typical historical conditions for longer-range dates |
| Open-Meteo Geocoding | Destination → latitude and longitude |
| Sunrise-Sunset API | Astronomical twilight / darkness |

---

# 📚 Data Sources

### Historical Machine Learning Data

**NASA Space Weather Data — Kaggle**  
https://www.kaggle.com/datasets/arashnic/soalr-wind/data

### Scientific Background

**NOAA Magnetic Forecasting — DrivenData**  
https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/page/280/

---

# 🖥️ Streamlit Application

The application functions similarly to an aurora weather forecast.

The user can:

- 📍 Enter a destination
- 📅 Select a forecast date up to 45 days ahead
- 🌐 Retrieve space weather and environmental data
- 🌌 Receive an Estimated Observation Chance
- ☁️ Explore the conditions influencing the forecast
- 🤖 View Today's AI Aurora Estimate
- 🗺️ Explore NOAA's auroral oval

---

# 🛠️ Technologies

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
- NOAA Space Weather APIs
- NOAA Solar Cycle Forecast API
- NOAA 45-Day Forecast
- Open-Meteo Forecast API
- Open-Meteo Historical Weather API
- Open-Meteo Geocoding API
- Sunrise-Sunset API
- GitHub

---

# ✅ Minimum Viable Product

The final MVP includes:

- Historical dataset analysis
- Data cleaning and integration
- Exploratory Data Analysis
- Feature engineering and preprocessing
- Machine Learning regression model
- Model evaluation and comparison
- Hyperparameter tuning
- Real-time API integration
- Streamlit application
- Today's AI Aurora Estimate
- Personalized forecast for a selected location and date
- Interactive NOAA auroral oval visualization

---

# 🏆 Success Criteria

The project is considered successful if it:

- Demonstrates a complete end-to-end data science workflow
- Produces a reliable Machine Learning model
- Successfully integrates real-time and forecast APIs
- Provides an intuitive Streamlit application
- Allows users to understand their estimated chance of observing the Northern Lights
- Demonstrates how data can help identify favourable aurora observation conditions

---

# 🚀 Future Improvements

Potential future developments include:

- ⏱️ Hourly aurora intensity forecasts for future dates
- 🔔 Push notifications
- ✈️ Travel recommendations
- 🔍 Explainable AI
- 🤖 Exploring approaches for extending the Machine Learning model from real-time estimation to future geomagnetic forecasting

---

# 🔗 Project Links

🌌 **Live Application**  
https://northern-lights-forecast.streamlit.app

📋 **Trello Project Planning**  
https://trello.com/b/Q96iavWk/northern-lights-final-project-ironhack

📊 **Historical Dataset**  
https://www.kaggle.com/datasets/arashnic/soalr-wind/data

---

### 🎓 Ironhack Data Analytics Final Project

Developed as the final project for the **Ironhack Data Analytics Bootcamp**.
