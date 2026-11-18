import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_theme import apply_page_theme, page_header
st.set_page_config(
    page_title="Air Quality Forecasting",
    page_icon="🔮",
    layout="wide"
)

apply_page_theme()

page_header('Air Quality Forecasting', 'Generate future hourly pollutant estimates with the trained Prophet time-series models.', 'TIME-SERIES FORECASTING')
st.divider()


# -----------------------------------------
# LOAD DATASET
# -----------------------------------------

@st.cache_data
def load_data():

    df = pd.read_excel(
        "data/Cleaned_Air_Quality.xlsx"
    )

    # Convert date and time into datetime

    time_values = (
        df["Time"]
        .astype(str)
        .str.replace(".", ":", regex=False)
    )

    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + time_values,
        dayfirst=True,
        errors="coerce"
    )

    return df


# -----------------------------------------
# LOAD PROPHET MODELS
# -----------------------------------------

@st.cache_resource
def load_models():

    model_CO = joblib.load(
        "models/prophet_model_CO.pkl"
    )

    model_NO2 = joblib.load(
        "models/prophet_model_NO2.pkl"
    )

    model_NOx = joblib.load(
        "models/prophet_model_NOx.pkl"
    )

    return model_CO, model_NO2, model_NOx


# -----------------------------------------
# LOAD FILES SAFELY
# -----------------------------------------

try:

    df = load_data()

    model_CO, model_NO2, model_NOx = load_models()

except Exception as e:

    st.error(f"Error loading dataset or models: {e}")

    st.info(
        "Check that Prophet is installed and all three "
        "model files are present in the models folder."
    )

    st.stop()


# -----------------------------------------
# POLLUTANT SELECTION
# -----------------------------------------

st.subheader("Select Pollutant")

pollutant = st.selectbox(
    "Choose a pollutant to forecast:",
    [
        "CO(GT)",
        "NO2(GT)",
        "NOx(GT)"
    ]
)


# -----------------------------------------
# SELECT MODEL
# -----------------------------------------

if pollutant == "CO(GT)":

    selected_model = model_CO

    pollutant_name = "Carbon Monoxide (CO)"

elif pollutant == "NO2(GT)":

    selected_model = model_NO2

    pollutant_name = "Nitrogen Dioxide (NO2)"

else:

    selected_model = model_NOx

    pollutant_name = "Nitrogen Oxides (NOx)"


st.info(f"Selected Forecasting Model: {pollutant_name}")


st.divider()


# -----------------------------------------
# FORECAST DURATION
# -----------------------------------------

st.subheader("Forecast Settings")

forecast_days = st.slider(
    "Select Forecast Duration (Days)",
    min_value=1,
    max_value=30,
    value=7
)

forecast_hours = forecast_days * 24

st.write(
    f"Forecasting {forecast_hours} future hourly observations."
)


# -----------------------------------------
# GENERATE FORECAST
# -----------------------------------------

if st.button("Generate Forecast"):

    try:

        with st.spinner("Generating forecast..."):

            # Generate future hourly dates

            future = selected_model.make_future_dataframe(
                periods=forecast_hours,
                freq="h"
            )

            # Predict using the trained Prophet model

            forecast = selected_model.predict(future)

            # Extract future predictions only

            future_forecast = forecast.tail(
                forecast_hours
            ).copy()

            # ---------------------------------
            # DISPLAY FORECAST TABLE
            # ---------------------------------

            st.divider()

            st.subheader("Forecast Results")

            display_forecast = future_forecast[
                [
                    "ds",
                    "yhat",
                    "yhat_lower",
                    "yhat_upper"
                ]
            ].copy()

            display_forecast.columns = [
                "Date and Time",
                "Predicted Value",
                "Lower Limit",
                "Upper Limit"
            ]

            st.dataframe(
                display_forecast,
                width="stretch"
            )

            # ---------------------------------
            # FORECAST METRICS
            # ---------------------------------

            st.subheader("Forecast Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Forecast Duration",
                f"{forecast_days} Days"
            )

            col2.metric(
                "Forecasted Hours",
                forecast_hours
            )

            col3.metric(
                "Average Predicted Value",
                f"{future_forecast['yhat'].mean():.3f}"
            )

            # ---------------------------------
            # FORECAST GRAPH
            # ---------------------------------

            st.divider()

            st.subheader(
                f"{pollutant} Future Forecast"
            )

            fig = go.Figure()

            # Historical actual values

            historical = df[
                ["DateTime", pollutant]
            ].dropna().copy()

            historical[pollutant] = pd.to_numeric(
                historical[pollutant],
                errors="coerce"
            )

            historical = historical.dropna()

            fig.add_trace(
                go.Scatter(
                    x=historical["DateTime"],
                    y=historical[pollutant],
                    mode="lines",
                    name="Historical Actual Values"
                )
            )

            # Confidence interval - upper limit

            fig.add_trace(
                go.Scatter(
                    x=future_forecast["ds"],
                    y=future_forecast["yhat_upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    name="Upper Limit"
                )
            )

            # Confidence interval - lower limit

            fig.add_trace(
                go.Scatter(
                    x=future_forecast["ds"],
                    y=future_forecast["yhat_lower"],
                    mode="lines",
                    fill="tonexty",
                    name="Prediction Interval"
                )
            )

            # Predicted values

            fig.add_trace(
                go.Scatter(
                    x=future_forecast["ds"],
                    y=future_forecast["yhat"],
                    mode="lines",
                    name="Future Prediction"
                )
            )

            fig.update_layout(
                title=f"{pollutant} Forecast",
                xaxis_title="Date and Time",
                yaxis_title="Pollutant Concentration",
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            # ---------------------------------
            # DOWNLOAD FORECAST
            # ---------------------------------

            csv = display_forecast.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name=f"{pollutant}_forecast.csv",
                mime="text/csv"
            )

            st.success(
                "Forecast Generated Successfully!"
            )

    except Exception as e:

        st.error(
            f"Forecast generation failed: {e}"
        )


st.divider()


# -----------------------------------------
# ABOUT PROPHET
# -----------------------------------------

st.subheader("About Prophet Forecasting")

st.markdown("""
Prophet is a time-series forecasting algorithm.

It models time-series data using components such as:

1. Trend – Long-term increase or decrease.
2. Seasonality – Repeating patterns over time.
3. Holidays – Special events, when configured.

In this project, Prophet is used to forecast future
CO, NO2 and NOx concentrations using hourly observations.
""")

st.success("Forecasting Page Loaded Successfully!")