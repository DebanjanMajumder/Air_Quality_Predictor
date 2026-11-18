import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_theme import apply_page_theme, page_header
st.set_page_config(
    page_title="Regression Prediction",
    page_icon="📈",
    layout="wide"
)

apply_page_theme()

page_header('Air Quality Regression Prediction', 'Predict pollutant concentrations from sensor readings and atmospheric conditions using trained regression models.', 'POLLUTANT PREDICTION')
st.divider()


# -----------------------------------------
# LOAD DATASET
# -----------------------------------------

@st.cache_data
def load_data():
    return pd.read_excel("data/Cleaned_Air_Quality.xlsx")


# -----------------------------------------
# LOAD TRAINED MODELS
# -----------------------------------------

@st.cache_resource
def load_models():

    linear_model = joblib.load(
        "models/linear_regression_model.pkl"
    )

    random_forest_model = joblib.load(
        "models/random_forest_model.pkl"
    )

    return linear_model, random_forest_model


try:
    df = load_data()

    linear_model, random_forest_model = load_models()

except Exception as e:

    st.error(f"Error loading dataset or models: {e}")

    st.stop()


# -----------------------------------------
# MODEL SELECTION
# -----------------------------------------

st.subheader("Select Regression Model")

selected_model = st.selectbox(
    "Choose a model:",
    [
        "Linear Regression",
        "Random Forest Regression"
    ]
)


# -----------------------------------------
# FEATURE LISTS
# -----------------------------------------

linear_features = [
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
    "Hour",
    "Month",
    "Day"
]


rf_features = [
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH"
]


# -----------------------------------------
# INPUT FORM
# -----------------------------------------

st.divider()

st.subheader("Enter Air Quality Readings")

st.info(
    "Enter the sensor, pollutant and environmental "
    "readings below. Default values are taken from "
    "the dataset medians."
)


# Get median values for input defaults

def get_default(column):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    median_value = values.median()

    if pd.isna(median_value):
        return 0.0

    return float(median_value)


input_values = {}


# -----------------------------------------
# LINEAR REGRESSION INPUTS
# -----------------------------------------

if selected_model == "Linear Regression":

    st.write("### Linear Regression – CO(GT) Prediction")

    st.caption(
        "This model uses 15 input features to predict CO(GT)."
    )

    with st.form("linear_regression_form"):

        col1, col2 = st.columns(2)

        for i, feature in enumerate(linear_features):

            with [col1, col2][i % 2]:

                if feature == "Hour":

                    value = st.number_input(
                        "Hour",
                        min_value=0,
                        max_value=23,
                        value=12
                    )

                elif feature == "Month":

                    value = st.number_input(
                        "Month",
                        min_value=1,
                        max_value=12,
                        value=6
                    )

                elif feature == "Day":

                    value = st.number_input(
                        "Day",
                        min_value=1,
                        max_value=31,
                        value=15
                    )

                else:

                    value = st.number_input(
                        feature,
                        value=get_default(feature),
                        format="%.4f"
                    )

                input_values[feature] = value

        predict_button = st.form_submit_button(
            "Predict CO Concentration"
        )


# -----------------------------------------
# RANDOM FOREST INPUTS
# -----------------------------------------

else:

    st.write("### Random Forest – C6H6(GT) Prediction")

    st.caption(
        "This model uses 12 input features to predict C6H6(GT)."
    )

    with st.form("random_forest_form"):

        col1, col2 = st.columns(2)

        for i, feature in enumerate(rf_features):

            with [col1, col2][i % 2]:

                value = st.number_input(
                    feature,
                    value=get_default(feature),
                    format="%.4f"
                )

                input_values[feature] = value

        predict_button = st.form_submit_button(
            "Predict C6H6 Concentration"
        )


# -----------------------------------------
# MAKE PREDICTION
# -----------------------------------------

if predict_button:

    try:

        if selected_model == "Linear Regression":

            input_df = pd.DataFrame(
                [input_values],
                columns=linear_features
            )

            prediction = linear_model.predict(input_df)[0]

            target = "CO(GT)"

        else:

            input_df = pd.DataFrame(
                [input_values],
                columns=rf_features
            )

            prediction = random_forest_model.predict(input_df)[0]

            target = "C6H6(GT)"

        # ---------------------------------
        # DISPLAY RESULT
        # ---------------------------------

        st.divider()

        st.subheader("Prediction Result")

        st.success("Prediction Generated Successfully!")

        st.metric(
            label=f"Predicted {target} Concentration",
            value=f"{prediction:.4f}"
        )

        st.write("### Input Values Used")

        st.dataframe(
            input_df,
            width="stretch"
        )

        st.caption(
            "The prediction is expressed in the same units "
            "as the target variable in your training dataset."
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")


st.divider()

st.markdown("""
**About Regression:**

Regression is a supervised Machine Learning technique
used to predict continuous numerical values.

In this project, Linear Regression and Random Forest
Regression are used to predict air pollutant concentrations.
""")