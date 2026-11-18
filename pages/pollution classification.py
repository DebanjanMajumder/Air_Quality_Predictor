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
    page_title="Pollution Classification",
    page_icon="🌫️",
    layout="wide"
)

apply_page_theme()

page_header('Air Quality Pollution Classification', 'Use trained classification models to estimate the pollution category from sensor and environmental readings.', 'POLLUTION CLASSIFICATION')
st.divider()


# -----------------------------------------
# LOAD DATASET
# -----------------------------------------

@st.cache_data
def load_data():

    return pd.read_excel(
        "data/Cleaned_Air_Quality.xlsx"
    )


# -----------------------------------------
# LOAD TRAINED MODELS
# -----------------------------------------

@st.cache_resource
def load_models():

    decision_tree = joblib.load(
        "models/decision_tree_model.pkl"
    )

    xgb_model = joblib.load(
        "models/xgb_model.pkl"
    )

    return decision_tree, xgb_model


try:

    df = load_data()

    decision_tree, xgb_model = load_models()

except Exception as e:

    st.error(f"Error loading files: {e}")

    st.stop()


# -----------------------------------------
# FEATURE LIST
# -----------------------------------------

features = [
    "PT08.S1(CO)",
    "NMHC(GT)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH"
]


# -----------------------------------------
# POLLUTION CATEGORY MAPPING
# -----------------------------------------

category_mapping = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Very High"
}


# -----------------------------------------
# MODEL SELECTION
# -----------------------------------------

st.subheader("Select Classification Model")

selected_model = st.selectbox(
    "Choose a classification algorithm:",
    [
        "Decision Tree Classifier",
        "XGBoost Classifier"
    ]
)

st.divider()


# -----------------------------------------
# MODEL INFORMATION
# -----------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Decision Tree Accuracy",
        "60.74%"
    )

with col2:

    st.metric(
        "XGBoost Accuracy",
        "62%"
    )

st.caption(
    "Accuracy values are from the previously evaluated "
    "test results of your trained models."
)

st.divider()


# -----------------------------------------
# INPUT FORM
# -----------------------------------------

st.subheader("Enter Air Quality Readings")

st.info(
    "Enter the nine sensor and environmental readings "
    "to classify the pollution level."
)


# Function to obtain default values

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


with st.form("classification_form"):

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        with [col1, col2][i % 2]:

            value = st.number_input(
                feature,
                value=get_default(feature),
                format="%.4f"
            )

            input_values[feature] = value

    predict_button = st.form_submit_button(
        "Predict Pollution Category"
    )


# -----------------------------------------
# MAKE PREDICTION
# -----------------------------------------

if predict_button:

    try:

        input_df = pd.DataFrame(
            [input_values],
            columns=features
        )

        # Select model

        if selected_model == "Decision Tree Classifier":

            model = decision_tree

        else:

            model = xgb_model

        # Generate prediction

        prediction = model.predict(input_df)[0]

        # Convert XGBoost integer output into category

        if selected_model == "XGBoost Classifier":

            prediction = category_mapping[int(prediction)]

        else:

            prediction = str(prediction)

        # ---------------------------------
        # DISPLAY RESULT
        # ---------------------------------

        st.divider()

        st.subheader("Classification Result")

        st.success(
            "Pollution Category Predicted Successfully!"
        )

        # Category display

        if prediction == "Low":

            st.success("🟢 LOW POLLUTION")

        elif prediction == "Moderate":

            st.info("🟡 MODERATE POLLUTION")

        elif prediction == "High":

            st.warning("🟠 HIGH POLLUTION")

        else:

            st.error("🔴 VERY HIGH POLLUTION")

        st.metric(
            "Predicted Pollution Category",
            prediction
        )

        # ---------------------------------
        # PREDICTION PROBABILITY
        # ---------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_df
            )[0]

            # Get the classes in the model's actual order
            classes = model.classes_

            probability_data = []

            for class_value, probability in zip(
                classes, probabilities
            ):

                if selected_model == "XGBoost Classifier":

                    category = category_mapping[int(class_value)]

                else:

                    category = str(class_value)

                probability_data.append({
                    "Pollution Category": category,
                    "Probability (%)": round(
                        probability * 100, 2
                    )
                })

            probability_df = pd.DataFrame(
                probability_data
            )

            st.subheader("Prediction Probability")

            st.bar_chart(
                probability_df.set_index(
                    "Pollution Category"
                )
            )

            st.dataframe(
                probability_df,
                width="stretch"
            )

        # ---------------------------------
        # INPUT VALUES
        # ---------------------------------

        st.subheader("Input Values Used")

        st.dataframe(
            input_df,
            width="stretch"
        )

    except Exception as e:

        st.error(
            f"Classification failed: {e}"
        )


st.divider()


# -----------------------------------------
# CATEGORY EXPLANATION
# -----------------------------------------

st.subheader("Pollution Categories")

category_df = pd.DataFrame({

    "Category": [
        "Low",
        "Moderate",
        "High",
        "Very High"
    ],

    "Description": [
        "Relatively low pollution score",
        "Moderate pollution score",
        "High pollution score",
        "Very high pollution score"
    ]

})

st.table(category_df)


st.info(
    "Note: These are custom pollution categories "
    "created for this project using the combined "
    "pollution classification method. They are not "
    "official AQI categories."
)

st.success("Classification Page Loaded Successfully!")