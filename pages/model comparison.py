
import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_theme import apply_page_theme, page_header
st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

apply_page_theme()

page_header('Machine Learning Model Comparison', 'Compare regression, classification and forecasting performance using the evaluation metrics from your experiments.', 'MODEL EVALUATION')
st.divider()

# -----------------------------------------
# HELPER FUNCTION FOR METRIC INPUTS
# -----------------------------------------

def metric_input(label, key, default=None):
    return st.number_input(
        label,
        value=default,
        placeholder="Enter actual result",
        format="%.4f",
        key=key
    )

def show_comparison_chart(data, metric, title):
    valid_data = data.dropna(subset=[metric])

    if not valid_data.empty:
        fig = px.bar(
            valid_data,
            x="Model",
            y=metric,
            title=title,
            text=metric
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )
    else:
        st.info("Enter the evaluation results to display this chart.")

# -----------------------------------------
# CREATE TABS
# -----------------------------------------

tab1, tab2, tab3 = st.tabs([
    "Regression Comparison",
    "Classification Comparison",
    "Forecasting Comparison"
])

# =========================================
# TAB 1: REGRESSION COMPARISON
# =========================================

with tab1:

    st.header("Regression Model Performance")

    st.write("""
    Compare Linear Regression and Random Forest
    Regression using the following evaluation metrics.
    """)

    st.caption("""
    Enter the results obtained from your regression
    model evaluation notebooks.
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Linear Regression")

        lr_mae = metric_input("MAE", "lr_mae")
        lr_mse = metric_input("MSE", "lr_mse")
        lr_rmse = metric_input("RMSE", "lr_rmse")
        lr_r2 = metric_input("R² Score", "lr_r2")

    with col2:

        st.subheader("Random Forest Regression")

        rf_mae = metric_input("MAE", "rf_mae")
        rf_mse = metric_input("MSE", "rf_mse")
        rf_rmse = metric_input("RMSE", "rf_rmse")
        rf_r2 = metric_input("R² Score", "rf_r2")

    regression_data = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest Regression"
        ],
        "MAE": [
            lr_mae if lr_mae is not None else float("nan"),
            rf_mae if rf_mae is not None else float("nan")
        ],
        "MSE": [
            lr_mse if lr_mse is not None else float("nan"),
            rf_mse if rf_mse is not None else float("nan")
        ],
        "RMSE": [
            lr_rmse if lr_rmse is not None else float("nan"),
            rf_rmse if rf_rmse is not None else float("nan")
        ],
        "R2 Score": [
            lr_r2 if lr_r2 is not None else float("nan"),
            rf_r2 if rf_r2 is not None else float("nan")
        ]
    })

    st.divider()

    st.subheader("Regression Results Table")

    st.dataframe(
        regression_data,
        hide_index=True,
        width="stretch"
    )

    st.subheader("Regression Metric Charts")

    metric_choice = st.selectbox(
        "Select metric to compare:",
        ["MAE", "MSE", "RMSE", "R2 Score"],
        key="regression_metric"
    )

    show_comparison_chart(
        regression_data,
        metric_choice,
        f"Regression Comparison - {metric_choice}"
    )

    st.caption(
        "For MAE, MSE and RMSE, lower values are generally "
        "better. For R², higher values are generally better."
    )

# =========================================
# TAB 2: CLASSIFICATION COMPARISON
# =========================================

with tab2:

    st.header("Classification Model Performance")

    st.write("""
    Compare Decision Tree and XGBoost Classifiers
    using classification evaluation metrics.
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Decision Tree")

        dt_accuracy = metric_input(
            "Accuracy (%)",
            "dt_accuracy",
            60.74
        )

        dt_precision = metric_input(
            "Precision",
            "dt_precision"
        )

        dt_recall = metric_input(
            "Recall",
            "dt_recall"
        )

        dt_f1 = metric_input(
            "F1 Score",
            "dt_f1"
        )

    with col2:

        st.subheader("XGBoost")

        xgb_accuracy = metric_input(
            "Accuracy (%)",
            "xgb_accuracy",
            62.0
        )

        xgb_precision = metric_input(
            "Precision",
            "xgb_precision"
        )

        xgb_recall = metric_input(
            "Recall",
            "xgb_recall"
        )

        xgb_f1 = metric_input(
            "F1 Score",
            "xgb_f1"
        )

    classification_data = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "XGBoost"
        ],
        "Accuracy (%)": [
            dt_accuracy,
            xgb_accuracy
        ],
        "Precision": [
            dt_precision if dt_precision is not None else float("nan"),
            xgb_precision if xgb_precision is not None else float("nan")
        ],
        "Recall": [
            dt_recall if dt_recall is not None else float("nan"),
            xgb_recall if xgb_recall is not None else float("nan")
        ],
        "F1 Score": [
            dt_f1 if dt_f1 is not None else float("nan"),
            xgb_f1 if xgb_f1 is not None else float("nan")
        ]
    })

    st.divider()

    st.subheader("Classification Results Table")

    st.dataframe(
        classification_data,
        hide_index=True,
        width="stretch"
    )

    st.subheader("Classification Accuracy Comparison")

    fig = px.bar(
        classification_data,
        x="Model",
        y="Accuracy (%)",
        title="Decision Tree vs XGBoost Accuracy",
        text="Accuracy (%)",
        range_y=[0, 100]
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.caption(
        "Accuracy values are your previously reported "
        "test results. Enter the actual precision, recall "
        "and F1 results from your evaluation notebooks."
    )

    st.info("""
    For multiclass classification, macro-averaged
    precision, recall and F1 score are useful for
    comparing performance across all four categories.
    """)

# =========================================
# TAB 3: FORECASTING COMPARISON
# =========================================

with tab3:

    st.header("Prophet Forecasting Performance")

    st.write("""
    Compare the forecasting performance of the three
    Prophet models trained for CO, NO2 and NOx.
    """)

    st.caption("""
    Enter the evaluation metrics calculated on your
    forecasting test data. Use the same evaluation
    period and metric definitions for all three models.
    """)

    pollutants = ["CO", "NO2", "NOx"]

    forecast_metrics = {}

    for pollutant in pollutants:

        with st.expander(
            f"{pollutant} Forecasting Results",
            expanded=True
        ):

            col1, col2, col3 = st.columns(3)

            with col1:
                mae = metric_input(
                    f"{pollutant} MAE",
                    f"{pollutant}_mae"
                )

            with col2:
                rmse = metric_input(
                    f"{pollutant} RMSE",
                    f"{pollutant}_rmse"
                )

            with col3:
                mape = metric_input(
                    f"{pollutant} MAPE (%)",
                    f"{pollutant}_mape"
                )

            forecast_metrics[pollutant] = {
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape
            }

    forecast_data = pd.DataFrame([
        {
            "Model": pollutant,
            "MAE": values["MAE"],
            "RMSE": values["RMSE"],
            "MAPE (%)": values["MAPE (%)"]
        }
        for pollutant, values in forecast_metrics.items()
    ])

    st.divider()

    st.subheader("Forecasting Results Table")

    st.dataframe(
        forecast_data,
        hide_index=True,
        width="stretch"
    )

    forecast_metric_choice = st.selectbox(
        "Select forecasting metric:",
        ["MAE", "RMSE", "MAPE (%)"],
        key="forecast_metric"
    )

    show_comparison_chart(
        forecast_data,
        forecast_metric_choice,
        f"Prophet Model Comparison - {forecast_metric_choice}"
    )

    st.caption(
        "Lower MAE, RMSE and MAPE generally indicate "
        "better forecasting performance. MAPE may be "
        "misleading when actual values are zero or near zero."
    )

# =========================================
# FINAL PROJECT SUMMARY
# =========================================

st.divider()

st.header("Overall Model Comparison")

st.markdown("""
This section compares the three different Machine
Learning approaches used in the project:

1. Regression – predicts numerical pollutant concentrations.
2. Classification – identifies pollution categories.
3. Forecasting – predicts future pollutant concentrations.
""")

st.success(
    "Model Comparison Page Loaded Successfully!"
)