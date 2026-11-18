import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_theme import apply_page_theme, page_header
st.set_page_config(
    page_title="Dataset Analysis",
    page_icon="📊",
    layout="wide"
)

apply_page_theme()

page_header('Air Quality Dataset Analysis', 'Explore the cleaned monitoring dataset through pollutant trends, statistical summaries and interactive visualisations.', 'DATA EXPLORATION')
st.divider()


# --------------------------------
# LOAD DATASET
# --------------------------------

@st.cache_data
def load_data():
    df = pd.read_excel("data/Cleaned_Air_Quality.xlsx")

    # Create a DateTime column if Date and Time exist
    if "Date" in df.columns and "Time" in df.columns:

        time_values = df["Time"].astype(str).str.replace(
            ".", ":", regex=False
        )

        df["DateTime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + time_values,
            dayfirst=True,
            errors="coerce"
        )
    return df
df = load_data()


# --------------------------------
# DATASET OVERVIEW
# --------------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{len(df):,}")
col2.metric("Total Columns", len(df.columns))
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.divider()


# --------------------------------
# DATASET PREVIEW
# --------------------------------

st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    width="stretch",
)


with st.expander("View Statistical Summary"):

    st.write(df.describe().T)


st.divider()


# --------------------------------
# POLLUTANT SELECTION
# --------------------------------

st.subheader("Pollutant Analysis")

pollutants = [
    "CO(GT)",
    "NO2(GT)",
    "NOx(GT)",
    "C6H6(GT)"
]

available_pollutants = [
    col for col in pollutants if col in df.columns
]

selected_pollutant = st.selectbox(
    "Select a pollutant:",
    available_pollutants
)


# --------------------------------
# POLLUTANT TREND GRAPH
# --------------------------------

st.subheader("Pollutant Concentration Trend")

if "DateTime" in df.columns:

    plot_df = df[
        ["DateTime", selected_pollutant]
    ].dropna()

    fig = px.line(
        plot_df,
        x="DateTime",
        y=selected_pollutant,
        title=f"{selected_pollutant} Concentration Over Time",
        labels={
            "DateTime": "Date and Time",
            selected_pollutant: "Concentration"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.warning("DateTime information is unavailable.")


st.divider()


# --------------------------------
# POLLUTANT DISTRIBUTION
# --------------------------------

st.subheader("Pollutant Distribution")

fig = px.histogram(
    df,
    x=selected_pollutant,
    nbins=40,
    title=f"Distribution of {selected_pollutant}",
    labels={
        selected_pollutant: "Concentration"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.divider()


# --------------------------------
# CORRELATION HEATMAP
# --------------------------------

st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(
    include="number"
)

correlation = numeric_df.corr()

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

ax.set_title("Correlation Between Air Quality Features")

st.pyplot(fig)

plt.close(fig)


st.divider()


# --------------------------------
# POLLUTANT COMPARISON
# --------------------------------

st.subheader("Pollutant Comparison")

selected_pollutants = st.multiselect(
    "Choose pollutants to compare:",
    available_pollutants,
    default=available_pollutants
)

if selected_pollutants:

    comparison_df = df[selected_pollutants].dropna()

    fig = px.line(
        comparison_df,
        title="Comparison of Pollutant Concentrations"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
else:
    st.info("Select at least one pollutant to display the graph.")
st.divider()
st.success("Dataset Analysis Completed!")