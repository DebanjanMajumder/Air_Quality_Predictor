import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Air Quality Intelligence Dashboard", page_icon="🌿", layout="wide")

# ----------------------------- Theme -----------------------------------------
BG = "#F3F6F8"
NAVY = "#17324D"
TEAL = "#159A91"
MUTED = "#718096"
POLLUTANTS = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "NMHC(GT)"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#F3F6F8; color:#17324D;}
.block-container {padding-top:1.45rem; padding-bottom:2.5rem; max-width:1500px;}
[data-testid="stSidebar"] {background:#17324D;}
[data-testid="stSidebar"] * {color:#F4F8FB !important;}
.hero {background:linear-gradient(115deg,#17324D 0%,#20516A 58%,#159A91 100%);padding:28px 32px;border-radius:18px;color:white;margin-bottom:18px;box-shadow:0 8px 24px rgba(23,50,77,.12);}
.hero h1 {font-size:29px;font-weight:800;margin:0 0 7px 0;color:white;letter-spacing:-.5px;}
.hero p {font-size:14px;margin:0;color:#E1F0F1;max-width:850px;}
.eyebrow {font-size:11px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#A8E3DA;margin-bottom:8px;}
.section-title {font-size:18px;font-weight:800;color:#17324D;margin:12px 0 2px 0;}
.section-sub {font-size:12px;color:#718096;margin-bottom:10px;}
.kpi {background:white;border:1px solid #E5EBF0;border-radius:15px;padding:17px 19px;min-height:108px;box-shadow:0 3px 12px rgba(23,50,77,.045);}
.kpi-label {font-size:11px;color:#718096;font-weight:700;text-transform:uppercase;letter-spacing:.8px;}
.kpi-value {font-size:27px;font-weight:800;color:#17324D;margin-top:7px;line-height:1.1;}
.kpi-note {font-size:11px;color:#159A91;margin-top:6px;}
.panel {background:white;border:1px solid #E5EBF0;border-radius:15px;padding:14px 16px;box-shadow:0 3px 12px rgba(23,50,77,.035);}
div[data-testid="stSelectbox"] label, div[data-testid="stDateInput"] label {font-weight:700;color:#17324D;}
hr {border-color:#E3EAF0;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    path = Path("data/Cleaned_Air_Quality.xlsx")
    data = pd.read_excel(path)
    # The source dataset uses -200 as its missing-value marker.
    data = data.replace(-200, np.nan)
    time_values = data["Time"].astype(str).str.replace(".", ":", regex=False) if "Time" in data else "00:00:00"
    if "Date" in data:
        data["DateTime"] = pd.to_datetime(data["Date"].astype(str) + " " + time_values, dayfirst=True, errors="coerce")
        data["Year"] = data["DateTime"].dt.year
    return data

try:
    df = load_data()
except Exception as exc:
    st.error(f"Unable to load the dataset. Please check that data/Cleaned_Air_Quality.xlsx exists. Details: {exc}")
    st.stop()

# ----------------------------- Sidebar filters -------------------------------
st.sidebar.markdown("## 🌿 Air Quality\n### Intelligence Center")
st.sidebar.caption("DATA EXPLORATION · PREDICTION · FORECASTING")
st.sidebar.markdown("---")
valid_years = sorted(df["Year"].dropna().astype(int).unique().tolist()) if "Year" in df else []
selected_years = st.sidebar.multiselect("Analysis year", valid_years, default=valid_years)
selected_pollutant = st.sidebar.selectbox("Primary pollutant", [p for p in POLLUTANTS if p in df.columns], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Use the navigation menu above to explore dataset analysis, regression, classification, model comparison and forecasting.")

filtered = df.copy()
if selected_years and "Year" in filtered:
    filtered = filtered[filtered["Year"].isin(selected_years)]

# ----------------------------- Header ----------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Environmental data · Interactive overview</div>
  <h1>Air Quality Indexing & Prediction Dashboard</h1>
  <p>Explore pollutant measurements, sensor readings and atmospheric conditions across the monitoring period. Use the filters to focus the dashboard on a selected year and pollutant.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- KPI row ---------------------------------------
valid_dates = filtered["DateTime"].dropna() if "DateTime" in filtered else pd.Series(dtype="datetime64[ns]")
mean_val = filtered[selected_pollutant].mean() if selected_pollutant in filtered else np.nan
mean_pm = filtered["C6H6(GT)"].mean() if "C6H6(GT)" in filtered else np.nan
mean_no2 = filtered["NO2(GT)"].mean() if "NO2(GT)" in filtered else np.nan
coverage = f"{valid_dates.min():%b %Y} – {valid_dates.max():%b %Y}" if not valid_dates.empty else "Date unavailable"

def kpi(label, value, note):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
with c1: kpi("Records in view", f"{len(filtered):,}", f"of {len(df):,} total observations")
with c2: kpi(f"Average {selected_pollutant}", f"{mean_val:,.1f}" if pd.notna(mean_val) else "—", "Mean valid measurement")
with c3: kpi("Average Benzene", f"{mean_pm:,.1f}" if pd.notna(mean_pm) else "—", "C6H6(GT) · valid measurements")
with c4: kpi("Average Nitrogen Dioxide", f"{mean_no2:,.1f}" if pd.notna(mean_no2) else "—", "NO2(GT) · valid measurements")

st.markdown('<div class="section-title">Monitoring overview</div><div class="section-sub">A visual summary of the selected observations</div>', unsafe_allow_html=True)

# Chart data: aggregate large hourly dataset to monthly averages for legibility.
chart_df = filtered.dropna(subset=["DateTime", selected_pollutant]).copy() if "DateTime" in filtered else pd.DataFrame()
if not chart_df.empty:
    chart_df["Month"] = chart_df["DateTime"].dt.to_period("M").dt.to_timestamp()
    monthly = chart_df.groupby("Month", as_index=False)[selected_pollutant].mean()
else:
    monthly = pd.DataFrame(columns=["Month", selected_pollutant])

left, right = st.columns([1.55,1])
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Pollutant trend over time**")
    if not monthly.empty:
        fig = px.area(monthly, x="Month", y=selected_pollutant, markers=True)
        fig.update_traces(line_color=TEAL, fillcolor="rgba(21,154,145,.13)", marker=dict(size=5))
        fig.update_layout(height=315, margin=dict(l=10,r=10,t=15,b=10), paper_bgcolor="white", plot_bgcolor="white", font_color=NAVY, xaxis_title="", yaxis_title="Average concentration", hovermode="x unified")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#EAF0F4", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No valid time-series measurements for this selection.")
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Pollutant profile**")
    available = [p for p in POLLUTANTS if p in filtered.columns]
    profile = pd.DataFrame({"Pollutant": available, "Average": [filtered[p].mean() for p in available]}).dropna()
    if not profile.empty:
        fig = px.bar(profile.sort_values("Average"), x="Average", y="Pollutant", orientation="h", text_auto=".1f")
        fig.update_traces(marker_color=TEAL, textposition="outside", cliponaxis=False)
        fig.update_layout(height=315, margin=dict(l=5,r=35,t=15,b=10), paper_bgcolor="white", plot_bgcolor="white", font_color=NAVY, xaxis_title="Mean measurement", yaxis_title="")
        fig.update_xaxes(showgrid=True, gridcolor="#EAF0F4")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No pollutant measurements available.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Pollution & atmospheric patterns</div><div class="section-sub">Compare pollutant behaviour and the conditions recorded by the monitoring station</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Selected pollutant distribution**")
    vals = filtered[selected_pollutant].dropna() if selected_pollutant in filtered else pd.Series(dtype=float)
    if not vals.empty:
        fig = px.histogram(x=vals, nbins=35, labels={"x": selected_pollutant, "y":"Observation count"})
        fig.update_traces(marker_color="#3977A8", marker_line_color="white", marker_line_width=.5)
        fig.update_layout(height=290, margin=dict(l=10,r=10,t=15,b=10), paper_bgcolor="white", plot_bgcolor="white", font_color=NAVY, bargap=.08)
        fig.update_yaxes(gridcolor="#EAF0F4")
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No values available for this pollutant.")
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Temperature and humidity relationship**")
    climate = filtered[[c for c in ["T", "RH"] if c in filtered.columns]].dropna()
    if len(climate) and {"T","RH"}.issubset(climate.columns):
        fig = px.scatter(climate.sample(min(1800,len(climate)), random_state=7), x="T", y="RH", opacity=.48, labels={"T":"Temperature (°C)","RH":"Relative humidity (%)"})
        fig.update_traces(marker=dict(color=TEAL, size=5))
        fig.update_layout(height=290, margin=dict(l=10,r=10,t=15,b=10), paper_bgcolor="white", plot_bgcolor="white", font_color=NAVY)
        fig.update_xaxes(gridcolor="#EAF0F4"); fig.update_yaxes(gridcolor="#EAF0F4")
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Temperature and humidity fields are not both available.")
    st.markdown('</div>', unsafe_allow_html=True)

# Dataset snapshot and project summary
with st.expander("View dataset snapshot", expanded=False):
    st.caption("Showing the first 12 rows after applying the selected year filter. Missing source values are displayed as blank.")
    st.dataframe(filtered.drop(columns=["Year"], errors="ignore").head(12), use_container_width=True, hide_index=True)

st.markdown("""
<div style="background:#17324D;border-radius:15px;padding:19px 23px;margin-top:18px;color:#E8F1F5;">
 <div style="font-weight:800;font-size:15px;color:white;margin-bottom:5px;">About this project</div>
 <div style="font-size:12px;line-height:1.7;">This application combines exploratory data analysis, regression, pollution classification, model comparison and time-series forecasting. The supplied dataset contains pollutant concentrations, sensor responses, date/time and atmospheric measurements. It does not contain state, city, AQI or AQI-bucket fields, so this dashboard visualises the available measurements rather than presenting unsupported AQI or geographic values.</div>
 <div style="font-size:11px;color:#A8E3DA;margin-top:9px;">PYTHON · PANDAS · SCIKIT-LEARN · XGBOOST · PROPHET · STREAMLIT</div>
</div>
""", unsafe_allow_html=True)
