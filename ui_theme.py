from __future__ import annotations
import streamlit as st


def apply_page_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
    .stApp {background:#F3F6F8;color:#17324D;}
    .block-container {padding-top:1.35rem;padding-bottom:2.4rem;max-width:1500px;}
    [data-testid="stSidebar"] {background:#17324D;}
    [data-testid="stSidebar"] * {color:#F4F8FB !important;}
    [data-testid="stMetric"] {background:#fff;border:1px solid #E5EBF0;border-radius:14px;padding:14px 17px;box-shadow:0 3px 12px rgba(23,50,77,.045);}
    [data-testid="stMetricLabel"] {color:#718096;font-weight:700;}
    [data-testid="stMetricValue"] {color:#17324D;font-weight:800;}
    div[data-testid="stForm"] {background:#fff;border:1px solid #E5EBF0;border-radius:16px;padding:20px;box-shadow:0 3px 12px rgba(23,50,77,.04);}
    div[data-testid="stExpander"] {background:#fff;border:1px solid #E5EBF0;border-radius:12px;}
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {background:#fff;border-radius:12px;}
    div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label, div[data-testid="stSlider"] label {font-weight:700;color:#17324D;}
    .stButton>button, .stFormSubmitButton>button {background:linear-gradient(110deg,#17324D,#159A91);color:white;border:0;border-radius:9px;padding:.55rem 1.1rem;font-weight:700;}
    .stButton>button:hover, .stFormSubmitButton>button:hover {color:white;border:0;filter:brightness(1.06);}
    button[data-baseweb="tab"] {font-weight:700;color:#526579;}
    button[data-baseweb="tab"][aria-selected="true"] {color:#159A91;}
    hr {border-color:#E3EAF0;}
    h1,h2,h3 {color:#17324D;}
    footer {visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str = "AIR QUALITY INTELLIGENCE"):
    st.markdown(f"""
    <div style="background:linear-gradient(115deg,#17324D 0%,#20516A 58%,#159A91 100%);padding:25px 30px;border-radius:18px;color:white;margin:0 0 20px 0;box-shadow:0 8px 24px rgba(23,50,77,.12);">
      <div style="font-size:10px;font-weight:800;letter-spacing:1.8px;text-transform:uppercase;color:#A8E3DA;margin-bottom:8px;">{eyebrow}</div>
      <div style="font-size:27px;font-weight:800;line-height:1.2;color:white;margin-bottom:8px;">{title}</div>
      <div style="font-size:13px;line-height:1.55;color:#E1F0F1;max-width:900px;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)
