"""Shared premium charcoal / navy visual language."""
import streamlit as st

CSS = """
<style>
:root {color-scheme: dark;}
.stApp {background: radial-gradient(ellipse at 72% -20%, #202047 0%, #0B1120 45%, #0A101B 100%); color:#F3F5FA;}
.block-container {max-width:1480px; padding-top:2.4rem; padding-bottom:5rem;}
[data-testid="stSidebar"] {background:#101827; border-right:1px solid #29344A;}
[data-testid="stSidebar"] .block-container {padding-top:1.6rem;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:5px;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:10px 11px; border-radius:10px; transition:background .15s;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#202B40;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#302849; outline:1px solid #7055AD;}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-weight:550; font-size:.88rem;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {display:none;}
h1,h2,h3 {letter-spacing:-.035em;}
h1 {font-weight:760!important;}
h2,h3 {color:#F1F4FC!important;}
[data-testid="stMetric"] {background:#161F30; border:1px solid #2C3950; padding:18px 20px; border-radius:14px;}
[data-testid="stMetricValue"] {font-size:1.7rem;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#2D3850!important; border-radius:14px!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:8px; overflow-x:auto;}
[data-testid="stTabs"] button {border-radius:9px 9px 0 0;}
[data-testid="stDataFrame"] {border-radius:12px; overflow:hidden;}
hr {border-color:#2D3950;}
.ri-eyebrow {font-size:.72rem; font-weight:750; color:#B9A0FF; letter-spacing:.14em; text-transform:uppercase; margin-bottom:10px;}
.ri-subtitle {color:#9CAAC1; font-size:.95rem; margin-top:-5px; margin-bottom:22px;}
.ri-section {font-size:1.17rem; font-weight:730; color:#F4F5FC; letter-spacing:-.025em; margin:18px 0 14px;}
.ri-kpi {min-height:180px; background:linear-gradient(145deg,#1B2638,#151E2E); border:1px solid #303E54; border-radius:16px; padding:21px 20px;}
.ri-kpi-label {font-size:.81rem; color:#A8B5CC; font-weight:600; margin-bottom:10px;}
.ri-kpi-value {font-size:clamp(1.45rem,2vw,2.15rem); font-weight:770; color:#F8FAFF; letter-spacing:-.04em; line-height:1.1;}
.ri-chip {display:inline-block; border-radius:7px; padding:4px 8px; font-size:.76rem; font-weight:700; margin-top:11px;}
.ri-chip.pos {background:#113D35; color:#5DE0BB;}
.ri-chip.neg {background:#482634; color:#FF9AAF;}
.ri-meta {font-size:.72rem; color:#96A5BD; margin-top:10px; line-height:1.65;}
.ri-brief {background:linear-gradient(110deg,#24233D,#171F33 85%); border:1px solid #52446F; border-left:4px solid #A48AF3; padding:23px 25px; border-radius:15px;}
.ri-brief-title {color:#F5F0FF; font-size:1.05rem; font-weight:740; margin-bottom:13px;}
.ri-brief-body {color:#E0E5F1; font-size:.94rem; line-height:1.75;}
.ri-finding {background:#171F30; border:1px solid #2E3B51; border-radius:12px; padding:16px 18px; margin-bottom:10px;}
.ri-finding-title {font-size:.88rem; color:#F2F5FB; font-weight:700;}
.ri-finding-meta {font-size:.77rem; color:#A4B2C9; margin-top:8px;}
.ri-brand {font-size:1.34rem; font-weight:800; color:#F3F5FB; letter-spacing:-.04em;}
.ri-brand span {color:#B49AFF;}
.ri-sidebar-caption {color:#8393AC; font-size:.72rem; letter-spacing:.12em; margin:8px 0 18px;}
.ri-footnote {font-size:.73rem; color:#8291AA; margin-top:12px;}
</style>
"""

def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)
