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

.ri-kpi {min-height:155px!important; padding:19px 20px!important;}
.ri-kpi-bottom {font-size:.75rem; color:#9DAEC5; margin-top:12px;}
.ri-discovery {padding:25px 27px; background:linear-gradient(115deg,#24213A,#182336); border:1px solid #52456D; border-left:4px solid #6DD9B4; border-radius:16px; margin:8px 0 14px;}
.ri-discovery.negative {border-left-color:#FF8FA3;}
.ri-discovery-tags {display:flex; flex-wrap:wrap; gap:7px; margin-bottom:15px;}
.ri-discovery-tags span {font-size:.66rem; font-weight:750; letter-spacing:.07em; padding:5px 8px; border-radius:6px; color:#CDBCF8; background:#38304F;}
.ri-discovery-heading {font-size:1.2rem; font-weight:750; color:#FAFBFF; line-height:1.4;}
.ri-discovery-impact {font-size:2rem; font-weight:800; color:#70DFBA; margin:9px 0; letter-spacing:-.04em;}
.ri-discovery.negative .ri-discovery-impact {color:#FF91A8;}
.ri-discovery-impact small {font-size:.78rem; font-weight:500; color:#B6C4D7; letter-spacing:0;}
.ri-discovery-detail {font-size:.86rem; color:#C3D0E1; margin-bottom:12px;}
.ri-discovery-action {background:#332D44; color:#EADCB6; border-radius:9px; padding:12px 15px; font-size:.85rem;}
.ri-compact-discovery {display:flex; gap:12px; align-items:center; flex-wrap:wrap; padding:15px 18px; background:#1A2535; border:1px solid #304057; border-radius:11px; margin-bottom:9px;}
.ri-compact-discovery span {color:#F0F4FC; font-weight:650; flex:1; min-width:170px;}
.ri-compact-discovery strong {color:#C4AFF9;}
.ri-compact-discovery small {color:#9EADC3; font-size:.75rem;}
.ri-watch-sub {color:#A6B6CD; font-size:.87rem; margin-top:-6px; margin-bottom:15px;}
.ri-watch-caption {font-size:.75rem; color:#AAB9CD; margin-bottom:10px;}
.ri-news-direction {font-size:.75rem; font-weight:700; color:#C5B3FF; margin-bottom:6px;}
</style>
"""

def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)
