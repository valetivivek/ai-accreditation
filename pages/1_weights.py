from __future__ import annotations
import streamlit as st
from core import topbar, try_load, show_df, availability_badge, DATA_DIR, APP_TITLE

st.set_page_config(page_title=f"{APP_TITLE} – Weights", layout="wide")

topbar()
st.header("Weights")
st.write("View raw Delphi inputs and any pre-computed weights (if present).")

delphi_df, delphi_msg = try_load("delphi_round1")
st.subheader("Delphi Round 1 (expert importance 1–9)")
st.caption(delphi_msg)
if delphi_df is not None and not delphi_df.empty:
    show_df(delphi_df, "Raw expert inputs that we’ll turn into weights in Week-2.")
else:
    st.info("Add `delphi_round1_example.csv` to `/data` to view.")

st.subheader("Precomputed Weights (if provided)")
weights_df, weights_msg = try_load("weights_from_delphi")
st.caption(weights_msg)
if weights_df is not None and not weights_df.empty:
    show_df(weights_df, "Weights should sum to 1. Read-only here.")
else:
    st.info("Add `weights_from_delphi_example.csv` to `/data` to view.")

st.sidebar.write("Data folder:", f"`{DATA_DIR}`")
availability_badge()
