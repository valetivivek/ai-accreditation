from __future__ import annotations
import streamlit as st
from core import topbar, try_load, show_df, availability_badge, DATA_DIR, APP_TITLE

st.set_page_config(page_title=f"{APP_TITLE} – Scores", layout="wide")

topbar()
st.header("Scores")

crit_df, crit_msg = try_load("criteria_catalog")
st.subheader("Criteria Catalog")
st.caption(crit_msg)
if crit_df is not None and not crit_df.empty:
    show_df(crit_df, "Names, themes, and gatekeeper fields.")
else:
    st.info("Add `criteria_catalog.csv` to `/data` to view.")

ops_df, ops_msg = try_load("operator_scores")
st.subheader("Operator Scores")
st.caption(ops_msg)
if ops_df is not None and not ops_df.empty:
    show_df(ops_df, "Raw scores by operator and criterion.")
else:
    st.info("Add `operator_scores_dummy.csv` to `/data` to view.")

st.sidebar.write("Data folder:", f"`{DATA_DIR}`")
availability_badge()
