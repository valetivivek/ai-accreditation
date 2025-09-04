from helpers import topbar, try_load, show_df
import streamlit as st
topbar()

st.header("Scores")
st.write("Displays operator inputs per criterion (raw scores).")

crit_df, crit_msg = try_load("criteria_catalog")
st.subheader("Criteria Catalog (type + gate min/max)")
st.caption(crit_msg)
if crit_df is not None and not crit_df.empty:
    show_df(crit_df, "Criteria, themes, and gatekeeper fields (used in later weeks).")
else:
    st.info("Add `criteria_catalog.csv` to /data to view.")

st.subheader("Operator Scores")
ops_df, ops_msg = try_load("operator_scores")
st.caption(ops_msg)
if ops_df is not None and not ops_df.empty:
    show_df(ops_df, "Raw scores by operator and criterion.")
else:
    st.info("Add `operator_scores_dummy.csv` to /data to view.")
