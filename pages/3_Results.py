from helpers import topbar, try_load, show_df
import streamlit as st
topbar()

st.header("Results (placeholder for Week-3 to Week-5)")
st.write(
    "This page will show Gatekeeper pass/fail reasons, ARAS utility K, and the final tier "
    "(Platinum/Gold/Silver/Bronze) once implemented in later weeks."
)
st.info("For Week-1, this is only a placeholder view.")

col1, col2 = st.columns(2)
weights_df, _ = try_load("weights_from_delphi")
ops_df, _ = try_load("operator_scores")

with col1:
    st.metric("Weights available", "Yes" if (weights_df is not None and not weights_df.empty) else "No")
with col2:
    if ops_df is not None and not ops_df.empty and "operator_id" in ops_df:
        st.metric("Operator scores", f"{ops_df['operator_id'].nunique()} operators")
    else:
        st.metric("Operator scores", "Missing")

st.divider()
st.write("**Preview (non-final):** If both weights and scores exist, we’ll show a merged glance.")
if (weights_df is not None and not weights_df.empty) and (ops_df is not None and not ops_df.empty):
    preview = ops_df.merge(weights_df, on="criterion_id", how="left")
    show_df(preview.head(50), "Joined view (no normalization, no gates, no K yet).")
else:
    st.info("Provide both operator scores and weights to see a quick joined preview.")
