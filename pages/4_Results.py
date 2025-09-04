from __future__ import annotations
import streamlit as st
from core import topbar, try_load, show_df, availability_badge, DATA_DIR, APP_TITLE

st.set_page_config(page_title=f"{APP_TITLE} – Results", layout="wide")

# simple modal gate
if "results_gate_ok" not in st.session_state:
    st.session_state.results_gate_ok = False

def gate():
    try:
        with st.dialog("Week-3 placeholder"):
            st.write("This page will show Gatekeepers, ARAS K, and final tiers (Week-3+).")
            if st.button("OK, enter", type="primary"):
                st.session_state.results_gate_ok = True
                st.rerun()
    except Exception:
        st.warning("This is a placeholder for Week-3+.")
        if st.button("OK, enter"):
            st.session_state.results_gate_ok = True
            st.rerun()
    st.stop()

topbar()
st.header("Results (placeholder for Week-3+)")
st.write("Preview join only; no normalization/gates/K yet.")

if not st.session_state.results_gate_ok:
    gate()

weights_df, _ = try_load("weights_from_delphi")
ops_df, _ = try_load("operator_scores")

if (weights_df is not None and not weights_df.empty) and (ops_df is not None and not ops_df.empty):
    if "criterion_id" in ops_df.columns and "criterion_id" in weights_df.columns:
        preview = ops_df.merge(weights_df, on="criterion_id", how="left")
        show_df(preview.head(200), "Joined preview")
    else:
        st.info("`criterion_id` missing in one of the tables.")
else:
    st.info("Provide both operator scores and weights to see a joined preview.")

st.sidebar.write("Data folder:", f"`{DATA_DIR}`")
availability_badge()
