from helpers import topbar, try_load, show_df
import streamlit as st
topbar()

st.header("Weights")
st.write("Displays expert inputs (Delphi) and any precomputed weights (if present).")

delphi_df, delphi_msg = try_load("delphi_round1")
st.subheader("Delphi Round 1 (expert importance 1–9)")
st.caption(delphi_msg)
if delphi_df is not None and not delphi_df.empty:
    show_df(delphi_df, "Raw expert inputs used to derive weights (Week-2 will compute).")
else:
    st.info("Add `delphi_round1_example.csv` to /data to view.")

st.subheader("Precomputed Weights (if provided)")
weights_df, weights_msg = try_load("weights_from_delphi")
st.caption(weights_msg)
if weights_df is not None and not weights_df.empty:
    show_df(weights_df, "Weights should sum to 1. These are read-only in Week-1.")
    with st.expander("Quick sanity checks"):
        s = weights_df.select_dtypes(include="number").sum(numeric_only=True)
        st.write("Column sums:", dict(s))
else:
    st.info("Add `weights_from_delphi_example.csv` to /data to view precomputed weights.")
