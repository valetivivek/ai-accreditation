from __future__ import annotations
import io
import streamlit as st
import pandas as pd
from core import topbar, try_load, availability_badge, DATA_DIR, APP_TITLE

st.set_page_config(page_title=f"{APP_TITLE} – Export", layout="wide")

topbar()
st.header("Export snapshot")

parts = []
for key in ["criteria_catalog", "delphi_round1", "weights_from_delphi", "operator_scores"]:
    df, _ = try_load(key)
    if df is not None and not df.empty:
        temp = df.copy()
        temp.insert(0, "_table", key)
        parts.append(temp)

if parts:
    combined = pd.concat(parts, ignore_index=True)
    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download snapshot CSV",
        data=buf.getvalue().encode("utf-8"),
        file_name="week1_snapshot.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl-snapshot",
    )
else:
    st.info("Nothing to export yet — add CSVs to `/data` and try again.")

st.sidebar.write("Data folder:", f"`{DATA_DIR}`")
availability_badge()
