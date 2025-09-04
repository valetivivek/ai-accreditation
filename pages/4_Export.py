from helpers import topbar, try_load
import pandas as pd
import io
import streamlit as st
topbar()

st.header("Export")
st.write("Download a snapshot CSV of what’s currently loaded (Week-1 demo).")

parts = []
for key in ["criteria_catalog", "delphi_round1", "weights_from_delphi", "operator_scores"]:
    df, msg = try_load(key)
    if df is not None and not df.empty:
        df = df.copy()
        df.insert(0, "_table", key)
        parts.append(df)

if parts:
    combined = pd.concat(parts, ignore_index=True)
    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    st.download_button(
        label="Download snapshot CSV",
        data=buf.getvalue().encode("utf-8"),
        file_name="week1_snapshot.csv",
        mime="text/csv"
    )
    st.caption("Exports a simple concatenation with a `_table` column for provenance.")
else:
    st.info("No data loaded yet—add CSVs to `/data` and revisit.")
