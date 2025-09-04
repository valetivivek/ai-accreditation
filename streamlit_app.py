from __future__ import annotations
import io
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

APP_TITLE = "AI Accreditation – Week 1"
DATA_DIR = Path(__file__).parent / "data"

EXPECTED_FILES = {
    "criteria_catalog": "criteria_catalog.csv",
    "delphi_round1": "delphi_round1_example.csv",
    "weights_from_delphi": "weights_from_delphi_example.csv",
    "operator_scores": "operator_scores_dummy.csv",
}

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def try_load(name_key: str) -> tuple[pd.DataFrame | None, str]:
    filename = EXPECTED_FILES[name_key]
    fpath = DATA_DIR / filename
    if fpath.exists():
        try:
            df = load_csv(fpath)
            return df, f"Loaded {filename} ({len(df):,} rows)"
        except Exception as e:
            return None, f"Error reading {filename}: {e}"
    return None, f"Missing {filename}"

def show_df(df: pd.DataFrame, caption: str):
    st.caption(caption)
    st.dataframe(df, use_container_width=True, height=420)

def availability_badge():
    missing = []
    for k, fn in EXPECTED_FILES.items():
        if not (DATA_DIR / fn).exists():
            missing.append(fn)
    if missing:
        st.warning("Missing files: " + ", ".join(missing))
    else:
        st.success("All expected CSVs found in /data")

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    options=["Overview", "Weights", "Scores", "Results", "Export"],
    index=0
)

st.sidebar.divider()
st.sidebar.write("Data folder:", f"`{DATA_DIR}`")
availability_badge()

# ---------- Overview ----------
if page == "Overview":
    st.title(APP_TITLE)
    st.markdown(
        """
**Week-1 goals**  
- Confirm project structure and `/data` files exist  
- Load and display CSVs on dedicated pages  
- Prepare placeholders for later math (Gatekeepers → ARAS → Tiers)  

        """
    )

# ---------- Weights ----------
elif page == "Weights":
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

# ---------- Scores ----------
elif page == "Scores":
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

# ---------- Results ----------
elif page == "Results":
    st.header("Results (placeholder for Week-3 to Week-5)")
    st.write(
        "This page will show Gatekeeper pass/fail reasons, ARAS utility K, and the final tier "
        "(Platinum/Gold/Silver/Bronze) once implemented in later weeks."
    )
    st.info("For Week-1, this is only a placeholder view.")
    col1, col2 = st.columns(2)
    with col1:
        weights_df, _ = try_load("weights_from_delphi")
        if weights_df is not None and not weights_df.empty:
            st.metric("Weights available", "Yes")
        else:
            st.metric("Weights available", "No")

    with col2:
        ops_df, _ = try_load("operator_scores")
        if ops_df is not None and not ops_df.empty:
            st.metric("Operator scores", f"{ops_df['operator_id'].nunique()} operators")
        else:
            st.metric("Operator scores", "Missing")

    st.divider()
    st.write("**Preview (non-final):** If both weights and scores exist, we’ll show a merged glance.")
    weights_df, _ = try_load("weights_from_delphi")
    ops_df, _ = try_load("operator_scores")
    if weights_df is not None and ops_df is not None and not weights_df.empty and not ops_df.empty:
        preview = ops_df.merge(weights_df, on="criterion_id", how="left")
        show_df(preview.head(50), "Joined view (no normalization, no gates, no K yet).")
    else:
        st.info("Provide both operator scores and weights to see a quick joined preview.")

# ---------- Export ----------
elif page == "Export":
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
