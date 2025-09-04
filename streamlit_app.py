from __future__ import annotations
import io
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# =========================
# App basics
# =========================
APP_TITLE = "AI Accreditation – Week 1"
DATA_DIR = Path(__file__).parent / "data"

# We expect these files in /data. If a file is missing,
# the app will tell you exactly which one.
EXPECTED_FILES = {
    "criteria_catalog": "criteria_catalog.csv",
    "delphi_round1": "delphi_round1_example.csv",
    "weights_from_delphi": "weights_from_delphi_example.csv",
    "operator_scores": "operator_scores_dummy.csv",
}

st.set_page_config(page_title=APP_TITLE, layout="wide")


# =========================
# Helpers
# =========================
def _file_path(name_key: str) -> Path:
    """Return the absolute path of a known file key (e.g., 'operator_scores')."""
    return DATA_DIR / EXPECTED_FILES[name_key]

def _file_mtime(path: Path) -> float:
    """Safe 'last modified' time for a file. Returns -1.0 if missing."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return -1.0  # sentinel for "missing"

@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    """
    Read a CSV with caching.
    - The 'mtime' argument is part of the cache key, so if the file changes on disk,
      the cache auto-refreshes. No need to restart Streamlit.
    - Uses robust parsing for large/wide files.
    """
    path = Path(path_str)
    try:
        # Pandas 2.x: 'dtype_backend="pyarrow"' lowers memory and can speed up some ops.
        return pd.read_csv(path, low_memory=False, dtype_backend="pyarrow")
    except TypeError:
        # Older pandas versions don’t support dtype_backend—fall back gracefully.
        return pd.read_csv(path, low_memory=False)

def try_load(name_key: str) -> tuple[pd.DataFrame | None, str]:
    """
    Try to load a known CSV by key.
    Returns (df, message). If loading fails or file is missing, df is None
    and the message explains what happened.
    """
    fpath = _file_path(name_key)
    if fpath.exists():
        try:
            df = load_csv_cached(str(fpath), _file_mtime(fpath))
            return df, f"✅ Loaded {fpath.name} ({len(df):,} rows)"
        except Exception as e:
            return None, f"⚠️ Error reading {fpath.name}: {e}"
    return None, f"❌ Missing {fpath.name}"

def show_df(df: pd.DataFrame, caption: str, preview_rows: int = 300):
    """
    Display a DataFrame nicely:
    - Show up to 'preview_rows' rows by default (keeps the page responsive).
    - Offer a full view under an expander if the table is large.
    - Include a one-click download of the exact data being shown.
    """
    st.caption(caption)
    if len(df) <= preview_rows:
        st.dataframe(df, use_container_width=True, height=min(420, 40 + 28*len(df)))
    else:
        st.dataframe(df.head(preview_rows), use_container_width=True, height=420)
        with st.expander(f"Show all rows ({len(df):,}) – may be slower"):
            st.dataframe(df, use_container_width=True, height=560)

    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="preview.csv",
        mime="text/csv",
        use_container_width=True
    )

def availability_badge():
    """Show a friendly note about which expected CSVs are present or missing."""
    missing = [fn for fn in EXPECTED_FILES.values() if not (DATA_DIR / fn).exists()]
    if missing:
        st.warning("Missing files: " + ", ".join(missing))
    else:
        st.success("All expected CSVs found in `/data` ✅")

def load_needed(keys: list[str]) -> dict[str, tuple[pd.DataFrame | None, str]]:
    """
    Load only what the current page needs.
    Returns a dict mapping each key to (df, message).
    """
    return {k: try_load(k) for k in keys}


# =========================
# Sidebar (Navigation + Tools)
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    options=["Overview", "Weights", "Scores", "Results", "Export"],
    index=0
)

st.sidebar.divider()
st.sidebar.write("Data folder:", f"`{DATA_DIR}`")
availability_badge()

# One-click cache clear so you can reload changed CSVs without restarting the app.
if st.sidebar.button("🔄 Reload data (clear cache)", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Data will reload on next render.")


# =========================
# Pages
# =========================
if page == "Overview":
    st.title(APP_TITLE)
    st.markdown(
        """
**Week-1 goals**
- Make sure the project structure exists and the `/data` files are in place  
- Load and show each CSV on its own page  
- Set placeholders for the math we’ll add later (Gatekeepers → ARAS → Tiers)  
        """
    )

elif page == "Weights":
    st.header("Weights")
    st.write("Here you can see raw expert ratings (Delphi) and any pre-computed weights, if available.")

    loaded = load_needed(["delphi_round1", "weights_from_delphi"])

    # --- Delphi inputs ---
    st.subheader("Delphi Round 1 (expert importance 1–9)")
    delphi_df, delphi_msg = loaded["delphi_round1"]
    st.caption(delphi_msg)
    if delphi_df is not None and not delphi_df.empty:
        show_df(delphi_df, "Raw expert inputs that we’ll turn into weights in Week-2.")
    else:
        st.info("Add `delphi_round1_example.csv` to `/data` to view.")

    # --- Weights (if someone already computed them) ---
    st.subheader("Precomputed Weights (if provided)")
    weights_df, weights_msg = loaded["weights_from_delphi"]
    st.caption(weights_msg)
    if weights_df is not None and not weights_df.empty:
        show_df(weights_df, "Weights should sum to 1. These are read-only in Week-1.")
        with st.expander("Quick checks"):
            numeric = weights_df.select_dtypes(include="number")
            st.write("Column sums:", {c: float(numeric[c].sum()) for c in numeric.columns})
            if "weight" in numeric.columns:
                total = float(numeric["weight"].sum())
                if not np.isclose(total, 1.0, atol=1e-6):
                    st.warning(f"`weight` column does not sum to 1 (sum = {total:.6f}).")
    else:
        st.info("Add `weights_from_delphi_example.csv` to `/data` to view precomputed weights.")

elif page == "Scores":
    st.header("Scores")
    st.write("View the scoring criteria and the raw operator scores.")

    loaded = load_needed(["criteria_catalog", "operator_scores"])

    # --- Criteria catalog ---
    st.subheader("Criteria Catalog (type + gate min/max)")
    crit_df, crit_msg = loaded["criteria_catalog"]
    st.caption(crit_msg)
    if crit_df is not None and not crit_df.empty:
        show_df(crit_df, "Names, themes, and gatekeeper settings (used later).")
        with st.expander("Columns & counts"):
            st.write({"columns": list(crit_df.columns), "rows": len(crit_df)})
    else:
        st.info("Add `criteria_catalog.csv` to `/data` to view.")

    # --- Operator scores ---
    st.subheader("Operator Scores")
    ops_df, ops_msg = loaded["operator_scores"]
    st.caption(ops_msg)
    if ops_df is not None and not ops_df.empty:
        show_df(ops_df, "Raw scores by operator and criterion.")
        with st.expander("Quick stats"):
            if "operator_id" in ops_df.columns:
                st.metric("Unique operators", f"{ops_df['operator_id'].nunique():,}")
            st.write(ops_df.describe(include="all"))
    else:
        st.info("Add `operator_scores_dummy.csv` to `/data` to view.")

elif page == "Results":
    st.header("Results (placeholder for Week-3 to Week-5)")
    st.write(
        "This page will eventually show Gatekeeper pass/fail reasons, ARAS utility K, "
        "and the final tier (Platinum/Gold/Silver/Bronze). For Week-1, it’s just a preview."
    )
    st.info("No math yet — we’re only checking that data can join cleanly.")

    loaded = load_needed(["weights_from_delphi", "operator_scores"])
    weights_df, _ = loaded["weights_from_delphi"]
    ops_df, _ = loaded["operator_scores"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Weights available", "Yes" if (weights_df is not None and not weights_df.empty) else "No")
    with col2:
        if ops_df is not None and not ops_df.empty and "operator_id" in ops_df.columns:
            st.metric("Operator scores", f"{ops_df['operator_id'].nunique()} operators")
        else:
            st.metric("Operator scores", "Missing")

    st.divider()
    st.write("**Non-final preview:** if both tables exist, we show a simple join.")
    if (weights_df is not None and not weights_df.empty) and (ops_df is not None and not ops_df.empty):
        join_key = "criterion_id"
        if join_key in ops_df.columns and join_key in weights_df.columns:
            preview = ops_df.merge(weights_df, on=join_key, how="left")
            show_df(preview.head(200), "Joined view only. No normalization, no gates, no K yet.")
        else:
            st.info(f"Can’t join: `{join_key}` column is missing in one of the tables.")
    else:
        st.info("Supply both operator scores and weights to see the preview.")

elif page == "Export":
    st.header("Export")
    st.write("Download a single CSV snapshot of whatever is currently loaded.")

    # Collect loaded tables and tag each with its source name for easy provenance.
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
            label="⬇️ Download snapshot CSV",
            data=buf.getvalue().encode("utf-8"),
            file_name="week1_snapshot.csv",
            mime="text/csv"
        )
        st.caption("The `_table` column tells you which original file each row came from.")
    else:
        st.info("Nothing to export yet—add CSVs to `/data` and try again.")
