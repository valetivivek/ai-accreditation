from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

APP_TITLE = "AI Accreditation"   # TITLE
DATA_DIR = Path(__file__).parent / "data"
EXPECTED_FILES = {
    "criteria_catalog": "criteria_catalog.csv",
    "delphi_round1": "delphi_round1_example.csv",
    "weights_from_delphi": "weights_from_delphi_example.csv",
    "operator_scores": "operator_scores_dummy.csv",
}

st.set_page_config(page_title=APP_TITLE, layout="wide")

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def try_load(name_key: str):
    filename = EXPECTED_FILES[name_key]
    fpath = DATA_DIR / filename
    if fpath.exists():
        try:
            df = load_csv(fpath)
            return df, f"Loaded {filename} ({len(df):,} rows)"
        except Exception as e:
            return None, f"Error reading {filename}: {e}"
    return None, f"Missing {filename}"

def show_df(df: pd.DataFrame, caption: str = ""):
    if caption:
        st.caption(caption)
    st.dataframe(df, use_container_width=True, height=420)

def availability_badge():
    missing = [fn for fn in EXPECTED_FILES.values() if not (DATA_DIR / fn).exists()]
    if missing:
        st.warning("Missing files: " + ", ".join(missing))
    else:
        st.success("All expected CSVs found in /data")

def topbar(title: str = APP_TITLE):
    # sticky top bar with Home button (top-left)
    st.markdown(
        """
        <style>
            .topbar { position: sticky; top: 0; z-index: 999; background: var(--background-color);
                      padding: .5rem 0 .5rem 0; margin: -1rem 0 1rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown('<div class="topbar"></div>', unsafe_allow_html=True)
        cols = st.columns([1, 6, 3])
        with cols[0]:
            st.page_link("streamlit_app.py", label="🏠 Home")
        with cols[1]:
            st.title(title)
        with cols[2]:
            availability_badge()
