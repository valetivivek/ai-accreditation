from __future__ import annotations
import io, uuid
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

APP_TITLE = "AI Accreditation"
DATA_DIR = Path(__file__).parent / "data"

EXPECTED_FILES = {
    "criteria_catalog": "criteria_catalog.csv",
    "delphi_round1": "delphi_round1_example.csv",
    "weights_from_delphi": "weights_from_delphi_example.csv",
    "operator_scores": "operator_scores_dummy.csv",
}

# ---------- basics ----------
def data_dir_label(show_real: bool = False) -> str:
    return f"`{DATA_DIR}`" if show_real else "`/data`"

def _file_path(name_key: str) -> Path:
    return DATA_DIR / EXPECTED_FILES[name_key]

def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return -1.0

@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    p = Path(path_str)
    try:
        return pd.read_csv(p, low_memory=False, dtype_backend="pyarrow")
    except TypeError:
        return pd.read_csv(p, low_memory=False)

def try_load(name_key: str) -> tuple[pd.DataFrame | None, str]:
    fpath = _file_path(name_key)
    if fpath.exists():
        try:
            df = load_csv_cached(str(fpath), _file_mtime(fpath))
            return df, f"✅ Loaded {fpath.name} ({len(df):,} rows)"
        except Exception as e:
            return None, f"⚠️ Error reading {fpath.name}: {e}"
    return None, f"❌ Missing {fpath.name}"

def show_df(df: pd.DataFrame, caption: str, preview_rows: int = 300):
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
        use_container_width=True,
        key=f"dl-{uuid.uuid4().hex}",
    )

def availability_badge():
    missing = [fn for fn in EXPECTED_FILES.values() if not (DATA_DIR / fn).exists()]
    if missing:
        st.warning("Missing files: " + ", ".join(missing))
    else:
        st.success("All expected CSVs found in /data ✅")


# ---------- toolbar CSS ----------
st.markdown(
    """
    <style>
      .toolbar {
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .toolbar .center-title {
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        font-weight: 700;
        font-size: 1.8rem;  /* bigger font */
        white-space: nowrap;
      }
      .icon-btn > button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        min-width: 2rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- top bar (Home left, Title centered, Reload right) ----------
def topbar():
    # 3-column row: [Home] [Centered Title] [Reload]
    left, center, right = st.columns([0.15, 0.70, 0.15], gap="small")

    # Home (left)
    with left:
        if st.button("🏠", key="home_btn", help="Go to Home"):
            try:
                st.switch_page("streamlit_app.py")
            except Exception:
                st.rerun()

    # Centered Title (middle)
    with center:
        st.markdown(
            f"""
            <div style="
                text-align:center;
                font-weight:700;
                font-size:1.8rem;   /* bump size here */
                line-height:1.2;
                margin:0.25rem 0 0.25rem 0;">
                {APP_TITLE}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Reload (right) — aligned to the right edge of the row
    with right:
        _sp, rbtn = st.columns([0.6, 0.4])  # small spacer + button
        with rbtn:
            if "last_cache_clear" not in st.session_state:
                st.session_state.last_cache_clear = 0.0
            if st.button("↻", help="Reload data cache", key="cache-reload-btn"):
                import time
                now = time.time()
                if now - st.session_state.last_cache_clear > 1.5:
                    with st.spinner("Clearing cache…"):
                        st.cache_data.clear()
                    st.session_state.last_cache_clear = now
                    st.toast("Cache cleared", icon="✅")

    st.divider()

