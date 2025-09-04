from __future__ import annotations
import streamlit as st
from core import data_dir_label, topbar, availability_badge, DATA_DIR, APP_TITLE

st.set_page_config(page_title=f"{APP_TITLE} – Overview", layout="wide")

topbar()
st.title("Overview")
st.markdown(
    """
**Week-1 goals**
- Confirm project structure and `/data` files exist  
- Load and display CSVs on dedicated pages  
- Prepare placeholders for later math (Gatekeepers → ARAS → Tiers)
    """
)

st.sidebar.header("Navigation")
st.sidebar.info("Use the page list (left) to jump around.")
show_debug = st.sidebar.toggle("Show debug paths", value=False)
st.sidebar.write("Data folder:", data_dir_label(show_real=show_debug))
availability_badge()
