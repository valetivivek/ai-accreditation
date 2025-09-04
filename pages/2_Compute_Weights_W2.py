from __future__ import annotations
import numpy as np
import pandas as pd      
import streamlit as st
from core import topbar, try_load, show_df, availability_badge, DATA_DIR, EXPECTED_FILES, APP_TITLE
from weights import compute_weights_from_delphi, kendalls_w


st.set_page_config(page_title=f"{APP_TITLE} – Compute Weights (W2)", layout="wide")

# --- column mapping helpers (inline) ---
def guess_cols(df):
    cols_lower = [c.lower() for c in df.columns]
    original = {c.lower(): c for c in df.columns}
    def pick(cands):
        for cand in cands:
            for col in cols_lower:
                if cand in col: return original[col]
        return None
    return {
        "criterion_id": pick(["criterion_id","criteria_id","criterion","criteria","item"]),
        "expert_id":    pick(["expert_id","expert","rater_id","rater","panelist","judge"]),
        "rating":       pick(["rating","score","importance","weight","vote"]),
    }

def apply_map(df, mapping):
    ren = {mapping[k]: k for k in ["criterion_id","expert_id","rating"] if mapping.get(k) in df.columns}
    out = df.rename(columns=ren).copy()
    out["rating"] = out["rating"].apply(pd.to_numeric, errors="coerce")
    return out

topbar()
st.header("Compute weights from Delphi (Week-2)")

delphi_df, msg = try_load("delphi_round1")
st.caption(msg)

if delphi_df is None or delphi_df.empty:
    st.info("Add `delphi_round1_example.csv` to `/data` first.")
else:
    st.subheader("Map your CSV columns")
    guess = guess_cols(delphi_df)
    c1,c2,c3 = st.columns(3)
    with c1:
        crit_col = st.selectbox("Criterion → `criterion_id`", ["<select>"]+list(delphi_df.columns),
                                index=(list(delphi_df.columns).index(guess["criterion_id"])+1) if guess["criterion_id"] else 0)
    with c2:
        exp_col  = st.selectbox("Expert → `expert_id`", ["<select>"]+list(delphi_df.columns),
                                index=(list(delphi_df.columns).index(guess["expert_id"])+1) if guess["expert_id"] else 0)
    with c3:
        rat_col  = st.selectbox("Rating → `rating`", ["<select>"]+list(delphi_df.columns),
                                index=(list(delphi_df.columns).index(guess["rating"])+1) if guess["rating"] else 0)

    if "<select>" in (crit_col, exp_col, rat_col):
        st.warning("Pick all three columns to continue.")
    else:
        mapping = {"criterion_id": crit_col, "expert_id": exp_col, "rating": rat_col}
        delphi_norm = apply_map(delphi_df, mapping)
        with st.expander("Preview normalized Delphi data"):
            show_df(delphi_norm.head(200), "First 200 rows")

        method = st.selectbox("Method", ["geom","trimmed","consensus"], index=0)
        trim = st.number_input("Trim proportion (trimmed only)", 0.0, 0.45, 0.10, 0.05)

        if st.button("Compute & Save weights", type="primary"):
            weights = compute_weights_from_delphi(delphi_norm, method=method, trim=trim)
            st.success("Weights computed.")
            show_df(weights, "Computed weights (sums to ~1).")
            out_path = DATA_DIR / EXPECTED_FILES["weights_from_delphi"]
            weights.to_csv(out_path, index=False)
            st.caption(f"Saved to `{out_path}`")

            W = kendalls_w(delphi_norm)
            if not np.isnan(W):
                st.metric("Kendall’s W (agreement)", f"{W:.3f}")

st.sidebar.write("Data folder:", f"`{DATA_DIR}`")
availability_badge()
