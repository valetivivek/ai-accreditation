import streamlit as st
import pandas as pd
from helpers import topbar, try_load, show_df
from helpers import topbar

topbar("AI Accreditation – Week 2: Compute Weights")

# --- load Delphi inputs ---
delphi_df, msg = try_load("delphi_round1")
st.caption(msg)

if delphi_df is None or delphi_df.empty:
    st.info("Add delphi_round1_example.csv to /data (needs columns like: criterion_id, expert_id, rating/score/importance/value).")
    st.stop()

df = delphi_df.copy()

# ---------- FLEXIBLE COLUMN DETECTION ----------
def pick_column(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None

crit_col = pick_column(["criterion_id", "criterion", "criteria_id", "criteria"], df.columns)
exp_col  = pick_column(["expert_id", "expert", "rater_id", "judge_id"], df.columns)
# NOTE: include 'importance_1to9' to match your professor's file
rate_col = pick_column(["rating", "score", "importance", "value", "importance_1to9"], df.columns)

missing = []
if crit_col is None: missing.append("criterion_id (or: criterion, criteria_id, criteria)")
if rate_col is None: missing.append("rating (or: score, importance, value, importance_1to9)")
if missing:
    st.error(f"Your CSV is missing required columns: {', '.join(missing)}\n\nFound columns: {list(df.columns)}")
    st.stop()

# --- controls ---
st.subheader("Aggregation settings")
method = st.radio("Method", ["Mean", "Median", "Trimmed mean (10%)"], horizontal=True)
do_expert_normalize = st.checkbox("Normalize per-expert scale before aggregation (optional, robust)")

st.divider()

# --- validation & cleaning ---
df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
before = len(df)
df = df.dropna(subset=[crit_col, rate_col])
dropped = before - len(df)
if dropped:
    st.warning(f"Dropped {dropped} rows with missing {crit_col} or {rate_col}.")

# Clamp to [1,9] (Delphi scale)
clamped = ((df[rate_col] < 1) | (df[rate_col] > 9)).sum()
if clamped:
    st.warning(f"{clamped} ratings outside [1,9] were clamped.")
df[rate_col] = df[rate_col].clip(lower=1, upper=9)

# Optional per-expert normalization (min–max to [0,1])
base_col = rate_col
if do_expert_normalize and exp_col is not None and exp_col in df.columns:
    df["_min"] = df.groupby(exp_col)[rate_col].transform("min")
    df["_max"] = df.groupby(exp_col)[rate_col].transform("max")
    span = (df["_max"] - df["_min"]).replace(0, pd.NA)
    df["rating_norm"] = (df[rate_col] - df["_min"]) / span
    df["rating_norm"] = df["rating_norm"].fillna(0.5)  # if span==0, fall back to 0.5
    base_col = "rating_norm"
    st.caption("Per-expert min–max normalization applied to ratings.")

# --- aggregation helpers ---
def trimmed_mean_10(x: pd.Series) -> float:
    x = x.sort_values()
    k = max(int(0.10 * len(x)), 0)
    return x.iloc[k: len(x)-k].mean() if len(x) > 2 * k else x.mean()

def aggregate(_df: pd.DataFrame, method_name: str) -> pd.Series:
    g = _df.groupby(crit_col)[base_col]
    if method_name == "Mean":
        return g.mean()
    elif method_name == "Median":
        return g.median()
    else:
        return g.apply(trimmed_mean_10)

# --- compute on click ---
if st.button("Compute weights"):
    s = aggregate(df, method)
    if s.sum() == 0 or s.isna().all():
        st.error("Aggregation produced empty/zero scores. Check input values or column mapping.")
        st.stop()

    weights = (s / s.sum()).rename("weight")
    out = pd.DataFrame({
        "criterion_id": s.index.astype(str),
        "weight": weights.values.round(6),
        "agg_stat": method.lower().replace(" ", "_"),
    })

    # optional n_experts
    if exp_col is not None and exp_col in df.columns:
        n_experts = df.groupby(crit_col)[exp_col].nunique()
        out["n_experts"] = n_experts.reindex(out["criterion_id"]).fillna(0).astype(int).values

   

    # download
    st.download_button(
        "Download weights_from_delphi.csv",
        out.to_csv(index=False).encode("utf-8"),
        file_name="weights_from_delphi.csv",
        mime="text/csv",
    )

st.divider()
