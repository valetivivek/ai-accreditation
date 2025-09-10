# pages/3_Results.py
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from helpers import topbar, try_load, show_df

topbar("AI Accreditation – Week 3: Results")

st.header("Results")
st.write(
    "Gatekeeper evaluation, ARAS utility computation, and final tier assignment."
)

# =========================
# Flexible column mapping
# =========================
def pick_column(candidates: List[str], columns: List[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None

def detect_columns_criteria(df: pd.DataFrame) -> Dict[str, str | None]:
    cols = list(df.columns)
    return {
        "criterion_id": pick_column(["criterion_id", "criteria_id", "criterion", "criteria"], cols),
        "type": pick_column(["type", "criterion_type", "sense"], cols),
        "gate_min": pick_column(["gate_min", "min", "threshold_min", "gatekeeper_min"], cols),
        "gate_max": pick_column(["gate_max", "max", "threshold_max", "gatekeeper_max"], cols),
    }

def detect_columns_ops(df: pd.DataFrame) -> Dict[str, str | None]:
    cols = list(df.columns)
    return {
        "operator_id": pick_column(["operator_id", "operator", "provider_id"], cols),
        "criterion_id": pick_column(["criterion_id", "criteria_id", "criterion", "criteria"], cols),
        "score": pick_column(["score", "value", "raw_score", "rating"], cols),
    }

def detect_columns_weights(df: pd.DataFrame) -> Dict[str, str | None]:
    cols = list(df.columns)
    return {
        "criterion_id": pick_column(["criterion_id", "criteria_id", "criterion", "criteria"], cols),
        "weight": pick_column(["weight", "w", "normalized_weight"], cols),
    }

# =========================
# Load data
# =========================
crit_df, crit_msg = try_load("criteria_catalog")
ops_df, ops_msg = try_load("operator_scores")
w_df, w_msg = try_load("weights_from_delphi")

st.subheader("Data status")
c1, c2, c3 = st.columns(3)
with c1: st.caption(crit_msg)
with c2: st.caption(ops_msg)
with c3: st.caption(w_msg)

if crit_df is None or crit_df.empty:
    st.error("Missing criteria_catalog. Add it to /data and retry.")
    st.stop()
if ops_df is None or ops_df.empty:
    st.error("Missing operator_scores. Add it to /data and retry.")
    st.stop()

# Column mappings
cmap = detect_columns_criteria(crit_df)
omap = detect_columns_ops(ops_df)

missing_c = [k for k, v in cmap.items() if k in ("criterion_id",) and v is None]
missing_o = [k for k, v in omap.items() if k in ("operator_id", "criterion_id", "score") and v is None]
if missing_c or missing_o:
    st.error(
        f"Cannot proceed due to missing required columns."
        f"\nCriteria missing: {missing_c or 'None'}"
        f"\nOperator scores missing: {missing_o or 'None'}"
        f"\nCriteria columns found: {list(crit_df.columns)}"
        f"\nOperator columns found: {list(ops_df.columns)}"
    )
    st.stop()

# Basic cleaning
ops = ops_df.copy()
ops[omap["score"]] = pd.to_numeric(ops[omap["score"]], errors="coerce")
ops = ops.dropna(subset=[omap["operator_id"], omap["criterion_id"], omap["score"]])

# Keep only criteria that appear in both tables
valid_criteria = set(crit_df[cmap["criterion_id"]]).intersection(set(ops[omap["criterion_id"]]))
crit = crit_df[crit_df[cmap["criterion_id"]].isin(valid_criteria)].copy()
ops = ops[ops[omap["criterion_id"]].isin(valid_criteria)].copy()

# Prepare weights
if w_df is not None and not w_df.empty:
    wmap = detect_columns_weights(w_df)
    if wmap["criterion_id"] and wmap["weight"]:
        weights = (
            w_df[[wmap["criterion_id"], wmap["weight"]]]
            .rename(columns={wmap["criterion_id"]: "criterion_id", wmap["weight"]: "weight"})
        )
        # keep only relevant criteria and re-normalize
        weights = weights[weights["criterion_id"].isin(valid_criteria)].copy()
        if weights["weight"].sum() <= 0 or weights["weight"].isna().all():
            st.warning("Weights file found but invalid. Falling back to equal weights.")
            weights = None
        else:
            weights["weight"] = weights["weight"] / weights["weight"].sum()
    else:
        st.warning("Weights file missing required columns. Falling back to equal weights.")
        weights = None
else:
    weights = None

# Equal weights if needed
if weights is None:
    weights = pd.DataFrame({"criterion_id": sorted(valid_criteria)})
    weights["weight"] = 1.0 / len(weights)

# =========================
# Gatekeeper logic
# =========================
def normalize_type(val: str | float | int) -> str:
    if pd.isna(val): 
        return ""
    s = str(val).strip().lower()
    # Treat aliases
    if s in {"benefit", "max", "higher", "higher_is_better", "pos"}:
        return "benefit"
    if s in {"cost", "min", "lower", "lower_is_better", "neg"}:
        return "cost"
    return s

crit_use = crit[[cmap["criterion_id"]] + [x for x in [cmap["type"], cmap["gate_min"], cmap["gate_max"]] if x is not None]].copy()
crit_use = crit_use.rename(columns={
    cmap["criterion_id"]: "criterion_id",
    (cmap["type"] or "type"): "type",
    (cmap["gate_min"] or "gate_min"): "gate_min",
    (cmap["gate_max"] or "gate_max"): "gate_max",
})
if "type" in crit_use.columns:
    crit_use["type"] = crit_use["type"].apply(normalize_type)
else:
    crit_use["type"] = "benefit"  # default

for col in ("gate_min", "gate_max"):
    if col in crit_use.columns:
        crit_use[col] = pd.to_numeric(crit_use[col], errors="coerce")
    else:
        crit_use[col] = np.nan

ops_use = ops[[omap["operator_id"], omap["criterion_id"], omap["score"]]].rename(
    columns={
        omap["operator_id"]: "operator_id",
        omap["criterion_id"]: "criterion_id",
        omap["score"]: "score",
    }
)

merged = ops_use.merge(crit_use, on="criterion_id", how="left")

def gatecheck_row(row) -> Tuple[bool, List[str]]:
    fails = []
    s = row["score"]
    gmin = row["gate_min"]
    gmax = row["gate_max"]
    cid = row["criterion_id"]

    if pd.notna(gmin) and s < gmin:
        fails.append(f"{cid} < min {gmin}")
    if pd.notna(gmax) and s > gmax:
        fails.append(f"{cid} > max {gmax}")
    return (len(fails) == 0, fails)

gate = (
    merged
    .assign(_gc=lambda d: d.apply(gatecheck_row, axis=1))
)

gate["crit_pass"] = gate["_gc"].apply(lambda x: x[0])
gate["crit_reason"] = gate["_gc"].apply(lambda x: "; ".join(x[1]) if x[1] else "")

# Aggregate to operator level
gate_op = (
    gate.groupby("operator_id")
        .agg(
            gate_pass=("crit_pass", "all"),
            failed_reasons=("crit_reason", lambda s: "; ".join([x for x in s if x]))
        )
        .reset_index()
)

# =========================
# ARAS computation
# =========================
st.subheader("ARAS Settings")
left, right = st.columns([1, 1])
with left:
    norm_method = st.radio(
        "Normalization method",
        ["Min–Max to [0,1]", "Z score then clamp to [0,1]"],
        horizontal=True,
    )
with right:
    show_debug = st.checkbox("Show debug tables")

# Build decision matrix: rows operators, columns criteria
pivot = ops_use.pivot_table(index="operator_id", columns="criterion_id", values="score", aggfunc="mean")
# Align to weight criteria set
pivot = pivot.reindex(columns=weights["criterion_id"], fill_value=np.nan)

# Per-criterion normalization
def nn_minmax(x: pd.Series, sense: str) -> pd.Series:
    # sense: benefit or cost
    if x.notna().sum() <= 1:
        return pd.Series(0.0, index=x.index)  # degenerate
    xmin, xmax = x.min(), x.max()
    if math.isclose(xmax, xmin):
        return pd.Series(0.5, index=x.index)
    z = (x - xmin) / (xmax - xmin)
    if sense == "cost":
        z = 1.0 - z
    return z.clip(0.0, 1.0)

def nn_zscore(x: pd.Series, sense: str) -> pd.Series:
    if x.notna().sum() <= 1:
        z = pd.Series(0.0, index=x.index)
    else:
        mu, sd = x.mean(), x.std(ddof=0)
        if math.isclose(sd, 0.0):
            z = pd.Series(0.5, index=x.index)
        else:
            z = (x - mu) / sd
            # map approx z in [-3, +3] to [0,1]
            z = (z + 3) / 6
    if sense == "cost":
        z = 1.0 - z
    return z.clip(0.0, 1.0)

# sense per criterion
sense_map = crit_use.set_index("criterion_id")["type"].reindex(pivot.columns).fillna("benefit")

norm = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
for cid in pivot.columns:
    s = pivot[cid].astype(float)
    sense = sense_map.get(cid, "benefit")
    if norm_method.startswith("Min"):
        norm[cid] = nn_minmax(s, sense)
    else:
        norm[cid] = nn_zscore(s, sense)

# Weighted sum
w = weights.set_index("criterion_id")["weight"].reindex(norm.columns).fillna(0.0)
K_raw = (norm * w).sum(axis=1)

# Normalize K to best = 1.0
if K_raw.max() > 0:
    K = K_raw / K_raw.max()
else:
    K = K_raw  # all zeros

aras_df = (
    pd.DataFrame({"operator_id": K.index, "K": K.values})
    .sort_values("K", ascending=False)
    .reset_index(drop=True)
)

# =========================
# Tier assignment
# =========================
def assign_tier(k: float) -> str:
    if pd.isna(k):
        return "Bronze"
    if k >= 0.85:
        return "Platinum"
    if k >= 0.70:
        return "Gold"
    if k >= 0.55:
        return "Silver"
    return "Bronze"

aras_df["Tier"] = aras_df["K"].apply(assign_tier)

# Merge Gatekeepers and ARAS
final = (
    aras_df.merge(gate_op, on="operator_id", how="left")
            .sort_values(["gate_pass", "K"], ascending=[False, False])
            .reset_index(drop=True)
)

# If an operator fails gates, you may want to optionally zero out K
apply_gate_zero = st.checkbox("Set K to 0 for gate failures", value=True)
if apply_gate_zero:
    final["K"] = np.where(final["gate_pass"].fillna(False), final["K"], 0.0)
    final["Tier"] = np.where(final["gate_pass"].fillna(False), final["Tier"], "Bronze")

# =========================
# Display
# =========================
st.subheader("Summary")
summary = final[["operator_id", "gate_pass", "K", "Tier", "failed_reasons"]].copy()
summary = summary.rename(columns={
    "gate_pass": "Gatekeeper",
    "failed_reasons": "Fail reasons",
})
summary["Gatekeeper"] = summary["Gatekeeper"].map({True: "Pass", False: "Fail", np.nan: "Unknown"})

show_df(summary, "Operator results with gate status, ARAS K, and assigned tier.")

if show_debug:
    with st.expander("Debug: Weights used"):
        show_df(weights, "Weights aligned to criteria used in ARAS.")
    with st.expander("Debug: Criteria, gatekeepers, and sense"):
        show_df(crit_use, "Criteria with sense and gate thresholds.")
    with st.expander("Debug: Decision matrix (raw scores)"):
        show_df(pivot.reset_index(), "Operators by criteria, raw scores.")
    with st.expander("Debug: Normalized matrix used for ARAS"):
        show_df(norm.reset_index(), "Per criterion normalized scores in [0,1].")

st.divider()
st.subheader("Charts")
c1, c2 = st.columns(2)
with c1:
    st.bar_chart(summary.set_index("operator_id")["K"])
with c2:
    tier_counts = summary["Tier"].value_counts().reindex(["Platinum", "Gold", "Silver", "Bronze"]).fillna(0).astype(int)
    st.bar_chart(tier_counts)

st.caption(
    "Notes: benefit vs cost is inferred from `type` in criteria_catalog. "
    "Gatekeepers use gate_min and gate_max if present. K is normalized so the best operator equals 1.0."
)
