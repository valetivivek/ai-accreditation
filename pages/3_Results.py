# pages/3_Results.py
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Tuple, Optional

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
# ARAS computation (Spec-compliant)
# =========================
st.subheader("ARAS Settings")
col1, col2 = st.columns(2)
with col1:
    show_debug = st.checkbox("Show debug tables and method explanation")
with col2:
    strict_gatekeeper = st.checkbox("Strict gatekeeper enforcement", value=True, 
                                   help="If unchecked, operators with gate failures can still get accredited tiers based on K score")

if show_debug:
    with st.expander("ARAS Method Explanation"):
        st.markdown("""
        **ARAS (Additive Ratio Assessment System) Steps:**
        
        1. **Gatekeepers**: Check if operator scores meet min/max thresholds per criterion
        2. **ARAS Normalization**: 
           - Benefit criteria: `x' = x / sum(x)` (linear ratio)
           - Cost criteria: `x' = (1/x) / sum(1/x)` (inverse ratio)
        3. **Weighted Sum**: `Si = Σ(w * x')` for each operator i
        4. **Ideal Reference**: `S0 = Σ(w * x'0)` where x'0 uses best value per criterion
           - Best = max for benefit criteria, min for cost criteria
        5. **Final Score**: `K = Si / S0` (ratio to ideal)
        6. **Tier Assignment**: 
           - Platinum: K ≥ 0.90
           - Gold: K ≥ 0.80  
           - Silver: K ≥ 0.65
           - Bronze: K ≥ 0.50
        """)

# Build decision matrix: rows operators, columns criteria
pivot = ops_use.pivot_table(index="operator_id", columns="criterion_id", values="score", aggfunc="mean")
# Ensure operator_id and criterion_id are strings
pivot.index = pivot.index.astype(str)
pivot.columns = pivot.columns.astype(str)

# Align to weight criteria set
pivot = pivot.reindex(columns=weights["criterion_id"].astype(str), fill_value=np.nan)

# Get criterion types
sense_map = crit_use.set_index("criterion_id")["type"].reindex(pivot.columns).fillna("benefit")

def aras_normalize_benefit(x: pd.Series) -> pd.Series:
    """ARAS benefit normalization: x' = x / sum(x)"""
    valid_x = x.dropna()
    if len(valid_x) < 2:
        return pd.Series(np.nan, index=x.index)
    sum_x = valid_x.sum()
    if math.isclose(sum_x, 0):
        return pd.Series(0.0, index=x.index)
    return x / sum_x

def aras_normalize_cost(x: pd.Series) -> pd.Series:
    """ARAS cost normalization: x' = (1/x) / sum(1/x)"""
    valid_x = x.dropna()
    if len(valid_x) < 2:
        return pd.Series(np.nan, index=x.index)
    
    # Handle zeros by replacing with NaN
    inv_x = np.where(valid_x == 0, np.nan, 1.0 / valid_x)
    inv_series = pd.Series(inv_x, index=valid_x.index)
    
    sum_inv = inv_series.sum()
    if math.isnan(sum_inv) or math.isclose(sum_inv, 0):
        return pd.Series(np.nan, index=x.index)
    
    # Apply normalization to original series
    result = pd.Series(np.nan, index=x.index)
    for idx in x.index:
        if not pd.isna(x.loc[idx]) and x.loc[idx] != 0:
            result.loc[idx] = (1.0 / x.loc[idx]) / sum_inv
    return result

# Apply ARAS normalization per criterion
norm = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
dropped_criteria = []

for cid in pivot.columns:
    s = pivot[cid].astype(float)
    sense = sense_map.get(cid, "benefit")
    
    # Check for sufficient valid values
    valid_count = s.notna().sum()
    if valid_count < 2:
        dropped_criteria.append(f"{cid} (only {valid_count} valid values)")
        continue
    
    if sense == "benefit":
        norm[cid] = aras_normalize_benefit(s)
    else:  # cost
        norm[cid] = aras_normalize_cost(s)

# Report dropped criteria
if dropped_criteria:
    st.warning(f"Dropped criteria with insufficient data: {', '.join(dropped_criteria)}")

# Update weights to match remaining criteria
remaining_criteria = [c for c in norm.columns if not norm[c].isna().all()]
if len(remaining_criteria) != len(weights):
    st.info(f"Using {len(remaining_criteria)} criteria (dropped {len(weights) - len(remaining_criteria)} due to insufficient data)")

# Re-normalize weights for remaining criteria
w_subset = weights[weights["criterion_id"].astype(str).isin(remaining_criteria)].copy()
if len(w_subset) > 0:
    w_subset["weight"] = w_subset["weight"] / w_subset["weight"].sum()
    w = w_subset.set_index("criterion_id")["weight"].reindex(remaining_criteria).fillna(0.0)
else:
    # Fallback to equal weights
    equal_weight = 1.0 / len(remaining_criteria)
    w = pd.Series(equal_weight, index=remaining_criteria)
    st.warning("No valid weights found, using equal weights")

# Compute weighted sums Si for each operator
S_i = (norm[remaining_criteria] * w).sum(axis=1)

# Compute ideal reference S0
S0_components = []
for cid in remaining_criteria:
    sense = sense_map.get(cid, "benefit")
    col_data = pivot[cid].dropna()
    
    if len(col_data) == 0:
        continue
        
    if sense == "benefit":
        best_val = col_data.max()
    else:  # cost
        best_val = col_data.min()
    
    # Normalize this best value using the same ARAS method
    if sense == "benefit":
        best_norm = aras_normalize_benefit(col_data).max()
    else:
        best_norm = aras_normalize_cost(col_data).min()
    
    S0_components.append(w[cid] * best_norm)

S0 = sum(S0_components)

# Compute final ARAS scores K = Si / S0
if S0 > 0:
    K = S_i / S0
else:
    K = pd.Series(0.0, index=S_i.index)
    st.error("Ideal reference S0 is zero - cannot compute ARAS scores")

aras_df = (
    pd.DataFrame({"operator_id": K.index.astype(str), "K": K.values})
    .reset_index(drop=True)
)

# Add criteria contribution count
criteria_counts = norm[remaining_criteria].notna().sum(axis=1)
aras_df["n_criteria"] = criteria_counts.reindex(K.index).fillna(0).astype(int).values

# =========================
# Tier assignment (Updated thresholds)
# =========================
def assign_tier(k: float, gate_passed: bool = True, strict: bool = True) -> str:
    """Assign tier based on K score and gate status"""
    if pd.isna(k):
        return "Not Accredited"
    
    # If strict gatekeeper enforcement is disabled, ignore gate status
    if not strict or gate_passed:
        if k >= 0.90:
            return "Platinum"
        if k >= 0.80:
            return "Gold"
        if k >= 0.65:
            return "Silver"
        if k >= 0.50:
            return "Bronze"
    
    return "Not Accredited"

# Merge Gatekeepers and ARAS
final = (
    aras_df.merge(gate_op, on="operator_id", how="left")
            .reset_index(drop=True)
)

# Assign tiers (K is computed regardless of gate status)
final["Tier"] = final.apply(lambda row: assign_tier(row["K"], row["gate_pass"], strict_gatekeeper), axis=1)

# Sort by gate_pass desc, then K desc
final = final.sort_values(["gate_pass", "K"], ascending=[False, False]).reset_index(drop=True)

# =========================
# Display
# =========================
st.subheader("Summary")
summary = final[["operator_id", "gate_pass", "K", "Tier", "n_criteria", "failed_reasons"]].copy()
summary = summary.rename(columns={
    "gate_pass": "Gatekeeper",
    "failed_reasons": "Fail reasons",
    "n_criteria": "Criteria used"
})
summary["Gatekeeper"] = summary["Gatekeeper"].map({True: "Pass", False: "Fail", np.nan: "Unknown"})

show_df(summary, "Operator results with gate status, ARAS K, criteria count, and assigned tier.")

if show_debug:
    with st.expander("Debug: Weights used"):
        w_debug = pd.DataFrame({
            "criterion_id": remaining_criteria,
            "weight": [w[c] for c in remaining_criteria],
            "weight_sum": w.sum()
        })
        show_df(w_debug, f"Weights aligned to {len(remaining_criteria)} criteria (sum = {w.sum():.6f})")
    
    with st.expander("Debug: ARAS Components"):
        aras_debug = pd.DataFrame({
            "operator_id": S_i.index,
            "S_i": S_i.values,
            "S0": S0,
            "K": K.values
        })
        show_df(aras_debug, f"ARAS computation: Si={S_i.values}, S0={S0:.6f}")
    
    with st.expander("Debug: Criteria, gatekeepers, and sense"):
        show_df(crit_use, "Criteria with sense and gate thresholds.")
    
    with st.expander("Debug: Decision matrix (raw scores)"):
        show_df(pivot.reset_index(), "Operators by criteria, raw scores.")
    
    with st.expander("Debug: ARAS normalized matrix"):
        norm_display = norm[remaining_criteria].copy()
        norm_display.index.name = "operator_id"
        show_df(norm_display.reset_index(), "ARAS normalized scores (x/sum(x) for benefit, (1/x)/sum(1/x) for cost)")
    
    with st.expander("Debug: Gatekeeper details"):
        gate_debug = gate[["operator_id", "criterion_id", "score", "type", "gate_min", "gate_max", "crit_pass", "crit_reason"]].copy()
        show_df(gate_debug, "Detailed gatekeeper evaluation per operator-criterion pair")

st.divider()
st.subheader("Charts")

c1, c2 = st.columns(2)

with c1:
    st.write("**ARAS Scores by Operator**")
    chart_data = summary[["operator_id", "K", "Tier", "Gatekeeper"]].copy()
    
    # Sort by K for visualization
    chart_data = chart_data.sort_values("K", ascending=False)
    
    # Create a simple bar chart without color parameter
    st.bar_chart(chart_data.set_index("operator_id")["K"])
    
    # Add legend/explanation
    st.caption("**Legend**: All operators shown. Check 'Gatekeeper' column for pass/fail status.")

with c2:
    st.write("**Tier Distribution**")
    tier_counts = summary["Tier"].value_counts()
    tier_order = ["Platinum", "Gold", "Silver", "Bronze", "Not Accredited"]
    tier_counts = tier_counts.reindex(tier_order).fillna(0).astype(int)
    
    st.bar_chart(tier_counts)
    st.caption(f"Total operators: {len(summary)}")

# Add summary statistics
st.subheader("Summary Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Operators", len(summary))
with col2:
    st.metric("Gate Passed", summary["Gatekeeper"].eq("Pass").sum())
with col3:
    st.metric("Avg K Score", f"{summary['K'].mean():.3f}")
with col4:
    st.metric("Max K Score", f"{summary['K'].max():.3f}")

st.caption(
    f"**ARAS Method**: K = Si/S0 where Si = Σ(w×x') and S0 uses best values per criterion. "
    "Benefit: x' = x/sum(x), Cost: x' = (1/x)/sum(1/x). "
    f"Gatekeeper enforcement: {'Strict' if strict_gatekeeper else 'Lenient'} - "
    f"{'Gate failures result in Not Accredited' if strict_gatekeeper else 'K score determines tier regardless of gates'}."
)
