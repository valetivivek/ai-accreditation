from __future__ import annotations
import numpy as np
import pandas as pd

def _geom_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    # Requires positive values; Delphi scales like 1–9 are fine
    return float(np.exp(np.log(x).mean()))

def compute_weights_from_delphi(
    delphi_df: pd.DataFrame,
    method: str = "geom",      # "geom" | "trimmed" | "consensus"
    trim: float = 0.1          # used when method="trimmed" (10% each side)
) -> pd.DataFrame:
    # Expect canonical columns (streamlit_app maps arbitrary headers into these)
    if not {"criterion_id", "expert_id", "rating"}.issubset(delphi_df.columns):
        raise ValueError("delphi_df must have columns: criterion_id, expert_id, rating")

    df = delphi_df.dropna(subset=["rating"]).copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    # Keep positive values (geom mean needs > 0)
    df = df[df["rating"] > 0]

    def agg_func(g: pd.Series) -> float:
        vals = np.array(g, dtype=float)
        if method == "geom":
            return _geom_mean(vals)
        elif method == "trimmed":
            if len(vals) < 3:
                return float(vals.mean())
            k = int(np.floor(len(vals) * trim))
            if k == 0:
                return float(vals.mean())
            vals_sorted = np.sort(vals)[k: len(vals)-k]
            return float(vals_sorted.mean())
        elif method == "consensus":
            mu = float(vals.mean())
            sigma = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            return mu / (1.0 + max(sigma, 0.0))
        else:
            raise ValueError("method must be one of: 'geom', 'trimmed', 'consensus'")

    agg = (
        df.groupby("criterion_id")["rating"]
          .apply(agg_func)
          .rename("score")
          .reset_index()
    )
    total = agg["score"].sum()
    agg["weight"] = agg["score"] / total if total > 0 else 0.0

    # Diagnostics
    stats = (df.groupby("criterion_id")["rating"]
               .agg(mean_rating="mean", stdev="std", n_experts="count")
               .reset_index())
    out = agg.merge(stats, on="criterion_id", how="left")
    out = out[["criterion_id", "weight", "mean_rating", "stdev", "n_experts"]]
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    return out

def kendalls_w(delphi_df: pd.DataFrame) -> float:
    """
    Kendall's W across experts for ranked criteria.
    Assumes higher 'rating' = higher rank (ties allowed).
    """
    if not {"criterion_id", "expert_id", "rating"}.issubset(delphi_df.columns):
        return float("nan")

    P = delphi_df.pivot_table(index="criterion_id", columns="expert_id", values="rating", aggfunc="mean")
    P = P.dropna(axis=0, how="any")  # require complete cases
    if P.shape[0] < 2 or P.shape[1] < 2:
        return float("nan")

    R = P.rank(axis=0, method="average")
    m = R.shape[1]  # experts
    n = R.shape[0]  # items (criteria)
    S = ((R.sum(axis=1) - (m * (n + 1) / 2)) ** 2).sum()
    W = 12 * S / (m**2 * (n**3 - n))
    return float(W)
