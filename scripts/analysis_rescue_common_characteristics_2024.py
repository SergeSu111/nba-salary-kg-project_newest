#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analysis_rescue_common_characteristics_2024.py

Purpose
-------
For each reference baseline (Stats+Time and Stats+Time+Meta), and each graph model:
1) Build Eligible / Green(Rescue) / Red(Misguide) sets on 2024 test
2) Merge 2024 tabular feature table (features_tabular_common_test2024.csv)
3) Output:
   - sets/{model}__eligible.csv
   - sets/{model}__green.csv
   - sets/{model}__red.csv
   - profiles/{model}__green_vs_red_profile.csv (sorted by |Cliff's Delta| then p-value)
4) Also output a paper-ready summary table (Top-K traits per model):
   - paper_table__top_traits.csv

Definitions
-----------
Rescue definition: rescue = |y - base| - |y - model|
Eligible: |y - base| > BASE_ERR_MIN
Green: rescue > RESCUE_MIN
Red: rescue < -RESCUE_MIN
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# =========================
# CONFIG (EDIT THESE)
# =========================
PRED_DIR = Path("runs/final_eval_strict_v3/20260224_143439/predictions")
FEATURES_2024 = Path("runs/final_eval_strict_v3/20260224_143439/feature_tables/features_tabular_common_test2024.csv")

SEED = 0
INV_MODE = "log1p"  # "log" or "log1p"

# Two reference baselines
REF_BASELINES = {
    "Stats+Time": f"predictions_Baseline_(StatsplusTime)_RandomForest_seed{SEED}.csv",
    "Stats+Time+Meta": f"predictions_Baseline_(StatsplusTimeplusMeta)_RandomForest_seed{SEED}.csv",
}

# Thresholds per reference baseline (match your earlier design)
THRESHOLDS = {
    "Stats+Time": {"BASE_ERR_MIN": 1_000_000, "RESCUE_MIN": 500_000},
    "Stats+Time+Meta": {"BASE_ERR_MIN": 700_000, "RESCUE_MIN": 500_000},
}

# Graph models
MODELS = {
    "RotatE":   f"predictions_RotatE_plus_Stats_RandomForest_seed{SEED}.csv",
    "Node2Vec": f"predictions_Node2Vec_plus_Stats_RandomForest_seed{SEED}.csv",
    "V1":       f"predictions_V1_plus_Stats_RandomForest_seed{SEED}.csv",
    "V2_Ind":   f"predictions_V2_Ind_plus_Stats_RandomForest_seed{SEED}.csv",
    "V2_Trans": f"predictions_V2_Trans_plus_Stats_RandomForest_seed{SEED}.csv",
    "V2_Full":  f"predictions_V2_Full_MG_plus_Stats_RandomForest_seed{SEED}.csv",
}

# Output root
OUT_ROOT = PRED_DIR / "rescue_characteristics_2024"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Save eligible set too?
SAVE_ELIGIBLE = True

# Paper-table selection rule (edit freely)
PAPER_TOP_K = 8
PAPER_P_MAX = 0.10
PAPER_DELTA_MIN = 0.25  # use 0.25 if you want more rows


# =========================
# Utilities
# =========================
def inv_log(x: np.ndarray) -> np.ndarray:
    """Inverse transform and clamp to non-negative."""
    if INV_MODE == "log1p":
        val = np.expm1(x)
    elif INV_MODE == "log":
        val = np.exp(x)
    else:
        raise ValueError("INV_MODE must be 'log' or 'log1p'")
    return np.maximum(val, 0.0)


def fmt_money(x: float) -> str:
    return f"${x/1e6:.2f}M"


def load_pred(path: Path) -> pd.DataFrame:
    """Load prediction CSV and standardize columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(path)
    required = ["player_id", "season", "y_true", "y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    if "player_name" not in df.columns:
        df["player_name"] = "Unknown"

    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)

    df["salary_true_usd"] = inv_log(df["y_true"].to_numpy())
    df["salary_pred_usd"] = inv_log(df["y_pred"].to_numpy())

    return df[["player_id", "season", "player_name", "salary_true_usd", "salary_pred_usd"]]


# --- Cliff's Delta (vectorized; safe casting done upstream) ---
def cliffs_delta(x: np.ndarray, y: np.ndarray, max_pairs: int = 2_000_000) -> float:
    """
    Cliff's Delta in [-1, 1] computed as mean(sign(x_i - y_j)).
    Uses broadcasting; guarded by max_pairs for safety.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("nan")

    if n * m > max_pairs:
        warnings.warn(
            f"cliffs_delta: n*m={n*m} exceeds max_pairs={max_pairs}; "
            "broadcast matrix may be heavy. Consider rank-based implementation if scaling up.",
            RuntimeWarning
        )

    mat = np.sign(x[:, None] - y)  # -1,0,1
    return float(mat.mean())


# --- Mann-Whitney U robust wrapper ---
def mannwhitney_p(g: np.ndarray, r: np.ndarray) -> float:
    """
    Handles ties more robustly by forcing normal approximation (exact=False) when supported.
    Backward compatible with older SciPy versions.
    """
    try:
        return float(mannwhitneyu(g, r, alternative="two-sided",
                                  use_continuity=True, exact=False).pvalue)
    except TypeError:
        return float(mannwhitneyu(g, r, alternative="two-sided",
                                  use_continuity=True).pvalue)


def profile_table(green: pd.DataFrame, red: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """
    Compare green vs red for each feature:
    - means
    - mean diff
    - Cliff's Delta effect size
    - Mann-Whitney p-value
    Sorted by |delta| desc then p asc.
    """
    rows = []
    for f in feat_cols:
        if f not in green.columns or f not in red.columns:
            continue

        # BULLETPROOF numeric casting (fix bool/object contamination)
        g = pd.to_numeric(green[f], errors="coerce").dropna().astype(float).to_numpy()
        r = pd.to_numeric(red[f], errors="coerce").dropna().astype(float).to_numpy()

        if len(g) < 10 or len(r) < 10:
            continue

        p = mannwhitney_p(g, r)
        cd = cliffs_delta(g, r)

        rows.append({
            "feature": f,
            "green_mean": float(np.mean(g)),
            "red_mean": float(np.mean(r)),
            "diff(g-r)": float(np.mean(g) - np.mean(r)),
            "cliffs_delta": float(cd),
            "abs_cliffs_delta": float(abs(cd)),
            "p_value": float(p),
            "n_green": int(len(g)),
            "n_red": int(len(r)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["abs_cliffs_delta", "p_value"], ascending=[False, True]).drop(columns=["abs_cliffs_delta"])


def sanitize_name(s: str) -> str:
    return s.replace("+", "_").replace(" ", "").replace("-", "_")


def build_paper_table(all_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Build a paper-ready Top-K traits table from concatenated profiles across models.
    Rule: p <= PAPER_P_MAX and |delta| >= PAPER_DELTA_MIN, then top K by |delta| per model.
    """
    if all_profiles.empty:
        return all_profiles

    df = all_profiles.copy()
    df["abs_delta"] = df["cliffs_delta"].abs()
    df = df[(df["p_value"] <= PAPER_P_MAX) & (df["abs_delta"] >= PAPER_DELTA_MIN)].copy()
    if df.empty:
        return df

    df["direction"] = np.where(df["diff(g-r)"] >= 0, "Green>Red", "Green<Red")

    # top-k per model
    df = df.sort_values(["Model", "abs_delta", "p_value"], ascending=[True, False, True])
    df = df.groupby("Model", as_index=False).head(PAPER_TOP_K)

    # pretty column order
    cols = ["Model", "feature", "direction", "green_mean", "red_mean", "diff(g-r)", "cliffs_delta", "p_value", "n_green", "n_red"]
    return df[cols].reset_index(drop=True)


# =========================
# Main
# =========================
def main():
    print("=== Rescue Common Characteristics (2024) [Dual Baselines] ===")
    print(f"PRED_DIR: {PRED_DIR}")
    print(f"FEATURES_2024: {FEATURES_2024}")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print("-" * 90)

    # 1) Load features (2024 matched population)
    if not FEATURES_2024.exists():
        raise FileNotFoundError(f"Missing FEATURES_2024: {FEATURES_2024}")

    feat = pd.read_csv(FEATURES_2024)
    feat["player_id"] = feat["player_id"].astype(str)
    feat["season"] = feat["season"].astype(int)

    # Choose profiling features: numeric-like, excluding IDs and target.
    drop_cols = {"player_id", "season", "player_name", "log_salary"}
    feat_cols = []
    for c in feat.columns:
        if c in drop_cols:
            continue
        # keep numeric + bool + anything numeric-like (we cast later in profile_table)
        # Here we only exclude pure strings that are clearly non-numeric
        if pd.api.types.is_numeric_dtype(feat[c]) or pd.api.types.is_bool_dtype(feat[c]):
            feat_cols.append(c)
        else:
            # try: if it can be mostly converted to numeric, keep it
            conv = pd.to_numeric(feat[c], errors="coerce")
            if conv.notna().mean() >= 0.80:  # 80% convertible -> treat as numeric-like
                feat_cols.append(c)

    if len(feat_cols) == 0:
        raise RuntimeError("No usable feature columns found for profiling.")

    print(f"Features loaded: rows={len(feat)}, cols={len(feat.columns)}")
    print(f"Feature cols for profiling (numeric-like): {len(feat_cols)}")
    print("-" * 90)

    # 2) Loop over reference baselines
    for ref_name, base_file in REF_BASELINES.items():
        if ref_name not in THRESHOLDS:
            raise KeyError(f"Missing thresholds for ref baseline: {ref_name}")

        base_err_min = THRESHOLDS[ref_name]["BASE_ERR_MIN"]
        rescue_min = THRESHOLDS[ref_name]["RESCUE_MIN"]

        print(f"\n=== Reference Baseline: {ref_name} ===")
        print(f"Baseline file: {base_file}")
        print(f"Thresholds: BASE_ERR_MIN={fmt_money(base_err_min)}, RESCUE_MIN={fmt_money(rescue_min)}")

        # Output directories per reference
        ref_dir = OUT_ROOT / sanitize_name(ref_name)
        sets_dir = ref_dir / "sets"
        prof_dir = ref_dir / "profiles"
        sets_dir.mkdir(parents=True, exist_ok=True)
        prof_dir.mkdir(parents=True, exist_ok=True)

        # Load baseline predictions for this reference
        base = load_pred(PRED_DIR / base_file).rename(columns={"salary_pred_usd": "pred_base"})
        base_core = base[["player_id", "season", "player_name", "salary_true_usd", "pred_base"]].copy()

        summary_rows = []
        profiles_concat = []

        # 3) For each model
        for model_name, fn in MODELS.items():
            model_path = PRED_DIR / fn
            if not model_path.exists():
                print(f"[{model_name}] ⚠️ missing prediction file -> skip: {model_path.name}")
                continue

            mdf = load_pred(model_path).rename(columns={"salary_pred_usd": "pred_model"})
            merged = base_core.merge(
                mdf[["player_id", "season", "pred_model"]],
                on=["player_id", "season"],
                how="inner"
            )

            merged["err_base"] = (merged["salary_true_usd"] - merged["pred_base"]).abs()
            merged["err_model"] = (merged["salary_true_usd"] - merged["pred_model"]).abs()
            merged["rescue"] = merged["err_base"] - merged["err_model"]

            eligible = merged[merged["err_base"] > base_err_min].copy()
            green = eligible[eligible["rescue"] > rescue_min].copy()
            red = eligible[eligible["rescue"] < -rescue_min].copy()

            # Merge features with indicator to diagnose coverage
            green = green.merge(feat, on=["player_id", "season"], how="left", indicator=True)
            red = red.merge(feat, on=["player_id", "season"], how="left", indicator=True)
            eligible_dbg = eligible.merge(feat, on=["player_id", "season"], how="left", indicator=True)

            green_merge_miss = int((green["_merge"] != "both").sum())
            red_merge_miss = int((red["_merge"] != "both").sum())
            eligible_merge_miss = int((eligible_dbg["_merge"] != "both").sum())

            green_missing_any = int(green.drop(columns=["_merge"]).isna().any(axis=1).sum()) if len(green) else 0
            red_missing_any = int(red.drop(columns=["_merge"]).isna().any(axis=1).sum()) if len(red) else 0

            # Drop indicator
            green = green.drop(columns=["_merge"])
            red = red.drop(columns=["_merge"])
            eligible_dbg = eligible_dbg.drop(columns=["_merge"])

            print(
                f"[{model_name}] eligible={len(eligible)} green={len(green)} red={len(red)} | "
                f"merge-miss: eligible={eligible_merge_miss}, green={green_merge_miss}, red={red_merge_miss} | "
                f"missing-any: green={green_missing_any}, red={red_missing_any}"
            )

            # Save sets
            green.to_csv(sets_dir / f"{model_name}__green.csv", index=False)
            red.to_csv(sets_dir / f"{model_name}__red.csv", index=False)
            if SAVE_ELIGIBLE:
                eligible_dbg.to_csv(sets_dir / f"{model_name}__eligible.csv", index=False)

            # Profile table
            prof = profile_table(green, red, feat_cols)
            prof_path = prof_dir / f"{model_name}__green_vs_red_profile.csv"
            prof.to_csv(prof_path, index=False)

            # Collect profile for paper-table building
            if not prof.empty:
                prof2 = prof.copy()
                prof2.insert(0, "Model", model_name)
                profiles_concat.append(prof2)

            # Summary row
            summary_rows.append({
                "RefBaseline": ref_name,
                "Model": model_name,
                "Eligible": int(len(eligible)),
                "Green": int(len(green)),
                "Red": int(len(red)),
                "Green_rate": float(len(green) / len(eligible)) if len(eligible) else 0.0,
                "Red_rate": float(len(red) / len(eligible)) if len(eligible) else 0.0,
                "Green_merge_miss": green_merge_miss,
                "Red_merge_miss": red_merge_miss,
                "Green_missing_any": green_missing_any,
                "Red_missing_any": red_missing_any,
                "Profile_rows": int(len(prof)),
                "Profile_path": str(prof_path),
            })

        # 4) Save summary + paper table for this reference baseline
        if summary_rows:
            df_sum = pd.DataFrame(summary_rows).sort_values(["Green_rate", "Red_rate"], ascending=[False, True])
            df_sum.to_csv(ref_dir / "summary__green_red_and_profile_coverage.csv", index=False)
            print(f"✅ Saved summary: {ref_dir / 'summary__green_red_and_profile_coverage.csv'}")

        # Paper table
        if profiles_concat:
            all_prof = pd.concat(profiles_concat, axis=0, ignore_index=True)
            paper_df = build_paper_table(all_prof)
            paper_path = ref_dir / "paper_table__top_traits.csv"
            paper_df.to_csv(paper_path, index=False)
            print(f"✅ Saved paper table: {paper_path}")
            if paper_df.empty:
                print("   (Paper table is empty under current thresholds; try PAPER_DELTA_MIN=0.25 or PAPER_P_MAX=0.10.)")
        else:
            print("⚠️ No profile rows to build paper table. Check if green/red sizes are too small.")

    print("\n🎉 Done. All outputs saved under:")
    print(f"   {OUT_ROOT}")


if __name__ == "__main__":
    main()