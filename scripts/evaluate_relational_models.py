#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/evaluate_relational_models_v2.py

Paper-grade aligned evaluation including GNN V2 (Inductive/Transductive).
Supports both Static Embeddings (Node2Vec/RotatE) and Dynamic Embeddings (GNN V2).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

import numpy as np
import pandas as pd
import sys

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===== ensure project root on PYTHONPATH =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ============================================

from src.features.get_just_oncourt import get_oncourt_cols

# ---------------------------
# Config
# ---------------------------
TEST_SEASON = 2024
TARGET_COL = "log_salary"
ID_COLS = ["player_id", "season"]

TIME_FEATS = ["age_now", "years_since_draft"]
SEEDS = [0, 1, 2, 3, 4]

PAPER_MODE_REQUIRE_ALL = True  # set False if you want to skip missing files
EVAL_COLD_START_2024 = True
COLD_START_MIN_ROWS = 30
ALSO_RUN_FULL_BASELINE = False

_L0_BAD_KEYWORDS = ["team", "agent", "draft", "salary_cap", "award_", "injury_"]

# Default paths
TAB = Path("data/processed/training_level1_full.csv")

# ### <--- MODIFIED: Update paths to include your uploaded V2 files
NODE2VEC = Path("graph/embeddings/node2vec_L1A_player_embeddings.csv")
ROTATE = Path("graph/embeddings/rotate_L1B_cpu_player_embeddings.csv")
GNN_V0 = Path("graph/embeddings/gnn_v0_sage_player_embeddings.csv")
GNN_V1 = Path("graph/embeddings/gnn_v1_sage_player_embeddings.csv")

# V2 Paths (Ensure these match your actual file locations)
GNN_V2_IND = Path("graph/embeddings/gnn_v2(baseline)_sage_playerseason_inductive.csv")
GNN_V2_TRANS = Path("graph/embeddings/gnn_v2(baseline)_sage_playerseason_transductive.csv")


# ---------------------------
# Utilities
# ---------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def eval_reg(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def split_train_test(df: pd.DataFrame, test_season: int = TEST_SEASON) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"] < test_season].copy()
    test = df[df["season"] == test_season].copy()
    if len(test) == 0:
        raise ValueError(f"Test set empty for season=={test_season}. Check seasons.")
    return train, test


def overlap_ratio_train_test_players(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    train_players = set(train_df["player_id"].astype(str).unique())
    test_players = set(test_df["player_id"].astype(str).unique())
    if len(test_players) == 0:
        return float("nan")
    return len(train_players & test_players) / len(test_players)


def cold_start_2024_subset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_players = set(train_df["player_id"].astype(str).unique())
    return test_df[~test_df["player_id"].astype(str).isin(train_players)].copy()


# ---------------------------
# Loaders
# ---------------------------
def load_tabular(tabular_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not tabular_path.exists():
        raise FileNotFoundError(f"Tabular file not found: {tabular_path}")

    df = pd.read_csv(tabular_path)
    oncourt_cols = get_oncourt_cols(df)
    oncourt_cols = [c for c in oncourt_cols if not c.lower().startswith(("award_", "injury_"))]

    keep_cols = ID_COLS + [TARGET_COL] + oncourt_cols
    df = df[keep_cols].dropna(subset=[TARGET_COL]).copy()

    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)

    bad = [c for c in oncourt_cols if any(k in c.lower() for k in _L0_BAD_KEYWORDS)]
    if bad:
        raise ValueError(f"L0' contamination detected: {bad[:5]}")
        
    return df, oncourt_cols


def load_time_feats(tabular_path: Path, time_cols: List[str]) -> Optional[pd.DataFrame]:
    raw = pd.read_csv(tabular_path, nrows=5)
    missing = [c for c in time_cols if c not in raw.columns]
    if missing:
        print(f"[WARN] Time feats missing: {missing}")
        return None
    df = pd.read_csv(tabular_path, usecols=["player_id", "season"] + time_cols)
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    for c in time_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_time_feats(df: pd.DataFrame, time_df: pd.DataFrame, time_cols: List[str]) -> pd.DataFrame:
    out = df.merge(time_df, on=["player_id", "season"], how="left")
    train_mask = out["season"] < TEST_SEASON
    for c in time_cols:
        med = out.loc[train_mask, c].median()
        out[c] = out[c].fillna(med)
    return out


# ---------------------------
# Embedding Logic (Major Changes Here)
# ---------------------------
def get_player_set(emb_path: Path) -> Set[str]:
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    emb = pd.read_csv(emb_path, usecols=["player_id"])
    return set(emb["player_id"].astype(str))


def _detect_rotate_complex_columns(df_emb: pd.DataFrame) -> List[str]:
    emb_cols = [c for c in df_emb.columns if c.startswith("e")]
    if not emb_cols: return []
    v = str(df_emb[emb_cols[0]].iloc[0])
    if ("j" in v) or ("i" in v) or ("(" in v):
        return emb_cols
    return []


def _parse_complex_safe(x) -> complex:
    s = str(x).strip().replace("i", "j")
    if s.startswith("(") and s.endswith(")"): s = s[1:-1].strip()
    try: return complex(s)
    except: return complex(0)


def load_embedding_features(emb_path: Path, allowed_players: Set[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    ### <--- MODIFIED: Returns df, emb_cols, AND merge_keys
    Detects if embedding is Static (player_id) or Dynamic (player_id + season)
    """
    # Read header first to check for season column
    header = pd.read_csv(emb_path, nrows=0)
    has_season = "season" in header.columns
    
    # Load full file
    emb = pd.read_csv(emb_path)
    emb["player_id"] = emb["player_id"].astype(str)
    
    # Filter by allowed players (intersection set)
    emb = emb[emb["player_id"].isin(allowed_players)].copy()
    
    # Determine Merge Keys
    if has_season:
        emb["season"] = emb["season"].astype(int)
        merge_keys = ["player_id", "season"]
        # Check uniqueness for V2
        if emb.duplicated(subset=merge_keys).any():
             # Fallback: drop dupes if any (defensive)
             emb = emb.drop_duplicates(subset=merge_keys)
    else:
        merge_keys = ["player_id"]
        # Check uniqueness for V0/V1
        if emb.duplicated(subset="player_id").any():
            raise ValueError(f"Duplicate players in static embedding {emb_path}")

    # Detect Columns
    emb_cols = [c for c in emb.columns if c.startswith("e")]
    
    # Handle Complex Numbers (RotatE)
    rotate_cols = _detect_rotate_complex_columns(emb)
    if rotate_cols:
        Z = emb[rotate_cols].map(_parse_complex_safe).to_numpy() # updated map/apply
        Z_re, Z_im = np.real(Z), np.imag(Z)
        re_cols = [f"{c}_re" for c in rotate_cols]
        im_cols = [f"{c}_im" for c in rotate_cols]
        
        # Reconstruct DataFrame preserving keys
        meta_cols = merge_keys
        df_meta = emb[meta_cols].reset_index(drop=True)
        df_re = pd.DataFrame(Z_re, columns=re_cols)
        df_im = pd.DataFrame(Z_im, columns=im_cols)
        emb_num = pd.concat([df_meta, df_re, df_im], axis=1)
        return emb_num, re_cols + im_cols, merge_keys

    # Standard Floats
    emb_num = emb[merge_keys + emb_cols].copy()
    for c in emb_cols:
        emb_num[c] = pd.to_numeric(emb_num[c], errors="coerce") # safe cast
        
    return emb_num, emb_cols, merge_keys


def merge_tabular_and_embedding(df_tab: pd.DataFrame, emb_df: pd.DataFrame, emb_cols: List[str], merge_keys: List[str]) -> pd.DataFrame:
    """
    ### <--- MODIFIED: Uses dynamic merge_keys
    If keys=['player_id'], it broadcasts static emb to all seasons.
    If keys=['player_id', 'season'], it matches specific seasons (V2).
    """
    # Inner join enforces "Matched Information" principle
    df = df_tab.merge(emb_df, on=merge_keys, how="inner")
    
    if df[emb_cols].isna().any().any():
        print("[WARN] NaNs detected after merge. Filling with 0 (defensive).")
        df[emb_cols] = df[emb_cols].fillna(0)
        
    return df


# ---------------------------
# Models
# ---------------------------
def run_models(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], seed: int) -> List[Dict[str, float]]:
    Xtr = train_df[feature_cols].to_numpy()
    ytr = train_df[TARGET_COL].to_numpy(dtype=float)
    Xte = test_df[feature_cols].to_numpy()
    yte = test_df[TARGET_COL].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(Xtr)
    Xte = imputer.transform(Xte)

    out = []

    # Ridge
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    ridge.fit(Xtr, ytr)
    pred = ridge.predict(Xte)
    r = eval_reg(yte, pred)
    r["model"] = "Ridge"
    out.append(r)

    # RF
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    r = eval_reg(yte, pred)
    r["model"] = "RandomForest"
    out.append(r)

    return out


# ---------------------------
# Evaluation Wrapper
# ---------------------------
def evaluate_one_setting(
    df_base: pd.DataFrame,
    oncourt_cols: List[str],
    time_df: Optional[pd.DataFrame],
    time_cols: List[str],
    emb_path: Optional[Path],
    setting: str,
    use_oncourt: bool,
    use_emb: bool,
    use_time: bool,
    seed: int,
) -> List[Dict[str, float]]:
    df = df_base.copy()

    emb_cols = []
    if use_emb:
        assert emb_path is not None
        allowed_players = set(df["player_id"].astype(str))
        # ### <--- MODIFIED: unpack merge_keys
        emb_df, emb_cols, merge_keys = load_embedding_features(emb_path, allowed_players)
        df = merge_tabular_and_embedding(df, emb_df, emb_cols, merge_keys)

    if use_time:
        df = merge_time_feats(df, time_df, time_cols)

    train, test = split_train_test(df, TEST_SEASON)
    ov = overlap_ratio_train_test_players(train, test)

    feature_cols = []
    if use_oncourt: feature_cols += oncourt_cols
    if use_emb: feature_cols += emb_cols
    if use_time: feature_cols += time_cols

    rows = run_models(train, test, feature_cols, seed=seed)

    # Cold Start Logic
    cold_rows = []
    cold_ratio = float("nan")
    if EVAL_COLD_START_2024:
        cold = cold_start_2024_subset(train, test)
        n_cold = int(len(cold))
        if len(test) > 0: cold_ratio = n_cold / len(test)
        
        if n_cold >= COLD_START_MIN_ROWS:
            cold_eval = run_models(train, cold, feature_cols, seed=seed)
            for r in cold_eval:
                r.update({
                    "setting": setting + " (Cold-Start)", # Mark clearly
                    "seed": seed,
                    "n_test": n_cold,
                    "is_cold_start": True
                })
            cold_rows += cold_eval

    out_rows = []
    for r in rows:
        r.update({
            "setting": setting,
            "seed": seed,
            "n_test": int(len(test)),
            "overlap_ratio": ov,
            "cold_ratio": cold_ratio,
            "is_cold_start": False
        })
        out_rows.append(r)

    out_rows.extend(cold_rows)
    return out_rows


def summarize_mean_std(raw: pd.DataFrame) -> pd.DataFrame:
    # Group by setting/model and calc Mean/Std
    grp = raw.groupby(["setting", "model"], as_index=False)
    out = grp.agg(
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        cold_ratio=("cold_ratio", "mean")
    )
    
    def _pm(a, b): return f"{a:.4f} ± {b:.4f}"
    
    out["R2"] = out.apply(lambda r: _pm(r["R2_mean"], r["R2_std"]), axis=1)
    out["MAE"] = out.apply(lambda r: _pm(r["MAE_mean"], r["MAE_std"]), axis=1)
    out["RMSE"] = out.apply(lambda r: _pm(r["RMSE_mean"], r["RMSE_std"]), axis=1)
    
    return out[["setting", "model", "R2", "MAE", "RMSE", "cold_ratio"]].sort_values("setting")


def main():
    print("=== NBA Salary Prediction Evaluation Pipeline (V2 Ready) ===")
    
    df_tab, oncourt_cols = load_tabular(TAB)
    time_df = load_time_feats(TAB, TIME_FEATS)
    time_cols = TIME_FEATS if time_df is not None else []

    # ### <--- MODIFIED: Include V2 Inductive and Transductive here
    emb_paths = {
        "Node2Vec": NODE2VEC,   # Uncomment if needed
        "RotatE": ROTATE,       # Uncomment if needed
        "GNN_V0": GNN_V0,       # Uncomment if needed
        "GNN_V1": GNN_V1,
        "V2_Inductive": GNN_V2_IND,      # Your New File
        "V2_Transductive": GNN_V2_TRANS, # Your New File
    }

    # Filter available
    available = {k: p for k, p in emb_paths.items() if p.exists()}
    if not available:
        print("No embedding files found! Check paths.")
        return

    # Intersection of players (Common Sample)
    player_sets = {k: get_player_set(p) for k, p in available.items()}
    common_players = set.intersection(*player_sets.values())
    print(f"Common Players across all methods: {len(common_players)}")
    
    df_common = df_tab[df_tab["player_id"].isin(common_players)].copy()

    results = []
    
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        
        # 1. Baseline (L0')
        results.extend(evaluate_one_setting(
            df_common, oncourt_cols, time_df, time_cols, None,
            "Baseline (L0')", True, False, False, seed
        ))

        # 2. Embeddings
        for name, path in available.items():
            # L0' + Emb
            results.extend(evaluate_one_setting(
                df_common, oncourt_cols, time_df, time_cols, path,
                f"{name} + L0'", True, True, False, seed
            ))
            # Emb Only (Optional, to prove structure contains signal)
            results.extend(evaluate_one_setting(
                df_common, oncourt_cols, time_df, time_cols, path,
                f"{name} Only", False, True, False, seed
            ))

    # Save
    raw = pd.DataFrame(results)
    _ensure_dir(Path("results"))
    raw.to_csv("results/final_results_raw.csv", index=False)
    
    summary = summarize_mean_std(raw)
    summary.to_csv("results/final_results_summary.csv", index=False)
    
    print("\n=== Final Results Summary ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
