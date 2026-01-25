#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/evaluate_v2_strongest.py

THE FINAL BOSS: V2 Strongest Experiment.
Fixes: Removed redundant merge that caused KeyError 'age_now'.
Now processes time features directly within the dataframe.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

import numpy as np
import pandas as pd
import sys

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# === THE STRONGEST FEATURES ===
TIME_FEATS = ["age_now", "years_since_draft"] 

SEEDS = [0, 1, 2, 3, 4]

PAPER_MODE_REQUIRE_ALL = True  
EVAL_COLD_START_2024 = True
COLD_START_MIN_ROWS = 30

_META_KEYWORDS = ["team", "agent", "draft", "pick", "round", "market", "value"]
_FORBIDDEN_KEYWORDS = ["award_", "injury_"] 

# Paths
TAB = Path("data/processed/training_level1_full.csv")
NODE2VEC = Path("graph/embeddings/node2vec_L1A_player_embeddings.csv")
ROTATE = Path("graph/embeddings/rotate_L1B_cpu_player_embeddings.csv")
GNN_V0 = Path("graph/embeddings/gnn_v0_sage_player_embeddings.csv")
GNN_V1 = Path("graph/embeddings/gnn_v1_sage_player_embeddings.csv")
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
    return train, test

def cold_start_2024_subset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_players = set(train_df["player_id"].astype(str).unique())
    return test_df[~test_df["player_id"].astype(str).isin(train_players)].copy()

# ---------------------------
# Loaders
# ---------------------------
def load_tabular(tabular_path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if not tabular_path.exists():
        raise FileNotFoundError(f"Tabular file not found: {tabular_path}")

    df = pd.read_csv(tabular_path)
    
    # 1. On-court Stats
    raw_oncourt = get_oncourt_cols(df)
    stats_cols = [c for c in raw_oncourt 
                  if not any(k in c.lower() for k in _FORBIDDEN_KEYWORDS)
                  and "salary" not in c.lower()] 
    
    # 2. Meta Cols
    used_cols = set(stats_cols + ID_COLS + [TARGET_COL] + TIME_FEATS)
    potential_meta = [c for c in df.columns if c not in used_cols]
    meta_cols = [c for c in potential_meta 
                 if any(k in c.lower() for k in _META_KEYWORDS)
                 and not any(k in c.lower() for k in _FORBIDDEN_KEYWORDS)]
    
    # 3. Leakage Removal
    LEAKAGE_COLS = [
        "salary", "salary_usd", "salary_cap", "cap_hit",
        "salary_cap_ratio", "log_salary_cap_ratio", "salary_cap_equiv",
        "sign_trade_bonus", "incentive_likely", "incentive_unlikely"
    ]
    stats_cols = [c for c in stats_cols if c not in LEAKAGE_COLS]
    meta_cols = [c for c in meta_cols if c not in LEAKAGE_COLS]
    
    # 4. Filter DF (Ensure TIME_FEATS are included here!)
    keep_cols = ID_COLS + [TARGET_COL] + stats_cols + meta_cols + TIME_FEATS
    
    # Defensive check: ensure time feats exist
    for tf in TIME_FEATS:
        if tf not in df.columns:
            print(f"[WARN] Time feature '{tf}' not found in CSV. Creating dummy.")
            df[tf] = 0.0

    df = df[keep_cols].dropna(subset=[TARGET_COL]).copy()
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)

    # 5. Encode Strings
    le = LabelEncoder()
    for col in meta_cols + stats_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown").astype(str)
            df[col] = le.fit_transform(df[col])

    # 6. Ensure Time Features are Numeric
    for tf in TIME_FEATS:
        df[tf] = pd.to_numeric(df[tf], errors='coerce').fillna(0.0)

    return df, stats_cols, meta_cols

# REPLACED: No more separate load/merge for time feats.
# Instead, we just impute them in place if needed.
def impute_time_feats(df: pd.DataFrame, time_cols: List[str]) -> pd.DataFrame:
    # Just simple median fill if any NaNs remain
    train_mask = df["season"] < TEST_SEASON
    for c in time_cols:
        if c in df.columns:
            med = df.loc[train_mask, c].median()
            df[c] = df[c].fillna(med)
    return df

# ---------------------------
# Embedding Logic
# ---------------------------
def get_player_set(emb_path: Path) -> Set[str]:
    emb = pd.read_csv(emb_path, usecols=["player_id"])
    return set(emb["player_id"].astype(str))

def _parse_complex_safe(x) -> complex:
    s = str(x).strip().replace("i", "j")
    if s.startswith("(") and s.endswith(")"): s = s[1:-1].strip()
    try: return complex(s)
    except: return complex(0)

def load_embedding_features(emb_path: Path, allowed_players: Set[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    emb = pd.read_csv(emb_path)
    emb["player_id"] = emb["player_id"].astype(str)
    emb = emb[emb["player_id"].isin(allowed_players)].copy()
    
    if "season" in emb.columns:
        emb["season"] = emb["season"].astype(int)
        merge_keys = ["player_id", "season"]
        if emb.duplicated(subset=merge_keys).any(): emb = emb.drop_duplicates(subset=merge_keys)
    else:
        merge_keys = ["player_id"]
        if emb.duplicated(subset="player_id").any(): raise ValueError(f"Duplicate players in static embedding")

    emb_cols = [c for c in emb.columns if c.startswith("e")]
    
    # RotatE check
    if any("j" in str(x) for x in emb[emb_cols[0]].head(5)):
        rotate_cols = emb_cols
        Z = emb[rotate_cols].map(_parse_complex_safe).to_numpy()
        Z_re, Z_im = np.real(Z), np.imag(Z)
        re_cols = [f"{c}_re" for c in rotate_cols]
        im_cols = [f"{c}_im" for c in rotate_cols]
        
        meta_df = emb[merge_keys].reset_index(drop=True)
        df_re = pd.DataFrame(Z_re, columns=re_cols)
        df_im = pd.DataFrame(Z_im, columns=im_cols)
        emb_num = pd.concat([meta_df, df_re, df_im], axis=1)
        return emb_num, re_cols + im_cols, merge_keys

    emb_num = emb[merge_keys + emb_cols].copy()
    for c in emb_cols:
        emb_num[c] = pd.to_numeric(emb_num[c], errors="coerce")
    return emb_num, emb_cols, merge_keys

def merge_tabular_and_embedding(df_tab: pd.DataFrame, emb_df: pd.DataFrame, emb_cols: List[str], merge_keys: List[str]) -> pd.DataFrame:
    df = df_tab.merge(emb_df, on=merge_keys, how="inner")
    df[emb_cols] = df[emb_cols].fillna(0)
    return df

# ---------------------------
# Models
# ---------------------------
def run_models(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], seed: int, setting_name: str = "") -> List[Dict[str, float]]:
    Xtr = train_df[feature_cols].to_numpy()
    ytr = train_df[TARGET_COL].to_numpy(dtype=float)
    Xte = test_df[feature_cols].to_numpy()
    yte = test_df[TARGET_COL].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(Xtr)
    Xte = imputer.transform(Xte)
    
    # Truth Check (Once)
    if seed == 0 and "Baseline" in setting_name and "Cold" not in setting_name:
        print(f"\n[TRUTH CHECK] {setting_name}: Features={len(feature_cols)}")
        if any(k in str(feature_cols).lower() for k in ['years', 'age']):
            print("✅ Time/Age Features Detected.")
        else:
            print("⚠️ No Time Features found.")

    out = []
    # RF Only
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    r = eval_reg(yte, pred)
    r["model"] = "RandomForest"
    out.append(r)
    return out

# ---------------------------
# Eval Wrapper
# ---------------------------
def evaluate_one_setting(
    df_base: pd.DataFrame,
    stats_cols: List[str],      
    meta_cols: List[str],       
    time_cols: List[str],
    emb_path: Optional[Path],
    setting: str,
    use_stats: bool,
    use_meta: bool,     
    use_emb: bool,
    use_time: bool,
    seed: int,
) -> List[Dict[str, float]]:
    df = df_base.copy()

    # 1. Prepare Embeddings
    emb_cols = []
    if use_emb and emb_path:
        allowed = set(df["player_id"].astype(str))
        emb_df, emb_cols, merge_keys = load_embedding_features(emb_path, allowed)
        df = merge_tabular_and_embedding(df, emb_df, emb_cols, merge_keys)

    # 2. Prepare Time (Impute only, no merge)
    if use_time:
        df = impute_time_feats(df, time_cols)

    # 3. Split
    train, test = split_train_test(df, TEST_SEASON)
    
    # 4. Feature Assembly
    feature_cols = []
    if use_stats: feature_cols += stats_cols
    if use_meta:  feature_cols += meta_cols 
    if use_emb:   feature_cols += emb_cols
    if use_time:  feature_cols += time_cols 

    rows = run_models(train, test, feature_cols, seed=seed, setting_name=setting)

    # 5. Cold Start
    if EVAL_COLD_START_2024:
        cold = cold_start_2024_subset(train, test)
        n_cold = int(len(cold))
        if n_cold >= COLD_START_MIN_ROWS:
            cold_eval = run_models(train, cold, feature_cols, seed=seed, setting_name=setting + " (Cold)")
            for r in cold_eval:
                r.update({"setting": setting + " (Cold)", "is_cold_start": True})
            rows += cold_eval

    for r in rows:
        if "setting" not in r: r.update({"setting": setting, "is_cold_start": False})
        r["seed"] = seed
    return rows

def summarize(raw):
    grp = raw.groupby(["setting", "model"], as_index=False)
    return grp.agg(R2_mean=("R2", "mean"), R2_std=("R2", "std")).sort_values("R2_mean", ascending=False)

def main():
    print("=== V2 STRONGEST: The Final Test ===")
    
    df_tab, stats_cols, meta_cols = load_tabular(TAB)
    time_cols = TIME_FEATS # Already in df_tab
    
    emb_paths = {
        "RotatE": ROTATE,
        "V2_Inductive": GNN_V2_IND, 
    }
    
    # Intersection
    player_sets = {k: get_player_set(p) for k, p in emb_paths.items()}
    common = set.intersection(*player_sets.values())
    df_common = df_tab[df_tab["player_id"].isin(common)].copy()
    print(f"Common Players: {len(common)}")

    results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        
        # 1. Baseline: Stats + Time (No Meta, just pure stats + age)
        results.extend(evaluate_one_setting(
            df_common, stats_cols, meta_cols, time_cols, None,
            "Baseline (Stats + Time)", 
            use_stats=True, use_meta=False, use_emb=False, use_time=True, 
            seed=seed
        ))

        # 2. RotatE: Stats + Time + Emb (Can V2 beat this?)
        results.extend(evaluate_one_setting(
            df_common, stats_cols, meta_cols, time_cols, ROTATE,
            "RotatE + Stats + Time", 
            use_stats=True, use_meta=False, use_emb=True, use_time=True, 
            seed=seed
        ))

        # 3. V2 Strongest: Stats + Time + Emb (The Challenger)
        results.extend(evaluate_one_setting(
            df_common, stats_cols, meta_cols, time_cols, GNN_V2_IND,
            "V2 Inductive + Stats + Time", 
            use_stats=True, use_meta=False, use_emb=True, use_time=True, 
            seed=seed
        ))

    summary = summarize(pd.DataFrame(results))
    print("\n=== V2 STRONGEST RESULTS ===")
    print(summary.to_string())

if __name__ == "__main__":
    main()