#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/evaluate_final_strict_v4_bulletproof.py

THE GRAND FINALE EVALUATION SCRIPT (BULLETPROOF VERSION).
-------------------------------------------
Prerequisite: 'training_level1_full.csv' must be CORRECT (no column shifts).

Updates (v4):
1. SAFETY: Strict assertions for empty intersection sets.
2. SAFETY: Enforces all stats_cols are numeric (prevents object/string crashes).
3. LOGIC: Strict Train-only imputation (Anti-Leakage).
4. LOGIC: Deep Copy for Cold Start (Anti-Double-Encoding).
5. SCOPE: Strict allowlist for categorical metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

import sys
import time
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
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
# Seasons: Train < 2023, Val == 2023, Test == 2024
TEST_SEASON = 2024
VAL_SEASON = 2023
TARGET_COL = "log_salary"

ID_COLS = ["player_id", "season", "player_name"]
TIME_FEATS = ["age_now", "years_since_draft"]
SEEDS = [0, 1, 2, 3, 4]

EVAL_COLD_START_2024 = True
COLD_START_MIN_ROWS = 30

# Explicit Allowlist for Categorical Encoding (Strict Control)
# Only these columns will be Ordinal Encoded for the Strong Baseline.
ALLOWED_CATS = {"team_abbr", "agent_name", "city", "state", "region"}

_FORBIDDEN_KEYWORDS = ["award_", "injury_", "salary"]
_META_KEYWORDS = ["team", "agent", "draft", "pick", "round", "market", "value", "city", "state"]

# 1. Base Data
TAB = Path("data/processed/training_level1_full.csv")
PLAYER_ID_MAP = Path("data/raw_on_court/unique_player_id.csv")

# 2. Embeddings
NODE2VEC = Path("graph/embeddings/node2vec_L1A_player_embeddings.csv")
ROTATE = Path("graph/embeddings/rotate_L1B_cpu_player_embeddings.csv")
V1_PLAYER = Path("graph/embeddings/gnn_v1_sage_player_embeddings.csv")
V2_IND = Path("graph/embeddings/gnn_v2(baseline)_sage_playerseason_inductive.csv")
V2_TRANS = Path("graph/embeddings/gnn_v2(baseline)_sage_playerseason_transductive.csv")
V2_FULL_SG_PT = Path("runs/v2_full_sg_rgcn_paper/20260201_200454/artifacts/node_embeddings_graph.pt")
V2_FULL_MG_PT = Path("runs/v2_full_mg_rgcn_paper/20260201_201328/artifacts/node_embeddings_graph.pt")

MASTER_MAPPING = Path("graph/mappings/master_node_id_to_idx.csv")
BRIDGE = Path("graph/mappings/playerSeason.csv")

OUT_ROOT = Path("runs/final_eval_strict_v3")


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

def split_time_strict(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = df["season"].astype(int)
    train = df[s < VAL_SEASON].copy()
    val = df[s == VAL_SEASON].copy()
    test = df[s == TEST_SEASON].copy()
    return train, val, test

def cold_start_2024_subset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_players = set(train_df["player_id"].astype(str).unique())
    return test_df[~test_df["player_id"].astype(str).isin(train_players)].copy()

def _parse_complex_safe(x) -> complex:
    s = str(x).strip().replace("i", "j")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    try:
        return complex(s)
    except Exception:
        return complex(0)


# ---------------------------
# Data Loaders (Clean Version + Safety Checks)
# ---------------------------
def load_tabular(tabular_path: Path) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    if not tabular_path.exists():
        raise FileNotFoundError(f"Tabular file not found: {tabular_path}")

    df = pd.read_csv(tabular_path)

    # 1. Merge Names if needed
    if "player_name" not in df.columns:
        print("ℹ️  'player_name' missing. Merging from unique_player_id.csv...")
        if PLAYER_ID_MAP.exists():
            try:
                name_map = pd.read_csv(PLAYER_ID_MAP)
                name_map = name_map.rename(columns={"Player_id": "player_id", "Player": "player_name"})
                name_map["player_id"] = name_map["player_id"].astype(str)
                df["player_id"] = df["player_id"].astype(str)
                name_map = name_map.drop_duplicates(subset="player_id")
                df = df.merge(name_map[["player_id", "player_name"]], on="player_id", how="left")
                df["player_name"] = df["player_name"].fillna("Unknown")
            except Exception as e:
                print(f"   ! Error reading ID Map: {e}")
                df["player_name"] = "Unknown"
        else:
            df["player_name"] = "Unknown"

    # 2. Identify Stats Cols (With Numeric Safety Check)
    raw_oncourt = get_oncourt_cols(df)
    stats_cols = [
        c for c in raw_oncourt
        if not any(k in c.lower() for k in _FORBIDDEN_KEYWORDS) 
        and "salary" not in c.lower()
    ]
    # ✅ Safety Check 1: Force numeric only
    stats_cols = [c for c in stats_cols if pd.api.types.is_numeric_dtype(df[c])]

    # 3. Identify Meta Cols (Numeric vs Categorical)
    used_cols = set(stats_cols + ID_COLS + [TARGET_COL] + TIME_FEATS)
    potential_meta = [c for c in df.columns if c not in used_cols]
    
    # Filter by keywords (team, agent, draft...) and NOT forbidden
    valid_meta = [
        c for c in potential_meta
        if any(k in c.lower() for k in _META_KEYWORDS)
        and not any(k in c.lower() for k in _FORBIDDEN_KEYWORDS)
        and c not in ["salary", "salary_usd"]
    ]

    meta_num = [c for c in valid_meta if pd.api.types.is_numeric_dtype(df[c])]
    
    # STRICT FILTER for Categorical: Only allow approved IDs
    meta_cat_candidates = [c for c in valid_meta if not pd.api.types.is_numeric_dtype(df[c])]
    meta_cat = [c for c in meta_cat_candidates if c in ALLOWED_CATS]

    print(f"   > Detected Meta (Numeric): {meta_num}")
    print(f"   > Detected Meta (Categorical - Approved): {meta_cat}")
    
    # 4. Final Cleanup
    keep_cols = ID_COLS + [TARGET_COL] + stats_cols + meta_num + meta_cat + TIME_FEATS
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    df = df[keep_cols].dropna(subset=[TARGET_COL]).copy()

    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    for tf in TIME_FEATS:
        df[tf] = pd.to_numeric(df[tf], errors="coerce").fillna(0.0)
    
    # Convert cat columns to string to ensure Encoder works
    for c in meta_cat:
        df[c] = df[c].astype(str)
    
    # Convert numeric columns to float
    for c in meta_num:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df, stats_cols, meta_num, meta_cat

def impute_time_feats_inplace(df: pd.DataFrame, time_cols: List[str]):
    # STRICT ANTI-LEAKAGE: Compute median ONLY from Train set (< VAL_SEASON)
    # Train = 2020, 2021, 2022. Val = 2023.
    train_mask = df["season"] < VAL_SEASON
    
    for c in time_cols:
        if c in df.columns:
            if train_mask.sum() > 0:
                med = df.loc[train_mask, c].median()
            else:
                med = df[c].median() # Fallback only if train empty
            if pd.isna(med): med = 0
            df[c] = df[c].fillna(med)
    return df


# ---------------------------
# Embedding Loaders
# ---------------------------
def load_embedding_csv(emb_path: Path, allowed_players: Set[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    emb = pd.read_csv(emb_path)
    emb["player_id"] = emb["player_id"].astype(str)
    emb = emb[emb["player_id"].isin(allowed_players)].copy()

    if "season" in emb.columns:
        emb["season"] = pd.to_numeric(emb["season"], errors="coerce").fillna(-1).astype(int)
        merge_keys = ["player_id", "season"]
        emb = emb.drop_duplicates(subset=merge_keys)
    else:
        merge_keys = ["player_id"]
        emb = emb.drop_duplicates(subset="player_id")

    emb_cols = [c for c in emb.columns if c.startswith("e")]
    
    head = emb[emb_cols[0]].head(5).astype(str).tolist()
    if any(("j" in s or "i" in s) for s in head):
        Z = emb[emb_cols].applymap(_parse_complex_safe).to_numpy()
        Z_re, Z_im = np.real(Z), np.imag(Z)
        re_cols = [f"{c}_re" for c in emb_cols]
        im_cols = [f"{c}_im" for c in emb_cols]
        meta = emb[merge_keys].reset_index(drop=True)
        out = pd.concat([meta, pd.DataFrame(Z_re, columns=re_cols), pd.DataFrame(Z_im, columns=im_cols)], axis=1)
        return out, re_cols + im_cols, merge_keys

    out = emb[merge_keys + emb_cols].copy()
    for c in emb_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out, emb_cols, merge_keys

def build_v2_full_df(pt_path: Path, mm_path: Path, br_path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if not pt_path.exists(): raise FileNotFoundError(f"{pt_path} not found")
    ckpt = torch.load(pt_path, map_location="cpu")
    Z = ckpt["node_embeddings"] if isinstance(ckpt, dict) and "node_embeddings" in ckpt else ckpt
    if isinstance(Z, dict): Z = Z["node_embeddings"]
    Z = Z.detach().cpu().numpy()
    
    mm = pd.read_csv(mm_path)
    mm.columns = [c.lower().strip() for c in mm.columns]
    node_map = dict(zip(mm["node_id"].astype(str), mm["idx"].astype(int)))

    br = pd.read_csv(br_path)
    br.columns = [c.lower().strip() for c in br.columns]
    node_col = next(c for c in br.columns if c in ["node_id", "element_id", "id"])
    br["player_id"] = br["player_id"].astype(str)
    br["season"] = pd.to_numeric(br["season"], errors="coerce").fillna(-1).astype(int)
    br["node_idx"] = br[node_col].astype(str).map(node_map)
    br = br.dropna(subset=["node_idx"]).copy()
    br["node_idx"] = br["node_idx"].astype(int)
    br = br[(br["node_idx"] >= 0) & (br["node_idx"] < Z.shape[0])].copy()

    E = Z[br["node_idx"].values]
    cols = [f"e{i}" for i in range(Z.shape[1])]
    out = pd.concat([br[["player_id", "season"]].reset_index(drop=True), pd.DataFrame(E, columns=cols)], axis=1)
    return out, cols, ["player_id", "season"]

def get_test_season_players(emb_df: pd.DataFrame, merge_keys: List[str]) -> Set[str]:
    if "season" in merge_keys:
        return set(emb_df.loc[emb_df["season"] == TEST_SEASON, "player_id"].astype(str))
    return set(emb_df["player_id"].astype(str))


# ---------------------------
# Models (SAFE Encoding + Copies)
# ---------------------------
def run_models_strict(
    train_full: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    seed: int,
    out_dir: Optional[Path] = None,
    setting_name: str = ""
) -> List[Dict]:

    # ✅ 1. PREVENT DOUBLE-ENCODING & MUTATION
    # Copy dataframes to ensure original string columns are preserved for subsequent runs (Cold Start)
    train_full = train_full.copy()
    val = val.copy()
    test = test.copy()

    # 2. Encode Categorical Features
    if len(cat_cols) > 0:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # Fit on Train
        train_full[cat_cols] = enc.fit_transform(train_full[cat_cols].astype(str))
        # Transform Val/Test
        val[cat_cols] = enc.transform(val[cat_cols].astype(str))
        test[cat_cols] = enc.transform(test[cat_cols].astype(str))

    # 3. Combine for Final Training
    # Standard practice: Train on Train+Val for final Test prediction
    train_combined = pd.concat([train_full, val], axis=0)

    # 4. Impute
    imp = SimpleImputer(strategy="median")
    X_combined = imp.fit_transform(train_combined[feature_cols].values)
    y_combined = train_combined[TARGET_COL].values
    X_test = imp.transform(test[feature_cols].values)
    y_test = test[TARGET_COL].values

    results = []
    if out_dir:
        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        safe_setting = setting_name.replace(" ", "_").replace("+", "plus")

    # RF
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=seed)
    rf.fit(X_combined, y_combined)
    pred_rf = rf.predict(X_test)
    res_rf = eval_reg(y_test, pred_rf)
    res_rf["model_type"] = "RandomForest"
    results.append(res_rf)
    
    if out_dir:
        res_df = test[ID_COLS].copy()
        res_df["y_true"] = y_test
        res_df["y_pred"] = pred_rf
        res_df.to_csv(pred_dir / f"predictions_{safe_setting}_RandomForest_seed{seed}.csv", index=False)

    # XGB
    xgb_search = xgb.XGBRegressor(
        n_estimators=5000, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1, random_state=seed, early_stopping_rounds=100
    )
    # Search/Validation setup (Use simple impute just for eval set)
    imp_s = SimpleImputer(strategy="median")
    X_train_s = imp_s.fit_transform(train_full[feature_cols].values)
    y_train_s = train_full[TARGET_COL].values
    X_val_s = imp_s.transform(val[feature_cols].values)
    y_val_s = val[TARGET_COL].values

    if len(X_val_s) < 20:
        best_n = 500
    else:
        xgb_search.fit(X_train_s, y_train_s, eval_set=[(X_val_s, y_val_s)], verbose=False)
        best_n = int(xgb_search.best_iteration) + 1

    xgb_final = xgb.XGBRegressor(
        n_estimators=best_n, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1, random_state=seed
    )
    xgb_final.fit(X_combined, y_combined)
    pred_xgb = xgb_final.predict(X_test)
    res_xgb = eval_reg(y_test, pred_xgb)
    res_xgb["model_type"] = "XGBoost"
    results.append(res_xgb)

    if out_dir:
        res_df = test[ID_COLS].copy()
        res_df["y_true"] = y_test
        res_df["y_pred"] = pred_xgb
        res_df.to_csv(pred_dir / f"predictions_{safe_setting}_XGBoost_seed{seed}.csv", index=False)

    return results


# ---------------------------
# Evaluation Logic
# ---------------------------
def evaluate_setting(
    df_base: pd.DataFrame,
    stats_cols: List[str], 
    meta_num: List[str], 
    meta_cat: List[str],
    time_cols: List[str],
    setting_name: str,
    seed: int,
    use_emb: bool,
    baseline_test_keys: Set[Tuple[str, int]],
    emb_cache: Optional[pd.DataFrame] = None,
    emb_cols: Optional[List[str]] = None,
    merge_keys: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    use_meta: bool = False,
) -> List[Dict]:

    df = df_base.copy()
    current_emb_cols: List[str] = []

    if use_emb:
        assert emb_cache is not None
        df = df.merge(emb_cache, on=merge_keys, how="inner")
        current_emb_cols = emb_cols
        df[current_emb_cols] = df[current_emb_cols].fillna(0)

    # Impute Time Feats (Strictly from Train)
    df = impute_time_feats_inplace(df, time_cols)
    
    train, val, test = split_time_strict(df)

    # Intersection Check (Matched Information Protocol)
    if set(zip(test["player_id"].astype(str), test["season"].astype(int))) != baseline_test_keys:
        raise ValueError(f"[{setting_name}] Test Set Mismatch!")

    feats = stats_cols + time_cols
    cat_feats = []

    if use_meta:
        feats += meta_num
        feats += meta_cat
        cat_feats = meta_cat
    
    if use_emb:
        feats += current_emb_cols

    out = []
    # 1. Warm Start
    res_list = run_models_strict(
        train, val, test, feats, cat_cols=cat_feats, seed=seed,
        out_dir=out_dir, setting_name=setting_name
    )
    for r in res_list:
        r.update({"setting": setting_name, "seed": seed, "is_cold_start": False})
        out.append(r)

    # 2. Cold Start (Safe now)
    if EVAL_COLD_START_2024:
        train_full_hist = pd.concat([train, val], axis=0)
        cold_test = cold_start_2024_subset(train_full_hist, test)
        if len(cold_test) >= COLD_START_MIN_ROWS:
            res_cold = run_models_strict(
                train, val, cold_test, feats, cat_cols=cat_feats, seed=seed,
                out_dir=out_dir, setting_name=setting_name + "_Cold"
            )
            for r in res_cold:
                r.update({"setting": setting_name + " (Cold)", "seed": seed, "is_cold_start": True})
                out.append(r)

    return out


# ---------------------------
# Main
# ---------------------------
def main():
    print("=== FINAL STRICT EVAL v4 (BULLETPROOF EDITION) ===")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    _ensure_dir(out_dir)

    # 1. Load Tabular
    df_tab, stats_cols, meta_num, meta_cat = load_tabular(TAB)
    print(f"[Tabular] Stats={len(stats_cols)} | MetaNum={len(meta_num)} | MetaCat={len(meta_cat)}")
    print(f"   Categorical columns used for Strong Baseline: {meta_cat}")

    # 2. Embeddings
    emb_data = {}
    csv_paths = { "Node2Vec": NODE2VEC, "RotatE": ROTATE, "V1": V1_PLAYER, "V2_Ind": V2_IND, "V2_Trans": V2_TRANS }
    for name, p in csv_paths.items():
        if p.exists():
            print(f"Loading {name}...")
            emb_data[name] = load_embedding_csv(p, set(df_tab["player_id"]))

    pt_paths = {"V2_Full_SG": V2_FULL_SG_PT, "V2_Full_MG": V2_FULL_MG_PT}
    for name, p in pt_paths.items():
        if p.exists():
            try:
                print(f"Loading {name} (PT)...")
                emb_data[name] = build_v2_full_df(p, MASTER_MAPPING, BRIDGE)
            except Exception as e:
                print(f"[Err] {name}: {e}")

    # 3. Intersection
    valid_test_players = set(df_tab.loc[df_tab["season"] == TEST_SEASON, "player_id"].astype(str))
    for name, (edf, _, keys) in emb_data.items():
        players_in_emb = get_test_season_players(edf, keys)
        valid_test_players = valid_test_players.intersection(players_in_emb)
    
    print(f"Final Common Test Players: {len(valid_test_players)}")
    
    # ✅ Safety Check 2: Assert Common Population Exists
    if len(valid_test_players) == 0:
        raise RuntimeError("❌ CRITICAL: No common test players found! Check embedding files for 2024 coverage.")

    df_common = df_tab[df_tab["player_id"].isin(valid_test_players)].copy()
    base_test_rows = df_common[df_common["season"] == TEST_SEASON]
    BASELINE_TEST_KEYS = set(zip(base_test_rows["player_id"].astype(str), base_test_rows["season"].astype(int)))

    # 4. Loop
    results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        
        # Weak Baseline (Stats + Time)
        results += evaluate_setting(
            df_common, stats_cols, meta_num, meta_cat, TIME_FEATS,
            setting_name="Baseline (Stats+Time)", seed=seed,
            use_emb=False, baseline_test_keys=BASELINE_TEST_KEYS, out_dir=out_dir,
            use_meta=False
        )

        # Strong Baseline (Stats + Time + Meta[Numeric+Encoded Cat])
        results += evaluate_setting(
            df_common, stats_cols, meta_num, meta_cat, TIME_FEATS,
            setting_name="Baseline (Stats+Time+Meta)", seed=seed,
            use_emb=False, baseline_test_keys=BASELINE_TEST_KEYS, out_dir=out_dir,
            use_meta=True
        )

        # Graph Models (vs Weak Info Set)
        for name, (edf, cols, keys) in emb_data.items():
            edf_small = edf[edf["player_id"].isin(valid_test_players)].copy()
            results += evaluate_setting(
                df_common, stats_cols, meta_num, meta_cat, TIME_FEATS,
                setting_name=f"{name} + Stats", seed=seed,
                use_emb=True, emb_cache=edf_small, emb_cols=cols, merge_keys=keys,
                baseline_test_keys=BASELINE_TEST_KEYS, out_dir=out_dir,
                use_meta=False
            )

    # 5. Output
    df_res = pd.DataFrame(results)
    grp = df_res.groupby(["setting", "model_type"], as_index=False)
    summ = grp.agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std")
    ).sort_values(["model_type", "RMSE_mean"])

    df_res.to_csv(out_dir / "raw_results.csv", index=False)
    summ.to_csv(out_dir / "summary_results.csv", index=False)
    print("\n=== FINAL RANKING (RMSE Ascending) ===")
    print(summ.to_string(index=False))
    print(f"\nSaved to {out_dir}")

if __name__ == "__main__":
    main()