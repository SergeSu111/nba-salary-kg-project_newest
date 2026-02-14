#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/evaluate_final_strict_v3_fixed.py

THE GRAND FINALE EVALUATION SCRIPT (FINAL PERFECTED VERSION).
-------------------------------------------
Updates:
1. Uses 'unique_player_id.csv' to recover player names (Critical for concrete examples).
2. Restores 'player_name' to ID_COLS so output CSVs have names.
3. Keeps all strict evaluation protocols intact.

NEW (for on+off baseline):
4. Adds a strict, oracle-strength tabular baseline: Stats + Time + Meta (off-court tabular).
5. Ensures meta cols are numeric-only (so median imputer won't crash).
6. Does NOT change any embedding/graph settings (they remain Stats+Time(+Embeddings) only),
   so previous results are unaffected except for the added new baseline row.
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
VAL_SEASON = 2023
TARGET_COL = "log_salary"

# Now we have name mapping, keep player_name for outputs
ID_COLS = ["player_id", "season", "player_name"]

TIME_FEATS = ["age_now", "years_since_draft"]
SEEDS = [0, 1, 2, 3, 4]

EVAL_COLD_START_2024 = True
COLD_START_MIN_ROWS = 30

_META_KEYWORDS = ["team", "agent", "draft", "pick", "round", "market", "value"]
_FORBIDDEN_KEYWORDS = ["award_", "injury_"]

# 1. Base Data
TAB = Path("data/processed/training_level1_full.csv")

# Name mapping
PLAYER_ID_MAP = Path("data/raw_on_court/unique_player_id.csv")

# 2. Old Embeddings (CSV)
NODE2VEC = Path("graph/embeddings/node2vec_L1A_player_embeddings.csv")
ROTATE = Path("graph/embeddings/rotate_L1B_cpu_player_embeddings.csv")
V1_PLAYER = Path("graph/embeddings/gnn_v1_sage_player_embeddings.csv")
V2_IND = Path("graph/embeddings/gnn_v2(baseline)_sage_playerseason_inductive.csv")
V2_TRANS = Path("graph/embeddings/gnn_v2(baseline)_sage_playerseason_transductive.csv")

# 3. New V2 Full Embeddings (PT)
V2_FULL_SG_PT = Path("runs/v2_full_sg_rgcn_paper/20260201_200454/artifacts/node_embeddings_graph.pt")
V2_FULL_MG_PT = Path("runs/v2_full_mg_rgcn_paper/20260201_201328/artifacts/node_embeddings_graph.pt")

# Mappings for V2 Full Restore
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
# Data Loaders (FIXED WITH ID MAP + META NUMERIC SAFE)
# ---------------------------
def load_tabular(tabular_path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if not tabular_path.exists():
        raise FileNotFoundError(f"Tabular file not found: {tabular_path}")

    df = pd.read_csv(tabular_path)

    # === Merge player_name from unique_player_id.csv if missing ===
    if "player_name" not in df.columns:
        print("ℹ️  'player_name' missing. Merging from unique_player_id.csv...")
        if PLAYER_ID_MAP.exists():
            try:
                name_map = pd.read_csv(PLAYER_ID_MAP)
                # normalize columns: Player_id -> player_id, Player -> player_name
                name_map = name_map.rename(columns={"Player_id": "player_id", "Player": "player_name"})

                name_map["player_id"] = name_map["player_id"].astype(str)
                df["player_id"] = df["player_id"].astype(str)

                name_map = name_map.drop_duplicates(subset="player_id")
                df = df.merge(name_map[["player_id", "player_name"]], on="player_id", how="left")

                df["player_name"] = df["player_name"].fillna("Unknown")
                print(f"   > Merged names successfully. Unknowns: {df['player_name'].eq('Unknown').sum()}")
            except Exception as e:
                print(f"   ! Error reading ID Map: {e}. Using 'Unknown'.")
                df["player_name"] = "Unknown"
        else:
            print(f"   ! ID Map not found at {PLAYER_ID_MAP}. Using 'Unknown'.")
            df["player_name"] = "Unknown"

    # On-court columns from your project helper
    raw_oncourt = get_oncourt_cols(df)

    # Remove award_/injury_ from on-court and anything salary-like
    stats_cols = [
        c for c in raw_oncourt
        if not any(k in c.lower() for k in _FORBIDDEN_KEYWORDS) and "salary" not in c.lower()
    ]

    used_cols = set(stats_cols + ID_COLS + [TARGET_COL] + TIME_FEATS)
    potential_meta = [c for c in df.columns if c not in used_cols]

    meta_cols = [
        c for c in potential_meta
        if any(k in c.lower() for k in _META_KEYWORDS) and not any(k in c.lower() for k in _FORBIDDEN_KEYWORDS)
    ]

    # Explicit leakage columns (remove from both stats/meta)
    LEAKAGE_COLS = [
        "salary", "salary_usd", "salary_cap", "cap_hit", "salary_cap_ratio", "log_salary_cap_ratio",
        "salary_cap_equiv", "sign_trade_bonus", "incentive_likely", "incentive_unlikely"
    ]
    stats_cols = [c for c in stats_cols if c not in LEAKAGE_COLS]
    meta_cols = [c for c in meta_cols if c not in LEAKAGE_COLS]

    # Ensure time feats exist
    for tf in TIME_FEATS:
        if tf not in df.columns:
            df[tf] = 0.0

    # --- NEW: Keep meta cols numeric-only so median imputer won't crash ---
    # (team_abbr / agent_name-like strings should NOT be used here unless you add encoding)
    meta_cols = [c for c in meta_cols if pd.api.types.is_numeric_dtype(df[c])]

    keep_cols = ID_COLS + [TARGET_COL] + stats_cols + meta_cols + TIME_FEATS
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].dropna(subset=[TARGET_COL]).copy()

    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    for tf in TIME_FEATS:
        df[tf] = pd.to_numeric(df[tf], errors="coerce").fillna(0.0)

    return df, stats_cols, meta_cols

def impute_time_feats_inplace(df: pd.DataFrame, time_cols: List[str]):
    train_mask = df["season"] < TEST_SEASON
    for c in time_cols:
        if c in df.columns:
            med = df.loc[train_mask, c].median()
            if pd.isna(med):
                med = 0
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
        if emb.duplicated(subset=merge_keys).any():
            emb = emb.drop_duplicates(subset=merge_keys)
    else:
        merge_keys = ["player_id"]
        if emb.duplicated(subset="player_id").any():
            emb = emb.drop_duplicates(subset="player_id")

    emb_cols = [c for c in emb.columns if c.startswith("e")]
    if not emb_cols:
        raise ValueError(f"No embedding columns (e*) found in {emb_path}")

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
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} not found")

    ckpt = torch.load(pt_path, map_location="cpu")
    Z = None
    if torch.is_tensor(ckpt):
        Z = ckpt
    elif isinstance(ckpt, dict):
        if "node_embeddings" in ckpt:
            Z = ckpt["node_embeddings"]
            if isinstance(Z, dict) and "node_embeddings" in Z:
                Z = Z["node_embeddings"]

    if Z is None or not torch.is_tensor(Z):
        raise ValueError(f"Could not extract tensor from {pt_path}")

    Z = Z.detach().cpu().numpy()
    dim = Z.shape[1]

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
    cols = [f"e{i}" for i in range(dim)]
    df_emb = pd.DataFrame(E, columns=cols)

    out = pd.concat([br[["player_id", "season"]].reset_index(drop=True), df_emb], axis=1)
    return out, cols, ["player_id", "season"]

def get_test_season_players(emb_df: pd.DataFrame, merge_keys: List[str]) -> Set[str]:
    if "season" in merge_keys:
        return set(emb_df.loc[emb_df["season"] == TEST_SEASON, "player_id"].astype(str))
    else:
        return set(emb_df["player_id"].astype(str))


# ---------------------------
# Models (Strict Protocol)
# ---------------------------
def run_models_strict(
    train_full: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
    out_dir: Optional[Path] = None,
    setting_name: str = ""
) -> List[Dict]:

    train_combined = pd.concat([train_full, val], axis=0)

    imp_search = SimpleImputer(strategy="median")
    X_train_search = imp_search.fit_transform(train_full[feature_cols].values)
    y_train_search = train_full[TARGET_COL].values
    X_val_search = imp_search.transform(val[feature_cols].values)
    y_val_search = val[TARGET_COL].values

    imp_final = SimpleImputer(strategy="median")
    X_combined = imp_final.fit_transform(train_combined[feature_cols].values)
    y_combined = train_combined[TARGET_COL].values
    X_test = imp_final.transform(test[feature_cols].values)
    y_test = test[TARGET_COL].values

    results = []

    if out_dir:
        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        safe_setting = setting_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")

    # --- Model 1: Random Forest ---
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
        fname = f"predictions_{safe_setting}_RandomForest_seed{seed}.csv"
        res_df.to_csv(pred_dir / fname, index=False)

    # --- Model 2: XGBoost ---
    xgb_search = xgb.XGBRegressor(
        n_estimators=5000, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1, random_state=seed, early_stopping_rounds=100
    )

    if len(X_val_search) < 20:
        best_n = 500
    else:
        xgb_search.fit(
            X_train_search, y_train_search,
            eval_set=[(X_val_search, y_val_search)],
            verbose=False
        )
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
        fname = f"predictions_{safe_setting}_XGBoost_seed{seed}.csv"
        res_df.to_csv(pred_dir / fname, index=False)

    return results


# ---------------------------
# Evaluation Logic
# ---------------------------
def evaluate_setting(
    df_base: pd.DataFrame,
    stats_cols: List[str], meta_cols: List[str], time_cols: List[str],
    setting_name: str,
    seed: int,
    use_emb: bool,
    baseline_test_keys: Set[Tuple[str, int]],
    emb_cache: Optional[pd.DataFrame] = None,
    emb_cols: Optional[List[str]] = None,
    merge_keys: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    use_meta: bool = False,   # NEW
) -> List[Dict]:

    df = df_base.copy()
    current_emb_cols: List[str] = []

    if use_emb:
        assert emb_cache is not None
        assert emb_cols is not None
        assert merge_keys is not None
        df = df.merge(emb_cache, on=merge_keys, how="inner")
        current_emb_cols = emb_cols
        df[current_emb_cols] = df[current_emb_cols].fillna(0)

    df = impute_time_feats_inplace(df, time_cols)
    train, val, test = split_time_strict(df)

    # Strict Check: enforce identical test set keys for all settings
    current_test_keys = set(zip(test["player_id"].astype(str), test["season"].astype(int)))
    if current_test_keys != baseline_test_keys:
        missing = baseline_test_keys - current_test_keys
        extra = current_test_keys - baseline_test_keys
        msg = f"[{setting_name}] Test Set Mismatch! Missing={len(missing)}, Extra={len(extra)}."
        raise ValueError(msg)

    feats = stats_cols + time_cols
    if use_meta:
        feats += meta_cols
    if use_emb:
        feats += current_emb_cols

    out: List[Dict] = []

    # Overall
    res_list = run_models_strict(
        train, val, test, feats, seed,
        out_dir=out_dir, setting_name=setting_name
    )
    for r in res_list:
        r.update({"setting": setting_name, "seed": seed, "is_cold_start": False, "n_test": len(test)})
        out.append(r)

    # Cold Start
    if EVAL_COLD_START_2024:
        train_full_hist = pd.concat([train, val], axis=0)
        cold_test = cold_start_2024_subset(train_full_hist, test)

        if len(cold_test) >= COLD_START_MIN_ROWS:
            res_cold = run_models_strict(
                train, val, cold_test, feats, seed,
                out_dir=out_dir, setting_name=setting_name + "_Cold"
            )
            for r in res_cold:
                r.update({"setting": setting_name + " (Cold)", "seed": seed, "is_cold_start": True, "n_test": len(cold_test)})
                out.append(r)

    return out


# ---------------------------
# Main
# ---------------------------
def main():
    print("=== FINAL STRICT EVAL v3 (Fixed + On/Off Baseline): The Paper Protocol ===")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    _ensure_dir(out_dir)

    # 1. Load Tabular
    df_tab, stats_cols, meta_cols = load_tabular(TAB)

    # Persist column inventories for paper/appendix (helps answer: what is off-court?)
    try:
        (out_dir / "stats_cols_used.txt").write_text("\n".join(stats_cols), encoding="utf-8")
        (out_dir / "meta_cols_used_numeric.txt").write_text("\n".join(meta_cols), encoding="utf-8")
    except Exception as e:
        print(f"[Warn] Could not write col inventories: {e}")

    print(f"[Tabular] stats_cols={len(stats_cols)} meta_cols(numeric)={len(meta_cols)} time_feats={TIME_FEATS}")

    # 2. Register Embeddings
    emb_data: Dict[str, Tuple[pd.DataFrame, List[str], List[str]]] = {}

    # CSVs
    csv_paths = {
        "Node2Vec": NODE2VEC,
        "RotatE": ROTATE,
        "V1": V1_PLAYER,
        "V2_Ind": V2_IND,
        "V2_Trans": V2_TRANS
    }
    for name, p in csv_paths.items():
        if p.exists():
            print(f"Loading {name}...")
            emb_data[name] = load_embedding_csv(p, set(df_tab["player_id"]))
        else:
            print(f"[Skip] {name} missing")

    # PTs
    pt_paths = {
        "V2_Full_SG": V2_FULL_SG_PT,
        "V2_Full_MG": V2_FULL_MG_PT
    }
    for name, p in pt_paths.items():
        if "REPLACE" in str(p):
            continue
        if p.exists():
            try:
                print(f"Loading {name} (PT)...")
                emb_data[name] = build_v2_full_df(p, MASTER_MAPPING, BRIDGE)
            except Exception as e:
                print(f"[Err] {name}: {e}")
        else:
            print(f"[Skip] {name} missing")

    if not emb_data:
        raise ValueError("No embeddings loaded!")

    # 3. Compute Strict Intersection
    print("\nComputing Intersection (Players valid for 2024 Test)...")
    valid_test_players = set(df_tab.loc[df_tab["season"] == TEST_SEASON, "player_id"].astype(str))

    for name, (edf, _, keys) in emb_data.items():
        players_in_emb = get_test_season_players(edf, keys)
        before = len(valid_test_players)
        valid_test_players = valid_test_players.intersection(players_in_emb)
        print(f" > After {name}: {len(valid_test_players)} (-{before - len(valid_test_players)})")

    if len(valid_test_players) == 0:
        raise ValueError("Intersection resulted in 0 test players!")

    df_common = df_tab[df_tab["player_id"].isin(valid_test_players)].copy()
    base_test_rows = df_common[df_common["season"] == TEST_SEASON]
    BASELINE_TEST_KEYS = set(zip(base_test_rows["player_id"].astype(str), base_test_rows["season"].astype(int)))
    print(f"Target Test Rows: {len(BASELINE_TEST_KEYS)}")

    # 4. Run Loop
    results: List[Dict] = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # Baseline (unchanged)
        results += evaluate_setting(
            df_common, stats_cols, meta_cols, TIME_FEATS,
            setting_name="Baseline (Stats+Time)", seed=seed,
            use_emb=False, baseline_test_keys=BASELINE_TEST_KEYS,
            out_dir=out_dir,
            use_meta=False
        )

        # NEW: On-court + Off-court tabular baseline (numeric meta only)
        results += evaluate_setting(
            df_common, stats_cols, meta_cols, TIME_FEATS,
            setting_name="Baseline (Stats+Time+Meta)", seed=seed,
            use_emb=False, baseline_test_keys=BASELINE_TEST_KEYS,
            out_dir=out_dir,
            use_meta=True
        )

        # All Embeddings (UNCHANGED behavior: Stats+Time + Embeddings only)
        for name, (edf, cols, keys) in emb_data.items():
            edf_small = edf[edf["player_id"].isin(valid_test_players)].copy()

            results += evaluate_setting(
                df_common, stats_cols, meta_cols, TIME_FEATS,
                setting_name=f"{name} + Stats", seed=seed,
                use_emb=True, emb_cache=edf_small, emb_cols=cols, merge_keys=keys,
                baseline_test_keys=BASELINE_TEST_KEYS,
                out_dir=out_dir,
                use_meta=False  # keep unchanged to avoid affecting prior model comparisons
            )

    # 5. Summarize
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
