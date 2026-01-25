#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analysis_results_for_rotatE.py

Specialized Post-hoc analysis for RotatE (Complex Numbers) + Tabular.
FIXED: Ensures player_id is string in BOTH dataframes before merging.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor

# ===== path setup =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.get_just_oncourt import get_oncourt_cols

# ===== CONFIG: RotatE Path =====
EMB_PATH = Path("graph/embeddings/rotate_L1B_cpu_player_embeddings.csv") 
TAB_PATH = Path("data/processed/training_level1_full.csv")

TEST_SEASON = 2024
TARGET_COL = "log_salary"

# --- Complex Number Parsing Logic ---
def _detect_rotate_complex_columns(df_emb: pd.DataFrame):
    emb_cols = [c for c in df_emb.columns if c.startswith("e")]
    if not emb_cols: return []
    v = str(df_emb[emb_cols[0]].iloc[0])
    # Check for 'j', 'i', or parens which indicate complex strings
    if ("j" in v) or ("i" in v) or ("(" in v):
        return emb_cols
    return []

def _parse_complex_safe(x) -> complex:
    s = str(x).strip().replace("i", "j")
    if s.startswith("(") and s.endswith(")"): s = s[1:-1].strip()
    try: return complex(s)
    except: return complex(0)

def load_data():
    print(f"Loading Tabular: {TAB_PATH}")
    df_tab = pd.read_csv(TAB_PATH)
    
    # <--- FIX IS HERE: 强制转换为字符串 --->
    df_tab["player_id"] = df_tab["player_id"].astype(str)
    
    oncourt_cols = get_oncourt_cols(df_tab)
    oncourt_cols = [c for c in oncourt_cols if not c.lower().startswith(("award_", "injury_"))]
    
    print(f"Loading Embedding: {EMB_PATH}")
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Missing file: {EMB_PATH}")
        
    df_emb = pd.read_csv(EMB_PATH)
    
    # <--- This was already here, now both match --->
    df_emb["player_id"] = df_emb["player_id"].astype(str)
    
    # 1. Determine Merge Key (Static vs Dynamic)
    if "season" in df_emb.columns:
        # RotatE file usually doesn't have season, but just in case
        df_emb["season"] = df_emb["season"].astype(int)
        merge_keys = ["player_id", "season"]
    else:
        merge_keys = ["player_id"]
        
    # 2. Handle Complex Numbers
    complex_cols = _detect_rotate_complex_columns(df_emb)
    if complex_cols:
        print("Detected Complex Numbers (RotatE). Converting to Real/Imag...")
        # Convert columns to complex objects
        Z = df_emb[complex_cols].map(_parse_complex_safe).to_numpy()
        Z_re, Z_im = np.real(Z), np.imag(Z)
        
        re_cols = [f"{c}_re" for c in complex_cols]
        im_cols = [f"{c}_im" for c in complex_cols]
        
        # Reconstruct DataFrame
        df_meta = df_emb[merge_keys].reset_index(drop=True)
        df_re = pd.DataFrame(Z_re, columns=re_cols)
        df_im = pd.DataFrame(Z_im, columns=im_cols)
        df_emb_clean = pd.concat([df_meta, df_re, df_im], axis=1)
        emb_cols = re_cols + im_cols
    else:
        # Standard Floats
        emb_cols = [c for c in df_emb.columns if c.startswith("e")]
        df_emb_clean = df_emb
        
    # 3. Merge
    print(f"Merging on {merge_keys}...")
    df = df_tab.merge(df_emb_clean, on=merge_keys, how="inner")
    
    return df, oncourt_cols, emb_cols

def analyze_and_print(df, oncourt_cols, emb_cols):
    print("\n=== 1. Feature Importance (RotatE) ===")
    train = df[df["season"] < TEST_SEASON].fillna(0)
    
    X = train[oncourt_cols + emb_cols]
    y = train[TARGET_COL]
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    
    imps = rf.feature_importances_
    feat_names = np.array(oncourt_cols + emb_cols)
    
    # Aggregation
    is_emb = np.array([c in emb_cols for c in feat_names])
    total_emb = imps[is_emb].sum()
    total_tab = imps[~is_emb].sum()
    
    print(f"Total Importance (Tabular):   {total_tab:.4f}")
    print(f"Total Importance (Embedding): {total_emb:.4f}")
    print(f"Ratio (Emb/Tab): {total_emb/total_tab:.4f}")
    
    # Top 10
    indices = np.argsort(imps)[::-1][:10]
    print("\nTop 10 Features:")
    for idx in indices:
        tag = "[EMB]" if is_emb[idx] else "[TAB]"
        print(f"  {tag} {feat_names[idx]}: {imps[idx]:.4f}")
        
    # === Case Studies ===
    print("\n=== 2. Case Studies (RotatE Rescues) ===")
    test = df[df["season"] == TEST_SEASON].fillna(0)
    
    if len(test) == 0: return

    # Baseline Model
    rf_base = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_base.fit(train[oncourt_cols], train[TARGET_COL])
    pred_base = rf_base.predict(test[oncourt_cols])
    
    # Graph Model
    rf_graph = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_graph.fit(train[oncourt_cols + emb_cols], train[TARGET_COL])
    pred_graph = rf_graph.predict(test[oncourt_cols + emb_cols])
    
    # Results
    cols = ["player_id", "season", TARGET_COL]
    if "player_name" in test.columns: cols.insert(2, "player_name")
    res = test[cols].copy()
    if "player_name" not in res.columns: res["player_name"] = res["player_id"].astype(str)
        
    res["err_base"] = (pred_base - res[TARGET_COL]).abs()
    res["err_graph"] = (pred_graph - res[TARGET_COL]).abs()
    res["improvement"] = res["err_base"] - res["err_graph"]
    
    # Filter: Baseline error > 0.3
    winners = res[res["err_base"] > 0.3].sort_values("improvement", ascending=False).head(10)
    
    print("\nTop 10 'Graph Rescues':")
    print(winners.to_string(index=False))

if __name__ == "__main__":
    try:
        df, oncourt, emb = load_data()
        analyze_and_print(df, oncourt, emb)
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()