#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analyze_v2_drivers.py

POST-HOC ANALYSIS: V2 Strongest
Purpose: Why did V2 Inductive fail on Cold Start?
Method: Inspect Feature Importances to see if the model over-relied on noisy embeddings.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Config
TEST_SEASON = 2024
TARGET_COL = "log_salary"
ID_COLS = ["player_id", "season"]
TIME_FEATS = ["age_now", "years_since_draft"]
_META_KEYWORDS = ["team", "agent", "draft", "pick", "round", "market", "value"]
_FORBIDDEN_KEYWORDS = ["award_", "injury_"] 

# Paths
TAB = Path("data/processed/training_level1_full.csv")
GNN_V2_IND = Path("graph/embeddings/gnn_v2(baseline)_sage_playerseason_inductive.csv")

# ---------------------------
# Loaders (Simplified for Analysis)
# ---------------------------
def get_oncourt_cols(df):
    # Minimal version of your getter to ensure compatibility
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cols = df.select_dtypes(include=numerics).columns.tolist()
    exclude = ID_COLS + [TARGET_COL] + ['salary', 'salary_cap']
    return [c for c in cols if c not in exclude and not any(k in c.lower() for k in _FORBIDDEN_KEYWORDS)]

def load_data():
    print(">>> Loading Data...")
    df = pd.read_csv(TAB)
    
    # === FIX: Ensure player_id is string immediately ===
    df["player_id"] = df["player_id"].astype(str)
    
    # 1. Stats
    raw_oncourt = get_oncourt_cols(df)
    # Strict Leakage Removal
    LEAKAGE = ["salary", "salary_usd", "salary_cap", "cap_hit", "salary_cap_ratio", 
               "log_salary_cap_ratio", "salary_cap_equiv", "sign_trade_bonus", "incentive"]
    stats_cols = [c for c in raw_oncourt if not any(l in c.lower() for l in LEAKAGE)]
    
    # 2. Time
    for tf in TIME_FEATS:
        if tf not in df.columns:
            # Simple fallback calculation if missing
            if tf == "age_now" and "Age" in df.columns:
                 df["age_now"] = df["Age"]
            elif tf == "years_since_draft" and "draft_year" in df.columns:
                 df["years_since_draft"] = df["season"] - df["draft_year"]
            else:
                 df[tf] = 0.0
    
    # 3. Embeddings (V2 Inductive)
    print(f">>> Merging V2 Embeddings from: {GNN_V2_IND}")
    emb = pd.read_csv(GNN_V2_IND)
    emb["player_id"] = emb["player_id"].astype(str) # Ensure string here too
    
    if "season" in emb.columns: 
        emb["season"] = emb["season"].astype(int)
        merge_keys = ["player_id", "season"]
    else:
        merge_keys = ["player_id"]
        
    emb_cols = [c for c in emb.columns if c.startswith("e")]
    
    # Filter Intersection
    common_players = set(df["player_id"]) & set(emb["player_id"])
    print(f">>> Intersection Players: {len(common_players)}")
    
    df = df[df["player_id"].isin(common_players)].copy()
    emb = emb[emb["player_id"].isin(common_players)].copy()
    
    # Merge
    full_df = df.merge(emb, on=merge_keys, how="inner")
    
    # Fill NaNs
    full_df[stats_cols] = full_df[stats_cols].fillna(0)
    full_df[emb_cols] = full_df[emb_cols].fillna(0)
    full_df[TIME_FEATS] = full_df[TIME_FEATS].fillna(0)
    
    return full_df, stats_cols, TIME_FEATS, emb_cols

def analyze_importance(df, stats_cols, time_cols, emb_cols):
    print("\n>>> Training Random Forest for Inspection...")
    
    # Train on Pre-2024 (Warm History)
    train_df = df[df["season"] < TEST_SEASON]
    
    X_cols = stats_cols + time_cols + emb_cols
    y = train_df[TARGET_COL]
    X = train_df[X_cols]
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    
    # Extract Importance
    imps = rf.feature_importances_
    feat_imp = pd.DataFrame({"feature": X_cols, "importance": imps})
    feat_imp = feat_imp.sort_values("importance", ascending=False)
    
    # --- Group Analysis ---
    stats_imp = feat_imp[feat_imp["feature"].isin(stats_cols)]["importance"].sum()
    time_imp = feat_imp[feat_imp["feature"].isin(time_cols)]["importance"].sum()
    emb_imp = feat_imp[feat_imp["feature"].isin(emb_cols)]["importance"].sum()
    
    print("\n" + "="*40)
    print("📊 V2 STRONGEST: FEATURE ATTRIBUTION ANALYSIS")
    print("="*40)
    print(f"Total Features Used: {len(X_cols)}")
    print(f" - Stats Cols: {len(stats_cols)}")
    print(f" - Time Cols:  {len(time_cols)}")
    print(f" - Graph Cols: {len(emb_cols)}")
    
    print("\n🏆 GROUP IMPORTANCE (Who drives the prediction?)")
    print(f"📈 On-Court Stats:  {stats_imp:.4f} ({stats_imp*100:.1f}%)")
    print(f"⏳ Time / Age:      {time_imp:.4f}  ({time_imp*100:.1f}%)")
    print(f"🕸️ V2 Embeddings:   {emb_imp:.4f}  ({emb_imp*100:.1f}%)")
    
    print("\n🔝 TOP 20 INDIVIDUAL FEATURES")
    print(feat_imp.head(20).to_string(index=False))
    
    # Interpretation
    print("\n" + "-"*40)
    print("🧐 DIAGNOSIS:")
    if emb_imp > 0.15:
        print(f"⚠️ [OVER-RELIANCE DETECTED] Graph contributes {emb_imp*100:.1f}% to the model.")
        print("   This confirms why Cold Start failed: The model learned to trust Embeddings")
        print("   on old players, but those Embeddings are NOISE for new players.")
    else:
        print("✅ [SAFE] Model mostly ignores the graph. Cold start failure might be due to other shifts.")
    print("-" * 40)

if __name__ == "__main__":
    df, stats, time, emb = load_data()
    analyze_importance(df, stats, time, emb)