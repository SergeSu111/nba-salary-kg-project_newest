#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analyze_results.py

Post-hoc analysis:
1. Feature Importance (Did the model actually use the embeddings?)
2. Case Studies (Which players did the Graph model rescue?)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# ===== path setup =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.get_just_oncourt import get_oncourt_cols

# ===== CONFIG =====
# 咱们分析 "V2 Inductive" (最有说服力) 或者是 "RotatE" (效果最好)
# 这里先分析 V2 Inductive，因为它更符合你的 V2 Strongest 叙事


EMB_PATH = Path("graph/embeddings/gnn_v2(baseline)_sage_playerseason_inductive.csv") 
TAB_PATH = Path("data/processed/training_level1_full.csv")


TEST_SEASON = 2024
TARGET_COL = "log_salary"

def load_data():
    # 1. Load Tabular
    df_tab = pd.read_csv(TAB_PATH)
    oncourt_cols = get_oncourt_cols(df_tab)
    oncourt_cols = [c for c in oncourt_cols if not c.lower().startswith(("award_", "injury_"))]
    
    # 2. Load Embedding
    df_emb = pd.read_csv(EMB_PATH)
    # Check if dynamic (V2) or static
    merge_cols = ["player_id", "season"] if "season" in df_emb.columns else ["player_id"]
    
    # 3. Merge
    df = df_tab.merge(df_emb, on=merge_cols, how="inner")
    
    emb_cols = [c for c in df_emb.columns if c.startswith("e")]
    return df, oncourt_cols, emb_cols

def analyze_feature_importance(df, oncourt_cols, emb_cols):
    print("\n=== 1. Feature Importance Analysis ===")
    
    train = df[df["season"] < TEST_SEASON]
    features = oncourt_cols + emb_cols
    
    X = train[features].fillna(0).values
    y = train[TARGET_COL].values
    
    # Train a quick RF
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    
    # Extract Importances
    imps = rf.feature_importances_
    
    # Aggregate Embedding Importance
    # 把所有 e0, e1... e63 加起来，看整体 Embedding 贡献了多少
    emb_indices = [i for i, f in enumerate(features) if f in emb_cols]
    tab_indices = [i for i, f in enumerate(features) if f in oncourt_cols]
    
    total_emb_imp = np.sum(imps[emb_indices])
    total_tab_imp = np.sum(imps[tab_indices])
    
    print(f"Total Importance (Tabular):   {total_tab_imp:.4f}")
    print(f"Total Importance (Embedding): {total_emb_imp:.4f}")
    print(f"Ratio (Emb/Tab): {total_emb_imp/total_tab_imp:.4f}")
    
    # Top 10 Individual Features
    feat_imp_list = sorted(zip(features, imps), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Individual Features:")
    for f, v in feat_imp_list[:10]:
        tag = "[EMB]" if f in emb_cols else "[TAB]"
        print(f"  {tag} {f}: {v:.4f}")

def find_case_studies(df, oncourt_cols, emb_cols):
    print("\n=== 2. Case Study Candidates (The 'Graph Rescues') ===")
    
    train = df[df["season"] < TEST_SEASON].fillna(0)
    test = df[df["season"] == TEST_SEASON].fillna(0)
    
    if len(test) == 0:
        print("Test set empty, skipping case studies.")
        return

    # A. Baseline Model (Tabular Only)
    rf_base = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_base.fit(train[oncourt_cols], train[TARGET_COL])
    pred_base = rf_base.predict(test[oncourt_cols])
    
    # B. Graph Model (Tabular + Embedding)
    rf_graph = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    feats_graph = oncourt_cols + emb_cols
    rf_graph.fit(train[feats_graph], train[TARGET_COL])
    pred_graph = rf_graph.predict(test[feats_graph])
    
    # 构造结果集 (Safe Check for player_name)
    cols_to_keep = ["player_id", "season", TARGET_COL]
    if "player_name" in test.columns:
        cols_to_keep.insert(2, "player_name")
        
    results = test[cols_to_keep].copy()
    
    # Fallback if no name
    if "player_name" not in results.columns:
        results["player_name"] = results["player_id"].astype(str)
        
    results["pred_base"] = pred_base
    results["pred_graph"] = pred_graph
    results["err_base"] = (results["pred_base"] - results[TARGET_COL]).abs()
    results["err_graph"] = (results["pred_graph"] - results[TARGET_COL]).abs()
    
    # "Graph Wins": Improvement = Base Error - Graph Error
    # 正数表示 Graph 误差更小（预测更准）
    results["improvement"] = results["err_base"] - results["err_graph"]
    
    # 筛选出 Graph 真正“拯救”了的球员 (Baseline 预测很差，但 Graph 预测很好)
    # 过滤掉那些本来 Baseline 就预测得很准的 (err_base > 0.3) 这样更有说服力
    winners = results[results["err_base"] > 0.3].sort_values("improvement", ascending=False).head(10)
    
    print("\nTop 10 'Graph Rescues' (Players where Graph fixed the Baseline error):")
    print(winners[["player_name", "season", "improvement", "err_base", "err_graph", TARGET_COL]].to_string(index=False))

def main():
    if not EMB_PATH.exists():
        print(f"File not found: {EMB_PATH}. Please fix path in script.")
        return
        
    df, oncourt, emb = load_data()
    analyze_feature_importance(df, oncourt, emb)
    find_case_studies(df, oncourt, emb)

if __name__ == "__main__":
    main()