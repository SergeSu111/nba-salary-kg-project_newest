import pandas as pd
import numpy as np
from pathlib import Path

# ================= CONFIG =================
PRED_DIR = Path("runs/final_eval_strict_v3/20260207_161729/predictions")
SEED = 0
BASE_FILE = f"predictions_Baseline_StatsplusTime_RandomForest_seed{SEED}.csv"
INV_MODE = "log1p" 

# --- 阈值设置 ---
RESCUE_MIN = 500_000        # 图模型至少救回 $0.5M
BASE_ERR_MIN = 1_000_000    # 基线误差至少 $1M (确保不是捡漏)
UNIQUE_RESCUE_MIN = 1_500_000 
UNIQUE_ADV_MIN = 500_000      
MIN_EXAMPLES_PER_MODEL = 3  

MODELS = {
    "RotatE":    f"predictions_RotatE_plus_Stats_RandomForest_seed{SEED}.csv",
    "Node2Vec":  f"predictions_Node2Vec_plus_Stats_RandomForest_seed{SEED}.csv",
    "V1":        f"predictions_V1_plus_Stats_RandomForest_seed{SEED}.csv",
    "V2_Ind":    f"predictions_V2_Ind_plus_Stats_RandomForest_seed{SEED}.csv",
    "V2_Trans":  f"predictions_V2_Trans_plus_Stats_RandomForest_seed{SEED}.csv",
    "V2_Full":   f"predictions_V2_Full_MG_plus_Stats_RandomForest_seed{SEED}.csv" 
}

# ================= 工具函数 =================

def inv_log(x):
    """反变换并做非负保护"""
    if INV_MODE == "log":
        val = np.exp(x)
    elif INV_MODE == "log1p":
        val = np.expm1(x)
    else:
        raise ValueError("INV_MODE error")
    return np.maximum(val, 0.0)

def load_pred_file(path):
    if not path.exists(): return None
    df = pd.read_csv(path)
    
    required = ['player_id', 'season', 'y_true', 'y_pred']
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"❌ {path.name} missing: {missing}")

    if 'player_name' not in df.columns: df['player_name'] = "Unknown"
    
    df['player_id'] = df['player_id'].astype(str)
    df['season'] = df['season'].astype(int)
    df['salary_pred_usd'] = inv_log(df['y_pred'])
    df['salary_true_usd'] = inv_log(df['y_true'])
    
    return df[['player_id', 'season', 'player_name', 'salary_true_usd', 'salary_pred_usd']]

def fmt(x): 
    abs_x = abs(x)
    if abs_x < 1e6: return f"${x/1e6:.2f}M"
    return f"${x/1e6:.1f}M"

def categorize_rescue(row):
    """学术化分类 + 边界处理"""
    act = row['salary_true_usd']
    base = row['pred_base']
    graph = row['pred_graph']
    
    # 1. 边界处理 (Exact)
    if abs(act - base) < 10_000: return "Exact (Skip)" 

    # 2. 主分类
    if act > base: main_type = "Underrated"
    else: main_type = "Overrated"
        
    # 3. 子分类 (Precision vs Overshoot)
    is_overshoot = (base - act) * (graph - act) < 0
    sub_type = "Overshoot" if is_overshoot else "Precision"
        
    return f"{main_type} ({sub_type})"

# ================= 主逻辑 =================

def main():
    print(f"--- Configuration ---")
    print(f"Seed: {SEED} | Mode: {INV_MODE}")
    print("-" * 30)

    # 1. Load Baseline
    base_path = PRED_DIR / BASE_FILE
    df_base = load_pred_file(base_path)
    if df_base is None:
        print(f"❌ Baseline missing: {base_path}")
        return

    df_base = df_base.rename(columns={'salary_pred_usd': 'pred_base'})
    df_base_core = df_base[['player_id', 'season', 'player_name', 'salary_true_usd', 'pred_base']]

    global_rescue_map = {} 
    all_examples = []
    coverage_stats = [] 

    # 2. Analyze Models
    for model_name, filename in MODELS.items():
        print(f"\n🔵 Analyzing: {model_name} ...")
        df_model = load_pred_file(PRED_DIR / filename)
        
        if df_model is None:
            print(f"   ⚠️ Skip (Not Found)")
            coverage_stats.append({
                'Model': model_name,
                'Coverage': 0,
                'Success_Cases': 0,
                'Success_Rate': 0.0,
                'Success_Rate_Pct': "0.0%"
            })
            continue

        # Merge
        merged = pd.merge(
            df_base_core, 
            df_model[['player_id', 'season', 'salary_pred_usd']], 
            on=['player_id', 'season'], 
            how='inner'
        )
        merged = merged.rename(columns={'salary_pred_usd': 'pred_graph'})
        
        # Metrics
        merged['err_base'] = (merged['salary_true_usd'] - merged['pred_base']).abs()
        merged['err_graph'] = (merged['salary_true_usd'] - merged['pred_graph']).abs()
        merged['rescue'] = merged['err_base'] - merged['err_graph']

        # Fill Global Map
        for row in merged.itertuples(index=False):
            key = (row.player_id, row.season)
            if key not in global_rescue_map:
                global_rescue_map[key] = {'player_name': row.player_name, 'models': {}}
            global_rescue_map[key]['models'][model_name] = {
                'rescue': row.rescue,
                'err': row.err_graph,
                'err_base': row.err_base
            }

        # Filter Success Cases
        success_cases = merged[
            (merged['rescue'] > RESCUE_MIN) & 
            (merged['err_base'] > BASE_ERR_MIN)
        ].copy()
        
        # Coverage Stats
        n_cov = len(merged)
        n_succ = len(success_cases)
        rate = n_succ / n_cov if n_cov > 0 else 0.0
        
        coverage_stats.append({
            'Model': model_name, 
            'Coverage': n_cov,
            'Success_Cases': n_succ,
            'Success_Rate': rate,
            'Success_Rate_Pct': f"{rate*100:.1f}%"
        })
        print(f"   Coverage: {n_cov} | Success: {n_succ} ({rate*100:.1f}%)")

        if success_cases.empty: continue

        success_cases['rescue_type'] = success_cases.apply(categorize_rescue, axis=1)
        
        # --- Selection Strategy ---
        seen_ids_model = set()
        selected_rows = []

        # A. Priority Selection
        target_types = [
            "Underrated (Precision)", "Underrated (Overshoot)",
            "Overrated (Precision)", "Overrated (Overshoot)"
        ]
        
        for r_type in target_types:
            subset = success_cases[success_cases['rescue_type'] == r_type].sort_values('rescue', ascending=False)
            for _, cand in subset.iterrows():
                if cand['player_id'] not in seen_ids_model:
                    seen_ids_model.add(cand['player_id'])
                    cand_copy = cand.copy()
                    cand_copy['selection_method'] = 'Category'
                    selected_rows.append(cand_copy)
                    break 
        
        # B. Fallback Selection
        if len(selected_rows) < MIN_EXAMPLES_PER_MODEL:
            needed = MIN_EXAMPLES_PER_MODEL - len(selected_rows)
            remaining = success_cases[~success_cases['player_id'].isin(seen_ids_model)].sort_values('rescue', ascending=False)
            
            for _, cand in remaining.head(needed).iterrows():
                seen_ids_model.add(cand['player_id'])
                cand_copy = cand.copy()
                cand_copy['selection_method'] = 'Fallback'
                selected_rows.append(cand_copy)

        # C. Add to Final List
        for row in selected_rows:
            delta = row['pred_graph'] - row['pred_base']
            all_examples.append({
                'Model': model_name,
                'Type': row['rescue_type'],
                'Method': row.get('selection_method', 'Category'),
                'Player': row['player_name'],
                'Season': row['season'],
                'Actual': row['salary_true_usd'],
                'Base_Pred': row['pred_base'],
                'Graph_Pred': row['pred_graph'],
                'Delta_Pred': delta,
                'Base_Err': row['err_base'],
                'Graph_Err': row['err_graph'],
                'Rescue_Amount': row['rescue'],
                'Model_Coverage': n_cov
            })
            print(f"   + Selected: {row['player_name']} [{row['rescue_type']}] (+{fmt(row['rescue'])})")

    # ================= 3. Unique Insights Analysis =================
    print("\n" + "="*60)
    print("🏆 Unique Insights Analysis")
    print("="*60)
    
    unique_insights_list = []

    for target_model in MODELS.keys():
        candidates = []
        for key, info in global_rescue_map.items():
            res_dict = info['models']
            if target_model in res_dict:
                known_models = [m for m in MODELS if m in res_dict]
                if len(known_models) < 2: continue 

                my_rescue = res_dict[target_model]['rescue']
                my_err = res_dict[target_model]['err']

                # Peer Stats
                peers = [m for m in known_models if m != target_model]
                peer_rescues = [res_dict[m]['rescue'] for m in peers]
                peer_errs = [res_dict[m]['err'] for m in peers]
                
                max_other_rescue = max(peer_rescues) if peer_rescues else -9e9
                min_other_err = min(peer_errs) if peer_errs else 9e9
                
                candidates.append({
                    'player_name': info['player_name'],
                    'player_id': key[0],
                    'season': key[1],
                    'my_rescue': my_rescue,
                    'my_err': my_err,
                    'min_other_err': min_other_err,
                    'rescue_advantage': my_rescue - max_other_rescue,
                    'coverage': len(known_models)
                })
        
        if not candidates: continue
        df_cand = pd.DataFrame(candidates)
        
        # Dual Thresholds: High Rescue + Absolute Supremacy
        df_cand = df_cand[
            (df_cand['my_rescue'] > UNIQUE_RESCUE_MIN) & 
            (df_cand['my_err'] <= df_cand['min_other_err'] + 10_000)
        ]

        if df_cand.empty: continue

        hits = df_cand[df_cand['rescue_advantage'] > UNIQUE_ADV_MIN].sort_values('rescue_advantage', ascending=False).head(3)

        if not hits.empty:
            print(f"\n🌟 [{target_model}] Exclusive:")
            for _, row in hits.iterrows():
                print(f"   🏀 {row['player_name']:<20} | Rescue:+{fmt(row['my_rescue'])} | Adv:+{fmt(row['rescue_advantage'])} | MyErr:{fmt(row['my_err'])}")
                unique_insights_list.append({
                    'Model': target_model,
                    'Player': row['player_name'],
                    'Rescue': row['my_rescue'],
                    'My_Err': row['my_err'],
                    'Peer_Best_Err': row['min_other_err'],
                    'Rescue_Advantage': row['rescue_advantage'],
                    'Coverage': row['coverage']
                })

    # ================= 保存 CSV =================
    
    # 1. Concrete Examples
    if all_examples:
        df_ex = pd.DataFrame(all_examples)
        df_ex.to_csv(PRED_DIR / "summary_concrete_examples_numeric.csv", index=False)
        
        df_fmt = df_ex.copy()
        money_cols = ['Actual', 'Base_Pred', 'Graph_Pred', 'Delta_Pred', 'Base_Err', 'Graph_Err', 'Rescue_Amount']
        for col in money_cols:
            df_fmt[col] = df_fmt[col].apply(fmt)
            
        df_fmt.to_csv(PRED_DIR / "summary_concrete_examples.csv", index=False)
        print(f"\n✅ Concrete Examples saved.")

    # 2. Unique Insights
    if unique_insights_list:
        df_u = pd.DataFrame(unique_insights_list)
        # 【升级】保存 Numeric 版
        df_u.to_csv(PRED_DIR / "summary_unique_insights_numeric.csv", index=False)
        
        df_u_fmt = df_u.copy()
        money_cols_u = ['Rescue', 'My_Err', 'Peer_Best_Err', 'Rescue_Advantage']
        for col in money_cols_u:
            df_u_fmt[col] = df_u_fmt[col].apply(fmt)
            
        df_u_fmt.to_csv(PRED_DIR / "summary_unique_insights.csv", index=False)
        print(f"✅ Unique Insights saved.")

    # 3. Coverage Summary
    if coverage_stats:
        df_cov = pd.DataFrame(coverage_stats)
        df_cov.to_csv(PRED_DIR / "summary_model_coverage.csv", index=False)
        print(f"✅ Coverage Summary saved.")

if __name__ == "__main__":
    main()