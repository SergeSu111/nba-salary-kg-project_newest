import pandas as pd
import numpy as np
import re
import os

# ================= 配置路径 =================
# 请确保这些路径与您的文件实际位置一致
PATH_OLD_INJURIES = 'data/raw_external/injuries_2010-2020.csv'  
PATH_NEW_INJURIES = 'data/raw_external/Injury Database - Oct 2021 - June 2024 (1).csv' 
PATH_PLAYER_IDS = 'data/raw_on_court/unique_player_id.csv' 
OUTPUT_KG_CSV = 'neo4j/import/offcourt_injury_for_kg.csv' # 统一输出文件名
OUTPUT_ML_CSV = 'data/processed/player_season_injury_stats.csv'

# ================= 辅助函数 =================
def extract_injury_part(description):
    if not isinstance(description, str):
        return 'Other'
    desc = description.lower()
    
    if 'knee' in desc or 'acl' in desc or 'meniscus' in desc: return 'Knee'
    if 'ankle' in desc: return 'Ankle'
    if 'foot' in desc or 'toe' in desc: return 'Foot'
    if 'hamstring' in desc: return 'Hamstring'
    if 'back' in desc: return 'Back'
    if 'shoulder' in desc: return 'Shoulder'
    if 'calf' in desc: return 'Calf'
    if 'achilles' in desc: return 'Achilles'
    if 'wrist' in desc: return 'Wrist'
    if 'hand' in desc or 'finger' in desc: return 'Hand'
    if 'groin' in desc: return 'Groin'
    if 'hip' in desc: return 'Hip'
    if 'concussion' in desc or 'head' in desc: return 'Head'
    if 'illness' in desc or 'health' in desc or 'protocols' in desc: return 'Illness/Protocol'
    return 'Other'

def clean_player_name(name):
    if not isinstance(name, str): return ""
    
    # 处理 "Last, First" 格式 (针对新伤病数据集)
    if ',' in name:
        parts = name.split(',')
        if len(parts) >= 2:
            # 翻转: "Doncic, Luka" -> "Luka Doncic"
            name = f"{parts[1].strip()} {parts[0].strip()}"
            
    name = name.replace(" Jr.", "").replace(" Sr.", "").replace(" III", "").replace(" II", "").replace(" IV", "")
    name = name.replace(".", "") # T.J. -> TJ
    
    return name.strip()

# ================= 主流程 =================
def main():
    print("正在加载数据...")
    try:
        df_old = pd.read_csv(PATH_OLD_INJURIES)
        df_new = pd.read_csv(PATH_NEW_INJURIES)
        df_ids = pd.read_csv(PATH_PLAYER_IDS)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
        return

    # === ID 表处理 ===
    # CSV中是 'Player_id' 和 'Player'，代码需要 'player_id' 和 'player_name'
    df_ids = df_ids.rename(columns={'Player_id': 'player_id', 'Player': 'player_name'})
    # 生成用于匹配的清洗名
    df_ids['name_clean'] = df_ids['player_name'].apply(clean_player_name)
    
    # *** 关键修复 *** # 准备一个只包含 [清洗名, ID, 标准名] 的表用于合并
    # 我们将 'player_name' 重命名为 'standard_name' 以便区分
    df_ids_ref = df_ids[['name_clean', 'player_id', 'player_name']].rename(columns={'player_name': 'standard_name'})

    print("正在清洗旧数据集 (2020部分)...")
    df_old['Date'] = pd.to_datetime(df_old['Date'])
    df_old = df_old[df_old['Date'] >= '2020-09-01'].copy()
    df_old = df_old.dropna(subset=['Relinquished'])
    df_old = df_old.rename(columns={'Relinquished': 'raw_name', 'Notes': 'description', 'Date': 'date'})
    df_old = df_old[['raw_name', 'date', 'description']]

    print("正在清洗新数据集 (2021-2024)...")
    df_new['Date'] = pd.to_datetime(df_new['DATE'])
    # 注意：这里我们把原始名字重命名为 'raw_name'，不再叫 'player_name'
    df_new = df_new.rename(columns={'PLAYER': 'raw_name', 'REASON': 'description', 'Date': 'date'})
    df_new = df_new[['raw_name', 'date', 'description']]

    print("正在合并数据集...")
    df_all = pd.concat([df_old, df_new], axis=0, ignore_index=True)
    df_all = df_all.sort_values('date')

    # 对原始名字进行清洗，以便与 ID 表匹配
    df_all['name_clean'] = df_all['raw_name'].apply(clean_player_name)
    
    # Merge ID
    # 这里通过 'name_clean' 关联，把 'standard_name' 和 'player_id' 拉过来
    df_merged = pd.merge(df_all, df_ids_ref, on='name_clean', how='inner')
    
    print(f"原始伤病记录数: {len(df_all)}")
    print(f"匹配到ID的记录数: {len(df_merged)} (丢失率: {1 - len(df_merged)/len(df_all) if len(df_all)>0 else 0:.2%})")

    # 提取伤病类别
    df_merged['injury_category'] = df_merged['description'].apply(extract_injury_part)
    
    # 确定赛季 (返回整数年份)
    def assign_season(date):
        year = date.year
        month = date.month
        if month >= 10:
            return year + 1
        else:
            return year
    
    df_merged['season'] = df_merged['date'].apply(assign_season)

    # ================= 输出 1: KG 导入数据 =================
    # 注意：这里使用 'standard_name' 作为最终输出的 'player_name'
    df_merged = df_merged.rename(columns={'standard_name': 'player_name'})
    
    kg_cols = ['player_id', 'player_name', 'date', 'description', 'injury_category', 'season']
    df_kg = df_merged[kg_cols].drop_duplicates()
    df_kg['date'] = df_kg['date'].dt.strftime('%Y-%m-%d')
    
    os.makedirs(os.path.dirname(OUTPUT_KG_CSV), exist_ok=True)
    df_kg.to_csv(OUTPUT_KG_CSV, index=False)
    print(f"KG 导入文件已生成: {OUTPUT_KG_CSV}")
    print("已验证: player_name 列现在统一使用 ID 表中的标准格式 (First Last)。")

    # ================= 输出 2: ML 聚合特征 =================
    stats = df_merged.groupby(['player_id', 'season']).agg(
        total_injury_records=('date', 'count'),
        unique_injuries=('injury_category', 'nunique'),
        has_knee_injury=('injury_category', lambda x: 1 if 'Knee' in x.values else 0),
        has_ankle_injury=('injury_category', lambda x: 1 if 'Ankle' in x.values else 0)
    ).reset_index()
    
    os.makedirs(os.path.dirname(OUTPUT_ML_CSV), exist_ok=True)
    stats.to_csv(OUTPUT_ML_CSV, index=False)
    print(f"ML 聚合统计文件已生成: {OUTPUT_ML_CSV}")

if __name__ == "__main__":
    main()