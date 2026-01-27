import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
EMBEDDING_FILE = 'graph/embeddings/rotate_L1B_cpu_player_embeddings.csv'
META_FILE = 'data/processed/training_level1_full.csv'
OUTPUT_IMG = 'slide5_agent_clustering.png'

# 🎯 核心修改：只高亮这 5 位真实存在的超级经纪人
TARGET_AGENTS = [
    'Aaron Mintz',      # CAA (人数最多 104)
    'Rich Paul',        # Klutch (84人, 必须高亮)
    'Jeff Schwartz',    # Excel (75人, 必须高亮)
    'Bill Duffy',       # WME (67人)
    'Jason Glushon'     # Glushon Sports (67人)
]
# ===========================================

def parse_complex_string(s):
    """处理 RotatE 的复数格式 (-0.123+0.456j)"""
    try:
        return complex(s)
    except:
        return 0j

def generate_agent_cluster_plot():
    print("1. Loading Metadata...")
    if not os.path.exists(META_FILE):
        print(f"❌ Error: Meta file not found at {META_FILE}")
        return

    meta_df = pd.read_csv(META_FILE, low_memory=False)
    
    # === 自动修正列名逻辑 (智能侦察) ===
    # 先假设在 agent_name
    agent_col = 'agent_name'
    
    # 如果 log1p_3PM 列里包含 "Rich Paul"，说明表头错位了，改用这一列
    if 'log1p_3PM' in meta_df.columns:
        sample_vals = meta_df['log1p_3PM'].astype(str).head(100).values
        if any('Rich Paul' in s for s in sample_vals):
            print("   ⚠️ Detected column misalignment. Found agents in 'log1p_3PM'.")
            agent_col = 'log1p_3PM'
    
    print(f"   ✅ Using '{agent_col}' as Agent Name column.")
    meta_df['real_agent_name'] = meta_df[agent_col]

    # 准备 Metadata
    meta_clean = meta_df[['player_id', 'real_agent_name']].copy()
    meta_clean['player_id'] = meta_clean['player_id'].astype(str)
    meta_clean = meta_clean.drop_duplicates(subset=['player_id'])

    print("\n2. Loading RotatE Embeddings...")
    if not os.path.exists(EMBEDDING_FILE):
        print(f"❌ Error: Embedding file not found.")
        return

    emb_df = pd.read_csv(EMBEDDING_FILE)
    emb_df['player_id'] = emb_df['player_id'].astype(str)
    
    # 识别复数向量列 (e0, e1...)
    embedding_cols = [c for c in emb_df.columns if c.startswith('e') and c not in ['player_id', 'node_id']]
    print(f"   Detected {len(embedding_cols)} complex embedding columns.")

    print("   Converting complex strings to features (Real & Imaginary)...")
    # 转换为 Python 复数对象
    complex_data = emb_df[embedding_cols].applymap(parse_complex_string)
    
    # 拆分实部和虚部 (128复数 -> 256实数)
    real_part = complex_data.applymap(np.real)
    imag_part = complex_data.applymap(np.imag)
    
    # 合并特征
    real_part.columns = [f"{c}_re" for c in embedding_cols]
    imag_part.columns = [f"{c}_im" for c in embedding_cols]
    X_features = pd.concat([real_part, imag_part], axis=1)
    X_features['player_id'] = emb_df['player_id'] # 加回 ID 用于合并

    print("\n3. Merging & t-SNE...")
    merged_df = pd.merge(X_features, meta_clean, on='player_id', how='inner')
    merged_df = merged_df.drop_duplicates(subset=['player_id'])
    
    # 提取纯数值特征
    feature_cols = [c for c in merged_df.columns if c.endswith('_re') or c.endswith('_im')]
    X = merged_df[feature_cols].values
    
    print(f"   Running t-SNE on {X.shape[0]} players with {X.shape[1]} dimensions...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    merged_df['x'] = X_embedded[:, 0]
    merged_df['y'] = X_embedded[:, 1]

    print("4. Plotting...")
    # 标记：如果在 Target 列表里就保留名字，否则标为 Other
    merged_df['Agent Label'] = merged_df['real_agent_name'].apply(lambda x: x if x in TARGET_AGENTS else 'Other')
    
    plt.figure(figsize=(12, 10))
    
    # 颜色顺序：Other 放第一个（灰色），后面跟着 5 大佬
    hue_order = ['Other'] + TARGET_AGENTS
    # 颜色盘：第一个是浅灰，后面是鲜艳色
    palette = [(0.85, 0.85, 0.85)] + sns.color_palette("bright", len(TARGET_AGENTS))
    
    # 先画 Other 防止遮挡，再画 Target
    sns.scatterplot(
        data=merged_df.sort_values('Agent Label', key=lambda col: col.map(lambda x: 0 if x == 'Other' else 1)), 
        x='x', y='y', hue='Agent Label', style='Agent Label',
        hue_order=hue_order, palette=palette,
        s=80, alpha=0.9, edgecolor='w', linewidth=0.5
    )
    
    plt.title('Mechanism Analysis: Implicit Discovery of Agency Power (RotatE)', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Top Agents', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"✅ 完成！高清大图已保存为: {OUTPUT_IMG}")
    print("   这张图上应该能清晰看到 Rich Paul (Klutch) 和 Jeff Schwartz (Excel) 的聚类团。")

if __name__ == "__main__":
    generate_agent_cluster_plot()