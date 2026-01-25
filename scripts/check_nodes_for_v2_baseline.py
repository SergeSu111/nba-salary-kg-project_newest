import pandas as pd

# 你的边文件路径
file_path = "graph/edges/edges_gnn_v2_core_elementId_full.csv"

print(f"Reading {file_path}...")
df = pd.read_csv(file_path)

# 1. 统计全图节点总数
# 这一步非常关键：它定义了你的 GNN 矩阵大小 (Num Nodes)
all_src = set(df['src'].astype(str))
all_dst = set(df['dst'].astype(str))
all_nodes = all_src.union(all_dst)
print(f"\n=== 1. Graph Size ===")
print(f"Total Unique Nodes (Graph Vocabulary): {len(all_nodes)}")
print("  (This is the exact size of your embedding matrix)")

# 2. 通过关系类型反推节点类型
# 逻辑：如果我们知道 'FOR_TEAM' 指向的一定是 Team，那我们就可以统计有多少个 Team
print(f"\n=== 2. Node Type Census (Inferred from Relations) ===")

# 我们遍历每一种关系，看它连接了多少个“目标节点”
relations = df['rel'].unique()

for r in relations:
    # 找到这种关系的所有目标节点 (Destination Nodes)
    target_nodes = df[df['rel'] == r]['dst'].unique()
    count = len(target_nodes)
    
    # 打印统计信息
    print(f"Relation: [{r}]")
    print(f"  -> Connects to {count} unique nodes")
    
    # 自动判别身份
    identity = "Unknown"
    if r in ['FOR_TEAM', 'OF_TEAM', 'DRAFTED_BY']:
        identity = "TEAMS (球队)"
    elif r == 'REPRESENTED_BY':
        identity = "AGENTS (经纪人)"
    elif r in ['IN_SEASON', 'OF_SEASON']:
        identity = "SEASONS (赛季时间点)"
    elif r == 'UNDRAFTED':
        identity = "STATUS: UNDRAFTED (落选状态)"
    elif r in ['HAS_VALUE', 'VALUE_IN']:
        identity = "CONTRACT BUCKETS (薪资区间)"
    elif 'AWARD' in r or 'WON' in r:
        identity = "!!! AWARDS (荣誉) - ALERT !!!"
    elif 'INJURY' in r:
        identity = "!!! INJURIES (伤病) - ALERT !!!"
        
    print(f"  -> Interpretation: These are likely {identity}")
    print("-" * 30)

print("\n=== 3. Final Verdict ===")
if not any('AWARD' in r or 'WON' in r for r in relations):
    print("✅ CONFIRMED: No Award Nodes detected.")
else:
    print("❌ ALERT: Award Nodes detected!")
    
if not any('INJURY' in r for r in relations):
    print("✅ CONFIRMED: No Injury Nodes detected.")
else:
    print("❌ ALERT: Injury Nodes detected!")