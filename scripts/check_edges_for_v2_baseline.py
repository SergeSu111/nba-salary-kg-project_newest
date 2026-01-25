import pandas as pd

# 读取 CSV
path = "graph/edges/edges_gnn_v2_core_elementId_full.csv"
df = pd.read_csv(path)

# 1. 打印所有列名（让你看清楚到底叫什么）
print("=== Columns in File ===")
print(df.columns.tolist())

# 2. 自动尝试寻找关系列并打印内容
possible_names = ['type', 'rel', 'relation', 'edge_type', 'label', 'relationship']
found = False

print("\n=== Checking Relationships ===")
for col in possible_names:
    if col in df.columns:
        print(f"Found column: '{col}'")
        print(f"Unique Relations:\n{df[col].unique()}")
        found = True
        break

if not found:
    print("Could not guess the relationship column. Please check the column list above.")