import pandas as pd

df = pd.read_csv("graph/edges/V2_Full_Injury_multigraph_Edges_FULL_19073.csv")

# 取前四位年份（如果已是 int 就更简单）
df["psY"] = df["ps_season"].astype(str).str[:4].astype(int)
df["injY"] = df["injury_season"].astype(str).str[:4].astype(int)

leaks = df[df["injY"] >= df["psY"]]
print("Leak rows:", len(leaks))
print(leaks.head(20))