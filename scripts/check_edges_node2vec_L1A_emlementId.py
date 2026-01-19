import pandas as pd

df = pd.read_csv("graph/edges/edges_node2vec_L1A_elementId.csv")

print(df.columns.tolist())
print(df["rel"].value_counts())
