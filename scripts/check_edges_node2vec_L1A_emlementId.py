import pandas as pd

edges = pd.read_csv("graph/edges/edges_node2vec_L1A_elementId.csv")
print(edges.columns)
print(edges["rel"].value_counts(dropna=False))