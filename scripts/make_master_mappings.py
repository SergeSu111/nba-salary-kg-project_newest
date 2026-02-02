import pandas as pd
from pathlib import Path

inp = r"graph\mappings\nodes_master_raw.csv"
out = r"graph\mappings\master_node_id_to_idx.csv"

p = Path(inp)
print("Reading:", p.resolve())
if not p.exists():
    raise FileNotFoundError(f"Missing input CSV: {p}")

df = pd.read_csv(p)
df.columns = [c.lower() for c in df.columns]

if "node_id" not in df.columns:
    raise ValueError(f"Expected column 'node_id'. Got columns: {list(df.columns)}")

df["node_id"] = df["node_id"].astype(str)

df = df.drop_duplicates(subset=["node_id"]).copy()
df = df.sort_values(by=["node_id"]).reset_index(drop=True)

df["idx"] = range(len(df))

outp = Path(out)
outp.parent.mkdir(parents=True, exist_ok=True)

cols = ["node_id", "idx"]
if "label" in df.columns:
    cols.append("label")

df[cols].to_csv(outp, index=False)

print("Saved:", outp.resolve())
print("Num nodes:", len(df))
print(df[cols].head(3).to_string(index=False))
