import pandas as pd
from pathlib import Path

# 1. 读取 parquet 文件
parquet_path = "data/processed/training_oncourt.parquet"
df = pd.read_parquet(parquet_path)

# 2. 确保 data/processed 文件夹存在
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

# 3. 导出为 CSV 到同一个目录
csv_path = out_dir / "training_oncourt.csv"
df.to_csv(csv_path, index=False)

print(f"✅ 已成功将文件导出为 CSV：{csv_path}")