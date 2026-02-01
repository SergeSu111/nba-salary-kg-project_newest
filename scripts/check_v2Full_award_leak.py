import pandas as pd

# === 1) 读入你导出的非泄露版 award edges ===
# 改成你的实际文件名/路径
path = "graph/edges/V2_Full_Award_Edges.csv"
df = pd.read_csv(path)

# === 2) 统一把年份字段转成可比较的整数（取前4位防止 '2023-24' 这种格式） ===
def year4(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if len(s) < 4:
        return None
    try:
        return int(s[:4])
    except:
        return None

df["psY"] = df["ps_season"].apply(year4)
df["awY"] = df["award_year"].apply(year4)

# === 3) 基础数据质量检查 ===
bad_ps = df["psY"].isna().sum()
bad_aw = df["awY"].isna().sum()

print(f"Total rows: {len(df)}")
print(f"Bad ps_season (cannot parse year): {bad_ps}")
print(f"Bad award_year (cannot parse year): {bad_aw}")

# === 4) 泄露审计：award_year <= ps_season 必须成立 ===
# 注意：如果 psY/awY 有 NaN，会先被当成违规处理（更保守）
leak_mask = (df["awY"].isna()) | (df["psY"].isna()) | (df["awY"] > df["psY"])
leak_df = df[leak_mask].copy()

print(f"\nLeak edges (awY > psY OR unparsable): {len(leak_df)}")

# === 5) 输出前20条违规示例（方便你回 Neo4j 定位） ===
if len(leak_df) > 0:
    cols = [c for c in ["source_id","target_id","relation_type","ps_season","award_year","psY","awY"] if c in leak_df.columns]
    print("\nExamples of leaks / bad rows (first 20):")
    print(leak_df[cols].head(20).to_string(index=False))
else:
    print("\n✅ Audit passed: all rows satisfy award_year <= ps_season (after parsing first 4 digits).")

# === 6) 额外统计（可选） ===
print("\nYear distribution:")
print(df.groupby(["psY","awY"]).size().reset_index(name="cnt").sort_values("cnt", ascending=False).head(20))