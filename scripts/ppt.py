import pandas as pd

# 1. 读取文件
p = r'graph\edges\V2_Full_Award_Edges.csv'
print(f"Reading: {p}")
df = pd.read_csv(p)

# 2. 统一列名（转小写，去空格）
df.columns = [c.lower().strip() for c in df.columns]
print(f"Columns found: {df.columns.tolist()}")

# 3. 智能查找 ps_season 和 award_year 列
ps_col = next((c for c in df.columns if 'ps' in c and 'season' in c), None)
aw_col = next((c for c in df.columns if 'award' in c and 'year' in c), None)

if not ps_col or not aw_col:
    print(f"❌ 关键列缺失! 没找到 ps_season 或 award_year。")
    print(f"请检查你的 CSV 表头。")
    exit()

print(f"Using columns: ps='{ps_col}', award='{aw_col}'")

# 4. 解析年份的函数
def parse_year(x):
    try:
        s = str(x).strip()
        return int(s[:4])
    except:
        return None

df['ps_year_val'] = df[ps_col].apply(parse_year)
df['aw_year_val'] = df[aw_col].apply(parse_year)

# 5. 查找导致 "strict_past" 失败的行 (aw >= ps)
leak_strict = df[df['aw_year_val'] >= df['ps_year_val']]

print(f"\nTotal rows: {len(df)}")
print(f"Strict Past Leaks (Award >= Season): {len(leak_strict)}")

if len(leak_strict) > 0:
    print("\n=== 泄露样本预览 (Top 20) ===")
    print(leak_strict[[ps_col, aw_col, 'relation_type', 'source_id', 'target_id']].head(20).to_string())
    
    # 统计一下是不是全是相等的情况
    equal_count = len(df[df['aw_year_val'] == df['ps_year_val']])
    future_count = len(df[df['aw_year_val'] > df['ps_year_val']])
    print(f"\n其中完全相等 (Award == Season) 的行数: {equal_count}")
    print(f"其中真正未来 (Award > Season) 的行数: {future_count}")
    
    if future_count == 0 and equal_count > 0:
        print("\n✅ 结论: 所有的'泄露'都只是当季获奖 (Summer Award)。")
        print("因为 award_year (2023) == ps_season (2023)，这在业务上是合法的（赛季开始前已颁奖）。")
        print("这证明你把 audit 规则改成 'non_future' 是完全正确的决定。")