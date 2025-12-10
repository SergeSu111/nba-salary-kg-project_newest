# scripts/check_training_table.py

from pathlib import Path
import pandas as pd

def main():
    path = Path("data/processed/training_level1_full.csv")  # 你改成自己的路径
    df = pd.read_csv(path)

    print("="*50)
    print("TRAINING LEVEL1 FULL CHECK")
    print("="*50)
    print("\nShape:", df.shape)

    # ---------- key 完整性 ----------
    print("\nCheck duplicates (player_id + season):")
    print(df.duplicated(subset=["player_id", "season"]).sum())

    # ---------- 缺失值检查 ----------
    print("\nMissing ratio per column (top 20):")
    missing = df.isna().mean().sort_values(ascending=False)
    print(missing.head(20))

    # ---------- season 值检查 ----------
    print("\nSeason unique values:", sorted(df["season"].unique()))

    # ---------- team_abbr 检查 ----------
    if "team_abbr" in df.columns:
        print("\nTeam_abbr NA count:", df["team_abbr"].isna().sum())
        print("Distinct teams:", df["team_abbr"].nunique())

    # ---------- salary 校验 ----------
    if "salary_usd" in df.columns:
        print("\nSalary_min/max:", df["salary_usd"].min(), df["salary_usd"].max())

    # ---------- 年龄检查 ----------
    if "Age" in df.columns:
        print("\nAge_min/max:", df["Age"].min(), df["Age"].max())

    print("\nDone.")

if __name__ == "__main__":
    main()