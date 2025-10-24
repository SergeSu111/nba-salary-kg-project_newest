import os, re, glob, argparse, pandas as pd
from pathlib import Path

SEASON_RE = re.compile(r"(\d{4})-(\d{4})")

"""infer the season year from file name"""
def infer_season_from_fname(fname: str) -> int:
    m = SEASON_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot infer season from: {fname}")
    return int(m.group(1)) # 2023 for 2023-2024


"""get all unique_id_stats.csv in data/raw/,
    then add season column, check if we have Player_id column,
    combine all the data into one big DataFrame"""
def load_stats_files(raw_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(raw_dir, "unique_id_stats_*.csv")))
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        season = infer_season_from_fname(os.path.basename(p))
        df["season"] = season
        if "Player_id" not in df.columns:
            raise KeyError(f"Missing Player_id in {p}")
        frames.append(df)
    stats = pd.concat(frames, ignore_index = True)
    return stats


"""read all the salary.csv. Then make all Player_ID to be Player_id
    make NAME to be Player 
    make Salary and salary to be salary_usd
    delete duplicated rows 
    combine all seasons' salary tables into one big df"""
def load_salary_files(raw_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(raw_dir, "NBA_*_salary.csv")))
    frames = []
    if not paths:
        raise FileNotFoundError("No NBA_*_salary.csv found")

    for p in paths:
        df = pd.read_csv(p)

        # 1) 统一列名风格：小写、去空格、空格换下划线
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # 2) 从文件名推断赛季
        season = infer_season_from_fname(os.path.basename(p))
        df["season"] = season

        # 3) 映射到标准列名（尽可能兼容不同写法）
        rename_map = {}

        # player_id 可能叫 player_id / playerid / id
        if "player_id" in df.columns:
            rename_map["player_id"] = "Player_id"
        elif "playerid" in df.columns:
            rename_map["playerid"] = "Player_id"
        elif "id" in df.columns:
            rename_map["id"] = "Player_id"
        else:
            raise KeyError(f"Cannot find player_id in {p}. Columns: {list(df.columns)}")

        # name → Player（如果没有就可以不强求）
        if "name" in df.columns:
            rename_map["name"] = "Player"
        elif "player" in df.columns and "Player" not in df.columns:
            rename_map["player"] = "Player"

        # salary → salary_usd（尽量多兼容几种写法）
        salary_candidate = None
        for cand in ("salary", "salary_usd", "base_salary", "cap_hit", "amount", "value"):
            if cand in df.columns:
                salary_candidate = cand
                break
        if salary_candidate is None:
            raise KeyError(f"Cannot find salary column in {p}. Columns: {list(df.columns)}")
        if salary_candidate != "salary_usd":
            rename_map[salary_candidate] = "salary_usd"

        df = df.rename(columns=rename_map)

        # 4) 薪水清洗为数值（去 $, , 等）
        if "salary_usd" in df.columns:
            df["salary_usd"] = (
                df["salary_usd"].astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
            )
            df["salary_usd"] = pd.to_numeric(df["salary_usd"], errors="coerce")

        # 5) (Player_id, season) 去重
        df = df.drop_duplicates(subset=["Player_id", "season"], keep="last")

        # 6) 仅保留所需列
        keep_cols = [c for c in ["Player_id", "Player", "salary_usd", "season"] if c in df.columns]
        df = df[keep_cols]

        frames.append(df)

    # 7) 纵向合并
    sal = pd.concat(frames, ignore_index=True)
    return sal




def unify_stat_columns_across_seasons(stats: pd.DataFrame) -> pd.DataFrame:
    from collections import Counter
    counts = Counter()
    for season, g in stats.groupby("season"):
        for c in g.columns:
            counts[c] += 1
    base = ["Player_id", "Player", "Team", "season"]
    others = [c for c, _ in counts.most_common() if c not in base]
    target_cols = base + others
    # 去重保持顺序
    seen, aligned_cols = set(), []
    for c in target_cols:
        if c not in seen:
            seen.add(c)
            aligned_cols.append(c)
    # 缺的列补 NA
    for c in aligned_cols:
        if c not in stats.columns:
            stats[c] = pd.NA
    return stats[aligned_cols]


def build_training_table(raw_dir: str, out_path: str, min_gp: int = 0, min_min: float = 0.0) -> pd.DataFrame:
    stats = load_stats_files(raw_dir)
    sal   = load_salary_files(raw_dir)
    stats = unify_stat_columns_across_seasons(stats)

    if "GP" in stats.columns and min_gp > 0:
        stats = stats[stats["GP"] >= min_gp]
    if "Min" in stats.columns and min_min > 0:
        stats = stats[stats["Min"] >= min_min]

    joined = stats.merge(
        sal[["Player_id", "season", "salary_usd"]],
        on=["Player_id", "season"],
        how="inner",
        validate="m:1"
    )

    # 简单缺失填补（仅少量缺失时；树模型可识别NA，但写入 parquet 时可选填）
    # for c in joined.select_dtypes(include="number").columns:
    #     joined[c] = joined[c].fillna(joined[c].median())

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out_path, index=False)
    print(f"[OK] training table saved: {out_path}  shape={joined.shape}")

    # 生成覆盖率小报告
    cov_rows = []
    for season, sgrp in stats.groupby("season"):
        stat_ids = set(sgrp["Player_id"])
        sal_ids  = set(sal[sal["season"] == season]["Player_id"])
        cov_rows.append({
            "season": season,
            "stats_rows": len(sgrp),
            "salary_rows": (sal["season"] == season).sum(),
            "intersect": len(stat_ids & sal_ids),
            "only_in_stats": len(stat_ids - sal_ids),
            "only_in_salary": len(sal_ids - stat_ids),
        })
    cov = pd.DataFrame(cov_rows).sort_values("season")
    cov_path = os.path.join(os.path.dirname(out_path), "coverage_report.csv")
    cov.to_csv(cov_path, index=False)
    print(f"[OK] coverage report: {cov_path}")
    return joined


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_path", default="data/processed/training_oncourt.parquet")
    ap.add_argument("--min_gp", type=int, default=0, help="filter: minimum GP (e.g., 10)")
    ap.add_argument("--min_min", type=float, default=0.0, help="filter: minimum minutes per game (e.g., 8)")
    args = ap.parse_args()
    build_training_table(args.raw_dir, args.out_path, args.min_gp, args.min_min)


    