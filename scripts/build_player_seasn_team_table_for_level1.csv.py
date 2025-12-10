# scripts/build_player_season_team_table.py

from pathlib import Path
import pandas as pd

# 为了把player 场上数据 还有球队串联起来 的csv 然后放入level1 跑的. 

DATA_ROOT = Path("neo4j/import")


def build_player_season_team() -> pd.DataFrame:
    """
    从 oncourt_core_for_kg.csv 构造 player_season_team 映射表：
    每行 = 一个球员在某个赛季主要效力的球队

    输出列：
    - player_id
    - season
    - team_id（如果 oncourt_core_for_kg 里有）
    - team_abbr（如果有）
    """
    src_path = DATA_ROOT / "oncourt_core_for_kg.csv"
    df = pd.read_csv(src_path)

    # ---- 1. 统一列名 ----
    rename_map = {}

    # player id
    if "Player_id" in df.columns:
        rename_map["Player_id"] = "player_id"
    elif "playerId" in df.columns:
        rename_map["playerId"] = "player_id"

    # season
    if "season" not in df.columns:
        if "Season" in df.columns:
            rename_map["Season"] = "season"
        elif "Year" in df.columns:
            rename_map["Year"] = "season"

    # team id / abbr
    if "Team_id" in df.columns and "team_id" not in df.columns:
        rename_map["Team_id"] = "team_id"
    if "teamAbbr" in df.columns and "team_abbr" not in df.columns:
        rename_map["teamAbbr"] = "team_abbr"
    if "Team" in df.columns and "team_abbr" not in df.columns:
        # 比如列名就是 "Team"
        rename_map["Team"] = "team_abbr"

    df = df.rename(columns=rename_map)

    # 确认必需列
    required_cols = ["player_id", "season"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"oncourt_core_for_kg.csv 里必须有列: {c}")

    # 如果没有 team_id/team_abbr 其中任何一个，就提示一下
    if "team_id" not in df.columns and "team_abbr" not in df.columns:
        raise ValueError(
            "oncourt_core_for_kg.csv 中没有 team_id 或 team_abbr，"
            "无法构造 player_season_team 映射，请检查原始数据。"
        )

    # ---- 2. 选出我们关心的列 ----
    keep_cols = ["player_id", "season"]
    if "team_id" in df.columns:
        keep_cols.append("team_id")
    if "team_abbr" in df.columns:
        keep_cols.append("team_abbr")

    # 如果有数据列可以用来判断“主要效力球队”，顺便带上
    # 例如 total_minutes / games_played / gp 之类
    # 名字你可以按实际情况调整
    minute_cols = [c for c in ["mp", "MP", "minutes", "total_minutes"] if c in df.columns]
    game_cols = [c for c in ["gp", "GP", "games", "G"] if c in df.columns]

    score_col = None
    if minute_cols:
        score_col = minute_cols[0]  # 优先用出场时间
        keep_cols.append(score_col)
    elif game_cols:
        score_col = game_cols[0]    # 其次用出场场次
        keep_cols.append(score_col)

    df = df[keep_cols].copy()

    # ---- 3. 处理一名球员同一赛季多队的情况 ----
    # 正常情况下 oncourt_core_for_kg 已经是每季一条记录，
    # 但为了安全，这里做一个“选择主要球队”的逻辑。
    if df.duplicated(subset=["player_id", "season"]).any():
        if score_col is not None:
            # 按出场时间 / 场次排序，取得分最高的一支队
            df = (
                df.sort_values(["player_id", "season", score_col],
                               ascending=[True, True, False])
                  .drop_duplicates(subset=["player_id", "season"], keep="first")
            )
        else:
            # 没有任何出场指标，就简单去重
            df = df.drop_duplicates(subset=["player_id", "season"], keep="first")

    # ---- 4. 最终只保留 key 列 ----
    out_cols = ["player_id", "season"]
    if "team_id" in df.columns:
        out_cols.append("team_id")
    if "team_abbr" in df.columns:
        out_cols.append("team_abbr")

    df = df[out_cols].reset_index(drop=True)

    return df


def main():
    df = build_player_season_team()
    print("player_season_team shape:", df.shape)

    out_path = DATA_ROOT / "player_season_team_for_all_csv_run.csv"
    df.to_csv(out_path, index=False)
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()