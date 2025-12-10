# src/data/training_table.py

# 将已有的csv的场上和场下数据合在一起去跑模型
from pathlib import Path
from typing import Literal

import pandas as pd

DATA_ROOT = Path("neo4j/import")


def load_training_oncourt() -> pd.DataFrame:
    """
    Level0 用的 base 表（你现在的 training_oncourt_features）
    KEY: Player_id + season
    """
    path = DATA_ROOT / "training_oncourt_features.csv"
    df = pd.read_csv(path)

    # 统一主键名字
    if "Player_id" in df.columns:
        df = df.rename(columns={"Player_id": "player_id"})

    return df


# ------------ Draft 相关 ------------

def build_draft_features() -> pd.DataFrame:
    """
    读 draft + undrafted，做成每个 player 一条记录的表
    KEY: player_id
    """
    draft_path = DATA_ROOT / "offcourt_draft_for_kg.csv"
    undrafted_path = DATA_ROOT / "offcourt_draft_undrafted_for_kg.csv"

    draft = pd.read_csv(draft_path)
    undrafted = pd.read_csv(undrafted_path)

    # 只保留最早一次 draft（一般就是唯一的）
    draft = (
        draft.sort_values(["player_id", "year"])
        .drop_duplicates(subset=["player_id"], keep="first")
    )

    draft_features = draft[[
        "player_id",
        "year",
        "round",
        "round_pick",
        "overall_pick",
    ]].rename(columns={"year": "draft_year"})

    # Lottery / round 标记
    draft_features["draft_is_lottery"] = draft_features["overall_pick"] <= 14
    draft_features["draft_is_first_round"] = draft_features["round"] == 1
    draft_features["draft_is_second_round"] = draft_features["round"] == 2

    # undrafted flag
    undrafted_flag = undrafted[["player_id", "undrafted_flag"]]
    draft_features = draft_features.merge(
        undrafted_flag, on="player_id", how="outer"
    )

    # 没在 draft 表里但在 undrafted 表里 -> undrafted_flag = True
    draft_features["undrafted_flag"] = draft_features["undrafted_flag"].fillna(False)

    return draft_features


# ------------ Age / experience 相关 ------------

def build_age_features() -> pd.DataFrame:
    """
    从 player_age.csv 读取每个 player 的当前年龄 (按 2025-26 赛季)
    KEY: player_id (不含 season)
    """
    path = DATA_ROOT / "player_age.csv"
    age = pd.read_csv(path)

    # 统一列名：假设是 Player_id, Age
    rename_map = {}
    if "Player_id" in age.columns:
        rename_map["Player_id"] = "player_id"
    if "Age" in age.columns:
        rename_map["Age"] = "age_now"

    age = age.rename(columns=rename_map)

    # 只保留 player_id + age_now
    age = age[["player_id", "age_now"]].drop_duplicates(subset=["player_id"])

    return age

# ------------ Team value 相关 ------------

def build_team_value_features() -> pd.DataFrame:
    """
    从 offcourt_team_value_for_kg.csv 构造 team 价值相关特征
    KEY: team_id + season
    """
    path = DATA_ROOT / "offcourt_team_value_for_kg.csv"
    df = pd.read_csv(path)

    # 假设有 team_id, season, team_value 这样的列
    # 按实际列名改
    rename_map = {}
    if "Team_id" in df.columns:
        rename_map["Team_id"] = "team_id"
    if "year" in df.columns:
        rename_map["year"] = "season"
    if "team_value_usd" in df.columns:
        rename_map["team_value_usd"] = "team_value_usd"

    df = df.rename(columns=rename_map)

    # 只保留我们关心的列
    keep_cols = ["team_id", "season", "team_value_usd"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["team_id", "season"])

    # 做一个百分位 rank，用来表示球队在联盟的价值位置
    if "team_value_usd" in df.columns:
        df["team_value_pct"] = (
            df.groupby("season")["team_value_usd"]
            .rank(pct=True)
        )
        # 大市场标记：前 25% 的球队
        df["team_big_market_flag"] = df["team_value_pct"] >= 0.75

    return df


# ------------ Team location / market 相关 ------------

def build_team_location_features() -> pd.DataFrame:
    """
    从 offcourt_team_location_for_kg.csv 提取球队所在城市/市场信息
    KEY: team_id (+ 可选 season)
    """
    path = DATA_ROOT / "offcourt_team_location_for_kg.csv"
    df = pd.read_csv(path)

    rename_map = {}
    if "team_id" in df.columns:
        rename_map["team_id"] = "team_id"
    if "Year" in df.columns and "season" not in df.columns:
        rename_map["Year"] = "season"

    df = df.rename(columns=rename_map)

    # 假设有这些列，按实际情况改：
    #   city, state, market_size, region, country
    #   如果没有 market_size，可以之后用 city 去映射
    keep = ["team_id"]
    for c in ["season", "city", "state", "market_size", "region"]:
        if c in df.columns:
            keep.append(c)

    df = df[keep].drop_duplicates(subset=[c for c in keep if c in ["team_id", "season"]])

    # 做一些简单编码：例如西海岸/东海岸/中部
    if "region" in df.columns:
        df["team_region_is_west"] = df["region"].str.contains("West", case=False, na=False)
        df["team_region_is_east"] = df["region"].str.contains("East", case=False, na=False)

    return df


# ------------ Award 相关 ------------

def build_award_features() -> pd.DataFrame:
    """
    从 award_std_fixed 统计每个 season 的 award 数量
    KEY: player_id + season
    """
    path = DATA_ROOT / "award_std_fixed.csv"
    awards = pd.read_csv(path)

    # 统一 season 名字
    awards = awards.rename(columns={"year": "season"})

    # 统计每种 award 的次数
    # 例如 award = 'All-Star', 'All-NBA', 'MVP', 'All-Defensive' 等
    awards["award_count"] = 1

    pivot = (
        awards
        .pivot_table(
            index=["player_id", "season"],
            columns="award",
            values="award_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    # 列名扁平化，比如 'All-Star' -> 'award_All-Star'
    pivot.columns = [
        "player_id" if c == "player_id"
        else "season" if c == "season"
        else f"award_{c}"
        for c in pivot.columns
    ]

    # 总 award 数
    award_cols = [c for c in pivot.columns if c.startswith("award_")]
    pivot["award_total"] = pivot[award_cols].sum(axis=1)

    return pivot


# ------------ Injury 相关 ------------

def build_injury_features() -> pd.DataFrame:
    """
    根据 offcourt_injury_for_kg，按 player_id + season 统计受伤情况
    KEY: player_id + season
    """
    path = DATA_ROOT / "offcourt_injury_for_kg.csv"
    inj = pd.read_csv(path)

    # 基本计数：每季有几条伤病记录
    grp = inj.groupby(["player_id", "season"])

    feat = grp.agg(
        injury_events=("date", "count"),
        injury_unique_dates=("date", "nunique"),
        injury_unique_categories=("injury_category", "nunique"),
    ).reset_index()

    feat["injury_any"] = feat["injury_events"] > 0

    return feat



# ------------ Agent 相关 ------------

def build_agent_features() -> pd.DataFrame:
    """
    Agent 是按 player 级别，不分 season
    这里做一些简单的数值特征，方便直接喂给模型
    KEY: player_id
    """
    path = DATA_ROOT / "offcourt_agents_for_kg.csv"
    agents = pd.read_csv(path)

    # 一个球员可能有多个 agent，这里简单处理：合并成一个字符串
    agg = (
        agents.groupby("player_id")["agent_name"]
        .apply(lambda x: " | ".join(sorted(set(x))))
        .reset_index()
    )

    # 统计每个 agent 的客户数量，用于 agent_power 特征
    # 展开到行
    exploded = (
        agents.assign(agent_name=agents["agent_name"].str.split(" / "))
        .explode("agent_name")
    )
    agent_counts = (
        exploded.groupby("agent_name")["player_id"]
        .nunique()
        .reset_index()
        .rename(columns={"player_id": "agent_client_count"})
    )

    # 取代表 agent（第一个）
    rep_agent = (
        exploded
        .drop_duplicates(subset=["player_id", "agent_name"])
        .sort_values(["player_id", "agent_name"])
        .drop_duplicates(subset=["player_id"], keep="first")
    )

    rep_agent = rep_agent.merge(agent_counts, on="agent_name", how="left")

    rep_agent = rep_agent[["player_id", "agent_name", "agent_client_count"]]

    # 合并两种信息（代表 agent + 全部 agent 串联）
    feat = rep_agent.merge(agg, on="player_id", how="left", suffixes=("", "_all"))

    # 简单做一个「大牌经纪人」标记：客户数前 10% 的 agent
    threshold = feat["agent_client_count"].quantile(0.9)
    feat["agent_is_top"] = feat["agent_client_count"] >= threshold

    return feat


def build_level1_training_table() -> pd.DataFrame:
    """
    Level1: on-court + 全部 off-court（draft/age/award/injury/agent/team）
    返回一张每行 = player_id + season 的训练表
    """
    base = load_training_oncourt()

    # 如果 season 列名不是 season，这里统一一下
    if "Year" in base.columns and "season" not in base.columns:
        base = base.rename(columns={"Year": "season"})
    if "Team_id" in base.columns and "team_id" not in base.columns:
        base = base.rename(columns={"Team_id": "team_id"})

    draft = build_draft_features()
    awards = build_award_features()
    injury = build_injury_features()
    agents = build_agent_features()
    age = build_age_features()
    team_value = build_team_value_features()
    team_loc = build_team_location_features()

    df = base.copy()

    # ---------- player 级别合并 ----------
    # draft / agents: 只按 player_id 合并（所有赛季共用）
    df = df.merge(draft, on="player_id", how="left")
    df = df.merge(agents, on="player_id", how="left")

    # age: 按 player_id + season 合并
    df = df.merge(age, on="player_id", how="left")

    # years since draft
    if "season" in df.columns and "draft_year" in df.columns:
        df["years_since_draft"] = df["season"] - df["draft_year"]

    # ---------- team 级别合并 ----------
    if "team_id" not in df.columns:
        raise ValueError("training_oncourt_features 中需要有 team_id（或 Team_id）列")

    # team value: 一般是 team_id + season
    merge_keys_tv = ["team_id", "season"] if "season" in team_value.columns else ["team_id"]
    df = df.merge(team_value, on=merge_keys_tv, how="left")

    # team location: 可能只有 team_id，也可能有 season
    merge_keys_loc = ["team_id", "season"] if "season" in team_loc.columns else ["team_id"]
    df = df.merge(team_loc, on=merge_keys_loc, how="left")

    # ---------- award / injury ----------
    df = df.merge(awards, on=["player_id", "season"], how="left")
    df = df.merge(injury, on=["player_id", "season"], how="left")

    # award / injury 缺失值处理
    for col in df.columns:
        if col.startswith("award_") or col.startswith("injury_"):
            df[col] = df[col].fillna(0)

    if "injury_any" in df.columns:
        df["injury_any"] = df["injury_any"].fillna(False)

    return df


def load_training_table(level: Literal["level0", "level1"] = "level1") -> pd.DataFrame:
    """
    对外统一入口：
    - level0: 只用 on-court 表（原始 baseline）
    - level1: on-court + off-court 合并后的 rich CSV
    """
    if level == "level0":
        return load_training_oncourt()
    elif level == "level1":
        return build_level1_training_table()
    else:
        raise ValueError(f"Unknown level: {level}")
    


def main():
    df = build_level1_training_table()
    print("Level1 shape:", df.shape)
    out = Path("data/processed/training_level1_full.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("saved to", out)

if __name__ == "__main__":
    main()