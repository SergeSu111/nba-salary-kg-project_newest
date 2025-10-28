# 放可复用的特征构造函数
from __future__ import annotations
import numpy as np
import pandas as pd
# src/features/transforms.py


EPS = 1e-9  # 防止除零

EPS = 1e-9  # 防止除零

def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / (denom.replace(0, np.nan))  # 先变 NaN，后统一 fillna


def add_log_salary(df: pd.DataFrame, salary_col: str = "salary_usd") -> pd.DataFrame:
    """目标的 log1p 版本，训练时可选用，也可只做分析用。"""
    if salary_col in df.columns:
        df["log_salary"] = np.log1p(df[salary_col].clip(lower=0))
    return df


def add_per_min_features(df: pd.DataFrame) -> pd.DataFrame:
    """把计数型数据转为“每分钟”强度：更能反映效率，不受上场时间直接影响。"""
    if "Min" not in df.columns:
        return df
    

    per_min_map = [
        ("PTS_per_min", "PTS"),
        ("REB_per_min", "REB"),
        ("AST_per_min", "AST"),
        ("STL_per_min", "STL"),
        ("BLK_per_min", "BLK"),
        ("TOV_per_min", "TOV"),
        ("FGA_per_min", "FGA"), #每分钟出手次数
        ("FGM_per_min", "FGM"), # 每分钟命中的投篮数
        ("3PA_per_min", "3PA"), # 每分钟三分球出手数
        ("3PM_per_min", "3PM"), # 每分钟命中三分球的次数
        ("FTA_per_min", "FTA"), # 每分钟罚球出手次数
        ("FTM_per_min", "FTM"), # 每分钟命中罚球的次数
        ("OREB_per_min","OREB"), # 每分钟进攻篮板数
        ("DREB_per_min","DREB"), # 每分钟防守篮板数
        ("PF_per_min","PF"), # 每分钟犯规数
    ]
    for new_col, base in per_min_map:
        if base in df.columns:
            df[new_col] = _safe_div(df[base], df["Min"])
    return df


def add_per_gp_features(df: pd.DataFrame) -> pd.DataFrame:
    """把计数型数据转为“每场”强度（如果你确认这些是赛季总和）。若原始列即为每场均值，可跳过。"""
    if "GP" not in df.columns:
        return df
    per_gp_map = [
        ("PTS_per_gp", "PTS"),
        ("REB_per_gp", "REB"),
        ("AST_per_gp", "AST"),
        ("STL_per_gp", "STL"),
        ("BLK_per_gp", "BLK"),
        ("TOV_per_gp", "TOV"),
        ("FGA_per_gp", "FGA"),
        ("FGM_per_gp", "FGM"),
        ("3PA_per_gp", "3PA"),
        ("3PM_per_gp", "3PM"),
        ("FTA_per_gp", "FTA"),
        ("FTM_per_gp", "FTM"),
        ("OREB_per_gp","OREB"),
        ("DREB_per_gp","DREB"),
        ("PF_per_gp","PF"),
    ]
    for new_col, base in per_gp_map:
        if base in df.columns:
            df[new_col] = _safe_div(df[base], df["GP"])
    return df



def add_shooting_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """投篮效率衍生：eFG%、TS% 等（公式标准）。"""
    # eFG% = (FGM + 0.5*3PM)/FGA  有效投篮命中率 没有罚球的
    if {"FGM","3PM","FGA"}.issubset(df.columns):
        df["eFG%_calc"] = _safe_div(df["FGM"] + 0.5 * df["3PM"], df["FGA"])
    # TS% = PTS / (2*(FGA + 0.44*FTA))  真实投篮命中率
    if {"PTS","FGA","FTA"}.issubset(df.columns):
        df["TS%_calc"] = _safe_div(df["PTS"], 2*(df["FGA"] + 0.44*df["FTA"]))
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """常见比率：助攻失误比、进攻篮板占比等。"""
    if {"AST","TOV"}.issubset(df.columns): # 助攻失误比
        df["AST_TO_ratio"] = _safe_div(df["AST"], df["TOV"])
    if {"OREB","REB"}.issubset(df.columns): # 进攻篮板比
        df["OREB_share"] = _safe_div(df["OREB"], df["REB"])
    if {"DREB","REB"}.issubset(df.columns): # 防守篮板比
        df["DREB_share"] = _safe_div(df["DREB"], df["REB"])
    return df


def add_log_transforms(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """对重尾的计数型特征做 log1p（不对百分比做）。"""
    for c in cols:
        if c in df.columns:
            df[f"log1p_{c}"] = np.log1p(df[c].clip(lower=0))
    return df


def finalize_fill(df: pd.DataFrame) -> pd.DataFrame:
    """把前面除零产生的 NaN 统一填 0（适合树模型）。"""
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)
    return df


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / (denom.replace(0, np.nan))  # 先变 NaN，后统一 fillna

def add_log_salary(df: pd.DataFrame, salary_col: str = "salary_usd") -> pd.DataFrame:
    """目标的 log1p 版本，训练时可选用，也可只做分析用。"""
    if salary_col in df.columns:
        df["log_salary"] = np.log1p(df[salary_col].clip(lower=0))
    return df

def add_per_min_features(df: pd.DataFrame) -> pd.DataFrame:
    """把计数型数据转为“每分钟”强度：更能反映效率，不受上场时间直接影响。"""
    if "Min" not in df.columns:
        return df

    per_min_map = [
        ("PTS_per_min", "PTS"),
        ("REB_per_min", "REB"),
        ("AST_per_min", "AST"),
        ("STL_per_min", "STL"),
        ("BLK_per_min", "BLK"),
        ("TOV_per_min", "TOV"),
        ("FGA_per_min", "FGA"),
        ("FGM_per_min", "FGM"),
        ("3PA_per_min", "3PA"),
        ("3PM_per_min", "3PM"),
        ("FTA_per_min", "FTA"),
        ("FTM_per_min", "FTM"),
        ("OREB_per_min","OREB"),
        ("DREB_per_min","DREB"),
        ("PF_per_min","PF"),
    ]
    for new_col, base in per_min_map:
        if base in df.columns:
            df[new_col] = _safe_div(df[base], df["Min"])
    return df

def add_per_gp_features(df: pd.DataFrame) -> pd.DataFrame:
    """把计数型数据转为“每场”强度（如果你确认这些是赛季总和）。若原始列即为每场均值，可跳过。"""
    if "GP" not in df.columns:
        return df
    per_gp_map = [
        ("PTS_per_gp", "PTS"),
        ("REB_per_gp", "REB"),
        ("AST_per_gp", "AST"),
        ("STL_per_gp", "STL"),
        ("BLK_per_gp", "BLK"),
        ("TOV_per_gp", "TOV"),
        ("FGA_per_gp", "FGA"),
        ("FGM_per_gp", "FGM"),
        ("3PA_per_gp", "3PA"),
        ("3PM_per_gp", "3PM"),
        ("FTA_per_gp", "FTA"),
        ("FTM_per_gp", "FTM"),
        ("OREB_per_gp","OREB"),
        ("DREB_per_gp","DREB"),
        ("PF_per_gp","PF"),
    ]
    for new_col, base in per_gp_map:
        if base in df.columns:
            df[new_col] = _safe_div(df[base], df["GP"])
    return df

def add_shooting_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """投篮效率衍生：eFG%、TS% 等（公式标准）。"""
    # eFG% = (FGM + 0.5*3PM)/FGA
    if {"FGM","3PM","FGA"}.issubset(df.columns):
        df["eFG%_calc"] = _safe_div(df["FGM"] + 0.5 * df["3PM"], df["FGA"])
    # TS% = PTS / (2*(FGA + 0.44*FTA))
    if {"PTS","FGA","FTA"}.issubset(df.columns):
        df["TS%_calc"] = _safe_div(df["PTS"], 2*(df["FGA"] + 0.44*df["FTA"]))
    return df

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """常见比率：助攻失误比、进攻篮板占比等。"""
    if {"AST","TOV"}.issubset(df.columns):
        df["AST_TO_ratio"] = _safe_div(df["AST"], df["TOV"])
    if {"OREB","REB"}.issubset(df.columns):
        df["OREB_share"] = _safe_div(df["OREB"], df["REB"])
    if {"DREB","REB"}.issubset(df.columns):
        df["DREB_share"] = _safe_div(df["DREB"], df["REB"])
    return df

def add_log_transforms(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """对重尾的计数型特征做 log1p（不对百分比做）。"""
    for c in cols:
        if c in df.columns:
            df[f"log1p_{c}"] = np.log1p(df[c].clip(lower=0))
    return df

def finalize_fill(df: pd.DataFrame) -> pd.DataFrame:
    """把前面除零产生的 NaN 统一填 0（适合树模型）。"""
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)
    return df

def build_feature_view(
    df: pd.DataFrame,
    drop_text_cols: list[str] = ["Player","Team"],
    add_logs_for: list[str] = None,
) -> pd.DataFrame:
    """
    串起上面的所有 transform，返回“可训练”的特征视图。
    - 默认删除文本列（Player/Team），保留 id/season 作为分组或回溯
    - 只输出数值列 + id/season + 目标
    """
    df = df.copy()

    # 1) 目标的 log 版本
    df = add_log_salary(df, "salary_usd")

    # 2) 基本强度特征
    df = add_per_min_features(df)
    df = add_per_gp_features(df)

    # 3) 投篮效率 & 比率
    df = add_shooting_efficiency(df)
    df = add_ratio_features(df)

    # 4) 对极度偏态的计数列做 log1p（挑一些常见字段；你可按需要增/减）
    if add_logs_for is None:
        add_logs_for = [
            "PTS","REB","AST","STL","BLK","TOV",
            "FGA","FGM","3PA","3PM","FTA","FTM",
            "OREB","DREB","PF","Min",
            "PTS_per_min","REB_per_min","AST_per_min","TOV_per_min"
        ]
    df = add_log_transforms(df, add_logs_for)

    # 5) 删除文本列（避免训练时报错）
    for c in drop_text_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 6) 填补因除零产生的 NaN
    df = finalize_fill(df)

    # 7) 仅保留：数值列 + id/season + 目标
    keep = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or c in ["Player_id","season"]:
            keep.append(c)
    df = df[keep]

    return df


SALARY_CAP_2020_2025 = {
    2020: 109_140_000,  # 2020-2021
    2021: 112_414_000,  # 2021-2022
    2022: 123_655_000,  # 2022-2023
    2023: 136_000_000,  # 2023-2024（官方约 136.021M，这里按 136M）
    2024: 140_588_000,  # 2024-2025
}


def add_salary_cap_features(
    df: pd.DataFrame,
    season_col: str = "season",
    salary_col: str = "salary_usd",
    cap_map: dict | None = None,
    baseline_season: int = 2024,   # 把名义薪资换算到该赛季的等价值（默认 2024-25）
) -> pd.DataFrame:
    """
    基于工资帽做标准化：
    - salary_cap: 当季工资帽
    - salary_cap_ratio: 名义薪资 / 当季工资帽（占帽比例）
    - log_salary_cap_ratio: 其 log1p
    - salary_cap_equiv: 以 baseline_season 的工资帽为基准换算的“等价薪资”（剔除通胀/帽增长）
    """
    cap_map = cap_map or SALARY_CAP_2020_2025
    df = df.copy()

    if season_col not in df.columns or salary_col not in df.columns:
        return df

    # 映射赛季工资帽
    df["salary_cap"] = df[season_col].map(cap_map).astype("float64")

    # 占帽比例（防除零）
    denom = df["salary_cap"].replace(0, np.nan)
    df["salary_cap_ratio"] = (df[salary_col] / denom).astype("float64")

    # log 比例
    df["log_salary_cap_ratio"] = np.log1p(df["salary_cap_ratio"].clip(lower=0))

    # 基准年等价薪资：把所有赛季按工资帽缩放到 baseline_season 的水平
    base_cap = cap_map.get(baseline_season, np.nan)
    df["salary_cap_equiv"] = (df["salary_cap_ratio"] * base_cap).astype("float64")

    # 缺失（极少数赛季不在映射表）统一补 0，便于树模型使用
    for c in ["salary_cap", "salary_cap_ratio", "log_salary_cap_ratio", "salary_cap_equiv"]:
        df[c] = df[c].fillna(0)

    return df
