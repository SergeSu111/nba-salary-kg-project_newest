# 根据第一步的eda1 然后对合并长表做预处理清洗
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys


def load_df(in_path: Path) -> pd.DataFrame:
    if in_path.suffix.lower() == ".parquet":
        return pd.read_parquet(in_path)
    elif in_path.suffix.lower() == ".csv":
        return pd.read_csv(in_path)
    else:
        # 尝试优先读 parquet，再读 csv
        pq = in_path.with_suffix(".parquet")
        cs = in_path.with_suffix(".csv")
        if pq.exists():
            return pd.read_parquet(pq)
        if cs.exists():
            return pd.read_csv(cs)
        raise FileNotFoundError(f"Cannot find parquet/csv: {in_path}")
    


def save_df(df: pd.DataFrame, out_path: Path, also_csv: bool = True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    if also_csv:
        df.to_csv(out_path.with_suffix(".csv"), index=False)


def find_all_na_cols(df: pd.DataFrame):
    return [c for c in df.columns if df[c].isna().all()]



def find_constant_cols(df: pd.DataFrame):
    const = []
    for c in df.columns:
        try:
            nun = df[c].nunique(dropna=True)
            if nun <= 1:
                const.append(c)
        except Exception:
            # 某些奇怪列就跳过
            pass
    return const


def high_missing_cols(df: pd.DataFrame, thresh: float = 0.5):
    na_ratio = df.isna().mean()
    return list(na_ratio[na_ratio > thresh].index), na_ratio


def drop_key_duplicates(df: pd.DataFrame, key_cols=("Player_id", "season")) -> pd.DataFrame:
    if not set(key_cols).issubset(df.columns):
        return df
    # 为了可复现：按所有列排序后去重（也可以自定义优先级）
    df2 = df.sort_values(by=list(df.columns)).drop_duplicates(subset=list(key_cols), keep="last")
    return df2


def filter_quality(df: pd.DataFrame, min_gp: int, min_min: float) -> pd.DataFrame:
    if "GP" in df.columns and min_gp > 0:
        df = df[df["GP"] >= min_gp]
    if "Min" in df.columns and min_min > 0:
        df = df[df["Min"] >= min_min]
    return df


def fillna_numeric(df: pd.DataFrame, strategy: str = "none") -> pd.DataFrame:
    """strategy in {"none","median","zero"}"""
    if strategy == "none":
        return df
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    if strategy == "median":
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
    elif strategy == "zero":
        df[num_cols] = df[num_cols].fillna(0)
    return df


def main():
    ap = argparse.ArgumentParser(description="Clean & preprocess training long table")
    ap.add_argument("--in_path",  default="data/processed/training_oncourt.parquet")
    ap.add_argument("--out_path", default="data/processed/training_oncourt_clean.parquet")
    ap.add_argument("--report_dir", default="reports/clean_preprocess")
    ap.add_argument("--high_na_thresh", type=float, default=0.50, help="缺失率>阈值的列会被删除")
    ap.add_argument("--min_gp", type=int, default=0, help="可选过滤：最低 GP")
    ap.add_argument("--min_min", type=float, default=0.0, help="可选过滤：最低 Min")
    ap.add_argument("--fill_strategy", choices=["none","median","zero"], default="none",
                    help="数值缺失填补策略（baseline 阶段默认 none）")
    args = ap.parse_args()

    in_path   = Path(args.in_path)
    out_path  = Path(args.out_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(in_path)
    orig_shape = df.shape

    # —— 统计与标记 —— #
    allna_cols   = find_all_na_cols(df)
    const_cols   = find_constant_cols(df)
    high_na_cols, na_ratio = high_missing_cols(df, args.high_na_thresh)

    # 在我的 EDA_1 中，'Unnamed: 30' 同时是全空/常数/高缺失列，这里统一删除
    to_drop = sorted(set(allna_cols) | set(const_cols) | set(high_na_cols))

    # —— 删除列 —— #
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

    # —— 复合键去重（如果有） —— #
    dup_cnt = 0
    if set(["Player_id","season"]).issubset(df.columns):
        dup_cnt = int(df.duplicated(subset=["Player_id","season"]).sum())
        if dup_cnt > 0:
            df = drop_key_duplicates(df, key_cols=("Player_id","season"))

    # —— 可选质量过滤（你当前 Min<=0/GP<=0 都为 0，这里只是保留参数） —— #
    df = filter_quality(df, args.min_gp, args.min_min)

    # —— 可选缺失填补 —— #
    df = fillna_numeric(df, strategy=args.fill_strategy)

    # —— 保存清洗后的数据（parquet + csv） —— #
    save_df(df, out_path, also_csv=True)

    # —— 输出清洗报告 —— #
    report = {
        "input_path": str(in_path),
        "output_path_parquet": str(out_path),
        "output_path_csv": str(out_path.with_suffix(".csv")),
        "orig_shape": list(orig_shape),
        "final_shape": list(df.shape),
        "dropped_cols": to_drop,
        "all_na_cols": allna_cols,
        "constant_cols": const_cols,
        "high_na_cols(>{:.0%})".format(args.high_na_thresh): high_na_cols,
        "duplicate_key_count_(Player_id,season)_before_drop": dup_cnt,
        "filters": {"min_gp": args.min_gp, "min_min": args.min_min},
        "fill_strategy": args.fill_strategy,
    }
    (report_dir / "clean_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    na_ratio.sort_values(ascending=False).to_csv(report_dir / "missing_ratio_before_drop.csv")

    print("✅ Clean finished.")
    print("   Input :", in_path)
    print("   Output:", out_path, "and", out_path.with_suffix(".csv"))
    print("   From  :", orig_shape, "→", df.shape)
    print("   Dropped columns:", to_drop)
    print("   Duplicate (Player_id, season) before drop:", dup_cnt)
    print("   Report saved at:", report_dir / "clean_report.json")

if __name__ == "__main__":
    main()