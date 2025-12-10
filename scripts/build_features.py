import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features.transforms import (
    add_salary_cap_features,
    build_feature_view,
)


def load_df(path: Path) -> pd.DataFrame:
    """容错读取 parquet/csv。"""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    # 没写后缀就尝试两个
    pq, cs = path.with_suffix(".parquet"), path.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if cs.exists():
        return pd.read_csv(cs)
    raise FileNotFoundError(f"Cannot find parquet/csv: {path}")


def save_df(df: pd.DataFrame, out_path: Path, also_csv: bool = True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    if also_csv:
        df.to_csv(out_path.with_suffix(".csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="Build features from clean long table")
    ap.add_argument("--in_path",  default="data/processed/training_oncourt_clean.parquet",
                    help="清洗后的长表路径（parquet/csv）")
    ap.add_argument("--out_path", default="data/processed/training_oncourt_features_with_team.parquet",
                    help="输出的特征表路径（parquet）")
    ap.add_argument("--report_path", default="reports/features/features_report_with_team.json",
                    help="特征构建报告（json）")
    ap.add_argument("--corr_out", default="reports/features/corr_with_targets_with_team.csv",
                    help="与目标相关性导出（csv）")
    # ✅ 只丢掉 Player，不丢 Team
    ap.add_argument("--drop_text_cols", nargs="*", default=["Player"])
    ap.add_argument("--target", choices=[
        "salary_usd", "log_salary", "salary_cap_ratio", "log_salary_cap_ratio", "salary_cap_equiv"
    ], default="salary_usd")
    args = ap.parse_args()

    in_path   = Path(args.in_path)
    out_path  = Path(args.out_path)
    rpt_path  = Path(args.report_path)
    corr_path = Path(args.corr_out)

    # 1) 读取清洗后长表
    df = load_df(in_path)
    orig_shape = df.shape
    orig_cols  = list(df.columns)

    # 👉 先把 Team 保存一份备用
    team_key_df = None
    if all(col in df.columns for col in ["Player_id", "season", "Team"]):
        team_key_df = df[["Player_id", "season", "Team"]].drop_duplicates()

    # 2) 先加入“工资帽标准化”特征
    df = add_salary_cap_features(df, season_col="season", salary_col="salary_usd")

    # 3) 再构造基础/效率/比率/log 等特征
    feats = build_feature_view(
        df,
        drop_text_cols=args.drop_text_cols,
        add_logs_for=None
    )

    # ✅ 把 Team 拼回去
    if team_key_df is not None:
        feats = feats.merge(team_key_df, on=["Player_id", "season"], how="left")

    # 4) 规范列顺序（把 id/season/目标放前面，便于看）
    front_cols = [c for c in ["Player_id", "season",
                              "salary_usd", "log_salary",
                              "salary_cap", "salary_cap_ratio", "log_salary_cap_ratio",
                              "salary_cap_equiv"] if c in feats.columns]
    other_cols = [c for c in feats.columns if c not in front_cols]
    feats = feats[front_cols + other_cols]

    # 5) 保存特征表
    save_df(feats, out_path, also_csv=True)

    # 6) 生成简单报告
    rpt_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "input_path": str(in_path),
        "output_path_parquet": str(out_path),
        "output_path_csv": str(out_path.with_suffix(".csv")),
        "orig_shape": list(orig_shape),
        "final_shape": list(feats.shape),
        "orig_columns_count": len(orig_cols),
        "final_columns_count": feats.shape[1],
        "dropped_text_cols": args.drop_text_cols,
        "target_selected": args.target,
        "contains": {k: (k in feats.columns) for k in [
            "salary_usd", "log_salary",
            "salary_cap", "salary_cap_ratio", "log_salary_cap_ratio", "salary_cap_equiv"
        ]},
    }
    rpt_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 7) 与几个“候选目标”的相关性导出
    corr_path.parent.mkdir(parents=True, exist_ok=True)
    numeric = feats.select_dtypes(include="number")
    targets = [t for t in ["salary_usd","log_salary","salary_cap_ratio","log_salary_cap_ratio","salary_cap_equiv"]
               if t in numeric.columns]

    corr_frames = []
    for tgt in targets:
        corr = numeric.drop(columns=[tgt]).corrwith(numeric[tgt]).sort_values(ascending=False)
        part = pd.DataFrame({"feature": corr.index, f"corr_with_{tgt}": corr.values})
        corr_frames.append(part.set_index("feature"))
    if corr_frames:
        corr_all = pd.concat(corr_frames, axis=1).sort_index()
        corr_all.to_csv(corr_path, index=True)

    # 8) 控制台信息
    print("✅ Features built.")
    print("   Input :", in_path, "shape:", orig_shape)
    print("   Output:", out_path, "shape:", feats.shape)
    print("   Report:", rpt_path)
    print("   Corr  :", corr_path)

if __name__ == "__main__":
    main()