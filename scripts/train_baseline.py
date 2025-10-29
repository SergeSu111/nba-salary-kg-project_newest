#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train baseline models on tabular features (core/full) with a configurable split.
- Split: random hold-out or time-based (by 'season')
- Models: Ridge, RF, XGB (if installed), LGBM (if installed), MLP
- Targets: salary_cap_ratio (default) + others
- Saves: metrics.csv (per feature set + grand), residual plots, importances, preds, models
"""

from __future__ import annotations
import argparse, json, os, sys, warnings, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# ---- headless plotting (avoid Tk) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")

# ----- Optional libs -----
XGB_OK = True
try:
    import xgboost as xgb
except Exception:
    XGB_OK = False

LGBM_OK = True
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBM_OK = False


# ============== Utils ==============

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def load_feature_list(p: Path) -> List[str]:
    """
    Load feature list from JSON({"features":[...]}) or plain text (one feature per line).
    """
    txt = p.read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "features" in obj:
            return list(obj["features"])
        elif isinstance(obj, list):
            return list(obj)
    except Exception:
        pass
    return [line.strip() for line in txt.splitlines() if line.strip()]

def make_xy(df: pd.DataFrame, features: List[str], target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[features].copy().fillna(0)
    y = df[target].copy().fillna(0)
    return X, y

def evaluate(y_true, y_pred) -> Tuple[float, float]:
    return r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred, squared=False)

def plot_residuals(y_true, y_pred, out_png: Path | None):
    resid = y_true - y_pred
    plt.figure(figsize=(5,4))
    plt.scatter(y_pred, resid, s=8, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted"); plt.ylabel("Residuals"); plt.title("Residuals")
    plt.tight_layout()
    if out_png: plt.savefig(out_png, dpi=160)
    plt.close()

def save_importance(series: pd.Series, out_csv: Path, topk_plot: int = 25):
    series = series.copy()
    series.index = series.index.astype(str)  # 避免混入非字符串 index
    series.sort_values(ascending=False).to_csv(out_csv, header=["importance"])
    top = series.sort_values().tail(topk_plot)
    plt.figure(figsize=(6, max(3, 0.3*len(top))))
    top.plot(kind="barh")
    plt.title(out_csv.stem)
    plt.tight_layout()
    plt.savefig(out_csv.with_suffix(".png"), dpi=160)
    plt.close()

def time_split(
    df: pd.DataFrame, season_col: str, cutoff_season: int, features: List[str], target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Train = season < cutoff, Valid = season >= cutoff
    """
    assert season_col in df.columns, f"Missing season column '{season_col}'"
    train_mask = df[season_col] < cutoff_season
    valid_mask = df[season_col] >= cutoff_season

    X, y = make_xy(df, features, target)
    Xtr, ytr = X[train_mask], y[train_mask]
    Xva, yva = X[valid_mask], y[valid_mask]

    meta = {
        "split": "time",
        "season_col": season_col,
        "cutoff": int(cutoff_season),
        "train_seasons": sorted(df.loc[train_mask, season_col].unique().tolist()),
        "valid_seasons": sorted(df.loc[valid_mask, season_col].unique().tolist()),
        "train_size": int(train_mask.sum()),
        "valid_size": int(valid_mask.sum()),
    }
    return Xtr, Xva, ytr, yva, meta


# ============== Model trainers ==============

def fit_ridge(Xtr, ytr, Xva, yva, alpha=1.0):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge",  Ridge(alpha=alpha, random_state=42))
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xva)
    coefs = np.abs(pipe.named_steps["ridge"].coef_)
    imp = pd.Series(coefs, index=Xtr.columns)
    return "Ridge", pipe, pred, imp

def fit_rf(Xtr, ytr, Xva, yva):
    rf = RandomForestRegressor(
        n_estimators=800, random_state=42, n_jobs=-1
    )
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xva)
    imp = pd.Series(rf.feature_importances_, index=Xtr.columns)
    return "RF", rf, pred, imp

def fit_xgb(Xtr, ytr, Xva, yva):
    if not XGB_OK:
        return None, None, None, None
    # DMatrix 会携带列名，get_score 可返回列名级重要性
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=Xtr.columns.tolist())
    dva = xgb.DMatrix(Xva, label=yva, feature_names=Xva.columns.tolist())
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    booster = xgb.train(
        params, dtr, num_boost_round=500,
        evals=[(dva, "val")], early_stopping_rounds=30, verbose_eval=False
    )
    pred = booster.predict(dva)
    score_map = booster.get_score(importance_type="gain")
    # 对齐列名（如果 score_map 缺某些列，补 0）
    imp = pd.Series({c: score_map.get(c, 0.0) for c in Xtr.columns})
    return "XGB", booster, pred, imp

def fit_lgbm(Xtr, ytr, Xva, yva):
    if not LGBM_OK:
        return None, None, None, None
    lgbm = LGBMRegressor(
        n_estimators=600, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42
    )
    # 兼容不同版本：不传 verbose
    lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l2")
    pred = lgbm.predict(Xva)
    imp = pd.Series(lgbm.feature_importances_, index=Xtr.columns)
    return "LGBM", lgbm, pred, imp

def fit_mlp(Xtr, ytr, Xva, yva):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPRegressor(hidden_layer_sizes=(128,64), activation="relu",
                             max_iter=1000, random_state=42))
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xva)
    # permutation importance（在验证集上），可能略慢，但特征数<=40 可接受
    pi = permutation_importance(pipe, Xva, yva, n_repeats=10, random_state=42, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=Xtr.columns)
    return "MLP", pipe, pred, imp


MODEL_REGISTRY = {
    "ridge": fit_ridge,
    "rf":    fit_rf,
    "xgb":   fit_xgb,
    "lgbm":  fit_lgbm,
    "mlp":   fit_mlp,
}

DEFAULT_TARGETS = [
    "salary_cap_ratio",       # 推荐 baseline target
    "salary_cap_equiv",
    "log_salary_cap_ratio",
    "salary_usd",
    "log_salary",
]


# ============== Runner ==============

def run_block(
    Xtr, Xva, ytr, yva,
    feature_set_name: str,
    target: str,
    models: List[str],
    out_dir: Path,
    save_pred: bool = True
) -> pd.DataFrame:

    rows = []
    for m in models:
        if m not in MODEL_REGISTRY:
            print(f"[Skip] Unknown model: {m}")
            continue
        name, model, pred, imp = MODEL_REGISTRY[m](Xtr, ytr, Xva, yva)
        if name is None:
            print(f"[Skip] {m} (dependency not available)")
            continue

        r2, rmse = evaluate(yva, pred)
        print(f"[{name}] set={feature_set_name}  target={target}  R^2={r2:.4f}  RMSE={rmse:.6f}")

        # Save residual plot / importance
        plot_residuals(yva, pred, out_dir / f"resid_{feature_set_name}_{name}_{target}.png")
        save_importance(imp, out_dir / f"importance_{feature_set_name}_{name}_{target}.csv")

        # Save predictions (optional)
        if save_pred:
            pred_df = pd.DataFrame({
                "y_true": yva.values,
                "y_pred": pred,
            }, index=yva.index)
            pred_df.to_csv(out_dir / f"pred_{feature_set_name}_{name}_{target}.csv", index=True)

        # Save model
        if name in ("Ridge", "RF", "MLP", "LGBM"):
            joblib.dump(model, out_dir / f"model_{feature_set_name}_{name}_{target}.joblib")
        elif name == "XGB":
            model.save_model(str(out_dir / f"model_{feature_set_name}_{name}_{target}.json"))

        rows.append({
            "feature_set": feature_set_name,
            "model": name,
            "target": target,
            "R2": r2,
            "RMSE": rmse,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Train baseline models on selected features.")
    ap.add_argument("--data", default="data/processed/training_oncourt_features.parquet")
    ap.add_argument("--features_core", default="notebooks/reports/features/selected_features_core.json",
                    help="Core feature list (json or txt)")
    ap.add_argument("--features_full", default="notebooks/reports/features/selected_features_full.json",
                    help="Full feature list (json or txt)")
    ap.add_argument("--feature_set", choices=["core","full","both"], default="both",
                    help="Which feature set to train on")

    # --- split options ---
    ap.add_argument("--split", choices=["random","time"], default="time",
                    help="random holdout or time-based split")
    ap.add_argument("--test_size", type=float, default=0.2, help="Hold-out ratio (random split)")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed")
    ap.add_argument("--season_col", default="season", help="Season column name for time split")
    ap.add_argument("--time_cutoff", type=int, default=2024,
                    help="Time split cutoff (train: season < cutoff; valid: season >= cutoff)")

    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                    help="Targets to run")
    ap.add_argument("--models", nargs="+", default=["ridge","rf","xgb","lgbm","mlp"],
                    help="Models to run")
    ap.add_argument("--outdir", default="notebooks/reports/baseline_cli", help="Output base dir")
    args = ap.parse_args()

    # ---- load data & features ----
    data_path = Path(args.data)
    df = pd.read_parquet(data_path)

    core_feats = load_feature_list(Path(args.features_core))
    full_feats = load_feature_list(Path(args.features_full))

    base_out = Path(args.outdir)
    base_out.mkdir(parents=True, exist_ok=True)
    stamp = now_tag()

    # ---- choose feature sets ----
    sets = []
    if args.feature_set in ("core","both"):
        sets.append(("core", core_feats))
    if args.feature_set in ("full","both"):
        sets.append(("full", full_feats))

    grand_metrics = []  # 所有组的汇总

    # ---- run per feature set ----
    for set_name, feats in sets:
        out_dir = base_out / f"{set_name}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        per_set_metrics = []  # 该组的汇总

        # --- split ---
        if args.split == "random":
            # 全量随机切分（注意：random 仅作为对照；正式实验建议用 time）
            # 这里的 random_state 保证复现
            # X/y 会在目标循环里生成；为了避免重复构造，先在循环里处理
            split_meta = {
                "split": "random",
                "test_size": args.test_size,
                "random_state": args.random_state
            }
        else:
            # time split 需要先验证 season 列
            assert args.season_col in df.columns, f"Missing season column '{args.season_col}' for time split"
            split_meta = {
                "split": "time",
                "season_col": args.season_col,
                "cutoff": int(args.time_cutoff)
            }

        # --- manifest（不含目标） ---
        manifest = {
            "data": str(data_path),
            "feature_set": set_name,
            "features_count": len(feats),
            "features": feats,
            "targets": args.targets,
            "models": args.models,
            "split_meta": split_meta,
            "timestamp": stamp,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        # --- iterate targets ---
        for tgt in args.targets:
            if tgt not in df.columns:
                print(f"[Warn] Target '{tgt}' not found in df. Skip.")
                continue

            if args.split == "random":
                X, y = make_xy(df, feats, tgt)
                Xtr, Xva, ytr, yva = train_test_split(
                    X, y, test_size=args.test_size, random_state=args.random_state
                )
                # 保存索引，方便复现
                pd.Index(Xtr.index).to_series().to_csv(out_dir / f"idx_train_{set_name}_{tgt}.csv", index=False)
                pd.Index(Xva.index).to_series().to_csv(out_dir / f"idx_valid_{set_name}_{tgt}.csv", index=False)
            else:
                Xtr, Xva, ytr, yva, meta = time_split(df, args.season_col, args.time_cutoff, feats, tgt)
                # 写入该目标的 split 详情
                (out_dir / f"split_{set_name}_{tgt}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                pd.Index(Xtr.index).to_series().to_csv(out_dir / f"idx_train_{set_name}_{tgt}.csv", index=False)
                pd.Index(Xva.index).to_series().to_csv(out_dir / f"idx_valid_{set_name}_{tgt}.csv", index=False)

            met = run_block(
                Xtr=Xtr, Xva=Xva, ytr=ytr, yva=yva,
                feature_set_name=set_name, target=tgt, models=args.models,
                out_dir=out_dir, save_pred=True
            )
            per_set_metrics.append(met)

        # ---- per-set metrics ----
        if per_set_metrics:
            merged = pd.concat(per_set_metrics, ignore_index=True)
            merged.to_csv(out_dir / "metrics.csv", index=False)
            print(f"[Saved] {out_dir / 'metrics.csv'}")
            grand_metrics.append(merged)

    # ---- grand metrics (all sets) ----
    if grand_metrics:
        grand = pd.concat(grand_metrics, ignore_index=True)
        grand.to_csv(base_out / f"metrics_{stamp}.csv", index=False)
        print(f"[Saved] {base_out / f'metrics_{stamp}.csv'}")


if __name__ == "__main__":
    main()
