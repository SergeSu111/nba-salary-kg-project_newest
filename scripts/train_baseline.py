#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train baseline models on tabular features (core/full) with a simple hold-out split.
- Supports: Ridge, RandomForest, XGBoost (if installed), LightGBM (if installed), MLP
- Targets: salary_cap_ratio (default) + others
- Saves: metrics.csv, residual plots, feature importances, trained models
"""
# 把eda4的内容脚本化 

from __future__ import annotations
import argparse, json, os, sys, warnings, time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # ← 无界面后端，避免 Tkinter
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

# ---------- Optional libs ----------
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
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Predicted"); plt.ylabel("Residuals"); plt.title("Residuals")
    plt.tight_layout()
    if out_png: plt.savefig(out_png, dpi=160)
    plt.close()

def save_importance(series: pd.Series, out_csv: Path, topk_plot: int = 25):
    series.sort_values(ascending=False).to_csv(out_csv, header=["importance"])
    top = series.sort_values().tail(topk_plot)
    plt.figure(figsize=(6, max(3, 0.3*len(top))))
    top.plot(kind="barh")
    plt.title(out_csv.stem)
    plt.tight_layout()
    plt.savefig(out_csv.with_suffix(".png"), dpi=160)
    plt.close()


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
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
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
    # importance (gain)
    score_map = booster.get_score(importance_type="gain")
    imp = pd.Series({k: score_map.get(k, 0.0) for k in Xtr.columns})
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
    # 用 permutation importance 近似（在验证集上），耗时适中
    pi = permutation_importance(pipe, Xva, yva, n_repeats=10, random_state=42, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=Xtr.columns)
    return "MLP", pipe, pred, imp


# ============== Runner ==============

MODEL_REGISTRY = {
    "ridge": fit_ridge,
    "rf":    fit_rf,
    "xgb":   fit_xgb,
    "lgbm":  fit_lgbm,
    "mlp":   fit_mlp,
}

DEFAULT_TARGETS = [
    "salary_cap_ratio",       # 推荐
    # 下面这些可按需追加
    "salary_cap_equiv",
    "log_salary_cap_ratio",
    "salary_usd",
    "log_salary",
]

def run_block(df: pd.DataFrame,
              features: List[str],
              target: str,
              models: List[str],
              test_size: float,
              random_state: int,
              out_dir: Path,
              save_pred: bool = True) -> pd.DataFrame:

    X, y = make_xy(df, features, target)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, random_state=random_state)

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
        print(f"[{name}] target={target}  R^2={r2:.4f}  RMSE={rmse:.6f}")

        # Save residual plot / importance
        plot_residuals(yva, pred, out_dir / f"resid_{name}_{target}.png")
        save_importance(imp, out_dir / f"importance_{name}_{target}.csv")

        # Save predictions (optional)
        if save_pred:
            pred_df = pd.DataFrame({
                "y_true": yva.values,
                "y_pred": pred,
            }, index=yva.index)
            pred_df.to_csv(out_dir / f"pred_{name}_{target}.csv", index=True)

        # Save model
        if name in ("Ridge", "RF", "MLP"):
            joblib.dump(model, out_dir / f"model_{name}_{target}.joblib")
        elif name == "LGBM":
            joblib.dump(model, out_dir / f"model_{name}_{target}.joblib")
        elif name == "XGB":
            # xgboost Booster
            model.save_model(str(out_dir / f"model_{name}_{target}.json"))

        rows.append({
            "feature_set": out_dir.name,  # 用子目录名标识 core/full
            "model": name,
            "target": target,
            "R2": r2,
            "RMSE": rmse,
        })  #fdsf

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
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                    help="Targets to run (default: salary_cap_ratio)")
    ap.add_argument("--models", nargs="+", default=["ridge","rf","xgb","lgbm","mlp"],
                    help="Models to run")
    ap.add_argument("--test_size", type=float, default=0.2, help="Hold-out ratio")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed")
    ap.add_argument("--outdir", default="notebooks/reports/baseline_cli", help="Output base dir")
    args = ap.parse_args()

    data_path = Path(args.data)
    df = pd.read_parquet(data_path)

    core_feats = load_feature_list(Path(args.features_core))
    full_feats = load_feature_list(Path(args.features_full))

    base_out = Path(args.outdir)
    base_out.mkdir(parents=True, exist_ok=True)
    stamp = now_tag()

    # Decide feature sets
    sets = []
    if args.feature_set in ("core","both"):
        sets.append(("core", core_feats))
    if args.feature_set in ("full","both"):
        sets.append(("full", full_feats))

    # Run
    all_metrics = []
    for set_name, feats in sets:
        out_dir = base_out / f"{set_name}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save a manifest
        manifest = {
            "data": str(data_path),
            "feature_set": set_name,
            "features_count": len(feats),
            "targets": args.targets,
            "models": args.models,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "timestamp": stamp,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        for tgt in args.targets:
            if tgt not in df.columns:
                print(f"[Warn] Target '{tgt}' not found in df. Skip.")
                continue
            met = run_block(
                df=df, features=feats, target=tgt, models=args.models,
                test_size=args.test_size, random_state=args.random_state,
                out_dir=out_dir, save_pred=True
            )
            all_metrics.append(met)

        # Merge & save metrics for this set
        if all_metrics:
            merged = pd.concat(all_metrics, ignore_index=True)
            merged.to_csv(out_dir / "metrics.csv", index=False)
            print(f"[Saved] {out_dir / 'metrics.csv'}")

    # Also save a grand metrics under base_out for convenience
    if all_metrics:
        grand = pd.concat(all_metrics, ignore_index=True)
        grand.to_csv(base_out / f"metrics_{stamp}.csv", index=False)
        print(f"[Saved] {base_out / f'metrics_{stamp}.csv'}")

if __name__ == "__main__":
    main()