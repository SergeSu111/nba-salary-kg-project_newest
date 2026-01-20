#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/evaluate_relational_models.py

Paper-grade aligned evaluation for:
- Baseline: L0' only, (optional) L0' + time
- Relational embeddings: Node2Vec / RotatE / GNN V0 / GNN V1
Settings per method:
  - L0' + emb
  - emb only
  - (optional) L0' + emb + time

Key guarantees:
- Same L0' definition via src.features.get_just_oncourt.get_oncourt_cols
- Same season split: train < TEST_SEASON, test == TEST_SEASON
- Same sample set across methods: intersection of player_id coverage across ALL embedding files (paper mode)
- Multi-seed reporting (mean ± std)
- Robust RotatE complex parsing (supports i/j/parentheses)

Important note (time leakage / transductive risk):
- Embeddings are merged by player_id to all seasons, which is correct for "static player embedding".
- But you must ensure embedding graphs are built without using test-season edges (or declare transductive setting).
  Keep this clear in notes/paper.

Run from repo root:
  conda run -n nba-research python scripts/evaluate_relational_models.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ===== ensure project root on PYTHONPATH =====
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # scripts/ -> project root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ============================================

from src.features.get_just_oncourt import get_oncourt_cols  # noqa: E402


# ---------------------------
# Config
# ---------------------------
TEST_SEASON = 2024
TARGET_COL = "log_salary"
ID_COLS = ["player_id", "season"]

TIME_FEATS = ["age_now", "years_since_draft"]  # optional
SEEDS = [0, 1, 2, 3, 4]

# paper mode: require all embedding files exist; else crash (fixed sample)
PAPER_MODE_REQUIRE_ALL = True  # set False for dev convenience

# cold-start evaluation inside test season
EVAL_COLD_START_2024 = True
COLD_START_MIN_ROWS = 30  # recommend >=30 or >=50 for stability

# optional: also run FULL baseline (no intersection) for appendix/context
ALSO_RUN_FULL_BASELINE = False

# sanity keywords that should NOT be in L0'
_L0_BAD_KEYWORDS = ["team", "agent", "draft", "salary_cap", "award_", "injury_"]

# Default paths
TAB = Path("data/processed/training_level1_full.csv")
NODE2VEC = Path("graph/embeddings/node2vec_L1A_player_embeddings.csv")
ROTATE = Path("graph/embeddings/rotate_L1B_cpu_player_embeddings.csv")
GNN_V0 = Path("graph/embeddings/gnn_v0_sage_player_embeddings.csv")
GNN_V1 = Path("graph/embeddings/gnn_v1_sage_player_embeddings.csv")


# ---------------------------
# Utilities
# ---------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def eval_reg(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def split_train_test(df: pd.DataFrame, test_season: int = TEST_SEASON) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"] < test_season].copy()
    test = df[df["season"] == test_season].copy()
    if len(test) == 0:
        raise ValueError(f"Test set empty for season=={test_season}. Check seasons in your data.")
    return train, test


def overlap_ratio_train_test_players(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    train_players = set(train_df["player_id"].astype(str).unique())
    test_players = set(test_df["player_id"].astype(str).unique())
    if len(test_players) == 0:
        return float("nan")
    return len(train_players & test_players) / len(test_players)


def cold_start_2024_subset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_players = set(train_df["player_id"].astype(str).unique())
    return test_df[~test_df["player_id"].astype(str).isin(train_players)].copy()


# ---------------------------
# L0' loader (tabular)
# ---------------------------
def load_tabular(tabular_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not tabular_path.exists():
        raise FileNotFoundError(f"Tabular file not found: {tabular_path}")

    df = pd.read_csv(tabular_path)

    # Freeze L0' definition from canonical function
    oncourt_cols = get_oncourt_cols(df)

    # Defensive: ensure no award_/injury_ sneak in
    oncourt_cols = [c for c in oncourt_cols if not c.lower().startswith(("award_", "injury_"))]

    keep_cols = ID_COLS + [TARGET_COL] + oncourt_cols
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in tabular: {missing[:20]}")

    df = df[keep_cols].dropna(subset=[TARGET_COL]).copy()

    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)

    bad = [c for c in oncourt_cols if any(k in c.lower() for k in _L0_BAD_KEYWORDS)]
    if bad:
        raise ValueError(f"L0' contamination detected. Examples: {bad[:20]}")

    print(f"[tabular] shape={df.shape} unique_players={df['player_id'].nunique()} seasons={sorted(df['season'].unique())[:3]}...{sorted(df['season'].unique())[-3:]}")
    print(f"[L0′] oncourt_cols={len(oncourt_cols)} sample={oncourt_cols[:10]}")
    return df, oncourt_cols


def load_time_feats(tabular_path: Path, time_cols: List[str]) -> Optional[pd.DataFrame]:
    raw = pd.read_csv(tabular_path, nrows=5)
    missing = [c for c in time_cols if c not in raw.columns]
    if missing:
        print(f"[WARN] Time feats missing in tabular: {missing}. Time settings will be skipped.")
        return None
    df = pd.read_csv(tabular_path, usecols=["player_id", "season"] + time_cols)
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    for c in time_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_time_feats(df: pd.DataFrame, time_df: pd.DataFrame, time_cols: List[str]) -> pd.DataFrame:
    out = df.merge(time_df, on=["player_id", "season"], how="left")
    train_mask = out["season"] < TEST_SEASON
    for c in time_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = out.loc[train_mask, c].median()
        out[c] = out[c].fillna(med)
    return out


# ---------------------------
# Embedding loaders
# ---------------------------
def get_player_set(emb_path: Path) -> Set[str]:
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    emb = pd.read_csv(emb_path, usecols=["player_id"])
    return set(emb["player_id"].astype(str))


def _detect_rotate_complex_columns(df_emb: pd.DataFrame) -> List[str]:
    emb_cols = [c for c in df_emb.columns if c.startswith("e")]
    if not emb_cols:
        return []
    s = df_emb[emb_cols[0]].dropna()
    if len(s) == 0:
        return []
    v = str(s.iloc[0])
    if ("j" in v) or ("i" in v) or ("(" in v and ")" in v):
        return emb_cols
    return []


def _parse_complex_safe(x) -> complex:
    s = str(x).strip().replace("i", "j")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    try:
        return complex(s)
    except Exception as e:
        raise ValueError(f"Cannot parse complex value: {x!r} -> {s!r}") from e


def load_embedding_player_features(emb_path: Path, allowed_players: Set[str]) -> Tuple[pd.DataFrame, List[str]]:
    emb = pd.read_csv(emb_path)
    if "player_id" not in emb.columns:
        raise ValueError(f"'player_id' column missing in {emb_path}")

    emb["player_id"] = emb["player_id"].astype(str)
    emb = emb[emb["player_id"].isin(allowed_players)].copy()

    dup = emb["player_id"].duplicated().sum()
    if dup > 0:
        raise ValueError(f"Duplicate player_id in {emb_path}: {dup}")

    emb_cols = [c for c in emb.columns if c.startswith("e")]
    if not emb_cols:
        raise ValueError(f"No embedding columns e* in {emb_path}")

    rotate_complex_cols = _detect_rotate_complex_columns(emb)
    if rotate_complex_cols:
        print(f"[emb] Detected complex embeddings in {emb_path.name}. Converting to Re/Im...")
        Z = emb[rotate_complex_cols].applymap(_parse_complex_safe).to_numpy()
        Z_re, Z_im = np.real(Z), np.imag(Z)
        re_cols = [f"{c}_re" for c in rotate_complex_cols]
        im_cols = [f"{c}_im" for c in rotate_complex_cols]
        emb_num = pd.concat(
            [
                emb[["player_id"]],
                pd.DataFrame(Z_re, columns=re_cols, index=emb.index),
                pd.DataFrame(Z_im, columns=im_cols, index=emb.index),
            ],
            axis=1,
        )
        return emb_num, re_cols + im_cols

    emb_num = emb[["player_id"] + emb_cols].copy()
    for c in emb_cols:
        emb_num[c] = pd.to_numeric(emb_num[c], errors="raise")
    return emb_num, emb_cols


def merge_tabular_and_embedding(df_tab: pd.DataFrame, emb_df: pd.DataFrame, emb_cols: List[str]) -> pd.DataFrame:
    df = df_tab.merge(emb_df, on="player_id", how="inner")
    if df[emb_cols].isna().any().any():
        bad_rate = df[emb_cols].isna().any(axis=1).mean()
        raise ValueError(f"Unexpected NaNs in emb cols after merge. NaN row rate={bad_rate:.2%}")
    return df


# ---------------------------
# Models
# ---------------------------
def run_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
) -> List[Dict[str, float]]:
    Xtr = train_df[feature_cols].to_numpy()
    ytr = train_df[TARGET_COL].to_numpy(dtype=float)
    Xte = test_df[feature_cols].to_numpy()
    yte = test_df[TARGET_COL].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(Xtr)
    Xte = imputer.transform(Xte)

    out: List[Dict[str, float]] = []

    # BUGFIX: Ridge has no random_state
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    ridge.fit(Xtr, ytr)
    pred = ridge.predict(Xte)
    r = eval_reg(yte, pred)
    r["model"] = "Ridge"
    out.append(r)

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    r = eval_reg(yte, pred)
    r["model"] = "RandomForest"
    out.append(r)

    return out


# ---------------------------
# Evaluation blocks
# ---------------------------
def evaluate_one_setting(
    df_base: pd.DataFrame,
    oncourt_cols: List[str],
    time_df: Optional[pd.DataFrame],
    time_cols: List[str],
    emb_path: Optional[Path],
    setting: str,
    use_oncourt: bool,
    use_emb: bool,
    use_time: bool,
    seed: int,
) -> List[Dict[str, float]]:
    df = df_base.copy()

    emb_cols: List[str] = []
    if use_emb:
        assert emb_path is not None
        allowed_players = set(df["player_id"].astype(str))
        emb_df, emb_cols = load_embedding_player_features(emb_path, allowed_players)
        df = merge_tabular_and_embedding(df, emb_df, emb_cols)

    if use_time:
        if time_df is None:
            raise ValueError("use_time=True but time_df is None (missing columns).")
        df = merge_time_feats(df, time_df, time_cols)

    train, test = split_train_test(df, TEST_SEASON)
    ov = overlap_ratio_train_test_players(train, test)

    feature_cols: List[str] = []
    if use_oncourt:
        feature_cols += oncourt_cols
    if use_emb:
        feature_cols += emb_cols
    if use_time:
        feature_cols += time_cols

    if len(feature_cols) == 0:
        raise ValueError(f"Setting {setting} has no features. Check flags.")

    rows = run_models(train, test, feature_cols, seed=seed)

    # Optional cold-start within 2024 (players unseen in train seasons)
    cold_rows: List[Dict[str, float]] = []
    cold_ratio = float("nan")
    n_cold = 0
    cold_players = 0
    if EVAL_COLD_START_2024:
        cold = cold_start_2024_subset(train, test)
        n_cold = int(len(cold))
        cold_players = int(cold["player_id"].nunique())
        cold_ratio = float(n_cold / len(test)) if len(test) > 0 else float("nan")
        if n_cold >= COLD_START_MIN_ROWS:
            cold_eval = run_models(train, cold, feature_cols, seed=seed)
            for r in cold_eval:
                r.update(
                    {
                        "setting": setting + " | cold_start_2024",
                        "seed": seed,
                        "n_train": int(len(train)),
                        "n_test": int(len(cold)),
                        "p": int(len(feature_cols)),
                        "train_players": int(train["player_id"].nunique()),
                        "test_players": int(cold["player_id"].nunique()),
                        "overlap_ratio_test": 0.0,
                        "cold_start_ratio": cold_ratio,
                        "n_cold_test": n_cold,
                        "cold_test_players": cold_players,
                        "emb_source": (emb_path.name if emb_path is not None else "none"),
                    }
                )
            cold_rows += cold_eval

    out_rows: List[Dict[str, float]] = []
    for r in rows:
        r.update(
            {
                "setting": setting,
                "seed": seed,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "p": int(len(feature_cols)),
                "train_players": int(train["player_id"].nunique()),
                "test_players": int(test["player_id"].nunique()),
                "overlap_ratio_test": float(ov),
                "cold_start_ratio": cold_ratio,
                "n_cold_test": n_cold,
                "cold_test_players": cold_players,
                "emb_source": (emb_path.name if emb_path is not None else "none"),
            }
        )
        out_rows.append(r)

    out_rows.extend(cold_rows)
    return out_rows


def summarize_mean_std(raw: pd.DataFrame) -> pd.DataFrame:
    grp = raw.groupby(["setting", "model"], as_index=False, observed=False)
    out = grp.agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        n_test_mean=("n_test", "mean"),
        n_test_min=("n_test", "min"),
        n_test_max=("n_test", "max"),
        test_players_mean=("test_players", "mean"),
        test_players_min=("test_players", "min"),
        test_players_max=("test_players", "max"),
        overlap_ratio_mean=("overlap_ratio_test", "mean"),
        overlap_ratio_std=("overlap_ratio_test", "std"),
        cold_ratio_mean=("cold_start_ratio", "mean"),
        cold_ratio_std=("cold_start_ratio", "std"),
        p=("p", "first"),
        emb_source=("emb_source", "first"),
    )

    def _pm(a, b) -> str:
        b = 0.0 if pd.isna(b) else float(b)
        return f"{float(a):.4f} ± {b:.4f}"

    out["R2"] = out.apply(lambda r: _pm(r["R2_mean"], r["R2_std"]), axis=1)
    out["RMSE"] = out.apply(lambda r: _pm(r["RMSE_mean"], r["RMSE_std"]), axis=1)
    out["MAE"] = out.apply(lambda r: _pm(r["MAE_mean"], r["MAE_std"]), axis=1)

    out["n_test"] = out.apply(lambda r: f"{float(r['n_test_mean']):.1f} (min={int(r['n_test_min'])}, max={int(r['n_test_max'])})", axis=1)
    out["test_players"] = out.apply(lambda r: f"{float(r['test_players_mean']):.1f} (min={int(r['test_players_min'])}, max={int(r['test_players_max'])})", axis=1)
    out["overlap_ratio_test"] = out.apply(lambda r: _pm(r["overlap_ratio_mean"], r["overlap_ratio_std"]), axis=1)
    out["cold_start_ratio"] = out.apply(lambda r: _pm(r["cold_ratio_mean"], r["cold_ratio_std"]), axis=1)

    keep = [
        "setting", "model", "R2", "RMSE", "MAE",
        "n_test", "test_players", "overlap_ratio_test", "cold_start_ratio",
        "p", "emb_source",
    ]
    out = out[keep].sort_values(["setting", "model"]).reset_index(drop=True)
    return out


def main() -> None:
    print("[NOTE] Embeddings are merged by player_id across seasons (static player embedding).")
    print("[NOTE] Please verify embedding graphs do NOT include test-season edges, or declare transductive setting in paper.\n")

    df_tab, oncourt_cols = load_tabular(TAB)
    time_df = load_time_feats(TAB, TIME_FEATS)
    time_cols = TIME_FEATS if time_df is not None else []

    emb_paths = {
        "Node2Vec": NODE2VEC,
        "RotatE": ROTATE,
        "GNN_V0": GNN_V0,
        "GNN_V1": GNN_V1,
    }

    if PAPER_MODE_REQUIRE_ALL:
        missing = [k for k, p in emb_paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"[PAPER_MODE] Missing embedding files: {missing}. "
                f"Generate them first or set PAPER_MODE_REQUIRE_ALL=False."
            )
        available = emb_paths
    else:
        available = {k: p for k, p in emb_paths.items() if p.exists()}
        for k, p in emb_paths.items():
            if not p.exists():
                print(f"[WARN] Missing embedding file for {k}: {p} (skipping)")

    # fixed common players across ALL available methods
    player_sets = {k: get_player_set(p) for k, p in available.items()}
    common_players: Set[str] = set.intersection(*player_sets.values())
    if len(common_players) == 0:
        raise ValueError("Intersection of players across embedding files is empty. Check exports.")

    df_common = df_tab[df_tab["player_id"].isin(common_players)].copy()

    print("\n[player coverage]")
    for k, s in player_sets.items():
        print(f"  {k}: {len(s)} players")
    print(f"  common across {list(available.keys())}: {len(common_players)} players")
    print(f"[common] tabular rows before={len(df_tab)} after={len(df_common)}")

    rows: List[Dict[str, float]] = []

    # Optional FULL baseline (no intersection) for appendix/context
    if ALSO_RUN_FULL_BASELINE:
        for seed in SEEDS:
            rows.extend(
                evaluate_one_setting(
                    df_base=df_tab,
                    oncourt_cols=oncourt_cols,
                    time_df=time_df,
                    time_cols=time_cols,
                    emb_path=None,
                    setting="FULL Baseline | L0' only",
                    use_oncourt=True,
                    use_emb=False,
                    use_time=False,
                    seed=seed,
                )
            )
            if time_df is not None and len(time_cols) > 0:
                rows.extend(
                    evaluate_one_setting(
                        df_base=df_tab,
                        oncourt_cols=oncourt_cols,
                        time_df=time_df,
                        time_cols=time_cols,
                        emb_path=None,
                        setting="FULL Baseline | L0' + time",
                        use_oncourt=True,
                        use_emb=False,
                        use_time=True,
                        seed=seed,
                    )
                )

    for seed in SEEDS:
        # -------- Baselines (must-have, on common subset for fairness) --------
        rows.extend(
            evaluate_one_setting(
                df_base=df_common,
                oncourt_cols=oncourt_cols,
                time_df=time_df,
                time_cols=time_cols,
                emb_path=None,
                setting="Baseline | L0' only",
                use_oncourt=True,
                use_emb=False,
                use_time=False,
                seed=seed,
            )
        )
        if time_df is not None and len(time_cols) > 0:
            rows.extend(
                evaluate_one_setting(
                    df_base=df_common,
                    oncourt_cols=oncourt_cols,
                    time_df=time_df,
                    time_cols=time_cols,
                    emb_path=None,
                    setting="Baseline | L0' + time",
                    use_oncourt=True,
                    use_emb=False,
                    use_time=True,
                    seed=seed,
                )
            )

        # -------- Embedding methods --------
        for method_key, emb_path in available.items():
            # L0' + emb
            rows.extend(
                evaluate_one_setting(
                    df_base=df_common,
                    oncourt_cols=oncourt_cols,
                    time_df=time_df,
                    time_cols=time_cols,
                    emb_path=emb_path,
                    setting=f"{method_key} | L0' + emb",
                    use_oncourt=True,
                    use_emb=True,
                    use_time=False,
                    seed=seed,
                )
            )
            # emb only
            rows.extend(
                evaluate_one_setting(
                    df_base=df_common,
                    oncourt_cols=oncourt_cols,
                    time_df=time_df,
                    time_cols=time_cols,
                    emb_path=emb_path,
                    setting=f"{method_key} | emb only",
                    use_oncourt=False,
                    use_emb=True,
                    use_time=False,
                    seed=seed,
                )
            )
            # L0' + emb + time
            if time_df is not None and len(time_cols) > 0:
                rows.extend(
                    evaluate_one_setting(
                        df_base=df_common,
                        oncourt_cols=oncourt_cols,
                        time_df=time_df,
                        time_cols=time_cols,
                        emb_path=emb_path,
                        setting=f"{method_key} | L0' + emb + time",
                        use_oncourt=True,
                        use_emb=True,
                        use_time=True,
                        seed=seed,
                    )
                )

    raw = pd.DataFrame(rows).sort_values(["setting", "model", "seed"]).reset_index(drop=True)

    _ensure_dir(Path("results"))
    raw_path = Path("results/paper_table_raw_by_seed.csv")
    raw.to_csv(raw_path, index=False)

    summary = summarize_mean_std(raw)
    summary_path = Path("results/paper_table_summary_mean_std.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=== Summary (mean ± std across seeds) ===")
    print(summary)
    print(f"\nSaved raw:     {raw_path.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
