# scripts/evaluate_relational_models.py
# Paper-grade aligned evaluation for Node2Vec / RotatE (+ future GNN)
# - Same L0' (on-court features)
# - Same train/test split by season
# - Same sample set across methods (intersection of player_id coverage)
# - Handles RotatE complex embeddings by converting to Re/Im numeric features

from __future__ import annotations
from sklearn.impute import SimpleImputer

from pathlib import Path
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===== ensure project root on PYTHONPATH =====
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]   # scripts/ -> project root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ============================================
from src.features.get_just_oncourt import get_oncourt_cols
import inspect
import src.features.get_just_oncourt as gj
print("[DEBUG] get_just_oncourt loaded from:", gj.__file__)
print("[DEBUG] get_oncourt_cols source file:", inspect.getsourcefile(get_oncourt_cols))

# ---------------------------
# Config (adjust if needed)
# ---------------------------
TEST_SEASON = 2024
TARGET_COL = "log_salary"
ID_COLS = ["player_id", "season"]

# sanity keywords that should NOT be in L0'
_L0_BAD_KEYWORDS = ["team", "agent", "draft", "salary_cap", "award_", "injury_"]


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


# ---------------------------
# L0' loader (tabular)
# ---------------------------
def load_tabular(tabular_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not tabular_path.exists():
        raise FileNotFoundError(f"Tabular file not found: {tabular_path}")

    df = pd.read_csv(tabular_path)

    # Freeze L0' definition from canonical function
    oncourt_cols = get_oncourt_cols(df)

    # Defensive: ensure no award_/injury_ sneak in (case-insensitive)
    oncourt_cols = [c for c in oncourt_cols if not c.lower().startswith(("award_", "injury_"))]

    keep_cols = ID_COLS + [TARGET_COL] + oncourt_cols
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in tabular: {missing[:20]}")

    df = df[keep_cols].dropna(subset=[TARGET_COL]).copy()

    # Normalize types
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)

    # L0' sanity checks
    bad = [c for c in oncourt_cols if any(k in c.lower() for k in _L0_BAD_KEYWORDS)]
    if len(bad) > 0:
        # Not always fatal, but for paper-grade we fail fast
        raise ValueError(f"L0' contamination detected (should be on-court only). Examples: {bad[:20]}")

    print(f"[tabular] shape={df.shape}  unique_players={df['player_id'].nunique()}  seasons={sorted(df['season'].unique())[:3]}...{sorted(df['season'].unique())[-3:]}")
    print(f"[L0′] oncourt_cols={len(oncourt_cols)}  sample={oncourt_cols[:10]}")
    return df, oncourt_cols


# ---------------------------
# Embedding loaders
# ---------------------------
def get_player_set(emb_path: Path) -> Set[str]:
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    emb = pd.read_csv(emb_path, usecols=["player_id"])
    return set(emb["player_id"].astype(str))


def _detect_rotate_complex_columns(df_emb: pd.DataFrame) -> List[str]:
    # Typical RotatE export: e0..e127 with complex numbers serialized as strings
    emb_cols = [c for c in df_emb.columns if c.startswith("e")]
    if not emb_cols:
        return []
    # Look at a few values to decide if complex strings
    s = df_emb[emb_cols[0]].dropna()
    if len(s) == 0:
        return []
    v = str(s.iloc[0])
    # if it contains 'j' or ends with ')', often complex format
    if ("j" in v) or ("(" in v and ")" in v):
        return emb_cols
    return []


def load_embedding_player_features(
    emb_path: Path,
    allowed_players: Set[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns:
      emb_df: columns = ["player_id"] + numeric embedding columns
      emb_cols: list of numeric embedding feature column names (excluding player_id)
    Handles:
      - Node2Vec numeric e0..e*
      - RotatE complex e0..e* by converting to re/im numeric columns
    """
    emb = pd.read_csv(emb_path)
    if "player_id" not in emb.columns:
        raise ValueError(f"'player_id' column missing in {emb_path}")

    emb["player_id"] = emb["player_id"].astype(str)

    # Restrict to common sample
    emb = emb[emb["player_id"].isin(allowed_players)].copy()

    # Duplicate check
    dup = emb["player_id"].duplicated().sum()
    if dup > 0:
        raise ValueError(f"Duplicate player_id found in embedding file {emb_path}: {dup}")

    # Identify embedding columns
    emb_cols = [c for c in emb.columns if c.startswith("e")]

    if len(emb_cols) == 0:
        raise ValueError(f"No embedding columns starting with 'e' found in {emb_path}")

    # RotatE complex -> Re/Im
    rotate_complex_cols = _detect_rotate_complex_columns(emb)
    if rotate_complex_cols:
        print(f"[emb] Detected RotatE complex embeddings in {emb_path.name}. Converting to Re/Im...")
        # Convert to complex then split
        Z = emb[rotate_complex_cols].applymap(lambda x: complex(x)).to_numpy()
        Z_re = np.real(Z)
        Z_im = np.imag(Z)

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
        emb_cols_num = re_cols + im_cols
        return emb_num, emb_cols_num

    # Node2Vec (numeric already)
    # Ensure numeric dtype
    emb_num = emb[["player_id"] + emb_cols].copy()
    # coerce to numeric, fail fast if not numeric
    for c in emb_cols:
        emb_num[c] = pd.to_numeric(emb_num[c], errors="raise")

    return emb_num, emb_cols


def merge_tabular_and_embedding(
    df_tab: pd.DataFrame,
    emb_df: pd.DataFrame,
    emb_cols: List[str],
) -> pd.DataFrame:
    # Inner merge to keep strict intersection
    df = df_tab.merge(emb_df, on="player_id", how="inner")

    # Sanity: after merge, embedding columns must be present and non-null
    if df[emb_cols].isna().any().any():
        # In strict intersection this should not happen; treat as bug
        bad_rate = df[emb_cols].isna().any(axis=1).mean()
        raise ValueError(f"Unexpected NaNs in embedding columns after merge. NaN row rate={bad_rate:.2%}")

    return df


# ---------------------------
# Models
# ---------------------------
def run_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oncourt_cols: List[str],
    emb_cols: List[str],
    seed: int = 42,
) -> List[Dict[str, float]]:

    Xtr = train_df[oncourt_cols + emb_cols].to_numpy()
    ytr = train_df[TARGET_COL].to_numpy(dtype=float)

    Xte = test_df[oncourt_cols + emb_cols].to_numpy()
    yte = test_df[TARGET_COL].to_numpy(dtype=float)

    # ---- impute missing values (paper-grade consistent preprocessing) ----
    imputer = SimpleImputer(strategy="median")
    Xtr_imp = imputer.fit_transform(Xtr)
    Xte_imp = imputer.transform(Xte)

    results: List[Dict[str, float]] = []

    # Ridge
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0))
    ])
    ridge.fit(Xtr_imp, ytr)
    pred = ridge.predict(Xte_imp)
    res = eval_reg(yte, pred)
    res["model"] = "Ridge"
    results.append(res)

    # RandomForest
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(Xtr_imp, ytr)
    pred = rf.predict(Xte_imp)
    res = eval_reg(yte, pred)
    res["model"] = "RandomForest"
    results.append(res)

    return results


# ---------------------------
# Evaluation runner
# ---------------------------
def evaluate_setting(
    df_tab_common: pd.DataFrame,
    oncourt_cols: List[str],
    emb_path: Path,
    label: str,
) -> List[Dict[str, float]]:
    allowed_players = set(df_tab_common["player_id"].astype(str))

    emb_df, emb_cols = load_embedding_player_features(emb_path, allowed_players)
    df = merge_tabular_and_embedding(df_tab_common, emb_df, emb_cols)

    train, test = split_train_test(df, TEST_SEASON)
    feat_cols = oncourt_cols + emb_cols
    nan_rate = df[feat_cols].isna().mean().sort_values(ascending=False)
    top = nan_rate[nan_rate > 0].head(10)
    if len(top) > 0:
        print(f"[{label}] NaN columns (top):")
        print(top)

    print(f"[{label}] merged shape={df.shape}  unique_players={df['player_id'].nunique()}  emb_dim={len(emb_cols)}")
    print(f"[{label}] n_train={len(train)}  n_test={len(test)}")

    rows = run_models(train, test, oncourt_cols, emb_cols)

    for r in rows:
        r["setting"] = label
        r["n_train"] = int(len(train))
        r["n_test"] = int(len(test))
        r["p"] = int(len(oncourt_cols) + len(emb_cols))
    return rows


def main() -> None:
    # Default paths (adjust if your repo differs)
    TAB = Path("data/processed/training_level1_full.csv")
    NODE2VEC = Path("graph/embeddings/node2vec_L1A_player_embeddings.csv")
    ROTATE = Path("graph/embeddings/rotate_L1B_cpu_player_embeddings.csv")

    # Load tabular + L0'
    df_tab, oncourt_cols = load_tabular(TAB)

    # Paper-grade alignment:
    # Use the same player subset across all relational methods (intersection of player coverage)
    p_node2vec = get_player_set(NODE2VEC)
    p_rotate = get_player_set(ROTATE)
    common_players = p_node2vec & p_rotate

    if len(common_players) == 0:
        raise ValueError("Intersection of players between Node2Vec and RotatE is empty. Check embedding files.")

    df_tab_common = df_tab[df_tab["player_id"].isin(common_players)].copy()

    print(f"[common] players node2vec={len(p_node2vec)} rotate={len(p_rotate)} common={len(common_players)}")
    print(f"[common] tabular rows before={len(df_tab)} after={len(df_tab_common)}")

    rows: List[Dict[str, float]] = []
    rows += evaluate_setting(df_tab_common, oncourt_cols, NODE2VEC, "L1-A (Node2Vec)")
    rows += evaluate_setting(df_tab_common, oncourt_cols, ROTATE, "L1-B (RotatE)")

    out = pd.DataFrame(rows).sort_values(["setting", "model"]).reset_index(drop=True)

    _ensure_dir(Path("results"))
    out_path = Path("results/paper_table_step2.csv")
    out.to_csv(out_path, index=False)

    print("\n=== Paper Table (Step 2) ===")
    print(out)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
