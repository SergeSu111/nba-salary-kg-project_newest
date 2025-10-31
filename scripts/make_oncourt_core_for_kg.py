# ★ 生成 oncourt_core_for_kg.csv
# -*- coding: utf-8 -*-
"""
make_oncourt_core_for_kg.py
---------------------------------
Purpose:
  - Build a clean CSV for Neo4j import: neo4j/import/oncourt_core_for_kg.csv
  - Source: parquet or csv (configurable & robust fallback)
  - Features list: configs/on_court_features_core_for_kg.txt (one column name per line)
  - Optional: also emit player_age.csv (Player node attrs) if 'Age' exists

Usage (examples):
  # 自动选择（有 parquet 优先 parquet，否则 csv）
  python scripts/make_oncourt_core_for_kg.py

  # 明确优先 parquet（读失败自动降级到 csv）
  python scripts/make_oncourt_core_for_kg.py --prefer parquet

  # 明确优先 csv（不会去读 parquet）
  python scripts/make_oncourt_core_for_kg.py --prefer csv \
      --input-csv data/processed/training_oncourt_features.csv

  # 指定配置与特征清单
  python scripts/make_oncourt_core_for_kg.py \
      --config configs/configs.yaml \
      --features-file configs/on_court_features_core_for_kg.txt

Outputs:
  - neo4j/import/oncourt_core_for_kg.csv
  - neo4j/import/player_age.csv (optional, only if Age exists)
  - neo4j/import/oncourt_core_for_kg.report.json
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import yaml
import pandas as pd
from typing import List, Dict, Any, Optional, Literal


# -------------------------- helpers --------------------------

def merge_cfg(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow recursive merge: base <- override (dict keys only)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_cfg(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_features_list(txt_path: str) -> List[str]:
    feats: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            feats.append(line)
    return feats


def ensure_int_series(series: pd.Series) -> pd.Series:
    """Try best-effort to coerce season/player/team id to integer.
       - accepts '2020-2021' (takes 2020)
       - accepts '2020.0'
       - accepts '2020'
    """
    def _to_int(x):
        if pd.isna(x):
            return None
        s = str(x)
        if "-" in s:
            s = s.split("-")[0]
        s = "".join([c for c in s if c.isdigit() or c == "."])
        if s == "":
            return None
        try:
            return int(float(s))
        except Exception:
            return None
    return series.map(_to_int)


def numeric_coerce(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def smart_read_table(
    preferred_parquet: Optional[str],
    fallback_csv: Optional[str],
    prefer: Literal["auto", "parquet", "csv"] = "auto",
) -> pd.DataFrame:
    """
    prefer: 'auto' | 'parquet' | 'csv'
      - 'csv'     : try CSV first (no parquet unless CSV missing)
      - 'parquet' : try parquet first; on failure/missing, fallback to CSV
      - 'auto'    : if parquet exists -> try parquet; else try CSV
    """
    def _read_csv(p: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"[WARN] CSV read failed ({type(e).__name__}): {e}")
            return None

    def _read_parquet(p: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_parquet(p)
        except Exception as e:
            print(f"[WARN] Parquet read failed ({type(e).__name__}): {e}")
            return None

    # Normalize empty strings to None
    preferred_parquet = preferred_parquet or None
    fallback_csv = fallback_csv or None

    if prefer == "csv":
        if fallback_csv and os.path.exists(fallback_csv):
            df = _read_csv(fallback_csv)
            if df is not None:
                return df
        if preferred_parquet and os.path.exists(preferred_parquet):
            df = _read_parquet(preferred_parquet)
            if df is not None:
                return df

    elif prefer == "parquet":
        if preferred_parquet and os.path.exists(preferred_parquet):
            df = _read_parquet(preferred_parquet)
            if df is not None:
                return df
        if fallback_csv and os.path.exists(fallback_csv):
            df = _read_csv(fallback_csv)
            if df is not None:
                return df

    else:  # auto
        if preferred_parquet and os.path.exists(preferred_parquet):
            df = _read_parquet(preferred_parquet)
            if df is not None:
                return df
        if fallback_csv and os.path.exists(fallback_csv):
            df = _read_csv(fallback_csv)
            if df is not None:
                return df

    raise FileNotFoundError(
        f"Neither usable: parquet={preferred_parquet} csv={fallback_csv}"
    )


def default_cfg() -> Dict[str, Any]:
    return {
        "paths": {
            "data_processed": "data/processed",
            "neo4j_import": "neo4j/import",
        },
        "ids": {
            "player_pk": "player_id",
            "team_pk": "team_id",
            "season_col": "season",
        },
    }


# -------------------------- main --------------------------

def build(args):
    # 1) load & merge config
    cfg = default_cfg()
    if args.config and os.path.exists(args.config):
        cfg = merge_cfg(cfg, load_yaml(args.config))

    processed_dir = cfg.get("paths", {}).get("data_processed", "data/processed")
    neo4j_import_dir = cfg.get("paths", {}).get("neo4j_import", "neo4j/import")
    os.makedirs(neo4j_import_dir, exist_ok=True)

    PLAYER_PK = cfg.get("ids", {}).get("player_pk", "player_id")
    TEAM_PK   = cfg.get("ids", {}).get("team_pk", "team_id")     # kept for future use
    SEASONCOL = cfg.get("ids", {}).get("season_col", "season")

    # 2) source table selection
    parquet_path = args.input_parquet or os.path.join(
        processed_dir, "training_oncourt_features.parquet"
    )
    csv_path = args.input_csv or os.path.join(
        processed_dir, "training_oncourt_features.csv"
    )

    # prefer decision: CLI > heuristic
    prefer = args.prefer
    if prefer == "auto":
        # If user explicitly supplied only CSV, make it 'csv'
        if args.input_csv and not args.input_parquet:
            prefer = "csv"
        # If user explicitly supplied parquet, prefer parquet (with fallback)
        elif args.input_parquet:
            prefer = "parquet"
        else:
            prefer = "auto"

    df = smart_read_table(parquet_path, csv_path, prefer=prefer)
    print(f"[INFO] Source loaded by prefer='{prefer}': shape={df.shape}")

    # 3) normalize common column names to canonical
    rename_plan: Dict[str, str] = {}

    # player id variants
    for cand in [PLAYER_PK, "Player_id", "playerId", "player_id_int"]:
        if cand in df.columns:
            rename_plan[cand] = PLAYER_PK
            break

    # season variants
    for cand in [SEASONCOL, "Season", "season_int", "year", "Year"]:
        if cand in df.columns:
            rename_plan[cand] = SEASONCOL
            break

    # age variants (optional)
    for cand in ["Age", "age"]:
        if cand in df.columns:
            rename_plan[cand] = "Age"
            break

    if rename_plan:
        df = df.rename(columns=rename_plan)

    # 4) standardize id types
    if PLAYER_PK in df.columns:
        df[PLAYER_PK] = ensure_int_series(df[PLAYER_PK])
    if SEASONCOL in df.columns:
        df[SEASONCOL] = ensure_int_series(df[SEASONCOL])

    # 5) features list
    feats_file = args.features_file or "configs/on_court_features_core_for_kg.txt"
    if not os.path.exists(feats_file):
        raise FileNotFoundError(f"Features file not found: {feats_file}")
    core_feats = read_features_list(feats_file)
    print(f"[INFO] Requested core features: {len(core_feats)}")

    # 6) determine targets (optional)
    target_cols = [c for c in ["salary_cap_ratio", "log_salary_cap_ratio"] if c in df.columns]

    # 7) assemble output columns
    base_cols = [c for c in [PLAYER_PK, SEASONCOL] if c in df.columns]
    existing_feats = [c for c in core_feats if c in df.columns]
    missing_feats = sorted(set(core_feats) - set(existing_feats))
    if missing_feats:
        print(f"[WARN] Missing features will be skipped: {missing_feats}")

    out_cols = base_cols + existing_feats + target_cols

    # unique & keep order
    seen = set()
    out_cols = [c for c in out_cols if not (c in seen or seen.add(c))]

    if not out_cols:
        raise ValueError(
            "No columns selected for output. "
            f"Check presence of IDs ({PLAYER_PK},{SEASONCOL}) and your features list."
        )

    df_out = df[out_cols].copy()

    # 8) numeric coercion for non-id cols
    num_cols = [c for c in df_out.columns if c not in [PLAYER_PK, SEASONCOL]]
    df_out = numeric_coerce(df_out, num_cols)

    # 9) drop rows missing ids
    before = len(df_out)
    df_out = df_out.dropna(subset=[PLAYER_PK, SEASONCOL])
    after = len(df_out)
    if after < before:
        print(f"[WARN] Dropped {before - after} rows due to missing {PLAYER_PK}/{SEASONCOL}")

    # 10) deduplicate per (player, season)
    df_out = (
        df_out.sort_values([PLAYER_PK, SEASONCOL])
              .drop_duplicates([PLAYER_PK, SEASONCOL], keep="last")
    )

    # 11) cast ids to int
    df_out[PLAYER_PK] = df_out[PLAYER_PK].astype(int)
    df_out[SEASONCOL] = df_out[SEASONCOL].astype(int)

    # 12) write main CSV
    out_csv = os.path.join(neo4j_import_dir, "oncourt_core_for_kg.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} with shape {df_out.shape}")

    # 13) optional: emit player_age table (latest season per player)
    age_csv_path = None
    if "Age" in df.columns and PLAYER_PK in df.columns and SEASONCOL in df.columns:
        age_df = df[[PLAYER_PK, SEASONCOL, "Age"]].dropna()
        if not age_df.empty:
            age_df = (
                age_df.sort_values([PLAYER_PK, SEASONCOL])
                      .drop_duplicates([PLAYER_PK], keep="last")
            )
            age_df[PLAYER_PK] = ensure_int_series(age_df[PLAYER_PK]).astype("Int64")
            age_df = age_df.dropna(subset=[PLAYER_PK])
            age_df[PLAYER_PK] = age_df[PLAYER_PK].astype(int)

            age_csv_path = os.path.join(neo4j_import_dir, "player_age.csv")
            age_df[[PLAYER_PK, "Age"]].to_csv(age_csv_path, index=False)
            print(f"[OK] Wrote {age_csv_path} with shape {age_df[[PLAYER_PK, 'Age']].shape}")

    # 14) simple report
    report = {
        "generated_at": pd.Timestamp.now(tz='UTC').isoformat(),
        "source_shape": list(df.shape),
        "output_shape": list(df_out.shape),
        "ids_present": {
            PLAYER_PK: PLAYER_PK in df.columns,
            SEASONCOL: SEASONCOL in df.columns
        },
        "targets_present": {c: c in df.columns for c in ["salary_cap_ratio", "log_salary_cap_ratio"]},
        "requested_features": core_feats,
        "kept_features": existing_feats,
        "missing_features": missing_feats,
        "age_csv": bool(age_csv_path),
        "prefer_mode": prefer,
        "source_used": ("parquet" if prefer == "parquet" and os.path.exists(parquet_path)
                        else "csv" if os.path.exists(csv_path) else "unknown")
    }
    report_path = os.path.join(neo4j_import_dir, "oncourt_core_for_kg.report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote report: {report_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/configs.yaml",
                    help="Path to configs.yaml (optional; merges into defaults).")
    ap.add_argument("--features-file", type=str, default="configs/on_court_features_core_for_kg.txt",
                    help="Path to features list (one column name per line).")
    ap.add_argument("--input-parquet", type=str, default=None,
                    help="Override input parquet path (optional).")
    ap.add_argument("--input-csv", type=str, default=None,
                    help="Override input csv path (optional).")
    ap.add_argument("--prefer", choices=["auto", "parquet", "csv"], default="auto",
                    help="Preferred engine when both parquet and csv are available.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args)