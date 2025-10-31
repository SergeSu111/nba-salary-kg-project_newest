# -*- coding: utf-8 -*-
"""
standardize_offcourt_team_value.py
----------------------------------
Purpose:
  - Clean & standardize team valuation data for Neo4j import
  - Build: neo4j/import/offcourt_team_value_for_kg.csv
  - Emit: neo4j/import/offcourt_team_value_for_kg.report.json

Inputs (defaults, can be overridden by args or configs.yaml):
  - data/processed/long_table_team_value.with_ids.csv

Usage:
  python scripts/standardize_offcourt_team_value.py \
      --config configs/configs.yaml
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import yaml

# -------------------------- helpers --------------------------
def load_yaml(path: str):
    """Safe YAML loader"""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_abbr(s: str) -> str:
    """Normalize team abbreviations"""
    if pd.isna(s): return np.nan
    return re.sub(r"\s+", "", str(s).upper().strip())

def money_to_usd(x):
    """
    Parse strings like "$4.3B", "5.0M", "2,100K", "8800000" into USD float.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("$", "").replace(",", "").upper()
    mult = 1.0
    if s.endswith("B"):
        mult = 1e9; s = s[:-1]
    elif s.endswith("M"):
        mult = 1e6; s = s[:-1]
    elif s.endswith("K"):
        mult = 1e3; s = s[:-1]
    try:
        return float(s) * mult
    except Exception:
        return np.nan

def to_int_safe(x):
    """Convert to int; return NaN if invalid"""
    try:
        if pd.isna(x): return np.nan
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return np.nan

# -------------------------- main builder --------------------------
def build(args):
    # Load config paths
    cfg = load_yaml(args.config)
    paths = cfg.get("paths", {})
    data_processed = paths.get("data_processed", "data/processed")
    neo4j_import = paths.get("neo4j_import", "neo4j/import")
    ensure_dir(neo4j_import)

    team_val_path = args.team_value or os.path.join(data_processed, "long_table_team_value.with_ids.csv")
    if not os.path.exists(team_val_path):
        raise FileNotFoundError(f"File not found: {team_val_path}")

    # Read file
    df = pd.read_csv(team_val_path)
    before_rows = len(df)

    # Expected columns (will adapt if names differ)
    rename_map = {
        "Team_id": "team_id",
        "Team": "team_abbr",
        "Team_Name": "team_name",
        "Season": "year",
        "Value": "team_value_usd",
        "Revenue": "revenue_usd",
        "Operating_Income": "operating_income_usd",
        "Brand_Value": "brand_value_usd",
        "Value_Rank": "value_rank",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Keep only relevant columns
    keep_cols = [
        "team_id", "team_abbr", "team_name", "year",
        "team_value_usd", "revenue_usd", "operating_income_usd", "brand_value_usd", "value_rank"
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Clean
    for c in ["team_abbr", "team_name"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "team_abbr" in df.columns:
        df["team_abbr"] = df["team_abbr"].map(normalize_abbr)

    for c in ["team_value_usd", "revenue_usd", "operating_income_usd", "brand_value_usd"]:
        if c in df.columns:
            df[c] = df[c].map(money_to_usd)

    for c in ["team_id", "year", "value_rank"]:
        if c in df.columns:
            df[c] = df[c].map(to_int_safe)

    # Drop incomplete rows
    df = df.dropna(subset=["team_id", "year"]).copy()

    # Deduplicate (some teams may appear multiple times per year — keep the latest)
    df = (
        df.sort_values(["team_id", "year"], ascending=[True, True])
        .drop_duplicates(["team_id", "year"], keep="last")
        .copy()
    )

    after_rows = len(df)

    # Output paths
    out_csv = os.path.join(neo4j_import, "offcourt_team_value_for_kg.csv")
    out_report = os.path.join(neo4j_import, "offcourt_team_value_for_kg.report.json")

    # Save CSV
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # Report
    report = {
        "input_rows": int(before_rows),
        "output_rows": int(after_rows),
        "unique_team_ids": int(df["team_id"].nunique()),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "columns_out": existing_cols,
        "sample": df.head(10).to_dict(orient="records"),
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Report: {out_report}")
    print(f"[Summary] {after_rows} rows | {df['team_id'].nunique()} teams | {df['year'].nunique()} years")

# -------------------------- CLI --------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/configs.yaml")
    ap.add_argument("--team-value", type=str, dest="team_value", default="data/raw_external/long_table_team_value.with_ids.csv",
                    help="Override path to long_table_team_value.with_ids.csv")
    return ap.parse_args()

if __name__ == "__main__":
    build(parse_args())