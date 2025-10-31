# -*- coding: utf-8 -*-
"""
standardize_offcourt_draft.py (Revised)
---------------------------------------
Purpose:
  - Clean & standardize draft data for Neo4j import
  - Build: neo4j/import/offcourt_draft_for_kg.csv
  - Also emit: neo4j/import/offcourt_draft_for_kg.report.json
  - Separate truly undrafted players into offcourt_draft_undrafted.csv

Inputs:
  - data/processed/player_draft_2020-2025.matched.csv
  - data/processed/long_table_team_value.with_ids.csv
  - data/processed/NBA_team_location.csv
"""

import os, re, json, argparse
import numpy as np
import pandas as pd
import yaml
from collections import Counter

# -------------------------- helpers --------------------------
def load_yaml(path: str):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\xa0"," ").strip())

def normalize_fullname(name: str) -> str:
    """Normalize team full names for consistent mapping"""
    if pd.isna(name): return np.nan
    s = normalize_spaces(name).lower().replace("&","and")
    hist = {
        "new jersey nets": "brooklyn nets",
        "charlotte bobcats": "charlotte hornets",
        "new orleans hornets": "new orleans pelicans",
        "new orleans/oklahoma city hornets": "new orleans pelicans",
        "seattle supersonics": "oklahoma city thunder",
        "washington bullets": "washington wizards",
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
    }
    return hist.get(s, s)

def is_undrafted_value(v):
    """Detect textual signs of undrafted players"""
    if pd.isna(v): return True
    s = str(v).strip().lower()
    if s in ("", "na", "n/a", "none", "undrafted", "udfa", "free agent", "-", "--"):
        return True
    if re.search(r"undraft", s): 
        return True
    return False

# -------------------------- core --------------------------
def build(args):
    cfg = load_yaml(args.config)
    paths = cfg.get("paths", {})
    data_processed = paths.get("data_processed", "data/processed")
    neo4j_import = paths.get("neo4j_import", "neo4j/import")
    ensure_dir(neo4j_import)

    draft_path = args.draft or os.path.join(data_processed, "player_draft_2020-2025.matched.csv")
    team_value_path = args.team_value or os.path.join(data_processed, "long_table_team_value.with_ids.csv")
    team_loc_path = args.team_loc or os.path.join(data_processed, "NBA_team_location.csv")

    draft = pd.read_csv(draft_path)
    team_val = pd.read_csv(team_value_path)
    team_loc = pd.read_csv(team_loc_path)

    # Rename core columns
    draft = draft.rename(columns={
        "Player_id":"player_id",
        "Player":"player_name",
        "Draft_Team":"draft_team_name",
        "Affiliation":"affiliation",
        "Year":"year",
        "Round_NUmber":"round",
        "Round_pick":"round_pick",
        "Overall_Pick":"overall_pick"
    })

    # Step 1️⃣ — separate drafted vs undrafted BEFORE mapping
    undrafted_mask = draft["draft_team_name"].map(is_undrafted_value) | draft["year"].map(is_undrafted_value)
    undrafted_true = draft.loc[undrafted_mask].copy()
    drafted_true = draft.loc[~undrafted_mask].copy()

    # Step 2️⃣ — build team mapping
    team_loc["Team_Fullname_norm"] = team_loc["Team_Fullname"].map(normalize_fullname)
    fullname_to_abbr = dict(zip(team_loc["Team_Fullname_norm"], team_loc["Team"].astype(str).str.upper()))

    team_val["Team_norm"] = team_val["Team"].astype(str).str.upper().map(normalize_spaces)
    abbr_to_id = team_val.groupby("Team_norm")["Team_id"].agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()

    # Step 3️⃣ — map drafted players
    drafted_true["draft_team_name_norm"] = drafted_true["draft_team_name"].map(normalize_fullname)
    drafted_true["draft_team_abbr"] = drafted_true["draft_team_name_norm"].map(fullname_to_abbr)
    drafted_true["team_id"] = drafted_true["draft_team_abbr"].map(lambda x: abbr_to_id.get(str(x).upper()) if pd.notna(x) else np.nan)

    for c in ["player_id","year","round","round_pick","overall_pick"]:
        drafted_true[c] = pd.to_numeric(drafted_true[c], errors="coerce")

    drafted_kg = (
        drafted_true
        .dropna(subset=["player_id","year","team_id"])
        .sort_values(["player_id","year","overall_pick"], na_position="last")
        .drop_duplicates(["player_id"], keep="first")
        .copy()
    )

    # Step 4️⃣ — outputs
    out_csv = os.path.join(neo4j_import, "offcourt_draft_for_kg.csv")
    drafted_out = drafted_kg[["player_id","player_name","team_id","year","draft_team_abbr","round","round_pick","overall_pick","affiliation"]]
    drafted_out.to_csv(out_csv, index=False, encoding="utf-8")

    # Undrafted players
    undrafted_out = (
        undrafted_true[["player_id","player_name","year","affiliation"]]
        .dropna(subset=["player_id"])
        .drop_duplicates(["player_id"], keep="first")
        .copy()
    )

    def clean_year(y):
        try: return int(float(str(y)))
        except Exception: return np.nan
    undrafted_out["year"] = undrafted_out["year"].map(clean_year).astype("Int64")

    undrafted_csv = os.path.join(neo4j_import, "offcourt_draft_undrafted.csv")
    undrafted_out.to_csv(undrafted_csv, index=False, encoding="utf-8")

    # Step 5️⃣ — report
    report = {
        "raw_rows": len(draft),
        "undrafted_detected": len(undrafted_true),
        "drafted_detected": len(drafted_true),
        "drafted_final_rows": len(drafted_out),
        "undrafted_final_rows": len(undrafted_out),
        "intersection_player_ids": len(set(drafted_out["player_id"]).intersection(set(undrafted_out["player_id"]))),
        "drafted_sample": drafted_out.head(8).to_dict(orient="records"),
        "undrafted_sample": undrafted_out.head(8).to_dict(orient="records")
    }

    report_path = os.path.join(neo4j_import, "offcourt_draft_for_kg.report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Drafted output: {out_csv}")
    print(f"[OK] Undrafted output: {undrafted_csv}")
    print(f"[OK] Report: {report_path}")

# -------------------------- CLI --------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/configs.yaml")
    ap.add_argument("--draft", type=str, default="data/raw_external/player_draft_2020-2025.matched.csv")
    ap.add_argument("--team-value", type=str, dest="team_value", default="data/raw_external/long_table_team_value.with_ids.csv")
    ap.add_argument("--team-loc", type=str, dest="team_loc", default="data/raw_external/NBA_team_location.csv")
    return ap.parse_args()

if __name__ == "__main__":
    build(parse_args())