# -*- coding: utf-8 -*-
"""
standardize_off_court_team_location.py
--------------------------------------
Purpose
  - From minimal team location file (Team, Team_Fullname, Location),
    build Neo4j-ready CSV with team_id joined from offcourt_team_value_for_kg.csv.

Inputs (fixed by your request):
  - data/raw_external/NBA_team_location.csv
      columns (minimal): Team, Team_Fullname, Location    # e.g., "Los Angeles, CA"
  - neo4j/import/offcourt_team_value_for_kg.csv
      columns: team_id, team_abbr, year, team_value_usd (year repeated per team)

Outputs:
  - neo4j/import/offcourt_team_location_for_kg.csv
      columns: team_id, team_abbr, city, state
  - neo4j/import/offcourt_team_location_for_kg.report.json
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd

INPUT_LOC   = "data/raw_external/NBA_team_location.csv"
XWALK_FROM  = "neo4j/import/offcourt_team_value_for_kg.csv"
OUT_CSV     = "neo4j/import/offcourt_team_location_for_kg.csv"
OUT_REPORT  = "neo4j/import/offcourt_team_location_for_kg.report.json"

def ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def norm_abbr(s: str) -> str:
    if pd.isna(s): return np.nan
    return re.sub(r"\s+", "", str(s).upper().strip())

def parse_location(loc):
    """Parse 'City, ST' or 'City, State' -> (city, state). Robust to extra commas."""
    if pd.isna(loc): return (None, None)
    s = str(loc).strip()
    # split by the last comma to be robust to 'Washington, D.C.'
    if "," in s:
        idx = s.rfind(",")
        city = s[:idx].strip()
        state = s[idx+1:].strip()
    else:
        # no comma -> fail soft
        city, state = (s, None)
    # normalize full-state to 2-letter if possible (best-effort)
    STATE_MAP = {
        "ALABAMA":"AL","ALASKA":"AK","ARIZONA":"AZ","ARKANSAS":"AR","CALIFORNIA":"CA","COLORADO":"CO",
        "CONNECTICUT":"CT","DELAWARE":"DE","FLORIDA":"FL","GEORGIA":"GA","HAWAII":"HI","IDAHO":"ID",
        "ILLINOIS":"IL","INDIANA":"IN","IOWA":"IA","KANSAS":"KS","KENTUCKY":"KY","LOUISIANA":"LA",
        "MAINE":"ME","MARYLAND":"MD","MASSACHUSETTS":"MA","MICHIGAN":"MI","MINNESOTA":"MN","MISSISSIPPI":"MS",
        "MISSOURI":"MO","MONTANA":"MT","NEBRASKA":"NE","NEVADA":"NV","NEW HAMPSHIRE":"NH","NEW JERSEY":"NJ",
        "NEW MEXICO":"NM","NEW YORK":"NY","NORTH CAROLINA":"NC","NORTH DAKOTA":"ND","OHIO":"OH","OKLAHOMA":"OK",
        "OREGON":"OR","PENNSYLVANIA":"PA","RHODE ISLAND":"RI","SOUTH CAROLINA":"SC","SOUTH DAKOTA":"SD",
        "TENNESSEE":"TN","TEXAS":"TX","UTAH":"UT","VERMONT":"VT","VIRGINIA":"VA","WASHINGTON":"WA",
        "WEST VIRGINIA":"WV","WISCONSIN":"WI","WYOMING":"WY","DISTRICT OF COLUMBIA":"DC",
        "WASHINGTON D.C.":"DC","WASHINGTON DC":"DC","D.C.":"DC","DC":"DC"
    }
    up = state.upper() if isinstance(state, str) else None
    if up and up not in STATE_MAP.values():
        state = STATE_MAP.get(up, state)
    return (city, state)

def build():
    # 1) read inputs
    if not os.path.exists(INPUT_LOC):
        raise FileNotFoundError(f"Team location not found: {INPUT_LOC}")
    if not os.path.exists(XWALK_FROM):
        raise FileNotFoundError(f"TeamId xwalk not found: {XWALK_FROM}")

    loc = pd.read_csv(INPUT_LOC)
    xwalk_src = pd.read_csv(XWALK_FROM)

    # 2) normalize columns
    loc = loc.rename(columns={
        "Team":"team_abbr",
        "Team_Fullname":"team_fullname",
        "Location":"location",
    })
    for c in ["team_abbr","team_fullname","location"]:
        if c not in loc.columns:
            raise ValueError(f"Missing column in location CSV: {c}")

    loc["team_abbr"] = loc["team_abbr"].map(norm_abbr)

    # 3) parse city/state
    parsed = loc["location"].map(parse_location)
    loc["city"]  = parsed.map(lambda t: t[0])
    loc["state"] = parsed.map(lambda t: t[1])

    # 4) build abbr->team_id from offcourt_team_value_for_kg.csv
    #    that file has many years per team; we only need a unique mapping
    if not {"team_abbr","team_id"}.issubset(xwalk_src.columns):
        raise ValueError("xwalk source must contain 'team_abbr' and 'team_id'.")

    xwalk = (
        xwalk_src[["team_abbr","team_id"]]
        .dropna()
        .assign(team_abbr=lambda d: d["team_abbr"].map(norm_abbr))
        .drop_duplicates(subset=["team_abbr"])
    )

    # 5) merge to get team_id
    out = loc.merge(xwalk, on="team_abbr", how="left")
    before = len(out)
    missing_team_id = int(out["team_id"].isna().sum())

    # 强约束：必须都有 team_id（NBA 30 队）
    out = out.dropna(subset=["team_id"]).copy()
    out["team_id"] = out["team_id"].astype("Int64")

    # 6) final columns & write
    cols_out = ["team_id","team_abbr","city","state"]
    ensure_dir(OUT_CSV)
    out[cols_out].to_csv(OUT_CSV, index=False, encoding="utf-8")

    # 7) report
    ensure_dir(OUT_REPORT)
    report = {
        "input_location_path": INPUT_LOC,
        "xwalk_from": XWALK_FROM,
        "source_rows": int(len(loc)),
        "rows_out": int(len(out)),
        "rows_missing_team_id_dropped": missing_team_id,
        "columns_out": cols_out,
        "unique_teams_out": int(out["team_id"].nunique()),
        "sample": out[cols_out].head(10).to_dict(orient="records"),
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved CSV: {OUT_CSV}")
    print(f"[OK] Report  : {OUT_REPORT}")
    print(f"[Summary] rows_out={len(out)} | missing_team_id_dropped={missing_team_id}")

if __name__ == "__main__":
    build()
