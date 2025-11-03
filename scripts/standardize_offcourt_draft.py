# -*- coding: utf-8 -*-
"""
standardize_offcourt_draft.py — FINAL
- Inputs:
    /mnt/data/player_draft_2020-2025.matched.corrected.csv
    /mnt/data/offcourt_team_location_for_kg.csv
    /mnt/data/offcourt_team_value_for_kg.csv
- Outputs:
    /mnt/data/neo4j/import/offcourt_draft_for_kg.csv
    /mnt/data/neo4j/import/offcourt_draft_undrafted.csv (columns: player_id, player_name, undrafted_flag)
"""
import os, re, json, unicodedata
import numpy as np
import pandas as pd

OUT_DIR = "neo4j/import"
os.makedirs(OUT_DIR, exist_ok=True)

DRAFT_PATH = "data/raw_external/player_draft_2020-2025.matched.corrected.csv"
TEAM_LOC_PATH = "neo4j/import/offcourt_team_location_for_kg.csv"
TEAM_VAL_PATH = "neo4j/import/offcourt_team_value_for_kg.csv"

def normalize_spaces(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return unicodedata.normalize("NFKC", s)

def is_undrafted_value(v):
    if pd.isna(v): 
        return True
    s = str(v).strip().lower()
    if s in ("", "na", "n/a", "none", "undrafted", "udfa", "free agent", "-", "--"):
        return True
    if re.search(r"undraft", s):
        return True
    return False

def to_int_or_none(x):
    try:
        return int(float(str(x).strip()))
    except:
        return None

def main():
    draft = pd.read_csv(DRAFT_PATH)
    team_loc = pd.read_csv(TEAM_LOC_PATH)
    team_val = pd.read_csv(TEAM_VAL_PATH)

    draft.columns = [c.strip().lower().replace(" ", "_") for c in draft.columns]
    team_loc.columns = [c.strip().lower().replace(" ", "_") for c in team_loc.columns]
    team_val.columns = [c.strip().lower().replace(" ", "_") for c in team_val.columns]

    for c in ["player","draft_team","affiliation","year","round_number","round_pick","overall_pick"]:
        if c in draft.columns:
            draft[c] = draft[c].map(normalize_spaces)

    undrafted_mask = draft["draft_team"].map(is_undrafted_value) | draft["year"].map(is_undrafted_value)
    undrafted_true = draft.loc[undrafted_mask].copy()
    drafted_true = draft.loc[~undrafted_mask].copy()

    abbr_to_id = {}
    if "team_abbr" in team_loc.columns:
        for _, r in team_loc.iterrows():
            abbr_to_id[str(r.team_abbr).upper()] = r.team_id
    if "team_abbr" in team_val.columns:
        for _, r in team_val.iterrows():
            abbr_to_id[str(r.team_abbr).upper()] = r.team_id

    full_to_abbr_manual = {
    "atlanta hawks": "ATL","boston celtics": "BOS","brooklyn nets": "BKN",
    "charlotte hornets": "CHA","chicago bulls": "CHI","cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL","denver nuggets": "DEN","detroit pistons": "DET",
    "golden state warriors": "GSW","houston rockets": "HOU","indiana pacers": "IND",
    "la clippers": "LAC","los angeles clippers": "LAC","los angeles lakers": "LAL",
    "memphis grizzlies": "MEM","miami heat": "MIA","milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN","new orleans pelicans": "NOP","new york knicks": "NYK",
    "oklahoma city thunder": "OKC","orlando magic": "ORL","philadelphia 76ers": "PHI",
    "phoenix suns": "PHX","portland trail blazers": "POR","sacramento kings": "SAC",
    "san antonio spurs": "SAS","toronto raptors": "TOR","utah jazz": "UTA","washington wizards": "WAS",
    # --- historical teams ---
    "seattle supersonics": "SEA","new jersey nets": "NJN","charlotte bobcats": "CHA",
    "washington bullets": "WSB",
    "new orleans hornets": "NOH",
    "oklahoma city hornets": "NOH",
    "new orleans/oklahoma city hornets": "NOH",
}

    legacy_map = {"SEA": "OKC", "NJN": "BKN", "WSB": "WAS", "NOH": "NOP"}

    def name_to_abbr(s):
        if pd.isna(s): 
            return np.nan
        key = normalize_spaces(s).lower()
        # prefer BRK if present in abbr_to_id else BKN handled here:
        if key == "brooklyn nets":
            return "BRK" if "BRK" in abbr_to_id else "BKN"
        return full_to_abbr_manual.get(key, np.nan)

    def translate_abbr(abbr):
        if pd.isna(abbr): 
            return abbr
        ab = str(abbr).upper()
        return legacy_map.get(ab, ab)

    def abbr_to_team_id(abbr):
        if pd.isna(abbr): 
            return np.nan
        return abbr_to_id.get(str(abbr).upper())

    drafted_true["draft_team_abbr"] = drafted_true["draft_team"].map(name_to_abbr)
    drafted_true["draft_team_abbr"] = drafted_true["draft_team_abbr"].map(translate_abbr)
    drafted_true["team_id"] = drafted_true["draft_team_abbr"].map(abbr_to_team_id)

    for c in ["player_id","round_number","round_pick","overall_pick"]:
        drafted_true[c] = pd.to_numeric(drafted_true[c], errors="coerce")
    drafted_true["year"] = drafted_true["year"].apply(to_int_or_none)

    drafted_kg = (
        drafted_true
        .dropna(subset=["player_id","year","team_id"])
        .sort_values(["player_id","year","overall_pick"], na_position="last")
        .drop_duplicates(["player_id"], keep="first")
        .copy()
    )

    drafted_out = drafted_kg.rename(columns={"player":"player_name"})[[
        "player_id","player_name","team_id","year","draft_team_abbr","round_number","round_pick","overall_pick"
    ]].copy()
    drafted_out = drafted_out.rename(columns={"round_number":"round"})
    drafted_out.to_csv(os.path.join(OUT_DIR,"offcourt_draft_for_kg.csv"), index=False, encoding="utf-8")

    undrafted_out = (
        undrafted_true[["player","player_id"]]
        .rename(columns={"player":"player_name"})
        .dropna(subset=["player_id"])
        .drop_duplicates(["player_id"], keep="first")
        .copy()
    )
    undrafted_out["undrafted_flag"] = True
    undrafted_out.to_csv(os.path.join(OUT_DIR,"offcourt_draft_undrafted.csv"), index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
