#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
standardize_offcourt_awards.py
---------------------------------
Purpose:
  - Standardize raw NBA awards CSV (award.csv or similar) into a clean schema
    suitable for Neo4j import.
  - Canonicalize award names (e.g., unify All-NBA naming), extract team-rank
    (1st/2nd/3rd) from award text when applicable, and de-duplicate on
    (player_id, award, year, team).
  - Optionally run validations (singleton awards per year, All-NBA team counts).

Input columns expected (case-insensitive, flexible):
  - player_id  (int)
  - player     (optional, not used for KG keying)
  - award      (str) e.g. "All-NBA First Team", "MVP", "Finals MVP*"
  - year       (int)
  - team       (optional; if absent or empty, will be inferred from award text
                for team-based awards like All-NBA/All-Defensive/All-Rookie)

Output columns:
  - player_id (int)
  - award     (canonical str; e.g., "All-NBA", "Most Valuable Player", ...)
  - year      (int)
  - team      (str or empty); '1st'|'2nd'|'3rd' for team-based awards; else empty

Usage:
  python standardize_offcourt_awards.py \
    --in award.csv \
    --out award_std_fixed.csv \
    --year-min 2020 --year-max 2024 \
    --validate

Notes:
  - This script is idempotent: running on an already standardized file is safe.
  - Designed to integrate with Neo4j MERGE pattern:
        MERGE (p:Player {player_id: pid})
        MERGE (a:Award  {name: award})
        MERGE (s:Season {year: yr})
        MERGE (p)-[:WON_AWARD {year: yr, team: team_or_null}]->(a)
        MERGE (a)-[:AWARDED_IN]->(s)

Author: ChatGPT
"""

from __future__ import annotations
import argparse
import sys
from typing import Optional, Tuple, Dict, List

import pandas as pd


# --------------------------- Configuration -------------------------------- #

# Canonical names for "singleton" awards (one winner per year).
SINGLETON_AWARD_MAP = {
    "mvp": "Most Valuable Player",
    "most valuable player": "Most Valuable Player",
    "finals mvp": "Finals MVP",
    "defensive player of the year": "Defensive Player of the Year",
    "rookie of the year": "Rookie of the Year",
    "sixth man of the year": "Sixth Man of the Year",
    "most improved player": "Most Improved Player",
    "coach of the year": "Coach of the Year",
    "clutch player of the year": "Clutch Player of the Year",
    # NBA Cup related (if present)
    "nba cup mvp": "NBA Cup MVP",
    "nba cup all-tournament team": "NBA Cup All-Tournament Team",
    # Conference finals MVP (if present)
    "nba eastern conference finals mvp": "NBA Eastern Conference Finals MVP",
    "nba western conference finals mvp": "NBA Western Conference Finals MVP",
    # Teammate award
    "twyman-stokes teammate of the year award": "Twyman-Stokes Teammate of the Year Award",
    "all-star mvp": "All-Star MVP",
}

# Team-based awards patterns: normalize to canonical award + team rank.
TEAM_AWARD_PREFIXES = {
    "all-nba": "All-NBA",
    "all defensive": "All-Defensive",
    "all-defensive": "All-Defensive",
    "all rookie": "All-Rookie",
    "all-rookie": "All-Rookie",
}

# Map common variants to rank tokens used in KG.
RANK_VARIANTS = {
    "first team": "1st",
    "1st team": "1st",
    "1st": "1st",
    "second team": "2nd",
    "2nd team": "2nd",
    "2nd": "2nd",
    "third team": "3rd",
    "3rd team": "3rd",
    "3rd": "3rd",
}


# ------------------------------ Helpers ----------------------------------- #

def _strip_award_text(x: str) -> str:
    """Trim award text, remove trailing asterisks and normalize spaces."""
    if not isinstance(x, str):
        return ""
    x = x.replace("\u200b", " ")  # zero-width
    x = x.strip()
    # Drop footnote-like asterisks at the end
    while x.endswith("*"):
        x = x[:-1].strip()
    # collapse spaces
    x = " ".join(x.split())
    return x


def _extract_team_rank(award_text: str) -> Optional[str]:
    """Extract 1st/2nd/3rd from award text if present."""
    low = award_text.lower()
    for pat, rank in RANK_VARIANTS.items():
        if pat in low:
            return rank
    return None


def _canonicalize_award_name(award_text: str) -> str:
    """Return canonical award name (without team rank)."""
    low = award_text.lower()

    # Team-based family first
    for pref, canon in TEAM_AWARD_PREFIXES.items():
        if low.startswith(pref):
            return canon

    # Singleton/other awards
    if low in SINGLETON_AWARD_MAP:
        return SINGLETON_AWARD_MAP[low]

    # Keep original capitalization if not recognized (but trimmed)
    return award_text


def _infer_team_for_award(award_canon: str, award_text: str, team_col: Optional[str]) -> Optional[str]:
    """Return team rank for team-based awards, else None.
    team_col: pre-existing value from 'team' column (may be None/empty)."""
    team_col = (team_col or "").strip()
    if award_canon in {"All-NBA", "All-Defensive", "All-Rookie"}:
        # Prefer explicit team column if valid, else parse from text
        if team_col in {"1st", "2nd", "3rd"}:
            return team_col
        return _extract_team_rank(award_text)
    # Non team-based
    return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to expected set."""
    col_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_map)

    # Flexible handling for common variants
    rename_variants = {
        "playerid": "player_id",
        "player_id": "player_id",
        "player id": "player_id",
        "player": "player",
        "award": "award",
        "year": "year",
        "season": "year",
        "team": "team",
    }
    for src, tgt in rename_variants.items():
        if src in df.columns:
            df = df.rename(columns={src: tgt})

    # Ensure required columns exist
    for req in ("player_id", "award", "year"):
        if req not in df.columns:
            raise ValueError(f"Missing required column: '{req}'")

    # Force types
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "team" not in df.columns:
        df["team"] = pd.Series([None] * len(df), dtype="object")

    # Clean text
    df["award"] = df["award"].astype(str).map(_strip_award_text)
    if "player" in df.columns:
        df["player"] = df["player"].astype(str).str.strip()

    return df


def standardize_awards(df: pd.DataFrame,
                       year_min: Optional[int] = None,
                       year_max: Optional[int] = None) -> pd.DataFrame:
    """Main transformation: canonicalize, infer team, filter, de-duplicate."""
    df = _standardize_columns(df)

    # Filter by year range if provided
    if year_min is not None:
        df = df[df["year"].astype("float64") >= float(year_min)]
    if year_max is not None:
        df = df[df["year"].astype("float64") <= float(year_max)]

    # Drop rows with missing key fields
    df = df.dropna(subset=["player_id", "award", "year"])

    # Canonicalize award names & infer team ranks
    awards_canon: List[str] = []
    teams: List[Optional[str]] = []
    for award_text, team_col in zip(df["award"].tolist(), df["team"].tolist()):
        canon = _canonicalize_award_name(award_text)
        rank = _infer_team_for_award(canon, award_text, team_col)
        awards_canon.append(canon)
        teams.append(rank)

    df = df.assign(award=awards_canon, team=teams)

    # For non-team awards, team must be None
    team_based_mask = df["award"].isin(["All-NBA", "All-Defensive", "All-Rookie"])
    df.loc[~team_based_mask, "team"] = None

    # De-duplicate
    df = df.drop_duplicates(subset=["player_id", "award", "year", "team"]).reset_index(drop=True)

    # Sort for stable output
    df = df.sort_values(by=["year", "award", "team", "player_id"], na_position="last")

    # Final schema
    return df[["player_id", "award", "year", "team"]]


def validate_awards(df: pd.DataFrame,
                    check_singletons: bool = True,
                    check_all_nba_counts: bool = True) -> Dict:
    """Basic validations: singleton awards (1 per year), All-NBA team counts (5 per team per year)."""
    report = {}

    if check_singletons:
        singletons = [
            "Most Valuable Player",
            "Defensive Player of the Year",
            "Rookie of the Year",
            "Sixth Man of the Year",
            "Most Improved Player",
            "Finals MVP",
        ]
        singles_per_year = {}
        for y, g in df.groupby("year"):
            singles_per_year[y] = {a: int((g["award"] == a).sum()) for a in singletons}
        report["singleton_counts_per_year"] = singles_per_year

    if check_all_nba_counts:
        # Verify 5 per team for All-NBA if present
        all_nba = df[df["award"] == "All-NBA"]
        if not all_nba.empty:
            counts = (all_nba.groupby(["year", "team"]).size().unstack(fill_value=0))
            report["all_nba_counts_per_year"] = counts.to_dict()

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Standardize NBA awards CSV for Neo4j KG import.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input CSV path (raw awards).")
    parser.add_argument("--out", dest="out_path", required=True, help="Output CSV path (standardized).")
    parser.add_argument("--year-min", type=int, default=None, help="Minimum year to keep (inclusive).")
    parser.add_argument("--year-max", type=int, default=None, help="Maximum year to keep (inclusive).")
    parser.add_argument("--validate", action="store_true", help="Run basic validations and print a report.")
    args = parser.parse_args(argv)

    # Load
    df = pd.read_csv(args.in_path)

    # Transform
    df_std = standardize_awards(df, year_min=args.year_min, year_max=args.year_max)

    # Save
    df_std.to_csv(args.out_path, index=False)

    # Validate
    if args.validate:
        rep = validate_awards(df_std)
        # Pretty print
        print("# Validation report")
        if "singleton_counts_per_year" in rep:
            print("Singleton awards (should be 1 per year):")
            for y in sorted(rep["singleton_counts_per_year"].keys()):
                print(f"  {y}: {rep['singleton_counts_per_year'][y]}")
        if "all_nba_counts_per_year" in rep:
            print("All-NBA counts per year (each team should be 5):")
            for y in sorted(rep["all_nba_counts_per_year"].keys()):
                print(f"  {y}: {rep['all_nba_counts_per_year'][y]}")

    print(f"✅ Wrote standardized file: {args.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
