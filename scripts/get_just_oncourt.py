import pandas as pd

ID_COLS = ["player_id", "season"]
TARGET_COL = "log_salary"

TARGET_LEAK_BASE = [
    "salary_usd", "log_salary",
    "salary_cap", "salary_cap_ratio", "log_salary_cap_ratio",
    "salary_cap_equiv",
]
TARGET_LEAK_COLS = [c for c in TARGET_LEAK_BASE if c != TARGET_COL]

TEXT_OR_ID_COLS = [
    "Player", "player_name",
    "team_abbr", "city", "state", "region",
    "agent_name", "agent_name_all",
]

OFFCOURT_PREFIXES = ("draft_", "team_", "agent_")
LEAKY_PREFIXES    = ("award_", "injury_")

OFFCOURT_EXACT = [
    "Age", "age_now", "years_since_draft",
    "overall_pick", "round", "round_pick", "draft_year",
    "undrafted_flag",
]
# IMPORTANT: include "draft" to catch messy names like "drafthe Year"
OFFCOURT_CONTAINS = ("draft", "team", "agent", "market", "city", "state", "region", "pick")

def is_offcourt(col: str) -> bool:
    lc = col.lower()
    if lc.startswith(OFFCOURT_PREFIXES):
        return True
    # exact rules (keep original case list, but compare lower for safety)
    if col in OFFCOURT_EXACT or lc in [x.lower() for x in OFFCOURT_EXACT]:
        return True
    if any(k in lc for k in OFFCOURT_CONTAINS):
        return True
    return False

def get_oncourt_cols(df: pd.DataFrame) -> list[str]:
    candidate_feature_cols = [
        c for c in df.columns
        if c not in ID_COLS + [TARGET_COL] + TARGET_LEAK_COLS + TEXT_OR_ID_COLS
    ]

    numeric_bool_cols = (
        df[candidate_feature_cols]
        .select_dtypes(include=["number", "bool"])
        .columns
        .tolist()
    )

    safe_cols = [c for c in numeric_bool_cols if not c.startswith(LEAKY_PREFIXES)]
    oncourt_cols = sorted([c for c in safe_cols if not is_offcourt(c)])

    return oncourt_cols