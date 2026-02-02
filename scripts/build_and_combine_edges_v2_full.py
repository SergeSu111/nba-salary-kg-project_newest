#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_and_combine_edges_v2_full.py (FINAL ROBUST VERSION)

Integrates strictly enforced audits (NaN checks), robust mapping, and consistent normalization.

Key Features:
1. Mapping: Strictly enforces 'node_id' and 'idx' columns from MASTER mapping.
2. Normalization: relation_type -> UPPER + REPLACE(" ", "_").
3. Audits: 
   - Raises ValueError if audit columns are missing.
   - Raises ValueError if years cannot be parsed (prevents silent failures).
4. Logic: Award audit rule parameterized via CLI.
5. Safety: SG derived from MG in memory (Single Source of Truth).

Usage:
  python build_and_combine_edges_v2_full.py \
    --mode v2_full \
    --injury-variant sg \
    --node-mapping master_node_id_to_idx.csv \
    --core graph/edges/edges_gnn_v2_core_elementId_full.csv \
    --award graph/edges/V2_Full_Award_Edges.csv \
    --injury-mg graph/edges/V2_Full_Injury_multigraph_Edges_FULL_19073.csv \
    --use-award --use-injury \
    --audit-award --audit-injury \
    --award-audit-rule non_future \
    --output graph/edges/edge_index_v2_full_sg.pt
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd
import torch

# ----------------------------
# Helpers
# ----------------------------

def _year4(x) -> Optional[int]:
    """Parse first 4 digits of a year-like field."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if len(s) < 4:
        return None
    try:
        return int(s[:4])
    except Exception:
        return None

def read_edges_csv(path: str) -> pd.DataFrame:
    """Read CSV and normalize to standard columns: source_id, target_id, relation_type."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Edges file not found: {path}")

    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]

    # --- column alias handling (core file uses src/dst/rel) ---
    col_map = {}
    if "source_id" not in df.columns:
        if "src" in df.columns: col_map["src"] = "source_id"
        elif "source" in df.columns: col_map["source"] = "source_id"
        elif "from" in df.columns: col_map["from"] = "source_id"
    if "target_id" not in df.columns:
        if "dst" in df.columns: col_map["dst"] = "target_id"
        elif "target" in df.columns: col_map["target"] = "target_id"
        elif "to" in df.columns: col_map["to"] = "target_id"
    if "relation_type" not in df.columns:
        if "rel" in df.columns: col_map["rel"] = "relation_type"
        elif "relation" in df.columns: col_map["relation"] = "relation_type"
        elif "type" in df.columns: col_map["type"] = "relation_type"

    if col_map:
        df = df.rename(columns=col_map)

    required = ["source_id", "target_id", "relation_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} missing required columns {missing}. "
            f"Detected columns: {list(df.columns)}"
        )

    df["source_id"] = df["source_id"].astype(str)
    df["target_id"] = df["target_id"].astype(str)

    # strict normalization: UPPER + replace spaces with underscore
    df["relation_type"] = (
        df["relation_type"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
    )
    return df


def load_node_mapping(mapping_path: str) -> Dict[str, int]:
    """Load explicit node mapping with strict validation."""
    p = Path(mapping_path)
    if not p.exists():
        raise FileNotFoundError(f"Node mapping not found: {mapping_path}")
    
    if p.suffix == '.json':
        with open(p, 'r') as f:
            obj = json.load(f)
        return {str(k): int(v) for k, v in obj.items()}
    
    elif p.suffix == '.csv':
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        
        required = {"node_id", "idx"}
        if not required.issubset(set(df.columns)):
             raise ValueError(f"Mapping CSV must contain columns {required}. Got: {list(df.columns)}")
        
        return dict(zip(df["node_id"].astype(str), df["idx"].astype(int)))
    else:
        raise ValueError("Mapping must be .json or .csv")

# ----------------------------
# Audits (Enhanced)
# ----------------------------

def audit_award_leak(df: pd.DataFrame, rule: str = "non_future") -> None:
    if "ps_season" not in df.columns or "award_year" not in df.columns:
        raise ValueError("Award audit requires columns: ps_season, award_year")

    psY = df["ps_season"].apply(_year4)
    awY = df["award_year"].apply(_year4)

    # FIX: Check for parsing failures (NaN)
    bad = psY.isna() | awY.isna()
    if bad.any():
        ex = df.loc[bad, ["source_id", "target_id", "relation_type", "ps_season", "award_year"]].head(10)
        raise ValueError("Award audit failed: cannot parse year in ps_season/award_year.\n"
                         f"Examples of bad data:\n{ex.to_string(index=False)}")

    if rule == "non_future":
        leaks = (awY > psY)
    else: # strict_past
        leaks = (awY >= psY)

    if leaks.any():
        raise AssertionError(f"Award Leak ({rule}): {leaks.sum()} rows violation.")
    print(f"✅ Award Audit Passed ({rule})")

def audit_injury_leak(df: pd.DataFrame) -> None:
    if "ps_season" not in df.columns or "injury_season" not in df.columns:
        raise ValueError("Injury audit requires columns: ps_season, injury_season")

    psY = df["ps_season"].apply(_year4)
    injY = df["injury_season"].apply(_year4)

    # FIX: Check for parsing failures (NaN)
    bad = psY.isna() | injY.isna()
    if bad.any():
        ex = df.loc[bad, ["source_id", "target_id", "relation_type", "ps_season", "injury_season"]].head(10)
        raise ValueError("Injury audit failed: cannot parse year in ps_season/injury_season.\n"
                         f"Examples of bad data:\n{ex.to_string(index=False)}")

    leaks = (injY >= psY)
    if leaks.any():
        raise AssertionError(f"Injury Leak: {leaks.sum()} rows violation (injury >= ps).")
    print("✅ Injury Audit Passed (Strict Past)")

# ----------------------------
# Build Logic
# ----------------------------

def map_edges(df: pd.DataFrame, mapping: Dict[str, int]) -> torch.Tensor:
    src = df["source_id"].map(mapping)
    dst = df["target_id"].map(mapping)

    if src.isna().any() or dst.isna().any():
        missing_src = df.loc[src.isna(), "source_id"].unique()[:3]
        missing_dst = df.loc[dst.isna(), "target_id"].unique()[:3]
        raise KeyError(
            f"Unmapped IDs found! Nodes in edges must exist in master mapping.\n"
            f"Examples Missing Source: {missing_src}\n"
            f"Examples Missing Target: {missing_dst}"
        )

    return torch.stack([
        torch.tensor(src.values.astype(int), dtype=torch.long),
        torch.tensor(dst.values.astype(int), dtype=torch.long)
    ], dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["v2_core", "v2_full"])
    parser.add_argument("--injury-variant", default="mg", choices=["mg", "sg"])
    
    parser.add_argument("--node-mapping", required=True, help="Path to MASTER node mapping csv/json")
    parser.add_argument("--core", required=True)
    parser.add_argument("--award", help="Path to Award edges")
    parser.add_argument("--injury-mg", help="Path to FULL Multigraph Injury edges")
    
    parser.add_argument("--use-award", action="store_true")
    parser.add_argument("--use-injury", action="store_true")
    parser.add_argument("--audit-award", action="store_true")
    parser.add_argument("--audit-injury", action="store_true")
    
    parser.add_argument("--award-audit-rule", default="non_future", 
                        choices=["non_future", "strict_past"])
    
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    # 1. Load Master Mapping
    print(f"Loading mapping from: {args.node_mapping}")
    node_mapping = load_node_mapping(args.node_mapping)
    print(f"Master Mapping loaded: {len(node_mapping)} nodes")

    # 2. Load Core
    core_df = read_edges_csv(args.core)
    frames = [core_df]
    print(f"Core edges: {len(core_df)}")

    # 3. Load Extras
    if args.mode == "v2_full":
        # --- AWARD ---
        if args.use_award:
            if not args.award:
                raise ValueError("--award path required when --use-award is set")
            aw_df = read_edges_csv(args.award)
            if args.audit_award:
                audit_award_leak(aw_df, rule=args.award_audit_rule)
            frames.append(aw_df)
            print(f"Award edges: {len(aw_df)}")

        # --- INJURY ---
        if args.use_injury:
            if not args.injury_mg:
                raise ValueError("--injury-mg path REQUIRED when --use-injury is set")
            
            # Always load MG
            inj_df = read_edges_csv(args.injury_mg)
            print(f"Injury (Master MG): {len(inj_df)}")

            if args.audit_injury:
                audit_injury_leak(inj_df)

            # Variant Logic
            if args.injury_variant == "sg":
                before = len(inj_df)
                inj_df = inj_df.drop_duplicates(subset=["source_id", "target_id", "relation_type"])
                print(f"Injury (Derived SG): {before} -> {len(inj_df)} (Deduped)")
            else:
                print(f"Injury (MG): Using full weighted edges")

            frames.append(inj_df)

    # 4. Combine & Convert
    full_df = pd.concat(frames, ignore_index=True)
    
    edge_index_dict = {}
    rel_counts = {}

    for rtype, group in full_df.groupby("relation_type"):
        print(f"Processing relation: {rtype}...")
        try:
            ei = map_edges(group, node_mapping)
            edge_index_dict[rtype] = ei
            rel_counts[rtype] = ei.size(1)
        except KeyError as e:
            print(f"❌ Error mapping relation {rtype}: {e}")
            exit(1)

    # 5. Save
    torch.save({
        "edge_index_dict": edge_index_dict,
        "rel_counts": rel_counts,
        "mode": args.mode,
        "injury_variant": args.injury_variant,
        "mapping_source": args.node_mapping 
    }, args.output)
    
    print(f"\n✅ Success! Saved to {args.output}")
    print("Rel Counts:", rel_counts)

if __name__ == "__main__":
    main()