#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_and_combine_edges_v2_full.py

Build PyG-style edge_index_dict from modular edge CSVs (core / award / injury),
and AUTO-generate node_id -> idx mapping from the SAME edges to ensure perfect alignment.

Key features:
- Combine edges in memory (no giant merged CSV required).
- Auto mapping: collect all source_id/target_id across used edge files, assign idx 0..N-1.
- Optional audits:
    Award:
      - non_future (default): leak if award_year > ps_season
      - strict_past          : leak if award_year >= ps_season
    Injury:
      - leak if injury_season >= ps_season
- Injury sg mode:
    If --injury-sg not provided, reuse --injury-mg and dedup in memory.
- Output:
    - torch .pt containing edge_index_dict + rel_counts + mapping_path
    - mapping CSV saved to graph/mappings by default
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
    """Parse first 4 digits of a year-like field. Returns None if cannot parse."""
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
    """
    Read an edges CSV and normalize column names to lowercase.

    Requires at least: source_id, target_id, relation_type
    Keeps all columns (for audits), but mapping uses only those 3.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Edges file not found: {path}")

    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]

    required = ["source_id", "target_id", "relation_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns {missing}. Columns: {list(df.columns)}")

    df["source_id"] = df["source_id"].astype(str)
    df["target_id"] = df["target_id"].astype(str)
    df["relation_type"] = df["relation_type"].astype(str).str.strip()
    return df


# ----------------------------
# Audits
# ----------------------------

def audit_award_leak(df: pd.DataFrame, rule: str = "non_future") -> None:
    """
    Award leak audit based on parsed first 4 digits.
    Requires columns: ps_season, award_year

    rule:
      - non_future : leak if award_year > ps_season   (allows ==)
      - strict_past: leak if award_year >= ps_season  (forbids ==)
    """
    if "ps_season" not in df.columns or "award_year" not in df.columns:
        raise ValueError("Award audit requires columns: ps_season, award_year")

    psY = df["ps_season"].apply(_year4)
    awY = df["award_year"].apply(_year4)

    if psY.isna().any() or awY.isna().any():
        bad_rows = df[psY.isna() | awY.isna()].head(10)
        raise AssertionError(
            "Award audit failed: unparsable ps_season or award_year.\n"
            f"Example rows:\n{bad_rows}"
        )

    if rule == "non_future":
        leaks = (awY > psY)
    elif rule == "strict_past":
        leaks = (awY >= psY)
    else:
        raise ValueError("award audit rule must be 'non_future' or 'strict_past'")

    if leaks.any():
        leak_df = df[leaks].head(20)
        raise AssertionError(
            f"Award leak detected under rule='{rule}': {int(leaks.sum())} rows.\n"
            f"Examples:\n{leak_df[['source_id','target_id','relation_type','ps_season','award_year']].to_string(index=False)}"
        )


def audit_injury_leak(df: pd.DataFrame) -> None:
    """
    Injury leak audit: require injury_season < ps_season
    Requires columns: ps_season, injury_season
    """
    if "ps_season" not in df.columns or "injury_season" not in df.columns:
        raise ValueError("Injury audit requires columns: ps_season, injury_season")

    psY = df["ps_season"].apply(_year4)
    injY = df["injury_season"].apply(_year4)

    if psY.isna().any() or injY.isna().any():
        bad_rows = df[psY.isna() | injY.isna()].head(10)
        raise AssertionError(
            "Injury audit failed: unparsable ps_season or injury_season.\n"
            f"Example rows:\n{bad_rows}"
        )

    leaks = (injY >= psY)
    if leaks.any():
        leak_df = df[leaks].head(20)
        raise AssertionError(
            f"Injury leak detected: {int(leaks.sum())} rows have injury_season >= ps_season.\n"
            f"Examples:\n{leak_df[['source_id','target_id','relation_type','ps_season','injury_season']].to_string(index=False)}"
        )


# ----------------------------
# Auto mapping
# ----------------------------

def build_node_id_to_idx_from_edges(edge_frames: List[pd.DataFrame]) -> Dict[str, int]:
    """
    Collect unique node_ids from source_id/target_id across edge_frames,
    then assign idx 0..N-1 in a deterministic order (sorted by string).
    """
    nodes = set()
    for df in edge_frames:
        nodes.update(df["source_id"].astype(str).tolist())
        nodes.update(df["target_id"].astype(str).tolist())

    # Deterministic ordering -> stable runs for same inputs
    nodes_sorted = sorted(nodes)
    return {nid: i for i, nid in enumerate(nodes_sorted)}


def save_mapping_csv(mapping: Dict[str, int], path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"node_id": list(mapping.keys()), "idx": list(mapping.values())})
    df.to_csv(p, index=False)
    return str(p)


# ----------------------------
# Mapping edges -> tensors
# ----------------------------

def map_edges_to_index(
    df: pd.DataFrame,
    node_mapping: Dict[str, int],
    strict: bool = True,
) -> torch.Tensor:
    """
    Map source_id/target_id strings to integer indices using node_mapping.
    Returns edge_index tensor of shape [2, E].
    """
    src_idx = df["source_id"].map(node_mapping)
    dst_idx = df["target_id"].map(node_mapping)

    if strict and (src_idx.isna().any() or dst_idx.isna().any()):
        bad_src = df[src_idx.isna()].head(10)["source_id"].tolist()
        bad_dst = df[dst_idx.isna()].head(10)["target_id"].tolist()
        raise KeyError(
            "Unmapped node ids found while mapping edges (should not happen if mapping is auto-built from same edges).\n"
            f"bad_src examples: {bad_src}\n"
            f"bad_dst examples: {bad_dst}\n"
        )

    src = torch.tensor(src_idx.astype(int).to_numpy(), dtype=torch.long)
    dst = torch.tensor(dst_idx.astype(int).to_numpy(), dtype=torch.long)
    return torch.stack([src, dst], dim=0)


# ----------------------------
# Build function
# ----------------------------

def load_and_select_edges(
    mode: str,
    injury_variant: str,
    core_path: str,
    award_path: Optional[str],
    injury_mg_path: Optional[str],
    injury_sg_path: Optional[str],
    use_award: bool,
    use_injury: bool,
    audit_award: bool,
    audit_injury: bool,
    award_audit_rule: str,
) -> List[pd.DataFrame]:
    """
    Load edge CSVs according to config and return list of DataFrames.
    """
    if mode not in ["v2_core", "v2_full"]:
        raise ValueError("mode must be 'v2_core' or 'v2_full'")
    if injury_variant not in ["mg", "sg"]:
        raise ValueError("injury_variant must be 'mg' or 'sg'")
    if award_audit_rule not in ["non_future", "strict_past"]:
        raise ValueError("award_audit_rule must be 'non_future' or 'strict_past'")

    edge_frames: List[pd.DataFrame] = []

    # Core always
    core_df = read_edges_csv(core_path)
    edge_frames.append(core_df)

    if mode == "v2_full":
        if use_award:
            if not award_path:
                raise ValueError("award_path is required when use_award=True in v2_full mode")
            award_df = read_edges_csv(award_path)
            if audit_award:
                audit_award_leak(award_df, rule=award_audit_rule)
            edge_frames.append(award_df)

        if use_injury:
            if injury_variant == "mg":
                if not injury_mg_path:
                    raise ValueError("injury_mg_path is required for injury_variant='mg'")
                inj_df = read_edges_csv(injury_mg_path)
            else:
                # sg: if injury_sg not provided, reuse injury_mg and dedup in memory
                key_path = injury_sg_path or injury_mg_path
                if not key_path:
                    raise ValueError("Provide injury_sg_path or injury_mg_path to derive singlegraph")
                inj_df = read_edges_csv(key_path)
                inj_df = inj_df.drop_duplicates(subset=["source_id", "target_id", "relation_type"])

            if audit_injury:
                audit_injury_leak(inj_df)
            edge_frames.append(inj_df)

    return edge_frames


def build_edge_index_dict_auto_mapping(
    mode: str,
    injury_variant: str,
    core_path: str,
    award_path: Optional[str],
    injury_mg_path: Optional[str],
    injury_sg_path: Optional[str],
    use_award: bool,
    use_injury: bool,
    audit_award: bool,
    audit_injury: bool,
    award_audit_rule: str,
    strict_mapping: bool,
    mapping_save_path: Optional[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, int], Optional[str]]:
    """
    Build edge_index_dict using auto-generated node_id->idx mapping from the SAME edges.
    Returns:
      edge_index_dict, rel_counts, node_mapping, saved_mapping_path
    """
    edge_frames = load_and_select_edges(
        mode=mode,
        injury_variant=injury_variant,
        core_path=core_path,
        award_path=award_path,
        injury_mg_path=injury_mg_path,
        injury_sg_path=injury_sg_path,
        use_award=use_award,
        use_injury=use_injury,
        audit_award=audit_award,
        audit_injury=audit_injury,
        award_audit_rule=award_audit_rule,
    )

    # Auto-build mapping from selected edges
    node_mapping = build_node_id_to_idx_from_edges(edge_frames)

    saved_path = None
    if mapping_save_path:
        saved_path = save_mapping_csv(node_mapping, mapping_save_path)

    # Combine edges then group by relation_type
    all_edges = pd.concat(edge_frames, ignore_index=True)

    edge_index_dict: Dict[str, torch.Tensor] = {}
    rel_counts: Dict[str, int] = {}
    for rel, rel_df in all_edges.groupby("relation_type", sort=True):
        edge_index = map_edges_to_index(rel_df, node_mapping, strict=strict_mapping)
        edge_index_dict[rel] = edge_index
        rel_counts[rel] = int(edge_index.size(1))

    return edge_index_dict, rel_counts, node_mapping, saved_path


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", type=str, required=True, choices=["v2_core", "v2_full"])
    ap.add_argument("--injury-variant", type=str, default="mg", choices=["mg", "sg"])

    ap.add_argument("--use-award", action="store_true", help="Include award edges (v2_full only).")
    ap.add_argument("--use-injury", action="store_true", help="Include injury edges (v2_full only).")

    ap.add_argument("--core", type=str, required=True)
    ap.add_argument("--award", type=str, default=None)
    ap.add_argument("--injury-mg", type=str, default=None)
    ap.add_argument("--injury-sg", type=str, default=None)

    ap.add_argument("--audit-award", action="store_true")
    ap.add_argument("--audit-injury", action="store_true")
    ap.add_argument("--award-audit-rule", type=str, default="non_future",
                    choices=["non_future", "strict_past"])

    ap.add_argument("--non-strict-mapping", action="store_true",
                    help="Do not error on unmapped ids (should never happen in auto-mapping).")

    # NEW: auto mapping save path
    ap.add_argument("--save-mapping", type=str, default=None,
                    help="Save auto-generated node_id->idx mapping CSV to this path. "
                         "If not provided, defaults to graph/mappings/node_id_to_idx__{mode}__{injury_variant}.csv")

    ap.add_argument("--output", type=str, default=None,
                    help="Path to save torch object (edge_index_dict + rel_counts + mapping path).")
    ap.add_argument("--print-counts", action="store_true",
                    help="Print relation_type edge counts.")
    ap.add_argument("--print-nodes", action="store_true",
                    help="Print number of nodes in the auto mapping.")

    args = ap.parse_args()

    # Default mapping save path if not given
    mapping_save_path = args.save_mapping
    if mapping_save_path is None:
        mapping_save_path = f"graph/mappings/node_id_to_idx__{args.mode}__{args.injury_variant}.csv"

    edge_index_dict, rel_counts, node_mapping, saved_mapping_path = build_edge_index_dict_auto_mapping(
        mode=args.mode,
        injury_variant=args.injury_variant,
        core_path=args.core,
        award_path=args.award,
        injury_mg_path=args.injury_mg,
        injury_sg_path=args.injury_sg,
        use_award=(args.use_award if args.mode == "v2_full" else False),
        use_injury=(args.use_injury if args.mode == "v2_full" else False),
        audit_award=args.audit_award,
        audit_injury=args.audit_injury,
        award_audit_rule=args.award_audit_rule,
        strict_mapping=not args.non_strict_mapping,
        mapping_save_path=mapping_save_path,
    )

    if args.print_nodes:
        print(f"\nAuto-mapping nodes: {len(node_mapping)}")
        if saved_mapping_path:
            print(f"Mapping saved to: {saved_mapping_path}")

    if args.print_counts:
        print("\nRelation edge counts:")
        for k in sorted(rel_counts.keys()):
            print(f"  {k}: {rel_counts[k]}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "edge_index_dict": edge_index_dict,
                "rel_counts": rel_counts,
                "mode": args.mode,
                "injury_variant": args.injury_variant,
                "award_audit_rule": args.award_audit_rule,
                "mapping_csv": saved_mapping_path,
                "num_nodes": len(node_mapping),
            },
            out,
        )
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
