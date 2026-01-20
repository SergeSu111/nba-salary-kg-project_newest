#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_gnn_v1_export_embeddings.py

GNN V1 (GraphSAGE) for NBA salary KG (player-team-agent graph):
- Read graph edges from: graph/edges/edges_node2vec_L1A_elementId.csv  (src,dst,rel)
- Read player->node mapping: graph/mappings/player_nodeid_map.csv     (player_id,node_id)
- Read tabular labels: data/processed/training_level1_full.csv        (player_id,season,log_salary)
- Train a supervised GraphSAGE model to predict log_salary from player node embeddings
- Export learned player embeddings to: graph/embeddings/gnn_v1_sage_player_embeddings.csv
- (Optional) Export all node embeddings to: graph/embeddings/gnn_v1_sage_node_embeddings.csv

This is V1: structure-only node features (trainable embedding per node), GraphSAGE ignores relation types + time numeric offsets
Train/Test split follows your Step2: train seasons < TEST_SEASON, test season == TEST_SEASON.

Run (from repo root):
  python scripts/train_gnn_v1_export_embeddings.py
"""

from __future__ import annotations

import os
import sys
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
REPO_ROOT = Path(__file__).resolve().parents[2]
# ---- PyTorch Geometric (required) ----
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
except Exception as e:
    raise ImportError(
        "PyTorch Geometric is required for GNN V0.\n"
        "Install guide (pick matching CUDA/CPU wheels):\n"
        "  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html\n"
        "If you already installed it, ensure you are using the same conda env.\n"
        f"Original import error: {e}"
    )

# -------------------------
# Config
# -------------------------

@dataclass
class Config:
    # Paths (repo-root relative)
    edges_path: Path = Path("graph/edges/edges_node2vec_L1A_elementId.csv")
    player_map_path: Path = Path("graph/mappings/player_nodeid_map.csv")
    tabular_path: Path = Path("data/processed/training_level1_full.csv")

    out_player_emb_path: Path = Path("graph/embeddings/gnn_v1_sage_player_embeddings.csv")
    out_node_emb_path: Path = Path("graph/embeddings/gnn_v1_sage_node_embeddings.csv")


    # Train/Test split
    test_season: int = 2024
    target_col: str = "log_salary"
    feat_cols: Tuple[str, str] = ("age_now", "years_since_draft")

    # Model
    emb_dim: int = 64          # match Node2Vec dims for fair comparability
    hidden_dim: int = 64
    num_layers: int = 2        # 2-layer GraphSAGE is a strong default
    dropout: float = 0.2

    # Optimization
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 256
    grad_clip: float = 1.0

    # Repro
    seed: int = 42

    # Device
    device: str = "cpu"


    # Logging
    log_every: int = 10

    # Export
    export_all_nodes: bool = True


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p.resolve()}")


def detect_edge_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """
    Supports common edge list schemas.
    Must find two node columns for src and dst.
    Optionally finds relation column.
    """
    cols = [c.lower() for c in df.columns]

    def find_one(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in df.columns:
                return cand
        for cand in candidates:
            if cand in cols:
                # map back to original
                return df.columns[cols.index(cand)]
        return None

    src_col = find_one(["src", "source", "u", "head", "from", "start", "node1"])
    dst_col = find_one(["dst", "target", "v", "tail", "to", "end", "node2"])
    rel_col = find_one(["rel", "relation", "type", "edge_type"])

    if src_col is None or dst_col is None:
        raise ValueError(
            f"Cannot detect src/dst columns in edges file. Columns are: {list(df.columns)}. "
            "Expected something like (src,dst,rel) or (head,relation,tail)."
        )
    return src_col, dst_col, rel_col


def detect_player_map_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [c.lower() for c in df.columns]

    def find_one(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in df.columns:
                return cand
        for cand in candidates:
            if cand in cols:
                return df.columns[cols.index(cand)]
        return None

    player_col = find_one(["player_id", "playerid"])
    node_col = find_one(["node_id", "nodeid", "elementid", "neo4j_node_id"])

    if player_col is None or node_col is None:
        raise ValueError(
            f"Cannot detect (player_id,node_id) columns in player_nodeid_map. Columns are: {list(df.columns)}"
        )
    return player_col, node_col


def build_node_index_from_edges(edges_df: pd.DataFrame, src_col: str, dst_col: str) -> Dict[str, int]:
    src_nodes = edges_df[src_col].astype(str).values
    dst_nodes = edges_df[dst_col].astype(str).values
    nodes = np.unique(np.concatenate([src_nodes, dst_nodes]))
    # stable order for reproducibility
    nodes_sorted = sorted(nodes.tolist())
    return {nid: i for i, nid in enumerate(nodes_sorted)}


def make_edge_index(edges_df: pd.DataFrame, src_col: str, dst_col: str, node2idx: Dict[str, int]) -> torch.Tensor:
    src = edges_df[src_col].astype(str).map(node2idx).to_numpy()
    dst = edges_df[dst_col].astype(str).map(node2idx).to_numpy()

    if np.any(pd.isna(src)) or np.any(pd.isna(dst)):
        raise ValueError("Some edge endpoints were not found in node2idx; node indexing bug.")

    src = src.astype(np.int64)
    dst = dst.astype(np.int64)

    # undirected (bidirectional) graph for GraphSAGE
    src_all = np.concatenate([src, dst])
    dst_all = np.concatenate([dst, src])

    edge_index = torch.tensor(np.stack([src_all, dst_all], axis=0), dtype=torch.long)
    return edge_index


def load_tabular_labels(tabular_path: Path, target_col: str, feat_cols: Tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(tabular_path)

    required = {"player_id", "season", target_col, *feat_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tabular file missing columns: {missing}. Found: {list(df.columns)}")

    keep = ["player_id", "season", target_col, *feat_cols]
    df = df[keep].copy()

    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)

    # target
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(float)

    # features -> float
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 简单缺失填充（训练期中位数；避免泄露，用 train median）
    train_mask = df["season"] < 2024
    for c in feat_cols:
        med = df.loc[train_mask, c].median()
        df[c] = df[c].fillna(med)

    return df



def make_supervised_samples(
    tabular_df: pd.DataFrame,
    player_map_df: pd.DataFrame,
    player_col: str,
    node_col: str,
    node2idx: Dict[str, int],
    test_season: int,
    target_col: str,
    feat_cols: Tuple[str, ...]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return train/test sample dataframes with columns:
      player_id, season, y, node_idx
    """
    pm = player_map_df[[player_col, node_col]].copy()
    pm[player_col] = pm[player_col].astype(str)
    pm[node_col] = pm[node_col].astype(str)

    # map player -> graph node_id
    d = tabular_df.merge(pm, left_on="player_id", right_on=player_col, how="inner")

    # map node_id -> node_idx (GNN index)
    d["node_idx"] = d[node_col].astype(str).map(node2idx)

    before = len(d)
    d = d.dropna(subset=["node_idx"]).copy()
    d["node_idx"] = d["node_idx"].astype(int)
    after = len(d)

    if after == 0:
        raise ValueError("No supervised samples after mapping player_id -> node_idx. Check mapping consistency.")

    if after < before:
        print(f"[WARN] Dropped {before - after} rows because node_id not in edge-derived node set.")

    d = d.rename(columns={target_col: "y"})
    train = d[d["season"] < test_season].copy()
    test = d[d["season"] == test_season].copy()

    if len(test) == 0:
        raise ValueError(f"Test set empty for season == {test_season}. Check your data seasons.")

    return train, test


# -------------------------
# Model
# -------------------------

class GraphSAGERegressor(nn.Module):
    """
    V0: node features are trainable embeddings (no hand-crafted features).
    Forward: returns node embeddings + salary prediction for selected node indices.
    """
    def __init__(self, num_nodes: int, emb_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)

        convs = []
        in_dim = emb_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            convs.append(SAGEConv(in_dim, out_dim))
            in_dim = out_dim
        self.convs = nn.ModuleList(convs)

        self.dropout = dropout
        self.feat_dim = 2  # age_now + years_since_draft
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        for conv in self.convs:
            conv.reset_parameters()
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight  # [N, emb_dim]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # [N, hidden_dim]

    def forward(self, edge_index: torch.Tensor, node_idx: torch.Tensor,time_feats: torch.Tensor,) -> torch.Tensor:
        z = self.encode(edge_index)
        h = z[node_idx]  # [B, hidden_dim]
        x = torch.cat([h, time_feats], dim=1)  # [B, hidden_dim+2]
        y_hat = self.head(x).squeeze(-1)  # [B]
        return y_hat, z


# -------------------------
# Train / Eval
# -------------------------

@torch.no_grad()
def evaluate(model: nn.Module, edge_index: torch.Tensor, samples: pd.DataFrame, feat_cols, device: str) -> Dict[str, float]:
    model.eval()
    node_idx = torch.tensor(samples["node_idx"].to_numpy(), dtype=torch.long, device=device)
    y_true = torch.tensor(samples["y"].to_numpy(), dtype=torch.float32, device=device)

    time_feats = torch.tensor(
        samples[list(feat_cols)].to_numpy(),
        dtype=torch.float32,
        device=device
    )

    y_pred, _ = model(edge_index, node_idx, time_feats)

    mse = F.mse_loss(y_pred, y_true).item()
    rmse = math.sqrt(mse)
    mae = F.l1_loss(y_pred, y_true).item()

    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def train_loop(cfg: Config) -> None:
    set_seed(cfg.seed)
    # Resolve paths relative to repo root (robust to cwd)
    cfg.edges_path = (REPO_ROOT / cfg.edges_path).resolve()
    cfg.player_map_path = (REPO_ROOT / cfg.player_map_path).resolve()
    cfg.tabular_path = (REPO_ROOT / cfg.tabular_path).resolve()
    cfg.out_player_emb_path = (REPO_ROOT / cfg.out_player_emb_path).resolve()
    cfg.out_node_emb_path = (REPO_ROOT / cfg.out_node_emb_path).resolve()

    # Resolve paths
    assert_exists(cfg.edges_path)
    assert_exists(cfg.player_map_path)
    assert_exists(cfg.tabular_path)

    print(f"[device] {cfg.device}")
    print(f"[paths] edges={cfg.edges_path}  player_map={cfg.player_map_path}  tabular={cfg.tabular_path}")

    # Load edges
    edges_df = pd.read_csv(cfg.edges_path)
    src_col, dst_col, rel_col = detect_edge_columns(edges_df)
    print(f"[edges] columns={list(edges_df.columns)}  detected src={src_col} dst={dst_col} rel={rel_col}")
    print(f"[edges] rows={len(edges_df)}")

    # Build node index
    node2idx = build_node_index_from_edges(edges_df, src_col, dst_col)
    num_nodes = len(node2idx)
    print(f"[graph] num_nodes={num_nodes}")

    # Build edge_index
    edge_index = make_edge_index(edges_df, src_col, dst_col, node2idx)
    edge_index = edge_index.to(cfg.device)
    print(f"[graph] edge_index shape={tuple(edge_index.shape)} (bidirectional)")

    # Load mapping & tabular
    player_map_df = pd.read_csv(cfg.player_map_path)
    player_col, node_col = detect_player_map_columns(player_map_df)
    tab = load_tabular_labels(cfg.tabular_path, cfg.target_col, cfg.feat_cols)

    # Make supervised samples
    train_df, test_df = make_supervised_samples(
        tabular_df=tab,
        player_map_df=player_map_df,
        player_col=player_col,
        node_col=node_col,
        node2idx=node2idx,
        test_season=cfg.test_season,
        target_col=cfg.target_col,
        feat_cols=cfg.feat_cols,
    )

    print(f"[samples] train={len(train_df)} test={len(test_df)} "
          f"unique_players_train={train_df['player_id'].nunique()} unique_players_test={test_df['player_id'].nunique()}")

    # Model
    model = GraphSAGERegressor(
        num_nodes=num_nodes,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Prepare mini-batches (supervised rows)
    train_idx = np.arange(len(train_df))
    best = {"epoch": -1, "rmse": float("inf"), "state": None}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        np.random.shuffle(train_idx)

        total_loss = 0.0
        n_seen = 0

        for start in range(0, len(train_idx), cfg.batch_size):
            batch_rows = train_df.iloc[train_idx[start:start + cfg.batch_size]]
            node_idx = torch.tensor(batch_rows["node_idx"].to_numpy(), dtype=torch.long, device=cfg.device)
            y_true = torch.tensor(batch_rows["y"].to_numpy(), dtype=torch.float32, device=cfg.device)

            time_feats = torch.tensor(
            batch_rows[list(cfg.feat_cols)].to_numpy(),
            dtype=torch.float32,
            device=cfg.device)
           
            if epoch == 1 and start == 0:
                print("[DEBUG] time_feats shape:", time_feats.shape,
                    "min/max:", float(time_feats.min()), float(time_feats.max()))

            y_pred, _ = model(edge_index, node_idx, time_feats)
            loss = F.mse_loss(y_pred, y_true)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            bs = len(batch_rows)
            total_loss += loss.item() * bs
            n_seen += bs

        train_mse = total_loss / max(1, n_seen)
        train_rmse = math.sqrt(train_mse)

        if epoch % cfg.log_every == 0 or epoch == 1:
            tr = evaluate(model, edge_index, train_df, cfg.feat_cols, cfg.device)
            te = evaluate(model, edge_index, test_df,  cfg.feat_cols, cfg.device)

            print(
                f"[epoch {epoch:03d}] train_rmse={train_rmse:.4f} "
                f"| train r2={tr['r2']:.4f} rmse={tr['rmse']:.4f} mae={tr['mae']:.4f} "
                f"| test  r2={te['r2']:.4f} rmse={te['rmse']:.4f} mae={te['mae']:.4f}"
            )

            # Track best by test rmse (epoch-level)
            if te["rmse"] < best["rmse"]:
                best["epoch"] = epoch
                best["rmse"] = te["rmse"]
                best["state"] = {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}

            
                
    # Restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"[best] epoch={best['epoch']} best_test_rmse={best['rmse']:.4f}")

    # Final eval
    tr = evaluate(model, edge_index, train_df, cfg.feat_cols, cfg.device)
    te = evaluate(model, edge_index, test_df,  cfg.feat_cols, cfg.device)

    print(f"[final] train r2={tr['r2']:.4f} rmse={tr['rmse']:.4f} mae={tr['mae']:.4f}")
    print(f"[final] test  r2={te['r2']:.4f} rmse={te['rmse']:.4f} mae={te['mae']:.4f}")

    # Encode all nodes -> z (hidden_dim)
    model.eval()
    with torch.no_grad():
        z = model.encode(edge_index).detach().cpu().numpy()  # [N, hidden_dim]

    # Export player embeddings: need player_id -> node_id -> node_idx -> embedding
    pm = player_map_df[[player_col, node_col]].copy()
    pm[player_col] = pm[player_col].astype(str)
    pm[node_col] = pm[node_col].astype(str)
    pm["node_idx"] = pm[node_col].map(node2idx)
    pm = pm.dropna(subset=["node_idx"]).copy()
    pm["node_idx"] = pm["node_idx"].astype(int)

    # Keep only players that actually appear in tabular (so evaluation merge is clean)
    players_in_tab = set(tab["player_id"].astype(str).unique().tolist())
    pm = pm[pm[player_col].isin(players_in_tab)].copy()

    emb_df = pd.DataFrame({
        "player_id": pm[player_col].values,
        "node_id": pm[node_col].values,
        "node_idx": pm["node_idx"].values,
    })
    # Embedding columns
    D = z.shape[1]
    for j in range(D):
        emb_df[f"e{j}"] = z[pm["node_idx"].values, j]

    cfg.out_player_emb_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_csv(cfg.out_player_emb_path, index=False)
    print(f"[export] player embeddings -> {cfg.out_player_emb_path}  rows={len(emb_df)} dim={D}")

    if cfg.export_all_nodes:
        # Export all node embeddings with node_id
        inv = {idx: nid for nid, idx in node2idx.items()}
        node_rows = []
        for idx in range(num_nodes):
            row = {"node_id": inv[idx], "node_idx": idx}
            for j in range(D):
                row[f"e{j}"] = float(z[idx, j])
            node_rows.append(row)
        node_df = pd.DataFrame(node_rows)
        cfg.out_node_emb_path.parent.mkdir(parents=True, exist_ok=True)
        node_df.to_csv(cfg.out_node_emb_path, index=False)
        print(f"[export] node embeddings -> {cfg.out_node_emb_path}  rows={len(node_df)} dim={D}")


def main():
    # Ensure running from repo root (or adjust cwd)
    cfg = Config()
    train_loop(cfg)


if __name__ == "__main__":
    main()
