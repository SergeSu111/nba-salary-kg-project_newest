#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sanity_checks_for_v2_baseline_ABCD_E.py

Adds the requested A–E checks on top of the V2 baseline sanity:
A) Mean-baseline (predict train mean) with train/val/test metrics
B) Report train/val/test R2+RMSE for every sanity mode (not just test)
C) node_idx mapping sanity: duplicates + label conflicts (same node_idx with multiple y)
D) y distribution describe (train/val/test) printed once per (graph_mode, time_feats)
E) 64-sample forced overfit test (should fit near-zero RMSE if pipeline is correct)

Run:
  conda run -n nba-research python graph/gnn/sanity_checks_for_v2_baseline_ABCD_E.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Iterable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- PyTorch Geometric ----
try:
    from torch_geometric.nn import SAGEConv
except Exception as e:
    raise ImportError(
        "PyTorch Geometric is required.\n"
        "Install guide (pick matching CUDA/CPU wheels):\n"
        "  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html\n"
        "If you already installed it, ensure you are using the same conda env.\n"
        f"Original import error: {e}"
    )

# -------------------------
# Repo root
# -------------------------
def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "graph").exists() and (p / "data").exists():
            if (p / "graph" / "edges").exists() and (p / "data" / "processed").exists():
                return p
    return start.parent


REPO_ROOT = find_repo_root(Path(__file__).resolve())
print(f"[repo_root] {REPO_ROOT}")

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    # Protocol splits
    test_season: int = 2024
    val_season: int = 2023

    # Files (repo-root relative)
    edges_path: Path = Path("graph/edges/edges_gnn_v2_core_elementId_full.csv")
    playerseason_map_path: Path = Path("graph/mappings/playerSeason.csv")
    tabular_path: Path = Path("data/processed/training_level1_full.csv")

    # Target + optional time feats
    target_col: str = "log_salary"
    feat_cols: Tuple[str, str] = ("age_now", "years_since_draft")

    # Model (baseline)
    emb_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Optimization
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    grad_clip: float = 1.0

    # Repro
    seed: int = 42

    # Device
    device: str = "cpu"

    # Graph protocol
    graph_mode: str = "transductive"  # or "inductive"

    # Inductive: freeze test PlayerSeason embedding rows
    freeze_test_ps: bool = True

    # Random-graph seed decoupling (recommended)
    random_graph_seed_offset: int = 10000

    # Overfit test (E)
    overfit_n: int = 64
    overfit_epochs: int = 1500
    overfit_lr: float = 1e-2


# -------------------------
# Sanity modes
# -------------------------
SANITY_MODES = [
    "mean_y_baseline",  # (A) added
    "orig",
    "id_only",  # (= no message passing; still learnable ID embeddings + MLP head)
    "random_graph_weak",
    "random_graph_strong",
    "random_graph_degree_preserving",
    "shuffle_labels",
    "cold_start_player_within_trainval",
    "cold_start_2024_unseen_players_only",
]


# -------------------------
# Utils
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
    cols_lower = [c.lower() for c in df.columns]

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        for c in cands:
            if c in cols_lower:
                return df.columns[cols_lower.index(c)]
        return None

    src = pick(["src", "source", "u", "head", "from", "start", "node1"])
    dst = pick(["dst", "target", "v", "tail", "to", "end", "node2"])
    rel = pick(["rel", "relation", "type", "edge_type"])
    if src is None or dst is None:
        raise ValueError(f"Cannot detect src/dst columns in edges file. columns={list(df.columns)}")
    return src, dst, rel


def detect_playerseason_map_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols_lower = [c.lower() for c in df.columns]

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        for c in cands:
            if c in cols_lower:
                return df.columns[cols_lower.index(c)]
        return None

    player = pick(["player_id", "playerid"])
    season = pick(["season", "year"])
    node = pick(["node_id", "nodeid", "elementid", "neo4j_node_id"])
    if player is None or season is None or node is None:
        raise ValueError(
            "Cannot detect (player_id, season, node_id) in playerSeason mapping. "
            f"columns={list(df.columns)}"
        )
    return player, season, node


def build_node2idx_from_edges(edges_df: pd.DataFrame, src_col: str, dst_col: str) -> Dict[str, int]:
    src_nodes = edges_df[src_col].astype(str).values
    dst_nodes = edges_df[dst_col].astype(str).values
    nodes = np.unique(np.concatenate([src_nodes, dst_nodes]))
    nodes = sorted(nodes.tolist())
    return {nid: i for i, nid in enumerate(nodes)}


def make_edge_index_bidirectional(edges_df: pd.DataFrame, src_col: str, dst_col: str, node2idx: Dict[str, int]) -> torch.Tensor:
    src = edges_df[src_col].astype(str).map(node2idx).to_numpy(dtype=np.int64)
    dst = edges_df[dst_col].astype(str).map(node2idx).to_numpy(dtype=np.int64)
    # bidirectional expansion
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    return torch.tensor(np.stack([s, d], axis=0), dtype=torch.long)


def edge_index_to_undirected_unique(edge_index: torch.Tensor) -> np.ndarray:
    ei = edge_index.detach().cpu().numpy()
    src = ei[0].astype(np.int64)
    dst = ei[1].astype(np.int64)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    und = np.stack([u, v], axis=1)
    und = np.unique(und, axis=0)
    return und


def make_edge_index_from_undirected(undirected_edges: np.ndarray) -> torch.Tensor:
    u = undirected_edges[:, 0].astype(np.int64)
    v = undirected_edges[:, 1].astype(np.int64)
    s = np.concatenate([u, v])
    d = np.concatenate([v, u])
    return torch.tensor(np.stack([s, d], axis=0), dtype=torch.long)


# (A) Mean baseline metrics
def eval_mean_baseline(train_df: pd.DataFrame, df: pd.DataFrame) -> Dict[str, float]:
    yhat = float(train_df["y"].mean())
    yt = df["y"].to_numpy(dtype=np.float64)
    yp = np.full_like(yt, yhat, dtype=np.float64)
    mse = float(np.mean((yt - yp) ** 2))
    rmse = math.sqrt(mse)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "r2": r2}


# (D) y describe printer
def describe_y(df: pd.DataFrame, name: str) -> str:
    s = df["y"].astype(float)
    q = s.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
    return (
        f"{name}: n={len(s)} mean={s.mean():.4f} std={s.std():.4f} "
        f"min={q[0.0]:.4f} p25={q[0.25]:.4f} p50={q[0.5]:.4f} p75={q[0.75]:.4f} max={q[1.0]:.4f}"
    )


# (C) node_idx mapping diagnostics
def node_idx_diagnostics(samples: pd.DataFrame) -> Dict[str, float]:
    # duplicates
    counts = samples.groupby("node_idx").size()
    n_total_nodes = int(counts.shape[0])
    n_dup_nodes = int((counts > 1).sum())
    max_count = int(counts.max()) if n_total_nodes else 0
    frac_dup = float(n_dup_nodes / max(1, n_total_nodes))

    # label conflicts: same node_idx with multiple distinct y values
    y_nunique = samples.groupby("node_idx")["y"].nunique()
    n_conflict_nodes = int((y_nunique > 1).sum())
    frac_conflict = float(n_conflict_nodes / max(1, n_total_nodes))

    return {
        "node_idx_unique_nodes": n_total_nodes,
        "node_idx_dup_nodes": n_dup_nodes,
        "node_idx_dup_frac": frac_dup,
        "node_idx_max_count": max_count,
        "node_idx_conflict_nodes": n_conflict_nodes,
        "node_idx_conflict_frac": frac_conflict,
    }


# -------------------------
# Standardization for time feats (train-only)
# -------------------------
def compute_stats(train_df: pd.DataFrame, feat_cols: Tuple[str, ...]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for c in feat_cols:
        mu = float(train_df[c].mean())
        sigma = float(train_df[c].std())
        if sigma == 0 or np.isnan(sigma):
            sigma = 1.0
        stats[c] = (mu, sigma)
    return stats


def apply_stats(df: pd.DataFrame, feat_cols: Tuple[str, ...], stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c in feat_cols:
        mu, sigma = stats[c]
        out[c] = (out[c] - mu) / sigma
    return out


# -------------------------
# Model
# -------------------------
class GraphSAGEBaseline(nn.Module):
    """
    Baseline:
      - node ID embedding always present
      - if mp_enabled: GraphSAGE message passing
      - if not: id_only path uses a projection to hidden_dim
    """
    def __init__(
        self,
        num_nodes: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        feat_dim: int,
        mp_enabled: bool = True,
        use_layernorm: bool = True,
        emb_max_norm: Optional[float] = 1.0,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim, max_norm=emb_max_norm)
        self.mp_enabled = mp_enabled
        self.dropout = dropout
        self.feat_dim = feat_dim

        self.id_proj = nn.Linear(emb_dim, hidden_dim, bias=False)

        convs = []
        in_dim = emb_dim
        for _ in range(num_layers):
            convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(convs)

        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.id_proj.weight)
        for conv in self.convs:
            conv.reset_parameters()
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(self.node_emb.num_embeddings, device=edge_index.device)
        x = self.node_emb(idx)

        if not self.mp_enabled:
            z = self.id_proj(x)
            return self.ln(z)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.ln(x)

    def predict(self, z: torch.Tensor, node_idx: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        h = z[node_idx]
        x = torch.cat([h, feats], dim=1) if feats.numel() else h
        y_hat = self.head(x).squeeze(-1)
        return y_hat


@torch.no_grad()
def eval_split(
    model: GraphSAGEBaseline,
    edge_index: torch.Tensor,
    df: pd.DataFrame,
    feat_cols: Tuple[str, ...],
    use_time_feats: bool,
    device: str,
) -> Dict[str, float]:
    model.eval()
    edge_index = edge_index.to(device)
    z = model.encode(edge_index)

    node_idx = torch.tensor(df["node_idx"].to_numpy(), dtype=torch.long, device=device)
    y_true = torch.tensor(df["y"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

    if use_time_feats:
        feats = torch.tensor(df[list(feat_cols)].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    else:
        feats = torch.zeros((len(df), 0), dtype=torch.float32, device=device)

    y_pred = model.predict(z, node_idx, feats)

    mse = F.mse_loss(y_pred, y_true).item()
    rmse = math.sqrt(mse)

    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"rmse": rmse, "r2": r2}


# -------------------------
# Graph protocol helpers
# -------------------------
def get_test_playerseason_node_ids(
    player_map_df: pd.DataFrame,
    player_col: str,
    season_col: str,
    node_col: str,
    test_season: int,
) -> List[str]:
    tmp = player_map_df[[season_col, node_col]].copy()
    tmp[season_col] = tmp[season_col].astype(int)
    tmp[node_col] = tmp[node_col].astype(str)
    return tmp.loc[tmp[season_col] == test_season, node_col].astype(str).tolist()


def filter_edges_remove_incident(edges_df: pd.DataFrame, src_col: str, dst_col: str, banned_nodes: set) -> pd.DataFrame:
    src = edges_df[src_col].astype(str)
    dst = edges_df[dst_col].astype(str)
    keep = (~src.isin(banned_nodes)) & (~dst.isin(banned_nodes))
    return edges_df.loc[keep].copy()


def freeze_test_ps_rows(model: GraphSAGEBaseline, node2idx: Dict[str, int], test_node_ids: List[str], device: str) -> str:
    idx_list = [node2idx[n] for n in test_node_ids if n in node2idx]
    if not idx_list:
        return "inductive-freeze: no test PlayerSeason nodes in node2idx; skipped."

    test_idx = torch.tensor(sorted(set(idx_list)), dtype=torch.long, device=device)

    def hook(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g.index_fill_(0, test_idx, 0.0)
        return g

    model.node_emb.weight.register_hook(hook)
    return f"inductive-freeze: zeroed gradients for {test_idx.numel()} test PlayerSeason embeddings."


# -------------------------
# Load + build samples
# -------------------------
def load_and_prepare(cfg: Config) -> Dict:
    # resolve paths
    edges_path = (REPO_ROOT / cfg.edges_path).resolve()
    map_path = (REPO_ROOT / cfg.playerseason_map_path).resolve()
    tab_path = (REPO_ROOT / cfg.tabular_path).resolve()

    for p in [edges_path, map_path, tab_path]:
        assert_exists(p)

    # edges
    edges_df = pd.read_csv(edges_path)
    src_col, dst_col, rel_col = detect_edge_columns(edges_df)
    # node index from FULL edges (keeps consistent node IDs)
    node2idx = build_node2idx_from_edges(edges_df, src_col, dst_col)
    num_nodes = len(node2idx)

    # mapping + tabular
    pm = pd.read_csv(map_path)
    player_col, season_col, node_col = detect_playerseason_map_columns(pm)
    pm[player_col] = pm[player_col].astype(str)
    pm[season_col] = pm[season_col].astype(int)
    pm[node_col] = pm[node_col].astype(str)

    tab = pd.read_csv(tab_path)
    required = {"player_id", "season", cfg.target_col, *cfg.feat_cols}
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(f"Tabular missing columns: {missing}. found={list(tab.columns)}")

    tab = tab[list(required)].copy()
    tab["player_id"] = tab["player_id"].astype(str)
    tab["season"] = tab["season"].astype(int)
    tab[cfg.target_col] = pd.to_numeric(tab[cfg.target_col], errors="coerce")
    tab = tab.dropna(subset=[cfg.target_col]).copy()
    for c in cfg.feat_cols:
        tab[c] = pd.to_numeric(tab[c], errors="coerce")

    # merge (player_id, season) -> PlayerSeason node_id -> node_idx
    merged = tab.merge(
        pm[[player_col, season_col, node_col]],
        left_on=["player_id", "season"],
        right_on=[player_col, season_col],
        how="inner",
    )
    merged["node_idx"] = merged[node_col].astype(str).map(node2idx)
    before = len(merged)
    merged = merged.dropna(subset=["node_idx"]).copy()
    merged["node_idx"] = merged["node_idx"].astype(int)
    after = len(merged)
    if after < before:
        print(f"[WARN] Dropped {before - after} rows: mapped node_id not in edge-derived node set.")

    merged = merged.rename(columns={cfg.target_col: "y"})
    merged["y"] = merged["y"].astype(float)

    # split (paper split)
    train_df = merged[merged["season"] < cfg.val_season].copy()
    val_df = merged[merged["season"] == cfg.val_season].copy()
    test_df = merged[merged["season"] == cfg.test_season].copy()

    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Val/Test empty. Check val_season/test_season config and data coverage.")

    # fill feat NaN using TRAIN median (no leakage)
    for c in cfg.feat_cols:
        med = float(train_df[c].median())
        train_df[c] = train_df[c].fillna(med)
        val_df[c] = val_df[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)

    # standardize feats using TRAIN only (applied once)
    stats = compute_stats(train_df, cfg.feat_cols)
    train_df = apply_stats(train_df, cfg.feat_cols, stats)
    val_df = apply_stats(val_df, cfg.feat_cols, stats)
    test_df = apply_stats(test_df, cfg.feat_cols, stats)

    # build edge_index_full (bidirectional)
    edge_index_full = make_edge_index_bidirectional(edges_df, src_col, dst_col, node2idx)

    # build training graph depending on graph_mode
    protocol_note = ""
    train_edges_df = edges_df
    freeze_note = ""

    if cfg.graph_mode == "transductive":
        edge_index_train = edge_index_full
        protocol_note = "transductive: train-time message passing sees full structure; labels only from train/val"
    elif cfg.graph_mode == "inductive":
        test_node_ids = get_test_playerseason_node_ids(pm, player_col, season_col, node_col, cfg.test_season)
        banned = set(test_node_ids)
        train_edges_df = filter_edges_remove_incident(edges_df, src_col, dst_col, banned)
        edge_index_train = make_edge_index_bidirectional(train_edges_df, src_col, dst_col, node2idx)
        protocol_note = (
            "inductive(train): remove test-season PlayerSeason nodes+incident edges during training-time message passing; "
            "evaluation uses full structure at inference (structure-visible); "
            "test-season nodes get representations via neighborhood aggregation at inference"
        )
    else:
        raise ValueError("graph_mode must be 'transductive' or 'inductive'")

    # undirected unique counts for randomization bookkeeping (train graph)
    und_train = edge_index_to_undirected_unique(edge_index_train)
    E_base_und_train = int(und_train.shape[0])

    # overlap ratio (orig only; computed later per-mode as well)
    seen_players = set(pd.concat([train_df, val_df])["player_id"].unique())
    test_players = set(test_df["player_id"].unique())
    overlap_ratio_test = len(seen_players & test_players) / max(1, len(test_players))

    # (C) global mapping diagnostics
    diag = node_idx_diagnostics(pd.concat([train_df, val_df, test_df], axis=0))

    pack = {
        "edges_df_full": edges_df,
        "edges_df_train": train_edges_df,
        "src_col": src_col,
        "dst_col": dst_col,
        "rel_col": rel_col,
        "node2idx": node2idx,
        "num_nodes": num_nodes,
        "pm": pm,
        "player_col": player_col,
        "season_col": season_col,
        "node_col": node_col,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "edge_index_full": edge_index_full,
        "edge_index_train": edge_index_train,
        "undirected_train": und_train,
        "E_base_und_train": E_base_und_train,
        "protocol_note": protocol_note,
        "overlap_ratio_test": overlap_ratio_test,
        "node_idx_diag": diag,
    }
    return pack


# -------------------------
# Random graphs (train-graph undirected base)
# -------------------------
def randomize_undirected_edges_weak(und: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = und[:, 0].copy()
    v = und[:, 1].copy()
    rng.shuffle(v)

    uu = np.minimum(u, v)
    vv = np.maximum(u, v)
    out = np.stack([uu, vv], axis=1)
    out = out[uu != vv]
    out = np.unique(out, axis=0)
    return out


def randomize_undirected_edges_strong(und: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = und[:, 0].copy()
    v = und[:, 1].copy()
    rng.shuffle(u)
    rng.shuffle(v)

    uu = np.minimum(u, v)
    vv = np.maximum(u, v)
    out = np.stack([uu, vv], axis=1)
    out = out[uu != vv]
    out = np.unique(out, axis=0)
    return out


def degree_preserving_swap_undirected(und: np.ndarray, seed: int, n_swaps_factor: int = 10) -> Optional[np.ndarray]:
    try:
        import networkx as nx
    except Exception:
        return None

    if und.shape[0] < 10:
        return None

    G = nx.Graph()
    G.add_edges_from([(int(a), int(b)) for a, b in und])

    nswap = n_swaps_factor * G.number_of_edges()
    max_tries = nswap * 20

    random.seed(seed)
    try:
        nx.double_edge_swap(G, nswap=nswap, max_tries=max_tries)
    except Exception:
        return None

    new_edges = np.array(list(G.edges()), dtype=np.int64)
    uu = np.minimum(new_edges[:, 0], new_edges[:, 1])
    vv = np.maximum(new_edges[:, 0], new_edges[:, 1])
    out = np.unique(np.stack([uu, vv], axis=1), axis=0)
    return out


# -------------------------
# Train + eval (full-batch per epoch; avoids the mini-batch backward bug)
# -------------------------
def train_fullbatch(
    cfg: Config,
    pack: Dict,
    mp_enabled: bool,
    use_time_feats: bool,
    edge_index_train: torch.Tensor,
    edge_index_test: torch.Tensor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    shuffle_y_train: bool = False,
    freeze_test_note_needed: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], str]:
    # defensive device handling
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # optional label shuffle (sanity)
    train_work = train_df.copy()
    if shuffle_y_train:
        y = train_work["y"].to_numpy().copy()
        np.random.default_rng(cfg.seed).shuffle(y)
        train_work["y"] = y

    model = GraphSAGEBaseline(
        num_nodes=pack["num_nodes"],
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        feat_dim=(len(cfg.feat_cols) if use_time_feats else 0),
        mp_enabled=mp_enabled,
        use_layernorm=True,
        emb_max_norm=1.0,
    ).to(device)

    freeze_note = ""
    if freeze_test_note_needed and cfg.graph_mode == "inductive" and cfg.freeze_test_ps:
        test_node_ids = get_test_playerseason_node_ids(
            pack["pm"], pack["player_col"], pack["season_col"], pack["node_col"], cfg.test_season
        )
        freeze_note = freeze_test_ps_rows(model, pack["node2idx"], test_node_ids, device)

    # AdamW: do NOT weight-decay node embeddings (stability)
    emb_params = list(model.node_emb.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("node_emb.")]
    opt = torch.optim.AdamW(
        [
            {"params": emb_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": cfg.weight_decay},
        ],
        lr=cfg.lr,
    )

    edge_index_train = edge_index_train.to(device)
    edge_index_test = edge_index_test.to(device)

    # tensors for train (full-batch)
    node_idx_tr = torch.tensor(train_work["node_idx"].to_numpy(), dtype=torch.long, device=device)
    y_tr = torch.tensor(train_work["y"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    if use_time_feats:
        feats_tr = torch.tensor(train_work[list(cfg.feat_cols)].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    else:
        feats_tr = torch.zeros((len(train_work), 0), dtype=torch.float32, device=device)

    best_val_rmse = float("inf")
    best_state = None

    for _epoch in range(1, cfg.epochs + 1):
        model.train()
        z = model.encode(edge_index_train)  # one encode per epoch
        yhat = model.predict(z, node_idx_tr, feats_tr)
        loss = F.mse_loss(yhat, y_tr)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # checkpoint by val RMSE
        va = eval_split(model, edge_index_train, val_df, cfg.feat_cols, use_time_feats, device)
        if va["rmse"] < best_val_rmse:
            best_val_rmse = va["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    tr = eval_split(model, edge_index_train, train_df, cfg.feat_cols, use_time_feats, device)
    va = eval_split(model, edge_index_train, val_df, cfg.feat_cols, use_time_feats, device)
    te = eval_split(model, edge_index_test, test_df, cfg.feat_cols, use_time_feats, device)

    return tr, va, te, freeze_note


# -------------------------
# (E) 64-sample overfit test
# -------------------------
def overfit_64_test(cfg: Config, pack: Dict, use_time_feats: bool) -> Dict[str, float]:
    """
    Train on 64 samples only. If the pipeline is correct, train RMSE should go very low.
    We make this as "easy" as possible:
      - dropout=0
      - no layernorm
      - no max_norm
      - weight_decay=0
      - higher lr
    """
    set_seed(cfg.seed)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_df = pack["train_df"].copy()
    if len(train_df) < cfg.overfit_n:
        n = max(8, len(train_df))
        print(f"[overfit64] WARN: train has only {len(train_df)} rows; using n={n}")
        n_use = n
    else:
        n_use = cfg.overfit_n

    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(np.arange(len(train_df)), size=n_use, replace=False)
    small = train_df.iloc[idx].copy()

    edge_train = pack["edge_index_train"].to(device)

    model = GraphSAGEBaseline(
        num_nodes=pack["num_nodes"],
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=0.0,
        feat_dim=(len(cfg.feat_cols) if use_time_feats else 0),
        mp_enabled=True,
        use_layernorm=False,
        emb_max_norm=None,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.overfit_lr, weight_decay=0.0)

    node_idx = torch.tensor(small["node_idx"].to_numpy(), dtype=torch.long, device=device)
    y_true = torch.tensor(small["y"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    if use_time_feats:
        feats = torch.tensor(small[list(cfg.feat_cols)].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    else:
        feats = torch.zeros((len(small), 0), dtype=torch.float32, device=device)

    for _ in range(cfg.overfit_epochs):
        model.train()
        z = model.encode(edge_train)
        y_pred = model.predict(z, node_idx, feats)
        loss = F.mse_loss(y_pred, y_true)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # evaluate on the same 64 points
    model.eval()
    with torch.no_grad():
        z = model.encode(edge_train)
        y_pred = model.predict(z, node_idx, feats)
        mse = F.mse_loss(y_pred, y_true).item()
        rmse = math.sqrt(mse)

        yt = y_true.detach().cpu().numpy()
        yp = y_pred.detach().cpu().numpy()
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"overfit_n": float(n_use), "overfit_train_rmse": float(rmse), "overfit_train_r2": float(r2)}


# -------------------------
# Split modifiers for cold-start modes
# -------------------------
def make_splits_paper(cfg: Config, pack: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    return pack["train_df"].copy(), pack["val_df"].copy(), pack["test_df"].copy(), "paper_split: train<val_season, val==val_season, test==test_season"


def split_cold_start_player_within_trainval(cfg: Config, pack: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Hold out ~20% players within pre-2024 seasons; val is max season in remaining cold_train.
    This is an auxiliary sanity; it is NOT the fixed val=2023 paper split.
    """
    base = pd.concat([pack["train_df"], pack["val_df"]], axis=0, ignore_index=True)
    rng = np.random.default_rng(cfg.seed)
    all_players = base["player_id"].unique()
    holdout = set(rng.choice(all_players, size=max(1, int(0.2 * len(all_players))), replace=False))

    cold_train = base[~base["player_id"].isin(holdout)].copy()
    cold_test = base[ base["player_id"].isin(holdout)].copy()

    if cold_train["season"].nunique() >= 2:
        v_season = int(cold_train["season"].max())
        val_df = cold_train[cold_train["season"] == v_season].copy()
        train_df = cold_train[cold_train["season"] < v_season].copy()
    else:
        # fallback random split
        idx = np.arange(len(cold_train))
        rng.shuffle(idx)
        cut = int(0.9 * len(idx))
        train_df = cold_train.iloc[idx[:cut]].copy()
        val_df = cold_train.iloc[idx[cut:]].copy()

    test_df = cold_test
    note = "aux_split: cold-start players within pre-2024; val is max(pre-2024 season in cold_train)"
    return train_df, val_df, test_df, note


def split_cold_start_2024_unseen_only(cfg: Config, pack: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    train_df = pack["train_df"].copy()
    val_df = pack["val_df"].copy()
    test_df = pack["test_df"].copy()

    seen = set(pd.concat([train_df, val_df])["player_id"].unique())
    orig_test_players = test_df["player_id"].nunique()
    test_df = test_df[~test_df["player_id"].isin(seen)].copy()
    new_test_players = test_df["player_id"].nunique()
    ratio = new_test_players / max(1, orig_test_players)
    note = f"paper_split: train<val_season, val==val_season, test==test_season | 2024_unseen_players_only player_ratio={ratio:.3f}"
    return train_df, val_df, test_df, note


# -------------------------
# One run (one mode, one seed)
# -------------------------
def run_one_mode(cfg: Config, pack: Dict, sanity_mode: str, use_time_feats: bool) -> Dict:
    set_seed(cfg.seed)

    # splits
    if sanity_mode in ["mean_y_baseline", "orig", "id_only", "random_graph_weak", "random_graph_strong", "random_graph_degree_preserving", "shuffle_labels"]:
        train_df, val_df, test_df, split_note = make_splits_paper(cfg, pack)
    elif sanity_mode == "cold_start_player_within_trainval":
        train_df, val_df, test_df, split_note = split_cold_start_player_within_trainval(cfg, pack)
    elif sanity_mode == "cold_start_2024_unseen_players_only":
        train_df, val_df, test_df, split_note = split_cold_start_2024_unseen_only(cfg, pack)
    else:
        raise ValueError(f"Unknown sanity_mode: {sanity_mode}")

    # emptiness guard
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        return {
            "graph_mode": cfg.graph_mode,
            "sanity_mode": sanity_mode,
            "mp_enabled": None,
            "use_time_feats": use_time_feats,
            "seed": cfg.seed,
            "train_r2": float("nan"), "train_rmse": float("nan"),
            "val_r2": float("nan"), "val_rmse": float("nan"),
            "test_r2": float("nan"), "test_rmse": float("nan"),
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "n_test_players": int(test_df["player_id"].nunique()) if len(test_df) else 0,
            "E_base_und_train": pack["E_base_und_train"],
            "E_after_und_train": pack["E_base_und_train"],
            "edge_ratio_train": 1.0,
            "overlap_ratio_test": pack["overlap_ratio_test"] if sanity_mode == "orig" else float("nan"),
            "protocol_note": pack["protocol_note"],
            "split_note": split_note + " | EMPTY SPLIT",
            "notes": "",
            **pack["node_idx_diag"],
        }

    # overlap only meaningful for orig on paper split
    overlap_ratio = float("nan")
    if sanity_mode == "orig":
        seen_players = set(pd.concat([train_df, val_df])["player_id"].unique())
        test_players = set(test_df["player_id"].unique())
        overlap_ratio = len(seen_players & test_players) / max(1, len(test_players))

    # edges: default uses training graph; test uses full graph for evaluation (structure-visible inference)
    edge_index_train = pack["edge_index_train"]
    edge_index_test = pack["edge_index_full"]  # always evaluate on full graph for test (matches your protocol)

    E_base = pack["E_base_und_train"]
    E_after = E_base
    edge_ratio = 1.0
    notes = ""
    mp_enabled = True
    shuffle_y = False

    # (A) mean baseline row
    if sanity_mode == "mean_y_baseline":
        mb_tr = eval_mean_baseline(train_df, train_df)
        mb_va = eval_mean_baseline(train_df, val_df)
        mb_te = eval_mean_baseline(train_df, test_df)
        return {
            "graph_mode": cfg.graph_mode,
            "sanity_mode": sanity_mode,
            "mp_enabled": False,
            "use_time_feats": use_time_feats,
            "seed": cfg.seed,
            "train_r2": mb_tr["r2"], "train_rmse": mb_tr["rmse"],
            "val_r2": mb_va["r2"], "val_rmse": mb_va["rmse"],
            "test_r2": mb_te["r2"], "test_rmse": mb_te["rmse"],
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "n_test_players": int(test_df["player_id"].nunique()),
            "E_base_und_train": E_base,
            "E_after_und_train": E_after,
            "edge_ratio_train": edge_ratio,
            "overlap_ratio_test": overlap_ratio,
            "protocol_note": pack["protocol_note"],
            "split_note": split_note,
            "notes": "predict y=mean(train)",
            **pack["node_idx_diag"],
        }

    # id_only
    if sanity_mode == "id_only":
        mp_enabled = False
        notes = "no message passing (ID-emb + proj + MLP)"

    # shuffle labels
    if sanity_mode == "shuffle_labels":
        shuffle_y = True
        notes = "shuffle_y_train"

    # random graphs (decouple seed)
    if sanity_mode.startswith("random_graph_"):
        und = pack["undirected_train"]
        rg_seed = cfg.seed + cfg.random_graph_seed_offset

        if sanity_mode == "random_graph_weak":
            und2 = randomize_undirected_edges_weak(und, seed=rg_seed)
            notes = f"weak randomization (seed={rg_seed})"
        elif sanity_mode == "random_graph_strong":
            und2 = randomize_undirected_edges_strong(und, seed=rg_seed)
            notes = f"strong randomization (seed={rg_seed})"
        elif sanity_mode == "random_graph_degree_preserving":
            und2 = degree_preserving_swap_undirected(und, seed=rg_seed, n_swaps_factor=10)
            if und2 is None:
                return {
                    "graph_mode": cfg.graph_mode,
                    "sanity_mode": sanity_mode,
                    "mp_enabled": True,
                    "use_time_feats": use_time_feats,
                    "seed": cfg.seed,
                    "train_r2": float("nan"), "train_rmse": float("nan"),
                    "val_r2": float("nan"), "val_rmse": float("nan"),
                    "test_r2": float("nan"), "test_rmse": float("nan"),
                    "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
                    "n_test_players": int(test_df["player_id"].nunique()),
                    "E_base_und_train": E_base,
                    "E_after_und_train": E_base,
                    "edge_ratio_train": 1.0,
                    "overlap_ratio_test": overlap_ratio,
                    "protocol_note": pack["protocol_note"],
                    "split_note": split_note,
                    "notes": f"skipped (networkx missing or swap failed) seed={rg_seed}",
                    **pack["node_idx_diag"],
                }
            notes = f"degree-preserving (seed={rg_seed}; networkx-version dependent)"
        else:
            raise ValueError(sanity_mode)

        E_after = int(und2.shape[0])
        edge_ratio = E_after / max(1, E_base)
        edge_index_train = make_edge_index_from_undirected(und2)

    # train model (B)
    tr, va, te, freeze_note = train_fullbatch(
        cfg=cfg,
        pack=pack,
        mp_enabled=mp_enabled,
        use_time_feats=use_time_feats,
        edge_index_train=edge_index_train,
        edge_index_test=edge_index_test,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        shuffle_y_train=shuffle_y,
        freeze_test_note_needed=True,
    )
    if freeze_note:
        notes = (notes + " | " + freeze_note).strip(" |")

    return {
        "graph_mode": cfg.graph_mode,
        "sanity_mode": sanity_mode,
        "mp_enabled": mp_enabled,
        "use_time_feats": use_time_feats,
        "seed": cfg.seed,
        "train_r2": tr["r2"], "train_rmse": tr["rmse"],
        "val_r2": va["r2"], "val_rmse": va["rmse"],
        "test_r2": te["r2"], "test_rmse": te["rmse"],
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "n_test_players": int(test_df["player_id"].nunique()),
        "E_base_und_train": E_base,
        "E_after_und_train": E_after,
        "edge_ratio_train": edge_ratio,
        "overlap_ratio_test": overlap_ratio,
        "protocol_note": pack["protocol_note"],
        "split_note": split_note,
        "notes": notes,
        **pack["node_idx_diag"],
    }


# -------------------------
# Run suite (single + multi-seed)
# -------------------------
def run_suite(cfg_base: Config, seeds: Iterable[int], use_time_feats: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []

    for s in seeds:
        cfg = Config(**{**cfg_base.__dict__, "seed": int(s)})
        set_seed(cfg.seed)

        pack = load_and_prepare(cfg)

        # (D) Print y describe once per (graph_mode, time_feats, seed==first seed only)
        if s == list(seeds)[0]:
            print("\n" + "-" * 110)
            print(f"=== (D) y distribution | graph_mode={cfg.graph_mode} | time_feats={use_time_feats} ===")
            print(describe_y(pack["train_df"], "train"))
            print(describe_y(pack["val_df"], "val"))
            print(describe_y(pack["test_df"], "test"))
            print("\n" + "-" * 110)
            print(f"=== (C) node_idx diagnostics | graph_mode={cfg.graph_mode} | time_feats={use_time_feats} ===")
            diag = pack["node_idx_diag"]
            print(
                f"unique_nodes={diag['node_idx_unique_nodes']} "
                f"dup_nodes={diag['node_idx_dup_nodes']} (frac={diag['node_idx_dup_frac']:.4f}) "
                f"max_count={diag['node_idx_max_count']} "
                f"conflict_nodes={diag['node_idx_conflict_nodes']} (frac={diag['node_idx_conflict_frac']:.4f})"
            )

            # (E) overfit test once per setting (use first seed)
            print("\n" + "-" * 110)
            print(f"=== (E) 64-sample overfit test | graph_mode={cfg.graph_mode} | time_feats={use_time_feats} ===")
            over = overfit_64_test(cfg, pack, use_time_feats=use_time_feats)
            print(
                f"overfit_n={int(over['overfit_n'])} "
                f"train_rmse={over['overfit_train_rmse']:.6f} "
                f"train_r2={over['overfit_train_r2']:.6f}"
            )
            print("-" * 110 + "\n")

        for mode in SANITY_MODES:
            raw_rows.append(run_one_mode(cfg, pack, sanity_mode=mode, use_time_feats=use_time_feats))

    raw_df = pd.DataFrame(raw_rows)

    # summary over seeds
    raw_df["sanity_mode"] = pd.Categorical(raw_df["sanity_mode"], categories=SANITY_MODES, ordered=True)
    grp = raw_df.groupby("sanity_mode", as_index=False, observed=False)

    summary = grp.agg(
        train_r2_mean=("train_r2", "mean"),
        train_r2_std=("train_r2", "std"),
        train_rmse_mean=("train_rmse", "mean"),
        train_rmse_std=("train_rmse", "std"),

        val_r2_mean=("val_r2", "mean"),
        val_r2_std=("val_r2", "std"),
        val_rmse_mean=("val_rmse", "mean"),
        val_rmse_std=("val_rmse", "std"),

        test_r2_mean=("test_r2", "mean"),
        test_r2_std=("test_r2", "std"),
        test_rmse_mean=("test_rmse", "mean"),
        test_rmse_std=("test_rmse", "std"),

        E_base_mean=("E_base_und_train", "mean"),
        E_after_mean=("E_after_und_train", "mean"),
        edge_ratio_mean=("edge_ratio_train", "mean"),
        edge_ratio_std=("edge_ratio_train", "std"),

        n_test_mean=("n_test", "mean"),
        n_test_players_mean=("n_test_players", "mean"),
        overlap_ratio_test_mean=("overlap_ratio_test", "mean"),
        overlap_ratio_test_std=("overlap_ratio_test", "std"),
    )

    def pm(mu: float, sd: float) -> str:
        if pd.isna(mu):
            return "nan"
        if pd.isna(sd):
            return f"{mu:.4f}"
        return f"{mu:.4f} ± {sd:.4f}"

    out = pd.DataFrame({
        "sanity_mode": summary["sanity_mode"].astype(str),
        "train_r2": [pm(m, s) for m, s in zip(summary["train_r2_mean"], summary["train_r2_std"])],
        "train_rmse": [pm(m, s) for m, s in zip(summary["train_rmse_mean"], summary["train_rmse_std"])],
        "val_r2": [pm(m, s) for m, s in zip(summary["val_r2_mean"], summary["val_r2_std"])],
        "val_rmse": [pm(m, s) for m, s in zip(summary["val_rmse_mean"], summary["val_rmse_std"])],
        "test_r2": [pm(m, s) for m, s in zip(summary["test_r2_mean"], summary["test_r2_std"])],
        "test_rmse": [pm(m, s) for m, s in zip(summary["test_rmse_mean"], summary["test_rmse_std"])],
        "E_base_und_train(mean)": summary["E_base_mean"].astype(int).astype(str),
        "E_after_und_train(mean)": summary["E_after_mean"].astype(int).astype(str),
        "edge_ratio_train": [pm(m, s) for m, s in zip(summary["edge_ratio_mean"], summary["edge_ratio_std"])],
        "n_test(mean)": summary["n_test_mean"].astype(float).round(1).astype(str),
        "n_test_players(mean)": summary["n_test_players_mean"].astype(float).round(1).astype(str),
        "overlap_ratio_test(orig_only)": [pm(m, s) for m, s in zip(summary["overlap_ratio_test_mean"], summary["overlap_ratio_test_std"])],
    })

    out["sanity_mode"] = pd.Categorical(out["sanity_mode"], categories=SANITY_MODES, ordered=True)
    out = out.sort_values("sanity_mode").reset_index(drop=True)

    return raw_df, out


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    base = Config()

    # you can change seeds here
    SEEDS = [0, 1, 2, 3, 4]

    for gm in ["transductive", "inductive"]:
        for tf in [False, True]:
            base.graph_mode = gm

            print("\n" + "=" * 110)
            print(f"=== V2 BASELINE SANITY + A–E | graph_mode={gm} | time_feats={tf} | multi-seed={SEEDS} ===")
            print("=" * 110)

            raw_df, sum_df = run_suite(base, seeds=SEEDS, use_time_feats=tf)

            # single-seed preview: seed=42 (consistent with your prior runs)
            preview_cfg = Config(**{**base.__dict__, "seed": 42, "graph_mode": gm})
            preview_pack = load_and_prepare(preview_cfg)
            print("\n" + "-" * 110)
            print(f"=== single-seed preview (seed=42) | graph_mode={gm} | time_feats={tf} ===")
            preview_rows = []
            for mode in SANITY_MODES:
                preview_rows.append(run_one_mode(preview_cfg, preview_pack, sanity_mode=mode, use_time_feats=tf))
            prev = pd.DataFrame(preview_rows)
            cols_show = [
                "graph_mode", "sanity_mode", "mp_enabled", "use_time_feats", "seed",
                "train_r2", "train_rmse", "val_r2", "val_rmse", "test_r2", "test_rmse",
                "n_train", "n_val", "n_test", "n_test_players",
                "E_base_und_train", "E_after_und_train", "edge_ratio_train",
                "overlap_ratio_test",
                "node_idx_unique_nodes", "node_idx_dup_nodes", "node_idx_max_count",
                "node_idx_conflict_nodes",
                "notes",
            ]
            print(prev[cols_show].to_string(index=False))

            print("\n" + "-" * 110)
            print(f"=== multi-seed summary | graph_mode={gm} | time_feats={tf} ===")
            print(sum_df.to_string(index=False))
            print("-" * 110)

    print("\n[done] V2 baseline sanity checks with A–E complete.")
