#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


# -------------------------
# 0) Config + constants
# -------------------------

@dataclass
class Cfg:
    edges_path: Path = Path("graph/edges/edges_node2vec_L1A_elementId.csv")
    player_map_path: Path = Path("graph/mappings/player_nodeid_map.csv")
    tabular_path: Path = Path("data/processed/training_level1_full.csv")

    test_season: int = 2024
    target_col: str = "log_salary"
    feat_cols: Tuple[str, str] = ("age_now", "years_since_draft")

    emb_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 256
    grad_clip: float = 1.0

    seed: int = 42
    device: str = "cpu"


MODES = [
    "orig",
    "no_graph",
    "random_graph_weak",
    "random_graph_strong",
    "random_graph_degree_preserving",
    "shuffle_labels",
    "cold_start_player",
    "cold_start_2024",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / "graph").exists() and (p / "data").exists():
            if (p / "graph" / "edges").exists() and (p / "data" / "processed").exists():
                return p
    return start.parent


# -------------------------
# 1) load_data()
# -------------------------

def detect_edge_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [c.lower() for c in df.columns]

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        for c in cands:
            if c in cols:
                return df.columns[cols.index(c)]
        return None

    src = pick(["src", "source", "u", "head", "from"])
    dst = pick(["dst", "target", "v", "tail", "to"])
    if src is None or dst is None:
        raise ValueError(f"Cannot detect src/dst in edges columns={list(df.columns)}")
    return src, dst


def detect_player_map_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [c.lower() for c in df.columns]

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        for c in cands:
            if c in cols:
                return df.columns[cols.index(c)]
        return None

    player = pick(["player_id", "playerid"])
    node = pick(["node_id", "nodeid", "elementid", "neo4j_node_id"])
    if player is None or node is None:
        raise ValueError(f"Cannot detect player/node columns in map columns={list(df.columns)}")
    return player, node


def build_node2idx(edges_df: pd.DataFrame, src_col: str, dst_col: str) -> Dict[str, int]:
    nodes = np.unique(np.concatenate([
        edges_df[src_col].astype(str).values,
        edges_df[dst_col].astype(str).values
    ]))
    nodes = sorted(nodes.tolist())
    return {nid: i for i, nid in enumerate(nodes)}


def make_edge_index_from_edges_df(edges_df: pd.DataFrame, src_col: str, dst_col: str, node2idx: Dict[str, int]) -> torch.Tensor:
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


def make_edge_index_bidirectional_from_undirected(undirected_edges: np.ndarray) -> torch.Tensor:
    u = undirected_edges[:, 0].astype(np.int64)
    v = undirected_edges[:, 1].astype(np.int64)
    s = np.concatenate([u, v])
    d = np.concatenate([v, u])
    return torch.tensor(np.stack([s, d], axis=0), dtype=torch.long)


def load_data(cfg: Cfg, repo_root: Path) -> dict:
    cfg.edges_path = (repo_root / cfg.edges_path).resolve()
    cfg.player_map_path = (repo_root / cfg.player_map_path).resolve()
    cfg.tabular_path = (repo_root / cfg.tabular_path).resolve()

    for p in [cfg.edges_path, cfg.player_map_path, cfg.tabular_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    edges = pd.read_csv(cfg.edges_path)
    src_col, dst_col = detect_edge_columns(edges)
    node2idx = build_node2idx(edges, src_col, dst_col)

    edge_index = make_edge_index_from_edges_df(edges, src_col, dst_col, node2idx)
    undirected_edges = edge_index_to_undirected_unique(edge_index)

    pm = pd.read_csv(cfg.player_map_path)
    player_col, node_col = detect_player_map_columns(pm)
    pm[player_col] = pm[player_col].astype(str)
    pm[node_col] = pm[node_col].astype(str)

    tab = pd.read_csv(cfg.tabular_path)
    need = {"player_id", "season", cfg.target_col, *cfg.feat_cols}
    miss = need - set(tab.columns)
    if miss:
        raise ValueError(f"Tabular missing columns: {miss}")

    tab = tab[list(need)].copy()
    tab["player_id"] = tab["player_id"].astype(str)
    tab["season"] = tab["season"].astype(int)
    tab[cfg.target_col] = pd.to_numeric(tab[cfg.target_col], errors="coerce")
    tab = tab.dropna(subset=[cfg.target_col]).copy()

    for c in cfg.feat_cols:
        tab[c] = pd.to_numeric(tab[c], errors="coerce")

    # fill feat NaN with TRAIN median (season < test_season)
    train_mask = tab["season"] < cfg.test_season
    for c in cfg.feat_cols:
        med = tab.loc[train_mask, c].median()
        tab[c] = tab[c].fillna(med)

    # merge to get node_idx
    d = tab.merge(pm[[player_col, node_col]], left_on="player_id", right_on=player_col, how="inner")
    d["node_idx"] = d[node_col].map(lambda x: node2idx.get(str(x), None))
    d = d.dropna(subset=["node_idx"]).copy()
    d["node_idx"] = d["node_idx"].astype(int)
    d = d.rename(columns={cfg.target_col: "y"})

    return {
        "node2idx": node2idx,
        "edge_index": edge_index,
        "undirected_edges": undirected_edges,  # base E (unique undirected)
        "samples": d,
        "n_edges_und_base": int(undirected_edges.shape[0]),
    }


# -------------------------
# 2) model + train/eval
# -------------------------

class GraphSAGERegressor(nn.Module):
    """
    no-graph mode returns hidden_dim via projection so emb_dim != hidden_dim is safe.
    """
    def __init__(
        self,
        num_nodes: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        feat_dim: int,
        use_graph: bool = True,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.use_graph = use_graph
        self.dropout = dropout

        self.no_graph_proj = nn.Linear(emb_dim, hidden_dim, bias=False)

        convs = []
        in_dim = emb_dim
        for _ in range(num_layers):
            convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(convs)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.no_graph_proj.weight)
        for c in self.convs:
            c.reset_parameters()
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight
        if not self.use_graph:
            return self.no_graph_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, edge_index: torch.Tensor, node_idx: torch.Tensor, feats: torch.Tensor):
        z = self.encode(edge_index)
        h = z[node_idx]
        x = torch.cat([h, feats], dim=1)
        y_hat = self.head(x).squeeze(-1)
        return y_hat, z


@torch.no_grad()
def eval_r2_rmse(model: nn.Module, edge_index: torch.Tensor, df: pd.DataFrame,
                feat_cols: Tuple[str, str], device: str, use_time_feats: bool) -> dict:
    model.eval()
    node_idx = torch.tensor(df["node_idx"].to_numpy(), dtype=torch.long, device=device)
    y_true = torch.tensor(df["y"].to_numpy(), dtype=torch.float32, device=device)

    if use_time_feats:
        feats = torch.tensor(df[list(feat_cols)].to_numpy(), dtype=torch.float32, device=device)
    else:
        feats = torch.zeros((len(df), 0), dtype=torch.float32, device=device)

    y_pred, _ = model(edge_index, node_idx, feats)

    mse = F.mse_loss(y_pred, y_true).item()
    rmse = math.sqrt(mse)

    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"rmse": rmse, "r2": r2}


def split_train_valid_test(cfg: Cfg, samples: pd.DataFrame):
    valid_season = cfg.test_season - 1
    train_df = samples[samples["season"] < valid_season].copy()
    valid_df = samples[samples["season"] == valid_season].copy()
    test_df  = samples[samples["season"] == cfg.test_season].copy()
    return train_df, valid_df, test_df


# -------------------------
# 3) graph randomizations (undirected base edges)
# -------------------------

def randomize_undirected_edges_weak(und: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = und[:, 0].copy()
    v = und[:, 1].copy()
    rng.shuffle(v)

    uu = np.minimum(u, v)
    vv = np.maximum(u, v)
    out = np.stack([uu, vv], axis=1)

    # filter self-loops + unique (may reduce edge count; we'll report ratio)
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
    """
    Degree-preserving randomization using NetworkX double_edge_swap on an undirected graph.
    NOTE: best-effort determinism; exact randomness may depend on networkx version.
    """
    try:
        import networkx as nx
    except Exception:
        return None

    if und.shape[0] < 10:
        return None

    G = nx.Graph()
    G.add_edges_from([(int(u), int(v)) for u, v in und])

    nswap = n_swaps_factor * G.number_of_edges()
    max_tries = nswap * 20

    # best-effort determinism: networkx often uses python random internally;
    # behavior can vary across versions.
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
# 4) one experiment
# -------------------------

def train_eval_one(cfg: Cfg, pack: dict, mode: str, use_time_feats: bool) -> dict:
    set_seed(cfg.seed)

    samples = pack["samples"].copy()
    und_base = pack["undirected_edges"]
    E_base = pack["n_edges_und_base"]

    train_df, valid_df, test_df = split_train_valid_test(cfg, samples)

    # overlap on orig definition: test_2024 players seen in train+valid
    overlap_ratio_test = float("nan")
    if mode == "orig":
        seen_players = set(pd.concat([train_df, valid_df])["player_id"].unique())
        test_players = set(test_df["player_id"].unique())
        overlap_ratio_test = len(seen_players & test_players) / max(1, len(test_players))

    # edge info to report
    n_edges_und = E_base
    edge_ratio = 1.0
    notes = ""
    use_graph = True

    # build edge_index
    if mode in ["orig", "no_graph", "shuffle_labels", "cold_start_player", "cold_start_2024"]:
        edge_index = pack["edge_index"].clone()
    else:
        if mode == "random_graph_weak":
            und2 = randomize_undirected_edges_weak(und_base, seed=cfg.seed)
            n_edges_und = int(und2.shape[0])
            edge_ratio = n_edges_und / max(1, E_base)
            edge_index = make_edge_index_bidirectional_from_undirected(und2)
            notes = "weak stress test (not degree-preserving); undirected unique then expand"

        elif mode == "random_graph_strong":
            und2 = randomize_undirected_edges_strong(und_base, seed=cfg.seed)
            n_edges_und = int(und2.shape[0])
            edge_ratio = n_edges_und / max(1, E_base)
            edge_index = make_edge_index_bidirectional_from_undirected(und2)
            notes = "strong stress test (not degree-preserving); undirected unique then expand"

        elif mode == "random_graph_degree_preserving":
            und2 = degree_preserving_swap_undirected(und_base, seed=cfg.seed, n_swaps_factor=10)
            if und2 is None:
                return {
                    "mode": mode, "seed": cfg.seed, "use_time_feats": use_time_feats,
                    "r2": float("nan"), "rmse": float("nan"),
                    "n_train": len(train_df), "n_valid": len(valid_df), "n_test": len(test_df),
                    "n_train_players": train_df["player_id"].nunique() if len(train_df) else 0,
                    "n_valid_players": valid_df["player_id"].nunique() if len(valid_df) else 0,
                    "n_test_players": test_df["player_id"].nunique() if len(test_df) else 0,
                    "n_edges_und": E_base, "edge_ratio": 1.0,
                    "overlap_ratio_test": overlap_ratio_test,
                    "notes": "skipped (networkx missing or swap failed)",
                }
            n_edges_und = int(und2.shape[0])
            edge_ratio = n_edges_und / max(1, E_base)
            edge_index = make_edge_index_bidirectional_from_undirected(und2)
            notes = "degree-preserving (double_edge_swap); randomness depends on networkx version"
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if mode == "shuffle_labels":
        y = train_df["y"].to_numpy().copy()
        np.random.shuffle(y)
        train_df["y"] = y
        notes = "shuffle y in train"

    if mode == "cold_start_player":
        base = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
        all_players = base["player_id"].unique()
        rng = np.random.default_rng(cfg.seed)
        holdout_players = set(rng.choice(all_players, size=max(1, int(0.2 * len(all_players))), replace=False))

        cold_train = base[~base["player_id"].isin(holdout_players)].copy()
        cold_test  = base[ base["player_id"].isin(holdout_players)].copy()

        # ensure we still have train/valid split
        if cold_train["season"].nunique() >= 2:
            v_season = cold_train["season"].max()
            valid_df = cold_train[cold_train["season"] == v_season].copy()
            train_df = cold_train[cold_train["season"] <  v_season].copy()
        else:
            idx = np.arange(len(cold_train))
            rng.shuffle(idx)
            cut = int(0.9 * len(idx))
            train_df = cold_train.iloc[idx[:cut]].copy()
            valid_df = cold_train.iloc[idx[cut:]].copy()

        test_df = cold_test
        notes = "holdout players within train/valid seasons"

    if mode == "cold_start_2024":
        seen_players = set(pd.concat([train_df, valid_df])["player_id"].unique())
        orig_test_players = test_df["player_id"].nunique()
        test_df = test_df[~test_df["player_id"].isin(seen_players)].copy()
        new_test_players = test_df["player_id"].nunique()
        ratio = new_test_players / max(1, orig_test_players)
        notes = f"2024 unseen players only (player_ratio={ratio:.3f})"

    if mode == "no_graph":
        use_graph = False
        notes = "no message passing (id-emb + proj)"

    # emptiness check
    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        return {
            "mode": mode, "seed": cfg.seed, "use_time_feats": use_time_feats,
            "r2": float("nan"), "rmse": float("nan"),
            "n_train": len(train_df), "n_valid": len(valid_df), "n_test": len(test_df),
            "n_train_players": train_df["player_id"].nunique() if len(train_df) else 0,
            "n_valid_players": valid_df["player_id"].nunique() if len(valid_df) else 0,
            "n_test_players": test_df["player_id"].nunique() if len(test_df) else 0,
            "n_edges_und": n_edges_und, "edge_ratio": edge_ratio,
            "overlap_ratio_test": overlap_ratio_test,
            "notes": notes + " | EMPTY SPLIT",
        }

    num_nodes = len(pack["node2idx"])
    feat_dim = len(cfg.feat_cols) if use_time_feats else 0

    model = GraphSAGERegressor(
        num_nodes=num_nodes,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        feat_dim=feat_dim,
        use_graph=use_graph,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    edge_index = edge_index.to(cfg.device)

    idx = np.arange(len(train_df))
    best_rmse = float("inf")
    best_state = None

    for _epoch in range(1, cfg.epochs + 1):
        model.train()
        np.random.shuffle(idx)

        for start in range(0, len(idx), cfg.batch_size):
            batch = train_df.iloc[idx[start:start + cfg.batch_size]]
            node_idx = torch.tensor(batch["node_idx"].to_numpy(), dtype=torch.long, device=cfg.device)
            y_true = torch.tensor(batch["y"].to_numpy(), dtype=torch.float32, device=cfg.device)

            if use_time_feats:
                feats = torch.tensor(batch[list(cfg.feat_cols)].to_numpy(), dtype=torch.float32, device=cfg.device)
            else:
                feats = torch.zeros((len(batch), 0), dtype=torch.float32, device=cfg.device)

            y_pred, _ = model(edge_index, node_idx, feats)
            loss = F.mse_loss(y_pred, y_true)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        cur = eval_r2_rmse(model, edge_index, valid_df, cfg.feat_cols, cfg.device, use_time_feats)
        if cur["rmse"] < best_rmse:
            best_rmse = cur["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    res = eval_r2_rmse(model, edge_index, test_df, cfg.feat_cols, cfg.device, use_time_feats)

    return {
        "mode": mode,
        "seed": cfg.seed,
        "use_time_feats": use_time_feats,
        "r2": res["r2"],
        "rmse": res["rmse"],
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
        "n_train_players": train_df["player_id"].nunique(),
        "n_valid_players": valid_df["player_id"].nunique(),
        "n_test_players": test_df["player_id"].nunique(),
        "n_edges_und": n_edges_und,
        "edge_ratio": edge_ratio,
        "overlap_ratio_test": overlap_ratio_test,
        "notes": notes,
    }


# -------------------------
# 5) run checks
# -------------------------

def run_checks_single_seed(repo_root: Path, use_time_feats: bool, seed: int = 42) -> pd.DataFrame:
    cfg = Cfg(seed=seed)
    pack = load_data(cfg, repo_root)

    rows = []
    for m in MODES:
        rows.append(train_eval_one(cfg, pack, m, use_time_feats=use_time_feats))

    df = pd.DataFrame(rows)
    df["mode"] = pd.Categorical(df["mode"], categories=MODES, ordered=True)
    df = df.sort_values("mode").reset_index(drop=True)
    return df


def _fmt_pm(mu: float, sd: float) -> str:
    if pd.isna(mu):
        return "nan"
    if pd.isna(sd):
        return f"{mu:.4f}"
    return f"{mu:.4f} ± {sd:.4f}"


def _fmt_minmax(a: float, b: float) -> str:
    if pd.isna(a) or pd.isna(b):
        return "nan"
    return f"[{a:.3f}, {b:.3f}]"


def run_checks_multi_seed(repo_root: Path, use_time_feats: bool, seeds: Iterable[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = []
    for s in seeds:
        raw.append(run_checks_single_seed(repo_root, use_time_feats=use_time_feats, seed=int(s)))
    raw_df = pd.concat(raw, ignore_index=True)

    # Ensure mode ordering without relying on .cat accessor
    raw_df["mode"] = pd.Categorical(raw_df["mode"], categories=MODES, ordered=True)

    grp = raw_df.groupby("mode", as_index=False)

    summary = grp.agg(
        # metrics
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),

        # split sizes
        n_train_mean=("n_train", "mean"),
        n_train_std=("n_train", "std"),
        n_train_min=("n_train", "min"),
        n_train_max=("n_train", "max"),

        n_valid_mean=("n_valid", "mean"),
        n_valid_std=("n_valid", "std"),
        n_valid_min=("n_valid", "min"),
        n_valid_max=("n_valid", "max"),

        n_test_mean=("n_test", "mean"),
        n_test_std=("n_test", "std"),
        n_test_min=("n_test", "min"),
        n_test_max=("n_test", "max"),

        n_train_players_mean=("n_train_players", "mean"),
        n_train_players_std=("n_train_players", "std"),
        n_train_players_min=("n_train_players", "min"),
        n_train_players_max=("n_train_players", "max"),

        n_valid_players_mean=("n_valid_players", "mean"),
        n_valid_players_std=("n_valid_players", "std"),
        n_valid_players_min=("n_valid_players", "min"),
        n_valid_players_max=("n_valid_players", "max"),

        n_test_players_mean=("n_test_players", "mean"),
        n_test_players_std=("n_test_players", "std"),
        n_test_players_min=("n_test_players", "min"),
        n_test_players_max=("n_test_players", "max"),

        # graph size / density proxy
        n_edges_und_mean=("n_edges_und", "mean"),
        n_edges_und_std=("n_edges_und", "std"),
        edge_ratio_mean=("edge_ratio", "mean"),
        edge_ratio_std=("edge_ratio", "std"),
        edge_ratio_min=("edge_ratio", "min"),
        edge_ratio_max=("edge_ratio", "max"),

        # overlap (only meaningful for orig; others are NaN)
        overlap_ratio_test_mean=("overlap_ratio_test", "mean"),
        overlap_ratio_test_std=("overlap_ratio_test", "std"),
    )

    out = pd.DataFrame({
        "mode": summary["mode"].astype(str),

        "r2": [_fmt_pm(m, s) for m, s in zip(summary["r2_mean"], summary["r2_std"])],
        "rmse": [_fmt_pm(m, s) for m, s in zip(summary["rmse_mean"], summary["rmse_std"])],

        "n_test": [
            f"{m:.1f}±{s:.1f} (min={mn}, max={mx})"
            for m, s, mn, mx in zip(summary["n_test_mean"], summary["n_test_std"], summary["n_test_min"], summary["n_test_max"])
        ],
        "n_test_players": [
            f"{m:.1f}±{s:.1f} (min={mn}, max={mx})"
            for m, s, mn, mx in zip(summary["n_test_players_mean"], summary["n_test_players_std"], summary["n_test_players_min"], summary["n_test_players_max"])
        ],

        "edge_ratio": [
            f"{m:.3f}±{s:.3f} (min={mn:.3f}, max={mx:.3f})"
            for m, s, mn, mx in zip(summary["edge_ratio_mean"], summary["edge_ratio_std"], summary["edge_ratio_min"], summary["edge_ratio_max"])
        ],

        "overlap_ratio_test": [
            _fmt_pm(m, s) for m, s in zip(summary["overlap_ratio_test_mean"], summary["overlap_ratio_test_std"])
        ],
    })

    out["mode"] = pd.Categorical(out["mode"], categories=MODES, ordered=True)
    out = out.sort_values("mode").reset_index(drop=True)

    return raw_df, out


if __name__ == "__main__":
    repo_root = find_repo_root(Path(__file__).resolve())
    print(f"[repo_root] {repo_root}")

    SEEDS = [0, 1, 2, 3, 4]

    print("\n=== V0 (no time feats) | single-seed preview (seed=42) ===")
    print(run_checks_single_seed(repo_root, use_time_feats=False, seed=42).to_string(index=False))

    print("\n=== V0 (no time feats) | multi-seed summary ===")
    _, v0_sum = run_checks_multi_seed(repo_root, use_time_feats=False, seeds=SEEDS)
    print(v0_sum.to_string(index=False))

    print("\n=== V1 (with time feats) | single-seed preview (seed=42) ===")
    print(run_checks_single_seed(repo_root, use_time_feats=True, seed=42).to_string(index=False))

    print("\n=== V1 (with time feats) | multi-seed summary ===")
    _, v1_sum = run_checks_multi_seed(repo_root, use_time_feats=True, seeds=SEEDS)
    print(v1_sum.to_string(index=False))
