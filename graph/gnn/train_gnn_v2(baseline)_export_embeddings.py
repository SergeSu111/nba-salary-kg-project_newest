from __future__ import annotations

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
# Config
# -------------------------

@dataclass
class Config:
    # Graph mode:
    # - "transductive": training sees full graph incl. 2024 structure (no 2024 labels used)
    # - "inductive": remove 2024 PlayerSeason nodes+incident edges during training encode
    #   NOTE: test evaluation uses full graph for inference (structure visible at test time).
    graph_mode: str = "inductive"  # or "inductive"

    # Paths (repo-root relative)
    edges_path: Path = Path("graph/edges/edges_gnn_v2_core_elementId_full.csv")
    playerseason_map_path: Path = Path("graph/mappings/playerSeason.csv")
    tabular_path: Path = Path("data/processed/training_level1_full.csv")

    # Splits (paper-grade)
    test_season: int = 2024
    val_season: int = 2023
    target_col: str = "log_salary"
    feat_cols: Tuple[str, str] = ("age_now", "years_since_draft")

    # Model
    emb_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Optimization
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    grad_clip: float = 1.0
    early_stop_patience: int = 30

    # Repro
    seed: int = 42

    # Device
    device: str = "cpu"  # set to "cuda" if available

    # Logging
    log_every: int = 10

    # Stability
    emb_max_norm: Optional[float] = 1.0  # None to disable
    use_layernorm: bool = True

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
    cols = [c.lower() for c in df.columns]

    def find_one(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in df.columns:
                return cand
        for cand in candidates:
            if cand in cols:
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


def detect_playerseason_map_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols_lower = [c.lower() for c in df.columns]

    def find_one(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in df.columns:
                return cand
        for cand in candidates:
            if cand in cols_lower:
                return df.columns[cols_lower.index(cand)]
        return None

    player_col = find_one(["player_id", "playerid"])
    season_col = find_one(["season", "year"])
    node_col = find_one(["node_id", "nodeid", "elementid", "neo4j_node_id"])

    if player_col is None or season_col is None or node_col is None:
        raise ValueError(
            "Cannot detect (player_id, season, node_id) in PlayerSeason mapping. "
            f"Columns are: {list(df.columns)}"
        )
    return player_col, season_col, node_col


def build_node_index_from_edges(edges_df: pd.DataFrame, src_col: str, dst_col: str) -> Dict[str, int]:
    src_nodes = edges_df[src_col].astype(str).values
    dst_nodes = edges_df[dst_col].astype(str).values
    nodes = np.unique(np.concatenate([src_nodes, dst_nodes]))
    nodes_sorted = sorted(nodes.tolist())
    return {nid: i for i, nid in enumerate(nodes_sorted)}


def make_edge_index(edges_df: pd.DataFrame, src_col: str, dst_col: str, node2idx: Dict[str, int]) -> torch.Tensor:
    src = edges_df[src_col].astype(str).map(node2idx).to_numpy()
    dst = edges_df[dst_col].astype(str).map(node2idx).to_numpy()

    if np.any(pd.isna(src)) or np.any(pd.isna(dst)):
        raise ValueError("Some edge endpoints were not found in node2idx; node indexing bug.")

    src = src.astype(np.int64)
    dst = dst.astype(np.int64)

    # bidirectional
    src_all = np.concatenate([src, dst])
    dst_all = np.concatenate([dst, src])

    return torch.tensor(np.stack([src_all, dst_all], axis=0), dtype=torch.long)


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

    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(float)

    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -------- FIX #1: correct standardization (NO double-standardize) --------

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
    df = df.copy()
    for c in feat_cols:
        mu, sigma = stats[c]
        df[c] = (df[c] - mu) / sigma
    return df


def make_supervised_samples(
    tabular_df: pd.DataFrame,
    player_map_df: pd.DataFrame,
    player_col: str,
    season_col: str,
    node_col: str,
    node2idx: Dict[str, int],
    test_season: int,
    val_season: int,
    target_col: str,
    feat_cols: Tuple[str, ...],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Return train/val/test dataframes with columns:
      player_id, season, y, node_idx, feat_cols...
    also return stats for time feature standardization.
    """
    pm = player_map_df[[player_col, season_col, node_col]].copy()
    pm[player_col] = pm[player_col].astype(str)
    pm[season_col] = pm[season_col].astype(int)
    pm[node_col] = pm[node_col].astype(str)

    d = tabular_df.merge(
        pm,
        left_on=["player_id", "season"],
        right_on=[player_col, season_col],
        how="inner",
    )

    d["node_idx"] = d[node_col].astype(str).map(node2idx)
    before = len(d)
    d = d.dropna(subset=["node_idx"]).copy()
    d["node_idx"] = d["node_idx"].astype(int)
    after = len(d)

    if after == 0:
        raise ValueError("No supervised samples after mapping (player_id, season) -> node_idx.")
    if after < before:
        print(f"[WARN] Dropped {before - after} rows because node_id not in edge-derived node set.")

    d = d.rename(columns={target_col: "y"})
    d["y"] = d["y"].astype(float)

    train = d[d["season"] < val_season].copy()
    val = d[d["season"] == val_season].copy()
    test = d[d["season"] == test_season].copy()

    if len(val) == 0 or len(test) == 0:
        raise ValueError("Val/Test set empty; check seasons and split config.")

    # Missing fill using TRAIN ONLY (no leakage)
    for c in feat_cols:
        med = float(train[c].median())
        train[c] = train[c].fillna(med)
        val[c] = val[c].fillna(med)
        test[c] = test[c].fillna(med)

    # Standardize using TRAIN RAW stats ONCE (FIXED)
    stats = compute_stats(train, feat_cols)
    train = apply_stats(train, feat_cols, stats)
    val = apply_stats(val, feat_cols, stats)
    test = apply_stats(test, feat_cols, stats)

    print("[time-feats] stats from TRAIN only:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in stats.items()})

    return train, val, test, stats


# -------------------------
# Model
# -------------------------

class GraphSAGERegressor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        feat_dim: int,
        emb_max_norm: Optional[float] = None,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim, max_norm=emb_max_norm)

        convs = []
        in_dim = emb_dim
        for _ in range(num_layers):
            convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(convs)

        self.dropout = dropout
        self.feat_dim = feat_dim
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + feat_dim, hidden_dim),
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
        # IMPORTANT: use embedding forward to trigger max_norm renorm (if enabled)
        idx = torch.arange(self.node_emb.num_embeddings, device=edge_index.device)
        x = self.node_emb(idx)  # [N, emb_dim]

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.ln(x)
        return x

    def predict_from_z(self, z: torch.Tensor, node_idx: torch.Tensor, time_feats: torch.Tensor) -> torch.Tensor:
        h = z[node_idx]
        x = torch.cat([h, time_feats], dim=1)
        y_hat = self.head(x).squeeze(-1)
        return y_hat


# -------------------------
# Eval
# -------------------------

@torch.no_grad()
def evaluate(
    model: GraphSAGERegressor,
    edge_index: torch.Tensor,
    samples: pd.DataFrame,
    feat_cols: Tuple[str, ...],
    device: str,
) -> Dict[str, float]:
    model.eval()
    edge_index = edge_index.to(device)  # device safety
    z = model.encode(edge_index)

    node_idx = torch.tensor(samples["node_idx"].to_numpy(), dtype=torch.long, device=device)
    y_true = torch.tensor(samples["y"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    time_feats = torch.tensor(samples[list(feat_cols)].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

    y_pred = model.predict_from_z(z, node_idx, time_feats)

    mse = F.mse_loss(y_pred, y_true).item()
    rmse = math.sqrt(mse)
    mae = F.l1_loss(y_pred, y_true).item()

    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# -------------------------
# Graph mode helpers
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


def build_training_edge_index(
    cfg: Config,
    edges_df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    node2idx: Dict[str, int],
    test_node_ids: List[str],
) -> torch.Tensor:
    """
    transductive: keep full edges
    inductive: remove edges incident to test-season PlayerSeason nodes (training encode graph)
    """
    if cfg.graph_mode == "transductive":
        return make_edge_index(edges_df, src_col, dst_col, node2idx)

    if cfg.graph_mode != "inductive":
        raise ValueError("cfg.graph_mode must be 'transductive' or 'inductive'")

    test_set = set(test_node_ids)
    src = edges_df[src_col].astype(str)
    dst = edges_df[dst_col].astype(str)
    keep_mask = (~src.isin(test_set)) & (~dst.isin(test_set))
    filtered = edges_df.loc[keep_mask].copy()

    print(f"[graph-mode:inductive] removed test PlayerSeason nodes={len(test_set)}; edges {len(edges_df)} -> {len(filtered)}")

    return make_edge_index(filtered, src_col, dst_col, node2idx)


def freeze_test_playerseason_rows(
    model: GraphSAGERegressor,
    node2idx: Dict[str, int],
    test_node_ids: List[str],
    device: str,
) -> None:
    """
    FIX #2: In inductive mode, freeze gradients for test PlayerSeason rows in embedding matrix.
    Prevents meaningless drift and makes the setting more defensible.
    """
    idx_list = [node2idx[n] for n in test_node_ids if n in node2idx]
    if len(idx_list) == 0:
        print("[inductive-freeze] no test nodes found in node2idx; skip.")
        return

    test_idx = torch.tensor(sorted(set(idx_list)), dtype=torch.long, device=device)

    def hook(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g.index_fill_(0, test_idx, 0.0)
        return g

    model.node_emb.weight.register_hook(hook)
    print(f"[inductive-freeze] zeroed gradients for {test_idx.numel()} test PlayerSeason embeddings.")


# -------------------------
# Train
# -------------------------

def train_loop(cfg: Config) -> None:
    set_seed(cfg.seed)

    # Resolve paths
    cfg.edges_path = (REPO_ROOT / cfg.edges_path).resolve()
    cfg.playerseason_map_path = (REPO_ROOT / cfg.playerseason_map_path).resolve()
    cfg.tabular_path = (REPO_ROOT / cfg.tabular_path).resolve()

    assert_exists(cfg.edges_path)
    assert_exists(cfg.playerseason_map_path)
    assert_exists(cfg.tabular_path)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cfg.device=cuda but CUDA not available. Falling back to cpu.")
        cfg.device = "cpu"

    # Output paths (embed graph_mode into filename)
    emb_dir = (REPO_ROOT / "graph/embeddings").resolve()
    emb_dir.mkdir(parents=True, exist_ok=True)
    out_playerseason_emb_path = (emb_dir / f"gnn_v2_sage_playerseason_{cfg.graph_mode}.csv").resolve()
    out_node_emb_path = (emb_dir / f"gnn_v2_sage_node_{cfg.graph_mode}.csv").resolve()

    print(f"[device] {cfg.device}")
    print(f"[graph_mode] {cfg.graph_mode}  (paper: disclose explicitly)")
    print(f"[paths] edges={cfg.edges_path}")
    print(f"[paths] playerseason_map={cfg.playerseason_map_path}")
    print(f"[paths] tabular={cfg.tabular_path}")

    # Load edges
    edges_df = pd.read_csv(cfg.edges_path)
    src_col, dst_col, rel_col = detect_edge_columns(edges_df)
    print(f"[edges] rows={len(edges_df)} cols={list(edges_df.columns)} detected src={src_col} dst={dst_col} rel={rel_col}")

    # Node index
    node2idx = build_node_index_from_edges(edges_df, src_col, dst_col)
    num_nodes = len(node2idx)
    print(f"[graph] num_nodes={num_nodes}")

    # Load mapping & tabular
    player_map_df = pd.read_csv(cfg.playerseason_map_path)
    player_col, season_col, node_col = detect_playerseason_map_columns(player_map_df)
    tab = load_tabular_labels(cfg.tabular_path, cfg.target_col, cfg.feat_cols)

    # Supervised samples + train-only stats
    train_df, val_df, test_df, _stats = make_supervised_samples(
        tabular_df=tab,
        player_map_df=player_map_df,
        player_col=player_col,
        season_col=season_col,
        node_col=node_col,
        node2idx=node2idx,
        test_season=cfg.test_season,
        val_season=cfg.val_season,
        target_col=cfg.target_col,
        feat_cols=cfg.feat_cols,
    )

    # Coverage reporting
    total_tab = len(tab)
    mapped_all = len(pd.concat([train_df, val_df, test_df], axis=0))
    print(f"[coverage] tabular_total={total_tab} mapped_total={mapped_all} ({mapped_all/total_tab:.3f})")
    print(f"[coverage] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    denom_test = len(tab[tab["season"] == cfg.test_season])
    print(f"[coverage] test_mapped_ratio={len(test_df) / max(1, denom_test):.3f}")

    # Identify test-season PlayerSeason node_ids
    test_node_ids = get_test_playerseason_node_ids(
        player_map_df=player_map_df,
        player_col=player_col,
        season_col=season_col,
        node_col=node_col,
        test_season=cfg.test_season,
    )

    # Build edge_index for training encode (inductive optionally removes test nodes)
    edge_index_train = build_training_edge_index(
        cfg=cfg,
        edges_df=edges_df,
        src_col=src_col,
        dst_col=dst_col,
        node2idx=node2idx,
        test_node_ids=test_node_ids,
    ).to(cfg.device)

    # Full graph edge_index for final export/inference
    edge_index_full = make_edge_index(edges_df, src_col, dst_col, node2idx).to(cfg.device)

    # Model
    model = GraphSAGERegressor(
        num_nodes=num_nodes,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        feat_dim=len(cfg.feat_cols),
        emb_max_norm=cfg.emb_max_norm,
        use_layernorm=cfg.use_layernorm,
    ).to(cfg.device)

    # FIX #2: inductive mode freeze test node rows
    if cfg.graph_mode == "inductive":
        freeze_test_playerseason_rows(model, node2idx, test_node_ids, cfg.device)

    # FIX #2A: do NOT weight_decay node embeddings (prevents drift; especially important for inductive)
    emb_params = list(model.node_emb.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("node_emb.")]
    opt = torch.optim.AdamW(
        [
            {"params": emb_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": cfg.weight_decay},
        ],
        lr=cfg.lr,
    )

    best = {"epoch": -1, "rmse": float("inf"), "state": None, "no_improve": 0}

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        # ONE full-graph encode per epoch (keeps compute graph for backprop)
        z = model.encode(edge_index_train)

        node_idx = torch.tensor(train_df["node_idx"].to_numpy(), dtype=torch.long, device=cfg.device)
        y_true = torch.tensor(train_df["y"].to_numpy(dtype=np.float32), dtype=torch.float32, device=cfg.device)
        time_feats = torch.tensor(train_df[list(cfg.feat_cols)].to_numpy(dtype=np.float32), dtype=torch.float32, device=cfg.device)

        y_pred = model.predict_from_z(z, node_idx, time_feats)
        loss = F.mse_loss(y_pred, y_true)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if epoch % cfg.log_every == 0 or epoch == 1:
            tr = evaluate(model, edge_index_train, train_df, cfg.feat_cols, cfg.device)
            va = evaluate(model, edge_index_train, val_df, cfg.feat_cols, cfg.device)
            te = evaluate(model, edge_index_full, test_df, cfg.feat_cols, cfg.device)

            print(
                f"[epoch {epoch:03d}] "
                f"| train r2={tr['r2']:.4f} rmse={tr['rmse']:.4f} "
                f"| val   r2={va['r2']:.4f} rmse={va['rmse']:.4f} "
                f"| test  r2={te['r2']:.4f} rmse={te['rmse']:.4f}"
            )

            # best checkpoint by VAL (not test)
            if va["rmse"] < best["rmse"]:
                best["epoch"] = epoch
                best["rmse"] = va["rmse"]
                best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best["no_improve"] = 0
            else:
                best["no_improve"] += 1

            if best["no_improve"] >= cfg.early_stop_patience:
                print(f"[early-stop] no val improvement for {cfg.early_stop_patience} eval steps. stopping.")
                break

    # Restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"[best] epoch={best['epoch']} best_val_rmse={best['rmse']:.4f}")

    # Final evaluation
    tr = evaluate(model, edge_index_train, train_df, cfg.feat_cols, cfg.device)
    va = evaluate(model, edge_index_train, val_df, cfg.feat_cols, cfg.device)
    te = evaluate(model, edge_index_full, test_df, cfg.feat_cols, cfg.device)

    print(f"[final] train r2={tr['r2']:.4f} rmse={tr['rmse']:.4f} mae={tr['mae']:.4f}")
    print(f"[final] val   r2={va['r2']:.4f} rmse={va['rmse']:.4f} mae={va['mae']:.4f}")
    print(f"[final] test  r2={te['r2']:.4f} rmse={te['rmse']:.4f} mae={te['mae']:.4f}")

    # Encode on FULL graph for export
    model.eval()
    with torch.no_grad():
        z_full = model.encode(edge_index_full).detach().cpu().numpy()

    # Export PlayerSeason embeddings (aligned to tabular pairs)
    pm = player_map_df[[player_col, season_col, node_col]].copy()
    pm[player_col] = pm[player_col].astype(str)
    pm[season_col] = pm[season_col].astype(int)
    pm[node_col] = pm[node_col].astype(str)
    pm["node_idx"] = pm[node_col].map(node2idx)
    pm = pm.dropna(subset=["node_idx"]).copy()
    pm["node_idx"] = pm["node_idx"].astype(int)

    pairs_in_tab = set(zip(tab["player_id"].astype(str), tab["season"].astype(int)))
    pm_pairs = list(zip(pm[player_col].astype(str), pm[season_col].astype(int)))
    mask = [p in pairs_in_tab for p in pm_pairs]
    pm = pm.loc[mask].copy()

    emb_df = pd.DataFrame(
        {
            "player_id": pm[player_col].astype(str).values,
            "season": pm[season_col].astype(int).values,
            "node_id": pm[node_col].values,
            "node_idx": pm["node_idx"].values,
        }
    )

    D = z_full.shape[1]
    idxs = pm["node_idx"].values
    for j in range(D):
        emb_df[f"e{j}"] = z_full[idxs, j]

    emb_df.to_csv(out_playerseason_emb_path, index=False)
    print(f"[export] playerseason embeddings -> {out_playerseason_emb_path} rows={len(emb_df)} dim={D}")

    if cfg.export_all_nodes:
        inv = {idx: nid for nid, idx in node2idx.items()}
        node_rows = []
        for idx in range(num_nodes):
            row = {"node_id": inv[idx], "node_idx": idx}
            for j in range(D):
                row[f"e{j}"] = float(z_full[idx, j])
            node_rows.append(row)
        node_df = pd.DataFrame(node_rows)
        node_df.to_csv(out_node_emb_path, index=False)
        print(f"[export] node embeddings -> {out_node_emb_path} rows={len(node_df)} dim={D}")


def main():
    cfg = Config()
    train_loop(cfg)


if __name__ == "__main__":
    main()
