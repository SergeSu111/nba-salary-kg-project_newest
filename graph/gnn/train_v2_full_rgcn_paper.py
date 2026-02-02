from __future__ import annotations

import json
import math
import time
import copy
import random
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv

# -------------------------
# Repo Root
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]


# -------------------------
# Utils: Logging / IO
# -------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def safe_json_dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def torch_load_safely(pt_path: Path):
    """
    Avoid torch FutureWarning if possible. Falls back for older torch versions.
    """
    try:
        return torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(pt_path, map_location="cpu")


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    # --- Experiment Control ---
    exp_name: str = "v2_full_mg_rgcn_paper"

    # Input Paths
    edge_index_pt_path: Path = Path("graph/edges/edge_index_v2_full_mg.pt")

    # Critical Mappings
    master_mapping_path: Path = Path("graph/mappings/master_node_id_to_idx.csv")
    bridge_path: Path = Path("graph/mappings/playerSeason.csv")

    # Features
    features_csv_path: Path = Path("data/processed/training_level1_full.csv")

    # Splits
    test_season: int = 2024
    val_season: int = 2023
    target_col: str = "log_salary"

    # Time features (always used in head)
    time_feat_cols: Tuple[str, str] = ("age_now", "years_since_draft")

    # GNN features (masked + zeroed for Val/Test)
    gnn_feat_cols: Tuple[str, ...] = (
        "pts_per_gp",
        "reb_per_gp",
        "ast_per_gp",
        "stl_per_gp",
        "blk_per_gp",
        "ts%_calc",
        "gp",
    )

    # Model Params
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # RGCN Specific
    num_bases: Optional[int] = 4

    # Optimization
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    early_stop_patience: int = 30
    log_every: int = 10

    seed: int = 42
    device: str = "cpu"  # or: "cuda" if torch.cuda.is_available() else "cpu"

    # Safety
    warn_memory_threshold: int = 200000

    # Output
    run_root: Path = Path("runs")  # runs/<exp_name>/<timestamp>/...

    # -------------------------
    # Ablation Control
    # -------------------------
    # 你要的 no-graph ablation：同一次运行跑两套（graph / nograph）
    run_modes: Tuple[str, ...] = ("graph", "nograph")
    # 如果你只想跑一种，把上面改成 ("graph",) 或 ("nograph",)


# -------------------------
# Dataset (RGCN ready)
# -------------------------
class V2FullDataset:
    """
    Builds:
      - x: [N, F] float
      - feature_mask: [N] bool (True = use projected features; False = use embedding)
      - full_edge_index: [2, E]
      - full_edge_type: [E]
      - labeled_samples: DataFrame with node_idx, season, y, ...
      - rel_to_idx: Dict[str, int]  (forward)
      - num_relations: int (forward + reverse)
    """

    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

        self._load_master_mapping()
        self._load_edges_rgcn()
        self._load_features_and_build_x()

    def _load_master_mapping(self) -> None:
        map_path = (REPO_ROOT / self.cfg.master_mapping_path).resolve()
        map_df = pd.read_csv(map_path)
        map_df.columns = [c.lower().strip() for c in map_df.columns]
        if "node_id" not in map_df.columns or "idx" not in map_df.columns:
            raise ValueError("Master mapping must have columns: node_id, idx")

        self.node_to_idx = dict(zip(map_df["node_id"].astype(str), map_df["idx"].astype(int)))
        self.num_nodes = len(self.node_to_idx)
        self.logger.info(f"[Mapping] Loaded master mapping: num_nodes={self.num_nodes}")

    def _load_edges_rgcn(self) -> None:
        pt_path = (REPO_ROOT / self.cfg.edge_index_pt_path).resolve()
        self.logger.info(f"[Graph] Loading edges from: {pt_path}")
        pt_data = torch_load_safely(pt_path)

        edge_index_dict: Dict[str, torch.Tensor] = pt_data["edge_index_dict"]
        self.edge_index_dict = edge_index_dict

        # Deterministic relation id assignment
        rel_names = sorted(edge_index_dict.keys())
        self.rel_to_idx = {name: i for i, name in enumerate(rel_names)}
        self.num_base_relations = len(rel_names)

        self.logger.info(f"[Graph] Base relations={self.num_base_relations} (sorted).")
        self.logger.info(f"[Graph] rel_to_idx keys: {list(self.rel_to_idx.keys())}")

        edge_lists: List[torch.Tensor] = []
        type_lists: List[torch.Tensor] = []
        edge_stats: Dict[str, int] = {}

        # Forward edges (skip empty)
        for rel_name in rel_names:
            ei = edge_index_dict.get(rel_name, None)
            if ei is None:
                edge_stats[rel_name] = 0
                continue
            if ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError(f"edge_index for {rel_name} must be [2, E], got {tuple(ei.shape)}")

            count = int(ei.size(1))
            edge_stats[rel_name] = count
            if count == 0:
                continue

            rel_idx = self.rel_to_idx[rel_name]
            edge_lists.append(ei)
            type_lists.append(torch.full((count,), rel_idx, dtype=torch.long))

        raw_forward_edges = int(sum(t.numel() for t in type_lists))
        self.logger.info(f"[Graph] Forward edges total: {raw_forward_edges}")
        self.logger.info(f"[Graph] Relation edge counts (forward): {edge_stats}")

        # Reverse edges with distinct type ids
        rev_edge_lists: List[torch.Tensor] = []
        rev_type_lists: List[torch.Tensor] = []
        for i, forward_ei in enumerate(edge_lists):
            row, col = forward_ei
            rev_ei = torch.stack([col, row], dim=0)
            orig_type = int(type_lists[i][0].item())
            rev_type = orig_type + self.num_base_relations
            rev_edge_lists.append(rev_ei)
            rev_type_lists.append(torch.full((rev_ei.size(1),), rev_type, dtype=torch.long))

        self.full_edge_index = (
            torch.cat(edge_lists + rev_edge_lists, dim=1) if edge_lists else torch.empty((2, 0), dtype=torch.long)
        )
        self.full_edge_type = (
            torch.cat(type_lists + rev_type_lists, dim=0) if type_lists else torch.empty((0,), dtype=torch.long)
        )
        self.num_relations = self.num_base_relations * 2

        E = int(self.full_edge_index.size(1))
        self.logger.info(f"[Graph] Final edges (forward+reverse): E={E}, num_relations={self.num_relations}")

        # Bounds check
        if E > 0:
            mx = int(self.full_edge_index.max())
            mn = int(self.full_edge_index.min())
            if mn < 0 or mx >= self.num_nodes:
                raise ValueError(f"Edge index out of bounds! min={mn}, max={mx}, num_nodes={self.num_nodes}")

        if E > self.cfg.warn_memory_threshold:
            self.logger.warning(
                f"[WARN] Edge count {E} > threshold {self.cfg.warn_memory_threshold}. Monitor GPU memory."
            )

        self.edge_stats = edge_stats

    def _load_features_and_build_x(self) -> None:
        cfg = self.cfg

        feat_path = (REPO_ROOT / cfg.features_csv_path).resolve()
        bridge_path = (REPO_ROOT / cfg.bridge_path).resolve()

        if not bridge_path.exists():
            raise FileNotFoundError(f"Bridge file not found: {bridge_path}")

        feat_df = pd.read_csv(feat_path)
        bridge_df = pd.read_csv(bridge_path)

        feat_df.columns = [c.lower().strip() for c in feat_df.columns]
        bridge_df.columns = [c.lower().strip() for c in bridge_df.columns]

        required_cols = ["player_id", "season", cfg.target_col] + list(cfg.time_feat_cols) + list(cfg.gnn_feat_cols)
        for c in required_cols:
            if c not in feat_df.columns:
                raise ValueError(f"Feature CSV missing required column: {c}")

        bridge_node_col = next((c for c in bridge_df.columns if c in ["node_id", "id", "element_id"]), None)
        if not bridge_node_col:
            raise ValueError("Bridge CSV missing node id column (node_id/id/element_id).")

        # Normalize join keys
        feat_df["player_id"] = feat_df["player_id"].astype(str)
        feat_df["season"] = feat_df["season"].astype(int)
        bridge_df["player_id"] = bridge_df["player_id"].astype(str)
        bridge_df["season"] = bridge_df["season"].astype(int)

        merged = feat_df.merge(
            bridge_df[["player_id", "season", bridge_node_col]],
            on=["player_id", "season"],
            how="inner",
        )
        merged["node_idx"] = merged[bridge_node_col].astype(str).map(self.node_to_idx)

        valid_rows = merged.dropna(subset=["node_idx"]).copy()
        valid_rows["node_idx"] = valid_rows["node_idx"].astype(int)

        # Dedup same node_idx
        before = len(valid_rows)
        valid_rows = valid_rows.drop_duplicates(subset=["node_idx"], keep="first")
        dropped = before - len(valid_rows)
        if dropped > 0:
            self.logger.warning(f"[Features] Dropped {dropped} duplicates by node_idx.")

        self.logger.info(f"[Features] Linked rows to graph nodes: {len(valid_rows)}")

        # Train-only normalization for GNN features
        self.num_features = len(cfg.gnn_feat_cols)
        train_mask = valid_rows["season"] < cfg.val_season
        train_subset = valid_rows[train_mask]
        if len(train_subset) == 0:
            raise ValueError("No training samples found for feature normalization. Check val_season.")

        stats = {}
        for col in cfg.gnn_feat_cols:
            vals = train_subset[col].astype(float).values
            mu = float(np.nanmean(vals))
            sigma = float(np.nanstd(vals))
            if sigma == 0 or np.isnan(sigma):
                sigma = 1.0
            stats[col] = (mu, sigma)

        self.logger.info(f"[Features] Computed GNN feature stats on train rows: {len(train_subset)}")

        # Build X / mask
        self.x = torch.zeros((self.num_nodes, self.num_features), dtype=torch.float)
        self.feature_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        node_indices = torch.tensor(valid_rows["node_idx"].values, dtype=torch.long)
        temp_feats = np.zeros((len(valid_rows), self.num_features), dtype=np.float32)

        for i, col in enumerate(cfg.gnn_feat_cols):
            mu, sigma = stats[col]
            raw = valid_rows[col].astype(float).values
            raw = np.nan_to_num(raw, nan=mu)
            temp_feats[:, i] = (raw - mu) / sigma

        self.x.index_copy_(0, node_indices, torch.tensor(temp_feats, dtype=torch.float))
        self.feature_mask.index_fill_(0, node_indices, True)

        # Val/Test leakage prevention: mask off + zero out features
        val_test_mask = valid_rows["season"] >= cfg.val_season
        val_test_node_idxs = torch.tensor(valid_rows.loc[val_test_mask, "node_idx"].values, dtype=torch.long)

        if len(val_test_node_idxs) > 0:
            self.feature_mask.index_fill_(0, val_test_node_idxs, False)
            self.x[val_test_node_idxs] = 0.0
            self.logger.info(f"[SECURITY] Zeroed GNN features for Val/Test nodes: {len(val_test_node_idxs)}")

        self.labeled_samples = valid_rows

    def get_data(self):
        return (
            self.x,
            self.full_edge_index,
            self.full_edge_type,
            self.feature_mask,
            self.labeled_samples,
            self.num_relations,
            self.rel_to_idx,
            self.edge_stats,
        )


# -------------------------
# Model
# -------------------------
class HybridRGCNRegressor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_bases: Optional[int],
        use_graph: bool = True,   # <-- 关键：控制是否启用消息传递
    ):
        super().__init__()
        self.use_graph = use_graph

        self.feat_encoder = nn.Linear(in_channels, hidden_dim)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)

        rgcn_kwargs = {"num_relations": num_relations}
        if num_bases is not None and num_bases > 0:
            rgcn_kwargs["num_bases"] = num_bases

        self.convs = nn.ModuleList()
        if num_layers > 0:
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, **rgcn_kwargs))
            for _ in range(num_layers - 1):
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, **rgcn_kwargs))

        self.dropout = dropout
        self.ln = nn.LayerNorm(hidden_dim)

        # Head uses time feats (2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, edge_index, edge_type, feature_mask):
        # 1) 输入表征：feat_encoder(x) 或 node_emb
        h_feat = self.feat_encoder(x)
        h_emb = self.node_emb(torch.arange(x.size(0), device=x.device))

        mask = feature_mask.unsqueeze(-1).expand_as(h_feat)
        h = torch.where(mask, h_feat, h_emb)

        # 2) 图消息传递（ablation 的关键点）
        if self.use_graph:
            for conv in self.convs:
                h = conv(h, edge_index, edge_type)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        # else: 完全跳过 conv，直接用 h

        return self.ln(h)

    def predict(self, z, node_idx, time_feats):
        h = z[node_idx]
        combined = torch.cat([h, time_feats], dim=1)
        return self.head(combined).squeeze(-1)


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, x, edge_index, edge_type, mask, df: pd.DataFrame, device: str):
    model.eval()
    z = model.encode(x, edge_index, edge_type, mask)

    node_idx = torch.tensor(df["node_idx"].values, dtype=torch.long, device=device)
    y_true = torch.tensor(df["y"].values, dtype=torch.float, device=device)
    t_feats = torch.tensor(df[["age_std", "ysd_std"]].values, dtype=torch.float, device=device)

    y_pred = model.predict(z, node_idx, t_feats)

    mse = F.mse_loss(y_pred, y_true).item()
    rmse = math.sqrt(mse)

    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    ssr = float(np.sum((yt - yp) ** 2))
    r2 = 1.0 - ssr / sst if sst > 1e-12 else 0.0

    return {"rmse": rmse, "r2": r2}


# -------------------------
# Train one mode
# -------------------------
def train_one_mode(
    cfg: Config,
    mode: str,
    dataset: V2FullDataset,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    feat_mask: torch.Tensor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    logger: logging.Logger,
    run_dir: Path,
    ckpt_dir: Path,
    baseline_out: dict,
    sanity_out: dict,
) -> dict:
    """
    mode in {"graph","nograph"}
    """
    assert mode in {"graph", "nograph"}
    use_graph = (mode == "graph")

    # 为了公平对比：每个 mode 都重新 set_seed，保证初始化一致（同 seed）
    set_seed(cfg.seed)

    model = HybridRGCNRegressor(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        in_channels=dataset.num_features,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        num_bases=cfg.num_bases,
        use_graph=use_graph,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    metrics_path = run_dir / f"metrics_{mode}.csv"
    metrics_path.write_text("epoch,train_loss,val_rmse,val_r2\n", encoding="utf-8")

    best_rmse = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0

    logger.info(f"[Mode={mode}] Training start. use_graph={use_graph}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(x, edge_index, edge_type, feat_mask)

        batch_idx = torch.tensor(train_df["node_idx"].values, dtype=torch.long, device=cfg.device)
        batch_y = torch.tensor(train_df["y"].values, dtype=torch.float, device=cfg.device)
        batch_t = torch.tensor(train_df[["age_std", "ysd_std"]].values, dtype=torch.float, device=cfg.device)

        preds = model.predict(z, batch_idx, batch_t)
        loss = F.mse_loss(preds, batch_y)
        loss.backward()
        optimizer.step()

        if epoch % cfg.log_every == 0:
            val_metrics = evaluate(model, x, edge_index, edge_type, feat_mask, val_df, cfg.device)

            logger.info(
                f"[Mode={mode}] Epoch {epoch:03d} | train_loss={loss.item():.6f} | "
                f"val_rmse={val_metrics['rmse']:.6f} | val_r2={val_metrics['r2']:.6f}"
            )

            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{loss.item():.8f},{val_metrics['rmse']:.8f},{val_metrics['r2']:.8f}\n")

            if val_metrics["rmse"] < best_rmse:
                best_rmse = val_metrics["rmse"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience = 0

                ckpt = {
                    "mode": mode,
                    "use_graph": use_graph,
                    "epoch": epoch,
                    "best_val_rmse": best_rmse,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "baselines": baseline_out,
                    "sanity": sanity_out,
                }
                torch.save(ckpt, ckpt_dir / f"best_{mode}.pt")
            else:
                patience += 1

            if patience >= cfg.early_stop_patience:
                logger.info(f"[Mode={mode}] Early stopping at epoch={epoch}, best_epoch={best_epoch}, best_val_rmse={best_rmse:.6f}")
                break

    if best_state is None:
        logger.warning(f"[Mode={mode}] No best_state captured (unexpected). Using final model weights.")
    else:
        model.load_state_dict(best_state)
        logger.info(f"[Mode={mode}] [Best] Loaded best state: epoch={best_epoch}, val_rmse={best_rmse:.6f}")

    test_metrics = evaluate(model, x, edge_index, edge_type, feat_mask, test_df, cfg.device)
    logger.info(f"[Mode={mode}] [Test] rmse={test_metrics['rmse']:.6f} | r2={test_metrics['r2']:.6f}")

    # export embeddings for this mode
    art_dir = run_dir / "artifacts"
    with torch.no_grad():
        model.eval()
        z = model.encode(x, edge_index, edge_type, feat_mask).detach().cpu()
        torch.save(
            {
                "mode": mode,
                "use_graph": use_graph,
                "node_embeddings": z,
                "hidden_dim": cfg.hidden_dim,
                "num_nodes": dataset.num_nodes,
                "best_epoch": best_epoch,
                "best_val_rmse": best_rmse,
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            },
            art_dir / f"node_embeddings_{mode}.pt",
        )

    return {
        "mode": mode,
        "use_graph": use_graph,
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_rmse),
        "test_rmse": float(test_metrics["rmse"]),
        "test_r2": float(test_metrics["r2"]),
        "metrics_csv": str(metrics_path),
        "best_ckpt": str(ckpt_dir / f"best_{mode}.pt"),
        "embeddings_pt": str(art_dir / f"node_embeddings_{mode}.pt"),
    }


# -------------------------
# Main
# -------------------------
def main():
    cfg = Config()
    set_seed(cfg.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(REPO_ROOT / cfg.run_root / cfg.exp_name / timestamp)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    art_dir = ensure_dir(run_dir / "artifacts")

    logger = setup_logger(run_dir / "train.log")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Run modes: {cfg.run_modes}")

    # Save config
    cfg_dict = asdict(cfg)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
        if isinstance(v, tuple):
            cfg_dict[k] = list(v)
    safe_json_dump(run_dir / "config.json", cfg_dict)

    # Dataset
    dataset = V2FullDataset(cfg, logger)
    x, edge_index, edge_type, feat_mask, samples, num_rels, rel_to_idx, edge_stats = dataset.get_data()

    # Persist rel_to_idx + edge_stats
    safe_json_dump(art_dir / "rel_to_idx.json", rel_to_idx)
    safe_json_dump(art_dir / "edge_stats.json", edge_stats)

    # Move tensors to device
    x = x.to(cfg.device)
    edge_index = edge_index.to(cfg.device)
    edge_type = edge_type.to(cfg.device)
    feat_mask = feat_mask.to(cfg.device)

    # Prepare supervised samples
    samples = samples.dropna(subset=[cfg.target_col]).copy()
    samples["y"] = samples[cfg.target_col].astype(float)

    # Time feature standardization (train-only)
    train_subset = samples[samples["season"] < cfg.val_season]
    if len(train_subset) == 0:
        raise ValueError("No train samples after dropping NaNs. Check seasons/target.")

    for col, new_col in zip(cfg.time_feat_cols, ["age_std", "ysd_std"]):
        mu = float(train_subset[col].mean())
        sigma = float(train_subset[col].std())
        if sigma == 0 or np.isnan(sigma):
            sigma = 1.0
        samples[new_col] = ((samples[col] - mu) / sigma).fillna(0.0)

    # Splits
    train_df = samples[samples["season"] < cfg.val_season].copy()
    val_df = samples[samples["season"] == cfg.val_season].copy()
    test_df = samples[samples["season"] == cfg.test_season].copy()

    logger.info(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One of splits is empty. Check val_season/test_season.")

    safe_json_dump(
        run_dir / "splits.json",
        {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "val_season": cfg.val_season,
            "test_season": cfg.test_season,
        },
    )

    # -------------------------
    # Baselines
    # -------------------------
    baseline_out = {}
    sanity_out = {}

    ytr = train_df["y"].astype(float).values
    yva = val_df["y"].astype(float).values
    yte = test_df["y"].astype(float).values

    mu_y = float(np.mean(ytr))
    med_y = float(np.median(ytr))

    baseline_out["mean"] = {
        "train_mean": mu_y,
        "val_rmse": rmse_np(yva, np.full_like(yva, mu_y)),
        "test_rmse": rmse_np(yte, np.full_like(yte, mu_y)),
    }
    baseline_out["median"] = {
        "train_median": med_y,
        "val_rmse": rmse_np(yva, np.full_like(yva, med_y)),
        "test_rmse": rmse_np(yte, np.full_like(yte, med_y)),
    }

    logger.info(
        f"[Baseline mean]   val_rmse={baseline_out['mean']['val_rmse']:.6f} | test_rmse={baseline_out['mean']['test_rmse']:.6f}"
    )
    logger.info(
        f"[Baseline median] val_rmse={baseline_out['median']['val_rmse']:.6f} | test_rmse={baseline_out['median']['test_rmse']:.6f}"
    )

    # Time-only Ridge baseline
    try:
        from sklearn.linear_model import Ridge  # type: ignore

        Xtr = train_df[["age_std", "ysd_std"]].astype(float).values
        Xva = val_df[["age_std", "ysd_std"]].astype(float).values
        Xte = test_df[["age_std", "ysd_std"]].astype(float).values

        ridge = Ridge(alpha=1.0, random_state=cfg.seed)
        ridge.fit(Xtr, ytr)

        baseline_out["time_only_ridge"] = {
            "alpha": 1.0,
            "val_rmse": rmse_np(yva, ridge.predict(Xva)),
            "test_rmse": rmse_np(yte, ridge.predict(Xte)),
            "coef": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_),
        }

        logger.info(
            f"[Baseline time-only Ridge] val_rmse={baseline_out['time_only_ridge']['val_rmse']:.6f} | "
            f"test_rmse={baseline_out['time_only_ridge']['test_rmse']:.6f}"
        )
    except Exception as e:
        baseline_out["time_only_ridge"] = {"skipped": True, "reason": str(e)}
        logger.warning(f"[Baseline time-only Ridge] skipped: {e}")

    safe_json_dump(art_dir / "baselines.json", baseline_out)

    # -------------------------
    # Sanity
    # -------------------------
    mask_true = int(feat_mask.detach().cpu().sum().item())
    sanity_out["feature_mask_true_count"] = mask_true
    logger.info(f"[Sanity] feature_mask True count = {mask_true}")

    vt_nodes = samples.loc[samples["season"] >= cfg.val_season, "node_idx"].astype(int).values
    sanity_out["val_test_node_count"] = int(len(vt_nodes))

    if len(vt_nodes) > 0:
        rng = np.random.default_rng(cfg.seed)
        pick = vt_nodes if len(vt_nodes) <= 500 else rng.choice(vt_nodes, size=500, replace=False)

        x_cpu = x.detach().cpu()
        pick_t = torch.tensor(pick, dtype=torch.long)

        abs_mean = float(x_cpu[pick_t].abs().mean().item())
        abs_max = float(x_cpu[pick_t].abs().max().item())
        sanity_out["val_test_x_abs_mean"] = abs_mean
        sanity_out["val_test_x_abs_max"] = abs_max

        logger.info(f"[Sanity] val/test x abs_mean={abs_mean:.12f}, abs_max={abs_max:.12f} (should be ~0)")
        if abs_max > 1e-6:
            logger.warning("[Sanity] WARNING: val/test x is not fully zeroed! Potential leakage risk.")

    tr_nodes = train_df["node_idx"].astype(int).values
    sanity_out["train_node_count"] = int(len(tr_nodes))
    if len(tr_nodes) > 0:
        rng = np.random.default_rng(cfg.seed + 1)
        pick = tr_nodes if len(tr_nodes) <= 500 else rng.choice(tr_nodes, size=500, replace=False)

        x_cpu = x.detach().cpu()
        pick_t = torch.tensor(pick, dtype=torch.long)
        abs_mean_tr = float(x_cpu[pick_t].abs().mean().item())
        abs_max_tr = float(x_cpu[pick_t].abs().max().item())

        sanity_out["train_x_abs_mean"] = abs_mean_tr
        sanity_out["train_x_abs_max"] = abs_max_tr
        logger.info(f"[Sanity] train x abs_mean={abs_mean_tr:.6f}, abs_max={abs_max_tr:.6f} (should be > 0)")

    sanity_out["y_train_mean"] = float(np.mean(ytr))
    sanity_out["y_train_std"] = float(np.std(ytr))
    logger.info(f"[Sanity] y train mean/std = {sanity_out['y_train_mean']:.4f}/{sanity_out['y_train_std']:.4f}")

    safe_json_dump(art_dir / "sanity.json", sanity_out)

    # SpotCheck
    try:
        map_df = pd.read_csv((REPO_ROOT / cfg.master_mapping_path).resolve())
        map_df.columns = [c.lower().strip() for c in map_df.columns]
        idx_to_node = dict(zip(map_df["idx"].astype(int), map_df["node_id"].astype(str)))

        spot = samples.sample(5, random_state=cfg.seed)[["player_id", "season", "node_idx"]].copy()
        spot["node_id_from_master"] = spot["node_idx"].astype(int).map(idx_to_node)
        logger.info("[SpotCheck] 5 random rows:\n" + spot.to_string(index=False))
        spot.to_csv(art_dir / "spotcheck.csv", index=False, encoding="utf-8")
    except Exception as e:
        logger.warning(f"[SpotCheck] failed: {e}")

    # -------------------------
    # Train both modes
    # -------------------------
    results = {}
    for mode in cfg.run_modes:
        res = train_one_mode(
            cfg=cfg,
            mode=mode,
            dataset=dataset,
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            feat_mask=feat_mask,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            logger=logger,
            run_dir=run_dir,
            ckpt_dir=ckpt_dir,
            baseline_out=baseline_out,
            sanity_out=sanity_out,
        )
        results[mode] = res

    # Summary compare
    safe_json_dump(art_dir / "summary_results.json", results)

    if "graph" in results and "nograph" in results:
        logger.info(
            "[Compare] graph vs nograph:\n"
            f"  graph   best_val={results['graph']['best_val_rmse']:.6f}, test={results['graph']['test_rmse']:.6f}\n"
            f"  nograph best_val={results['nograph']['best_val_rmse']:.6f}, test={results['nograph']['test_rmse']:.6f}"
        )
        delta = results["graph"]["test_rmse"] - results["nograph"]["test_rmse"]
        logger.info(f"[Compare] test_rmse(graph - nograph) = {delta:.6f}  (negative => graph better)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
