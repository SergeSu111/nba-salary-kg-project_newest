"""
scripts/rotate_L1B_train_and_export.py

Train RotatE (L1B: relation-aware KG embedding) and export Player embeddings
for downstream salary regression.

Inputs:
- Triples CSV with header: head,relation,tail  (strings; e.g., Neo4j elementId)
- Player map CSV with columns: node_id,player_id (node_id must match triples entity labels)

Outputs:
- PyKEEN artifacts saved to out_model_dir
- Player embeddings CSV saved to out_embeddings
- Vocab JSON saved to out_model_dir/vocab_entity_relation.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import random

import numpy as np
import pandas as pd
import torch

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # (optional) determinism; can slightly slow down
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_triples_csv(path: Path) -> pd.DataFrame:
    """Read triples from CSV with header: head,relation,tail."""
    if not path.exists():
        raise FileNotFoundError(f"Triples file not found: {path}")

    df = pd.read_csv(path)
    need = {"head", "relation", "tail"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(
            f"Triples file missing columns {missing}. "
            f"Expected header columns: head,relation,tail. "
            f"Got columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["head", "relation", "tail"]).drop_duplicates()

    # IMPORTANT: keep everything as string labels
    df["head"] = df["head"].astype(str)
    df["relation"] = df["relation"].astype(str)
    df["tail"] = df["tail"].astype(str)

    return df


def read_player_map(path: Path) -> pd.DataFrame:
    """Read player mapping: node_id -> player_id."""
    if not path.exists():
        raise FileNotFoundError(f"Player map file not found: {path}")

    pm = pd.read_csv(path)
    need = {"node_id", "player_id"}
    missing = need - set(pm.columns)
    if missing:
        raise ValueError(
            f"Player map missing columns {missing}. "
            f"Expected columns: node_id,player_id. "
            f"Got columns: {list(pm.columns)}"
        )

    pm = pm.dropna(subset=["node_id", "player_id"]).drop_duplicates()

    # node_id must match entity labels in triples exactly
    pm["node_id"] = pm["node_id"].astype(str)

    # normalize player_id to string to avoid 2030.0 issues
    pm["player_id"] = pm["player_id"].astype(str)

    return pm


def save_vocab(tf: TriplesFactory, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vocab_entity_relation.json"
    payload = {
        "num_entities": int(tf.num_entities),
        "num_relations": int(tf.num_relations),
        "entity_to_id": tf.entity_to_id,
        "relation_to_id": tf.relation_to_id,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path


@torch.no_grad()
def export_player_embeddings(
    *,
    tf: TriplesFactory,
    model,
    player_map: pd.DataFrame,
    out_path: Path,
    device: str,
) -> dict:
    """
    Export embeddings for Player nodes (node_id in player_map).
    Returns summary stats.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entity_to_id = tf.entity_to_id  # dict[str, int]

    model.eval()
    rep = model.entity_representations[0]

    # Get full embedding matrix in a version-stable way
    # rep(indices=None) is NOT stable across PyKEEN versions.
    indices = torch.arange(tf.num_entities, device=device, dtype=torch.long)
    emb_matrix = rep(indices=indices).detach().cpu().numpy()  # [num_entities, dim]
    dim = int(emb_matrix.shape[1])
    emb_cols = [f"e{i}" for i in range(dim)]

    rows = []
    missing = 0

    for _, r in player_map.iterrows():
        node_id = r["node_id"]
        player_id = r["player_id"]

        idx = entity_to_id.get(node_id)
        if idx is None:
            missing += 1
            continue

        vec = emb_matrix[idx]
        rows.append([player_id, node_id, *vec.tolist()])

    out_df = pd.DataFrame(rows, columns=["player_id", "node_id", *emb_cols])
    out_df.to_csv(out_path, index=False)

    return {
        "players_in_map": int(len(player_map)),
        "players_exported": int(len(out_df)),
        "players_missing_in_entity_vocab": int(missing),
        "embedding_dim_exported": int(dim),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--triples",
        type=str,
        default="graph/edges/triples_rotate_L1A.csv",
        help="Path to triples file (CSV with header head,relation,tail).",
    )
    parser.add_argument(
        "--player_map",
        type=str,
        default="graph/mappings/player_nodeid_map.csv",
        help="Path to player mapping file (columns: node_id,player_id).",
    )
    parser.add_argument(
        "--out_model_dir",
        type=str,
        default="graph/models/rotate_L1B",
        help="Directory to save PyKEEN pipeline artifacts.",
    )
    parser.add_argument(
        "--out_embeddings",
        type=str,
        default="graph/embeddings/rotate_L1B_player_embeddings.csv",
        help="Output CSV path for exported player embeddings.",
    )

    # training config
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_negs_per_pos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    # stability options
    parser.add_argument(
        "--create_inverse_triples",
        action="store_true",
        help="If set, add inverse relations (often improves KGE stability).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu. (You said you use GPU, so default=cuda.)",
    )

    args = parser.parse_args()

    triples_path = Path(args.triples)
    player_map_path = Path(args.player_map)
    out_model_dir = Path(args.out_model_dir)
    out_emb_path = Path(args.out_embeddings)

    # ---------- seeds ----------
    set_all_seeds(args.seed)

    # ---------- Load inputs ----------
    triples_df = read_triples_csv(triples_path)
    player_map_df = read_player_map(player_map_path)

    print(f"[Input] triples: {triples_path}")
    print(f"[Triples] num_triples={len(triples_df)}")
    print(f"[Triples] num_relations={triples_df['relation'].nunique()}")
    print(f"[Triples] relations={sorted(triples_df['relation'].unique().tolist())}")
    print(f"[Input] player_map: {player_map_path}")
    print(f"[PlayerMap] rows={len(player_map_df)}")

    # ---------- Build TriplesFactory ----------
    triples = triples_df[["head", "relation", "tail"]].to_numpy(dtype=str)
    tf = TriplesFactory.from_labeled_triples(
        triples,
        create_inverse_triples=bool(args.create_inverse_triples),
    )
    print(f"[TF] num_entities={tf.num_entities} num_relations={tf.num_relations}")

    # save vocab for reproducibility/debugging
    vocab_path = save_vocab(tf, out_model_dir)
    print(f"[Saved] vocab -> {vocab_path}")

    # ---------- Train RotatE ----------
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but not available; falling back to cpu", file=sys.stderr)
        device = "cpu"

    print(
        f"[Train] device={device} seed={args.seed} "
        f"dim={args.embedding_dim} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.lr} negs={args.num_negs_per_pos} "
        f"inverse={args.create_inverse_triples}"
    )

    # ---------- Split into train/valid/test (required by your PyKEEN version) ----------
    training_tf, testing_tf, validation_tf = tf.split(
        ratios=(0.8, 0.1, 0.1),
        random_state=args.seed,
    )

    # ---------- Train RotatE ----------
    result = pipeline(
        training=training_tf,
        testing=testing_tf,
        validation=validation_tf,
        model="RotatE",
        model_kwargs=dict(embedding_dim=args.embedding_dim),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=args.lr),
        loss="SoftplusLoss",
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=args.num_negs_per_pos),
        training_kwargs=dict(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    ),
    random_seed=args.seed,
    device=device,
    )


    out_model_dir.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(out_model_dir)
    print(f"[Saved] model artifacts -> {out_model_dir}")

    # ---------- Export player embeddings ----------
    stats = export_player_embeddings(
        tf=training_tf,
        model=result.model,
        player_map=player_map_df,
        out_path=out_emb_path,
        device=device,
    )
    print(f"[Saved] embeddings -> {out_emb_path}")
    print("[ExportStats]", json.dumps(stats, ensure_ascii=False))

    # ---------- Basic sanity checks ----------
    if stats["players_exported"] == 0:
        print(
            "[WARN] Exported 0 player embeddings. "
            "Usually node_id in player_map does not match entity labels in triples.",
            file=sys.stderr,
        )
    if stats["players_missing_in_entity_vocab"] > 0:
        print(
            f"[WARN] Missing players in entity vocab: {stats['players_missing_in_entity_vocab']} / {stats['players_in_map']}. "
            "This can happen if some Player nodes have no edges in the exported triples.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()