import pandas as pd
import numpy as np
from pathlib import Path

# ================= CONFIG =================
PRED_DIR = Path("runs/final_eval_strict_v3/20260213_131005/predictions")

SEED = 0
INV_MODE = "log1p"

# --- reference baselines ---
BASE_FILE = f"predictions_Baseline_StatsplusTime_RandomForest_seed{SEED}.csv"
BASE_META_FILE = f"predictions_Baseline_StatsplusTimeplusMeta_RandomForest_seed{SEED}.csv"

# --- thresholds ---
RESCUE_MIN = 500_000          # 至少救回 $0.5M
BASE_ERR_MIN = 1_000_000      # baseline 原本误差至少 $1M（避免捡漏）
UNIQUE_RESCUE_MIN = 1_500_000 # Unique insight: 我至少救回 $1.5M
UNIQUE_ADV_MIN = 500_000      # Unique insight: 比其他模型最好救回还多 $0.5M
MIN_EXAMPLES_PER_MODEL = 3    # 每个模型至少几个例子（尽量）

# --- candidate models ---
MODELS = {
    "Tabular_OnOff": f"predictions_Baseline_StatsplusTimeplusMeta_RandomForest_seed{SEED}.csv",
    "RotatE":        f"predictions_RotatE_plus_Stats_RandomForest_seed{SEED}.csv",
    "Node2Vec":      f"predictions_Node2Vec_plus_Stats_RandomForest_seed{SEED}.csv",
    "V1":            f"predictions_V1_plus_Stats_RandomForest_seed{SEED}.csv",
    "V2_Ind":        f"predictions_V2_Ind_plus_Stats_RandomForest_seed{SEED}.csv",
    "V2_Trans":      f"predictions_V2_Trans_plus_Stats_RandomForest_seed{SEED}.csv",
    "V2_Full":       f"predictions_V2_Full_MG_plus_Stats_RandomForest_seed{SEED}.csv",
}

# run rescue relative to both baselines
REF_BASELINES = {
    "Stats+Time": BASE_FILE,
    "Stats+Time+Meta": BASE_META_FILE,
}

# ================= Utilities =================

def inv_log(x: np.ndarray) -> np.ndarray:
    """Inverse transform and clamp to non-negative."""
    if INV_MODE == "log":
        val = np.exp(x)
    elif INV_MODE == "log1p":
        val = np.expm1(x)
    else:
        raise ValueError("INV_MODE error")
    return np.maximum(val, 0.0)

def load_pred_file(path: Path) -> pd.DataFrame | None:
    """Load prediction CSV and return standardized columns."""
    if not path.exists():
        return None
    df = pd.read_csv(path)

    required = ["player_id", "season", "y_true", "y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"❌ {path.name} missing columns: {missing}")

    if "player_name" not in df.columns:
        df["player_name"] = "Unknown"

    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)

    df["salary_true_usd"] = inv_log(df["y_true"].to_numpy())
    df["salary_pred_usd"] = inv_log(df["y_pred"].to_numpy())

    return df[["player_id", "season", "player_name", "salary_true_usd", "salary_pred_usd"]]

def fmt(x: float) -> str:
    return f"${x/1e6:.2f}M" if abs(x) < 1e7 else f"${x/1e6:.1f}M"

def categorize_rescue(row: pd.Series) -> str:
    """Academic-style categorization: Under/Over × Precision/Overshoot."""
    act = row["salary_true_usd"]
    base = row["pred_base"]
    pred = row["pred_model"]

    # boundary: baseline already essentially exact -> skip
    if abs(act - base) < 10_000:
        return "Exact (Skip)"

    main_type = "Underrated" if act > base else "Overrated"

    # overshoot if model crosses the true value to the other side
    is_overshoot = (base - act) * (pred - act) < 0
    sub_type = "Overshoot" if is_overshoot else "Precision"

    return f"{main_type} ({sub_type})"

def select_representative_players(
    success_cases: pd.DataFrame,
    min_examples: int = 3,
) -> pd.DataFrame:
    """
    Select representative players for one model under one reference baseline.

    Protocol:
      A) Category picks: for each of 4 rescue types, select top-rescue case.
      B) Fallback: if < min_examples, fill with remaining top-rescue cases.
    Produces:
      selection_method: Category / Fallback
      selection_bucket: which rescue_type (or TopRescue)
      selection_rank: 1..K
    """
    if success_cases.empty:
        return success_cases.head(0).copy()

    target_types = [
        "Underrated (Precision)", "Underrated (Overshoot)",
        "Overrated (Precision)",  "Overrated (Overshoot)"
    ]

    picked_rows = []
    seen_pids = set()

    # A) Category
    for r_type in target_types:
        subset = success_cases[success_cases["rescue_type"] == r_type].sort_values("rescue", ascending=False)
        if subset.empty:
            continue
        for _, cand in subset.iterrows():
            pid = cand["player_id"]
            if pid in seen_pids:
                continue
            c = cand.copy()
            c["selection_method"] = "Category"
            c["selection_bucket"] = r_type
            picked_rows.append(c)
            seen_pids.add(pid)
            break

    # B) Fallback
    if len(picked_rows) < min_examples:
        needed = min_examples - len(picked_rows)
        remaining = success_cases[~success_cases["player_id"].isin(seen_pids)] \
            .sort_values("rescue", ascending=False) \
            .head(needed)
        for _, cand in remaining.iterrows():
            pid = cand["player_id"]
            if pid in seen_pids:
                continue
            c = cand.copy()
            c["selection_method"] = "Fallback"
            c["selection_bucket"] = "TopRescue"
            picked_rows.append(c)
            seen_pids.add(pid)

    out = pd.DataFrame(picked_rows).reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1)
    return out

# ================= Core analysis per reference baseline =================

def run_one_reference(ref_name: str, ref_baseline_file: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print(f"🧭 Rescue Reference Baseline = {ref_name}  |  file = {ref_baseline_file}")
    print("=" * 70)

    # Load reference baseline
    base_path = PRED_DIR / ref_baseline_file
    df_base = load_pred_file(base_path)
    if df_base is None:
        raise FileNotFoundError(f"❌ Reference baseline missing: {base_path}")

    df_base = df_base.rename(columns={"salary_pred_usd": "pred_base"})
    df_base_core = df_base[["player_id", "season", "player_name", "salary_true_usd", "pred_base"]]

    global_rescue_map = {}  # for unique insights
    all_examples = []
    coverage_stats = []

    for model_name, filename in MODELS.items():
        # Skip if model equals reference baseline
        if filename == ref_baseline_file:
            print(f"\n⏭️  Skip: {model_name} (same as reference baseline: {ref_name})")
            continue

        print(f"\n🔵 [{ref_name}] Analyzing: {model_name} ...")
        df_model = load_pred_file(PRED_DIR / filename)

        if df_model is None:
            print("   ⚠️ Skip (Not Found)")
            coverage_stats.append({
                "RefBaseline": ref_name,
                "Model": model_name,
                "Coverage": 0,
                "Success_Cases": 0,
                "Success_Rate": 0.0,
                "Success_Rate_Pct": "0.0%"
            })
            continue

        # Merge + rename to pred_model
        merged = pd.merge(
            df_base_core,
            df_model[["player_id", "season", "salary_pred_usd"]],
            on=["player_id", "season"],
            how="inner"
        ).rename(columns={"salary_pred_usd": "pred_model"})

        # Metrics
        merged["err_base"] = (merged["salary_true_usd"] - merged["pred_base"]).abs()
        merged["err_model"] = (merged["salary_true_usd"] - merged["pred_model"]).abs()
        merged["rescue"] = merged["err_base"] - merged["err_model"]

        # Build global rescue map
        for row in merged.itertuples(index=False):
            key = (row.player_id, row.season)
            if key not in global_rescue_map:
                global_rescue_map[key] = {"player_name": row.player_name, "models": {}}
            global_rescue_map[key]["models"][model_name] = {
                "rescue": row.rescue,
                "err": row.err_model,
                "err_base": row.err_base
            }

        # Success filter
        success_cases = merged[
            (merged["rescue"] > RESCUE_MIN) &
            (merged["err_base"] > BASE_ERR_MIN)
        ].copy()

        # Coverage stats
        n_cov = len(merged)
        n_succ = len(success_cases)
        rate = (n_succ / n_cov) if n_cov > 0 else 0.0

        coverage_stats.append({
            "RefBaseline": ref_name,
            "Model": model_name,
            "Coverage": int(n_cov),
            "Success_Cases": int(n_succ),
            "Success_Rate": float(rate),
            "Success_Rate_Pct": f"{rate*100:.1f}%"
        })
        print(f"   Coverage: {n_cov} | Success: {n_succ} ({rate*100:.1f}%)")

        if success_cases.empty:
            continue

        # Categorize + select reps
        success_cases["rescue_type"] = success_cases.apply(categorize_rescue, axis=1)
        selected_df = select_representative_players(success_cases, min_examples=MIN_EXAMPLES_PER_MODEL)

        # Add to examples
        for _, row in selected_df.iterrows():
            delta = row["pred_model"] - row["pred_base"]
            all_examples.append({
                "RefBaseline": ref_name,
                "Model": model_name,
                "Type": row["rescue_type"],
                "Selection_Method": row.get("selection_method", "Category"),
                "Selection_Bucket": row.get("selection_bucket", ""),
                "Selection_Rank": int(row.get("selection_rank", 0)),
                "Player": row["player_name"],
                "Season": int(row["season"]),
                "Actual": float(row["salary_true_usd"]),
                "Base_Pred": float(row["pred_base"]),
                "Model_Pred": float(row["pred_model"]),
                "Delta_Pred": float(delta),
                "Base_Err": float(row["err_base"]),
                "Model_Err": float(row["err_model"]),
                "Rescue_Amount": float(row["rescue"]),
                "Model_Coverage": int(n_cov),
            })
            print(
                f"   + Selected(R{int(row.get('selection_rank',0))}): {row['player_name']} "
                f"[{row['rescue_type']}] ({row.get('selection_method','')}/{row.get('selection_bucket','')}) "
                f"(+{fmt(row['rescue'])})"
            )

    # ================= Unique Insights =================
    print("\n" + "=" * 60)
    print(f"🏆 Unique Insights Analysis  (RefBaseline={ref_name})")
    print("=" * 60)

    unique_insights_list = []

    for target_model in MODELS.keys():
        candidates = []
        for key, info in global_rescue_map.items():
            res_dict = info["models"]
            if target_model not in res_dict:
                continue

            known_models = [m for m in MODELS if m in res_dict]
            if len(known_models) < 2:
                continue

            my_rescue = res_dict[target_model]["rescue"]
            my_err = res_dict[target_model]["err"]

            peers = [m for m in known_models if m != target_model]
            peer_rescues = [res_dict[m]["rescue"] for m in peers]
            peer_errs = [res_dict[m]["err"] for m in peers]

            max_other_rescue = max(peer_rescues) if peer_rescues else -9e9
            min_other_err = min(peer_errs) if peer_errs else 9e9

            candidates.append({
                "player_name": info["player_name"],
                "player_id": key[0],
                "season": key[1],
                "my_rescue": my_rescue,
                "my_err": my_err,
                "min_other_err": min_other_err,
                "rescue_advantage": my_rescue - max_other_rescue,
                "coverage": len(known_models),
            })

        if not candidates:
            continue

        df_cand = pd.DataFrame(candidates)

        # Dual thresholds: big rescue + near-best absolute error
        df_cand = df_cand[
            (df_cand["my_rescue"] > UNIQUE_RESCUE_MIN) &
            (df_cand["my_err"] <= df_cand["min_other_err"] + 10_000)
        ]
        if df_cand.empty:
            continue

        hits = df_cand[df_cand["rescue_advantage"] > UNIQUE_ADV_MIN] \
            .sort_values("rescue_advantage", ascending=False).head(3)

        if not hits.empty:
            print(f"\n🌟 [{target_model}] Exclusive:")
            for _, row in hits.iterrows():
                print(
                    f"   🏀 {row['player_name']:<20} | "
                    f"Rescue:+{fmt(row['my_rescue'])} | "
                    f"Adv:+{fmt(row['rescue_advantage'])} | "
                    f"MyErr:{fmt(row['my_err'])}"
                )
                unique_insights_list.append({
                    "RefBaseline": ref_name,
                    "Model": target_model,
                    "Player": row["player_name"],
                    "Rescue": float(row["my_rescue"]),
                    "My_Err": float(row["my_err"]),
                    "Peer_Best_Err": float(row["min_other_err"]),
                    "Rescue_Advantage": float(row["rescue_advantage"]),
                    "Coverage": int(row["coverage"]),
                })

    df_examples = pd.DataFrame(all_examples) if all_examples else pd.DataFrame()
    df_unique = pd.DataFrame(unique_insights_list) if unique_insights_list else pd.DataFrame()
    df_cov = pd.DataFrame(coverage_stats) if coverage_stats else pd.DataFrame()
    return df_examples, df_unique, df_cov

# ================= Main =================

def main():
    print("--- Configuration ---")
    print(f"PRED_DIR: {PRED_DIR}")
    print(f"Seed: {SEED} | INV_MODE: {INV_MODE}")
    print("-" * 50)

    # Sanity check baseline files exist
    for ref_name, ref_file in REF_BASELINES.items():
        p = PRED_DIR / ref_file
        if not p.exists():
            print(f"❌ Missing baseline file for {ref_name}: {p}")
            return

    all_ex_list = []
    all_unique_list = []
    all_cov_list = []

    for ref_name, ref_file in REF_BASELINES.items():
        df_ex, df_u, df_cov = run_one_reference(ref_name, ref_file)
        if not df_ex.empty:
            all_ex_list.append(df_ex)
        if not df_u.empty:
            all_unique_list.append(df_u)
        if not df_cov.empty:
            all_cov_list.append(df_cov)

    # ===== Save outputs =====
    # 1) Concrete examples
    if all_ex_list:
        df_ex_all = pd.concat(all_ex_list, axis=0, ignore_index=True)
        df_ex_all.to_csv(PRED_DIR / "summary_concrete_examples_numeric__both_refs.csv", index=False)

        df_fmt = df_ex_all.copy()
        money_cols = ["Actual", "Base_Pred", "Model_Pred", "Delta_Pred", "Base_Err", "Model_Err", "Rescue_Amount"]
        for col in money_cols:
            df_fmt[col] = df_fmt[col].apply(fmt)
        df_fmt.to_csv(PRED_DIR / "summary_concrete_examples__both_refs.csv", index=False)

        print("\n✅ Concrete Examples saved:")
        print(f" - {PRED_DIR / 'summary_concrete_examples_numeric__both_refs.csv'}")
        print(f" - {PRED_DIR / 'summary_concrete_examples__both_refs.csv'}")

    # 2) Unique insights
    if all_unique_list:
        df_u_all = pd.concat(all_unique_list, axis=0, ignore_index=True)
        df_u_all.to_csv(PRED_DIR / "summary_unique_insights_numeric__both_refs.csv", index=False)

        df_u_fmt = df_u_all.copy()
        money_cols_u = ["Rescue", "My_Err", "Peer_Best_Err", "Rescue_Advantage"]
        for col in money_cols_u:
            df_u_fmt[col] = df_u_fmt[col].apply(fmt)
        df_u_fmt.to_csv(PRED_DIR / "summary_unique_insights__both_refs.csv", index=False)

        print("\n✅ Unique Insights saved:")
        print(f" - {PRED_DIR / 'summary_unique_insights_numeric__both_refs.csv'}")
        print(f" - {PRED_DIR / 'summary_unique_insights__both_refs.csv'}")

    # 3) Coverage summary
    if all_cov_list:
        df_cov_all = pd.concat(all_cov_list, axis=0, ignore_index=True)
        df_cov_all.to_csv(PRED_DIR / "summary_model_coverage__both_refs.csv", index=False)

        print("\n✅ Coverage Summary saved:")
        print(f" - {PRED_DIR / 'summary_model_coverage__both_refs.csv'}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
