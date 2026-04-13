from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.absolute()


def summarize():
    """Summarizes all benchmark results (*_results.parquet) in the directory.

    Aggregates metrics per model, computes ranks, and exports leaderboard files.
    """
    # 1. Find and Load all result parquet files
    result_files = sorted([f for f in DATA_DIR.glob("*_results.parquet") if f.name != "all_results.parquet"])

    if not result_files:
        print("No result parquet files found (*_results.parquet).")
        return None

    print(f"Found {len(result_files)} result files. Loading metrics...")

    dfs = []
    for f in result_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f.name}: {e}")

    if not dfs:
        print("No data could be loaded.")
        return None

    # Combine all individual results
    df_all = pd.concat(dfs, ignore_index=True)

    # 2. Ensure consistent dtypes before saving and summarizing
    # This prevents PyArrow errors (mixed float32/float64 or object types)
    for col in df_all.columns:
        if col in ["model", "structure", "formula", "id"]:
            df_all[col] = df_all[col].astype(str)
        elif col == "missing":
            df_all[col] = df_all[col].astype(bool)
        elif col in [
            "energy-diff-flip-times",
            "tortuosity",
            "spearman-compression-energy",
            "spearman-compression-derivative",
            "spearman-tension-energy",
        ]:
            # Convert single-value metrics to float64
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
        elif col in ["volume-ratio", "energy-delta-per-atom", "energy-delta-per-volume-b0"]:
            # Ensure array contents are consistent float64
            df_all[col] = df_all[col].apply(
                lambda x: np.array(x, dtype=np.float64) if isinstance(x, list | np.ndarray) else x
            )

    # Expose combined results
    results_fpath = DATA_DIR / "all_results.parquet"
    df_all.to_parquet(results_fpath)
    print(f"Combined {len(df_all)} results into {results_fpath.name}")

    # 3. Calculate summary metrics per model
    # Metrics to aggregate (means and stds)
    metrics = [
        "energy-diff-flip-times",
        "tortuosity",
        "spearman-compression-energy",
        "spearman-compression-derivative",
        "spearman-tension-energy",
    ]

    # Only include valid (not missing) results for means and stds
    df_valid = df_all[~df_all["missing"]]

    # Group by model
    summary_means = df_valid.groupby("model")[metrics].mean()
    summary_stds = df_valid.groupby("model")[metrics].std().rename(columns={m: f"{m}-std" for m in metrics})

    # Calculate missing count per model (includes all attempts)
    summary_missing = df_all.groupby("model")["missing"].sum().astype(int).to_frame("missing")

    # Combine all pieces into the leaderboard table
    leaderboard = pd.concat([summary_means, summary_stds, summary_missing], axis=1)
    leaderboard = leaderboard.reset_index()

    # 4. Ranking Logic
    # flip_rank: smaller absolute difference from 1 is better
    leaderboard["flip_rank"] = (leaderboard["energy-diff-flip-times"] - 1).abs().rank(ascending=True, method="min")

    # tortuosity_rank: smaller is better (minimum is 1)
    leaderboard["tortuosity_rank"] = leaderboard["tortuosity"].rank(ascending=True, method="min")

    # spearman_compression_energy_rank: smaller (more negative) is better
    leaderboard["spearman_compression_energy_rank"] = leaderboard["spearman-compression-energy"].rank(method="min")

    # spearman_compression_derivative_rank: larger is better
    leaderboard["spearman_compression_derivative_rank"] = leaderboard["spearman-compression-derivative"].rank(
        ascending=False, method="min"
    )

    # spearman_tension_energy_rank: larger is better
    leaderboard["spearman_tension_energy_rank"] = leaderboard["spearman-tension-energy"].rank(
        ascending=False, method="min"
    )

    # missing_rank: fewer failures/missing data is better
    leaderboard["missing_rank"] = leaderboard["missing"].rank(ascending=True, method="min")

    # Aggregate Rank
    leaderboard["rank-aggregation"] = (
        leaderboard["flip_rank"]
        + leaderboard["tortuosity_rank"]
        + leaderboard["spearman_compression_energy_rank"]
        + leaderboard["spearman_compression_derivative_rank"]
        + leaderboard["spearman_tension_energy_rank"]
        + leaderboard["missing_rank"]
    ).astype(int)
    leaderboard["rank"] = leaderboard["rank-aggregation"].rank(method="min").astype(int)

    # 5. Clean up and Export
    # Reorder columns to match original format
    cols_ordered = [
        "model",
        "rank",
        "rank-aggregation",
        "energy-diff-flip-times",
        "tortuosity",
        "spearman-compression-energy",
        "spearman-compression-derivative",
        "spearman-tension-energy",
        "missing",
        "energy-diff-flip-times-std",
        "tortuosity-std",
        "spearman-compression-energy-std",
        "spearman-compression-derivative-std",
        "spearman-tension-energy-std",
    ]
    # Ensure all columns exist
    for col in cols_ordered:
        if col not in leaderboard.columns:
            leaderboard[col] = np.nan

    leaderboard = leaderboard[cols_ordered].sort_values("rank")

    # Save to CSV and LaTeX
    leaderboard.to_csv(DATA_DIR / "leaderboard.csv", index=False)
    leaderboard.to_latex(DATA_DIR / "leaderboard.tex", index=False, float_format="%.3f")

    print("\nBenchmark Leaderboard:")
    print(leaderboard[["model", "rank", "missing"]].to_string(index=False))
    print(f"\nFinal results exported to {DATA_DIR / 'leaderboard.csv'} and {DATA_DIR / 'leaderboard.tex'}")

    return leaderboard


if __name__ == "__main__":
    summarize()
