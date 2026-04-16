from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st

from mlip_arena.models import REGISTRY

DATA_DIR = Path(__file__).parents[2] / "benchmarks" / "stability"


@st.cache_data
def get_data(model_list, run_type: Literal["heating", "compression"]) -> pd.DataFrame:
    """Load benchmarking data for the given models and run type.

    Parameters:
        model_list (Iterable): An iterable of model identifiers (used to locate each model's data file).
        run_type (Literal["heating", "compression"]): Which run dataset to load for each model.

    Returns:
        pd.DataFrame: Concatenated dataframe of all found model datasets with a `method` column set to the model identifier; returns an empty DataFrame if no files were found.
    """
    dfs = []
    for m in model_list:
        fpath = DATA_DIR / REGISTRY[str(m)]["family"].lower() / f"{m}-{run_type}.parquet"
        if not fpath.exists():
            continue
        df_local = pd.read_parquet(fpath)
        df_local["method"] = str(m)
        dfs.append(df_local)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data
def prepare_scatter_df(df_in: pd.DataFrame, max_points: int = 20000) -> pd.DataFrame:
    """Prepare scatter dataframe with marker sizes scaled by total steps."""
    dfp = df_in.dropna(subset=["natoms", "steps_per_second"]).copy()
    if dfp.empty:
        return dfp

    # Downsample if too many points
    if len(dfp) > max_points:
        dfp = dfp.sample(max_points, random_state=1)

    if "total_steps" in dfp.columns:
        ts_local = dfp["total_steps"].fillna(dfp["total_steps"].median()).astype(float)
        ts_range = ts_local.max() - ts_local.min()
        scaled = (ts_local - ts_local.min()) / (ts_range if ts_range != 0 else 1.0)
        dfp["_marker_size"] = (scaled * 40) + 5
    else:
        dfp["_marker_size"] = 8
    return dfp


@st.cache_data
def compute_power_law_fits(df_in: pd.DataFrame) -> dict:
    """Estimate per-method power-law scaling of throughput (steps per second) as a function of system size.

    For each group in `df_in` grouped by the `method` column, fits a power law of the form
    steps/s ≈ a * N^(-n) using a linear fit on log-transformed `natoms` and `steps_per_second`.
    Groups with fewer than 3 valid rows or with nonpositive/missing `natoms` or `steps_per_second`
    are skipped.

    Parameters:
        df_in (pd.DataFrame): DataFrame containing at minimum the columns
            `method`, `natoms`, and `steps_per_second`.

    Returns:
        dict: Mapping from `method` to a tuple `(a, n)` where `a` is the prefactor and `n`
        is the positive scaling exponent in the relation steps/s ≈ a * N^(-n).
    """
    fits = {}
    for name, grp in df_in.groupby("method"):
        grp_clean = grp.dropna(subset=["natoms", "steps_per_second"])
        grp_clean = grp_clean[(grp_clean["natoms"] > 0) & (grp_clean["steps_per_second"] > 0)]
        if len(grp_clean) < 3:
            continue
        try:
            logsx = np.log(grp_clean["natoms"].astype(float))
            logsy = np.log(grp_clean["steps_per_second"].astype(float))
            slope, intercept = np.polyfit(logsx, logsy, 1)
            fits[name] = (float(np.exp(intercept)), float(-slope))  # (a, n)
        except Exception:
            continue
    return fits


@st.cache_data
def compute_auc(df: pd.DataFrame) -> dict:
    """Compute the area under the "valid run" curve for each method.

    For each method this function drops duplicate runs by `formula`, builds a distribution of `normalized_final_step`,
    constructs the corresponding valid-run curve, and returns the area under that curve.

    Returns:
        dict: Mapping from method identifier to the computed AUC (float).
    """
    aucs = {}
    for method, dfm in df.groupby("method"):
        dfm = dfm.drop_duplicates(["formula"])
        if dfm.empty:
            continue
        hist, bin_edges = np.histogram(dfm["normalized_final_step"], bins=np.linspace(0, 1, 100))
        cumulative_population = np.cumsum(hist)
        valid_curve = (cumulative_population[-1] - cumulative_population) / len(dfm)
        aucs[method] = np.trapezoid(valid_curve, bin_edges[:-1])  # trapezoidal integration
    return aucs


# Load data
df_nvt = get_data(list(REGISTRY.keys()), run_type="heating")
df_npt = get_data(list(REGISTRY.keys()), run_type="compression")

# Compute metrics
aucs_nvt = compute_auc(df_nvt)
aucs_npt = compute_auc(df_npt)

fits_nvt = compute_power_law_fits(df_nvt)
fits_npt = compute_power_law_fits(df_npt)

# Build summary table
rows = []
for method in set(aucs_nvt) | set(aucs_npt):
    row = {
        "Model": method,
        "AUC (Heating)": aucs_nvt.get(method, np.nan),
        "AUC (Compression)": aucs_npt.get(method, np.nan),
        "Scaling exponent (Heating)": fits_nvt.get(method, (np.nan, np.nan))[1],
        "Scaling exponent (Compression)": fits_npt.get(method, (np.nan, np.nan))[1],
    }
    rows.append(row)

table = pd.DataFrame(rows).set_index("Model")

table["Rank"] = table["AUC (Heating)"].rank(ascending=False, na_option="bottom")
table["Rank"] += table["AUC (Compression)"].rank(ascending=False, na_option="bottom")
table["Rank"] += table["Scaling exponent (Heating)"].rank(ascending=True, na_option="bottom")
table["Rank"] += table["Scaling exponent (Compression)"].rank(ascending=True, na_option="bottom")

table.sort_values(["Rank"], ascending=True, inplace=True)

table["Rank aggr."] = table["Rank"].astype(int)
table["Rank"] = table["Rank aggr."].rank(method="min").astype(int)

table = table.reindex(
    columns=[
        "Rank",
        "Rank aggr.",
        "AUC (Heating)",
        "AUC (Compression)",
        "Scaling exponent (Heating)",
        "Scaling exponent (Compression)",
    ]
)


@st.cache_data
def get_table():
    return table


def render():
    # Style
    """Apply visual styling to the module-level summary table and render it in Streamlit.

    Applies a blue background gradient to the rank columns, a reversed-green gradient to the AUC columns,
    and a green gradient to the scaling-exponent columns. Formats the metric columns to three decimal
    places and represents missing values with a hyphen, then displays the resulting styled table via
    Streamlit's `st.dataframe` with container-width enabled.
    """
    s = (
        table.style.background_gradient(
            cmap="Blues",
            subset=["Rank", "Rank aggr."],
        )
        .background_gradient(cmap="Greens_r", subset=["AUC (Heating)", "AUC (Compression)"])
        .background_gradient(
            cmap="Greens",
            subset=["Scaling exponent (Heating)", "Scaling exponent (Compression)"],
        )
        .format(
            "{:.3f}",
            subset=[
                "AUC (Heating)",
                "AUC (Compression)",
                "Scaling exponent (Heating)",
                "Scaling exponent (Compression)",
            ],
            na_rep="-",
        )
    )
    st.dataframe(s, use_container_width=True)
