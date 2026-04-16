from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from mlip_arena.models import REGISTRY

st.title("Stability")

DATA_DIR = Path(__file__).parents[2] / "benchmarks" / "stability"

st.markdown("### Methods")
container = st.container(border=True)

# Filter models that have valid parquet results
valid_models = [
    model
    for model, metadata in REGISTRY.items()
    if (DATA_DIR / REGISTRY[str(model)]["family"].lower() / f"{model}-heating.parquet").exists()
]

models = container.multiselect(
    "MLIPs",
    valid_models,
    [
        "MACE-MP(M)",
        "CHGNet",
        "SevenNet",
        "ORBv2",
        "eqV2(OMat)",
        "M3GNet",
        "MatterSim",
        "MACE-MPA",
    ],
)

st.markdown("### Settings")
vis = st.container(border=True)

# Build available color palettes from Plotly
color_palettes = {
    attr: getattr(pcolors.qualitative, attr)
    for attr in dir(pcolors.qualitative)
    if isinstance(getattr(pcolors.qualitative, attr), list)
}
color_palettes.pop("__all__", None)

palette_name = vis.selectbox("Color sequence", options=list(color_palettes.keys()), index=22)
color_sequence = color_palettes[palette_name]

if not models:
    st.stop()


@st.cache_data
def get_data(model_list, run_type: Literal["heating", "compression"]) -> pd.DataFrame:
    """Load and concatenate parquet files for the given models and run type.

    Parameters:
        model_list (iterable): Iterable of model identifiers to load (elements convertible to str).
        run_type (Literal["heating", "compression"]): Which run variant to load for each model.

    Returns:
        pd.DataFrame: Concatenated dataframes from all found parquet files with an added
        "method" column set to the model identifier; returns an empty DataFrame if no files were found.
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


df_nvt = get_data(models, run_type="heating")
df_npt = get_data(models, run_type="compression")

# Map model → color
method_color_mapping = {
    method: color_sequence[i % len(color_sequence)] for i, method in enumerate(df_nvt["method"].unique())
}


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
    """Compute per-method power-law fits for inference speed as a function of system size.

    Groups rows by the "method" column and, for each method with at least three rows where
    "natoms" and "steps_per_second" are present and greater than zero, fits a linear model
    on log-transformed values to estimate parameters of the relation steps_per_second ≈ a * N^(-n).

    Parameters:
        df_in (pd.DataFrame): DataFrame containing at minimum the columns
            "method", "natoms", and "steps_per_second".

    Returns:
        dict: Mapping from method name to a tuple `(a, n)` where `a` is the prefactor
        and `n` is the exponent in the power law `steps_per_second ≈ a * N^(-n)`.
        Methods with fewer than three valid rows or for which fitting fails are omitted.
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
def build_speed_figure(df_in: pd.DataFrame, color_map: dict, show_scatter: bool) -> go.Figure:
    """Create a log-log Plotly figure showing steps per second versus number of atoms, optionally with scatter points and overlaid power-law fit lines per method.

    Returns:
        go.Figure: A Plotly figure containing a log-scaled x-axis ("Number of atoms") and y-axis ("Steps per second"), with optional scatter traces (colored by method) and power-law fit lines for each method.
    """
    fig = go.Figure()

    # Optionally add scatter points
    if show_scatter:
        dfp = prepare_scatter_df(df_in)
        scatter_fig = px.scatter(
            dfp,
            x="natoms",
            y="steps_per_second",
            color="method",
            size="_marker_size",
            hover_data=[c for c in ["material_id", "formula"] if c in dfp.columns],
            color_discrete_map=color_map,
            log_x=True,
            log_y=True,
            render_mode="webgl",
            labels={
                "steps_per_second": "Steps per second",
                "natoms": "Number of atoms",
            },
        )
        for trace in scatter_fig.data:
            fig.add_trace(trace)

    # Overlay fits
    fits = compute_power_law_fits(df_in)
    for method, (a, n) in fits.items():
        grp = df_in[df_in["method"] == method]
        if grp["natoms"].dropna().empty:
            continue
        xs = np.logspace(np.log10(grp["natoms"].min()), np.log10(grp["natoms"].max()), 200)
        ys = a * xs ** (-n)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color_map.get(method, "black"), width=2),
                showlegend=not show_scatter,
                name=f"{method}",
                # zorder=0,
                # text=hover_text,
                # hoverinfo='text',  # use the custom text
            )
        )

    fig.update_layout(
        height=520,
        title="Inference speed (steps/s)",
        xaxis=dict(type="log", title="Number of atoms"),
        yaxis=dict(type="log", title="Steps per second"),
    )
    return fig


@st.cache_data
def build_nvt_figure(df_in: pd.DataFrame, color_map: dict, show_scatter: bool) -> go.Figure:
    """Create a 1x2 Plotly figure showing cumulative valid runs and inference speed scaling.

    Left subplot shows the percentage of valid runs over normalized time for each method.
    Right subplot shows inference speed (steps per second) versus number of atoms on log-log axes; the right panel includes scatter points and fitted power-law lines when `show_scatter` is enabled.

    Parameters:
        df_in (pd.DataFrame): Input dataframe containing run records. Expected columns include
            'method', 'normalized_final_step', and 'formula' (used for the left subplot); the right
            subplot expects 'natoms' and 'steps_per_second' to be present for speed data.
        color_map (dict): Mapping from method identifier to a color string used for plotting.
        show_scatter (bool): If true, include scatter points in the speed subplot; if false, only
            fit lines are shown.

    Returns:
        fig (go.Figure): A Plotly Figure with two subplots:
            - Left: "Valid runs" — Normalized time (0-1) vs valid runs percentage.
            - Right: "Inference speed: steps/s vs N" — Number of atoms vs steps per second on log scales.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=("Valid runs", "Inference speed: steps/s vs N"),
    )

    # Right panel: speed scaling
    speed_fig = build_speed_figure(df_in, color_map, show_scatter)
    for trace in speed_fig.data:
        fig.add_trace(trace, row=1, col=2)

    # Left panel: cumulative valid runs
    for method, df_model in df_in.groupby("method"):
        df_model_grp = df_model.drop_duplicates(["formula"])
        hist, bin_edges = np.histogram(df_model_grp["normalized_final_step"], bins=np.linspace(0, 1, 50))
        cumulative_population = np.cumsum(hist)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig.add_trace(
            go.Scatter(
                x=bin_centers[:-1],
                y=(cumulative_population[-1] - cumulative_population[:-1]) / 120 * 100,
                mode="lines",
                line=dict(color=color_map.get(method)),
                name=str(method),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="Normalized time", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Valid runs (%)", row=1, col=1)
    fig.update_xaxes(type="log", row=1, col=2, title_text="Number of atoms")
    fig.update_yaxes(type="log", row=1, col=2, title_text="Steps per second")
    fig.update_layout(height=520, width=1000)
    return fig


@st.cache_data
def build_npt_figure(df_in: pd.DataFrame, color_map: dict, show_scatter: bool) -> go.Figure:
    """Create a 1x2 subplot for NPT data: cumulative valid runs (left) and inference speed vs number of atoms (right).

    The left panel plots the cumulative percentage of valid runs over normalized time using per-method deduplicated formulas and scales the result by a constant divisor of 80. The right panel contains the inference speed plot (steps/s vs number of atoms) and delegates scatter/fit rendering to the speed-building routine.

    Parameters:
        df_in (pd.DataFrame): Input dataframe containing at least the columns `method`, `normalized_final_step`, and `formula`; also used by the speed plot (e.g., `natoms`, `steps_per_second`) when present.
        color_map (dict): Mapping from method name to an HTML/CSS color string used for traces.
        show_scatter (bool): If True, include scatter points in the right-hand inference speed panel; otherwise only show fit lines.

    Returns:
        go.Figure: A Plotly Figure with two subplots: left shows "Valid runs (%)" over normalized time, right shows "Steps per second" vs "Number of atoms" (both axes on log scale).
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=("Valid runs", "Inference speed: steps/s vs N"),
    )

    # Right panel: speed scaling
    speed_fig = build_speed_figure(df_in, color_map, show_scatter)
    for trace in speed_fig.data:
        fig.add_trace(trace, row=1, col=2)

    # Left panel: cumulative valid runs
    for method, df_model in df_in.groupby("method"):
        df_model_grp = df_model.drop_duplicates(["formula"])
        hist, bin_edges = np.histogram(df_model_grp["normalized_final_step"], bins=np.linspace(0, 1, 50))
        cumulative_population = np.cumsum(hist)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig.add_trace(
            go.Scatter(
                x=bin_centers[:-1],
                y=(cumulative_population[-1] - cumulative_population[:-1]) / 80 * 100,
                mode="lines",
                line=dict(color=color_map.get(method)),
                name=str(method),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="Normalized time", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Valid runs (%)", row=1, col=1)
    fig.update_xaxes(type="log", row=1, col=2, title_text="Number of atoms")
    fig.update_yaxes(type="log", row=1, col=2, title_text="Steps per second")
    fig.update_layout(height=520, width=1000)
    return fig


if df_nvt.empty and df_npt.empty:
    st.info("No data available to display for selected models.")
else:
    st.markdown("""
    ## Heating
    Isochoric-isothermal (NVT) MD simulations on RM24 structures, with temperature ramp from 300K to 3000K over 10 ps.
    """)

    show_scatter_nvt = st.toggle("Show scatter points", key="show_scatter_nvt", value=True)
    # Toggle for scatter points
    # show_scatter = vis.checkbox("Show scatter points", value=True)
    st.plotly_chart(
        build_nvt_figure(df_nvt, method_color_mapping, show_scatter_nvt),
        use_container_width=True,
    )

    st.markdown("""
    ## Compression
    Isothermal-isobaric (NPT) MD simulations on RM24 structures, with pressure ramp from 0 GPa to 500 GPa and temperature ramp from 300K to 3000K over 10 ps.
    """)

    show_scatter_npt = st.toggle("Show scatter points", key="show_scatter_npt", value=True)
    # Toggle for scatter points
    # show_scatter = vis.checkbox("Show scatter points", value=True)
    st.plotly_chart(
        build_npt_figure(df_npt, method_color_mapping, show_scatter_npt),
        use_container_width=True,
    )
