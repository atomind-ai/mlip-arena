from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit

from mlip_arena.models import REGISTRY

DATA_DIR = Path("mlip_arena/tasks/stability")


st.markdown("""
# High Pressure Stability

Stable and accurate molecular dynamics (MD) simulations are important for understanding the properties of matters.
However, many MLIPs have unphysical potential energy surface (PES) at the short-range interatomic distances or under many-body effect. These are often manifested as softened repulsion and hole in the PES and can lead to incorrect and sampling of the phase space.

Here, we analyze the stability of the MD simulations under high pressure conditions by gradually increasing the pressure from 0 to 1000 GPa at 300K until the system crashes or completes 100 ps trajectory. This benchmark also explores faster the far-from-equilibrium dynamics of the system and the durability of the MLIPs under extreme conditions.
""")

st.markdown("### Methods")
container = st.container(border=True)
valid_models = [
    model
    for model, metadata in REGISTRY.items()
    if Path(__file__).stem in metadata.get("gpu-tasks", [])
]

models = container.multiselect(
    "MLIPs", valid_models, ["MACE-MP(M)", "CHGNet", "ORB", "SevenNet"]
)

st.markdown("### Settings")
vis = st.container(border=True)
# Get all attributes from pcolors.qualitative
all_attributes = dir(pcolors.qualitative)
color_palettes = {
    attr: getattr(pcolors.qualitative, attr)
    for attr in all_attributes
    if isinstance(getattr(pcolors.qualitative, attr), list)
}
color_palettes.pop("__all__", None)

palette_names = list(color_palettes.keys())
palette_colors = list(color_palettes.values())

palette_name = vis.selectbox("Color sequence", options=palette_names, index=22)

color_sequence = color_palettes[palette_name]

if not models:
    st.stop()


@st.cache_data
def get_data(models):
    families = [REGISTRY[str(model)]["family"] for model in models]

    dfs = [
        pd.read_json(DATA_DIR / family.lower() / "chloride-salts.json")
        for family in families
    ]
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(inplace=True, subset=["material_id", "formula", "method"])

    return df


df = get_data(models)

method_color_mapping = {
    method: color_sequence[i % len(color_sequence)]
    for i, method in enumerate(df["method"].unique())
}

###

# Determine the bin edges for the entire dataset to keep them consistent across groups

max_steps = df["total_steps"].max()
max_target_steps = df["target_steps"].max()

bins = np.append(np.arange(0, max_steps + 1, max_steps // 10), max_target_steps)
bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

num_bins = len(bin_labels)
colormap = px.colors.sequential.YlOrRd_r
indices = np.linspace(0, len(colormap) - 1, num_bins, dtype=int)
bin_colors = [colormap[i] for i in indices]

# Initialize a dictionary to hold the counts for each method and bin range
counts_per_method = {method: [0] * len(bin_labels) for method in df["method"].unique()}


# Populate the dictionary with counts
for method, group in df.groupby("method"):
    counts, _ = np.histogram(group["total_steps"], bins=bins)
    counts_per_method[method] = counts

# Sort the dictionary by the percentage of the last bin
counts_per_method = {
    k: v
    for k, v in sorted(
        counts_per_method.items(), key=lambda item: item[1][-1] / sum(item[1])
    )
}


count_or_percetange = st.toggle("show counts", False)


@st.experimental_fragment()
def plot_md_steps(counts_per_method, count_or_percetange):
    """Plot the distribution of the total number of MD steps before crash or completion."""
    # Create a figure
    fig = go.Figure()

    # Add a bar for each bin range across all methods
    for i, bin_label in enumerate(bin_labels):
        for method, counts in counts_per_method.items():
            fig.add_trace(
                go.Bar(
                    # name=method,  # This will be the legend entry
                    x=[counts[i] / counts.sum() * 100]
                    if not count_or_percetange
                    else [counts[i]],
                    y=[method],  # Method as the y-axis category
                    # name=bin_label,
                    orientation="h",  # Horizontal bars
                    marker=dict(
                        color=bin_colors[i],
                        line=dict(color="rgb(248, 248, 249)", width=1),
                    ),
                    text=f"{bin_label}: {counts[i]/counts.sum()*100:.0f}%",
                    width=0.5,
                )
            )

    # Update the layout to stack the bars
    fig.update_layout(
        barmode="stack",  # Stack the bars
        title="Total MD steps (before crash or completion)",
        xaxis_title="Percentage (%)" if not count_or_percetange else "Count",
        yaxis_title="Method",
        showlegend=False,
    )

    st.plotly_chart(fig)


plot_md_steps(counts_per_method, count_or_percetange)

st.caption(
"""
The histogram shows the distribution of the total number of MD steps before the system crashes or completes the trajectory. :red[The color of the bins indicates the number of steps in the bin]. :blue[The height of the bars indicates the number or percentage of each bin among all the runs].
"""
)

###

st.markdown(
"""
## Inference speed

The inference speed of the MLIPs is crucial for the high-throughput virutal screening. Under high pressure conditions, the atoms often move faster and closer to each other, which increases the size of neighbor list and local graph construction and hence slows down the inference speed.
"""
)


def func(x, a, n):
    return a * x ** (-n)


@st.experimental_fragment()
def plot_speed(df, method_color_mapping):
    """Plot the inference speed as a function of the number of atoms."""
    fig = px.scatter(
        df,
        x="natoms",
        y="steps_per_second",
        color="method",
        size="total_steps",
        hover_data=["material_id", "formula"],
        color_discrete_map=method_color_mapping,
        # trendline="ols",
        # trendline_options=dict(log_x=True),
        log_x=True,
        # log_y=True,
        # range_y=[1, 1e2],
        range_x=[df["natoms"].min() * 0.9, df["natoms"].max() * 1.1],
        # range_x=[1e3, 1e2],
        title="Inference speed (on single A100 GPU)",
        labels={"steps_per_second": "Steps per second", "natoms": "Number of atoms"},
    )

    x = np.linspace(df["natoms"].min(), df["natoms"].max(), 100)

    for method, data in df.groupby("method"):
        data.dropna(subset=["steps_per_second"], inplace=True)
        popt, pcov = curve_fit(func, data["natoms"], data["steps_per_second"])

        fig.add_trace(
            go.Scatter(
                x=x,
                y=func(x, *popt),
                mode="lines",
                # name='Fit',
                line=dict(color=method_color_mapping[method], width=3),
                showlegend=False,
                name=f"{popt[0]:.2f}N^{-popt[1]:.2f}",
                hovertext=f"{popt[0]:.2f}N^{-popt[1]:.2f}",
            )
        )

    st.plotly_chart(fig)


plot_speed(df, method_color_mapping)

st.caption(
"""
The plot shows the inference speed (steps per second) as a function of the number of atoms in the system. :red[The size of the points is proportional to the total number of steps in the MD trajectory before crash or completion (~49990)]. :blue[The lines show the fit of the data to the power law function $a N^{-n}$], where $N$ is the number of atoms and $a$ and $n$ are the fit parameters.
"""
)
