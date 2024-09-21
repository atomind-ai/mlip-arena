from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st

from mlip_arena.models import REGISTRY

DATA_DIR = Path("mlip_arena/tasks/combustion")


st.markdown("""
# Combustion
""")

st.markdown("### Methods")
container = st.container(border=True)
valid_models = [model for model, metadata in REGISTRY.items() if Path(__file__).stem in metadata.get("gpu-tasks", [])]

models = container.multiselect(
    "MLIPs", 
    valid_models, 
    ["MACE-MP(M)", "CHGNet", "M3GNet", "SevenNet", "ORB", "EquiformerV2(OC22)", "eSCN(OC20)"]
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

families = [REGISTRY[str(model)]["family"] for model in models]

dfs = [
    pd.read_json(DATA_DIR / family.lower() / "hydrogen.json")
    for family in families
]
df = pd.concat(dfs, ignore_index=True)
df.drop_duplicates(inplace=True, subset=["formula", "method"])

method_color_mapping = {
    method: color_sequence[i % len(color_sequence)]
    for i, method in enumerate(df["method"].unique())
}

###

# Number of products
fig = go.Figure()

for method in df["method"].unique():
    row = df[df["method"] == method].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=row["timestep"],
            y=row["nproducts"],
            mode="lines",
            name=method,
            line=dict(color=method_color_mapping[method]),
            showlegend=True,
        ),
    )

fig.update_layout(
    title="Hydrogen Combustion (2H2 + O2 -> 2H2O, 64 units)",
    xaxis_title="Timestep",
    yaxis_title="Number of water molecules",
)

st.plotly_chart(fig)

# tempearture

fig = go.Figure()

for method in df["method"].unique():
    row = df[df["method"] == method].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=row["timestep"],
            y=row["temperatures"],
            mode="markers",
            name=method,
            line=dict(color=method_color_mapping[method]),
            showlegend=True,
        ),
    )

target_steps = df["target_steps"].iloc[0]
fig.add_trace(
    go.Line(
        x=[0, target_steps/3, target_steps/3*2, target_steps],
        y=[300, 3000, 3000, 300],
        mode="lines",
        name="Target",
        line=dict(
            dash="dash",
        ),
        showlegend=True,
    ),
)

fig.update_layout(
    title="Hydrogen Combustion (2H2 + O2 -> 2H2O, 64 units)",
    xaxis_title="Timestep",
    yaxis_title="Temperatures",
    yaxis2=dict(
        title="Product Percentage (%)",
        overlaying="y",
        side="right",
        range=[0, 100],
        tickmode="sync"
    )
    # template="plotly_dark",
)

st.plotly_chart(fig)

# Final reaction rate

fig = go.Figure()

# df["yield"] = np.array(df["nproducts"]) / 128 * 100

df = df.sort_values("yield", ascending=True)

fig.add_trace(
    go.Bar(
        x=df["yield"] * 100,
        y=df["method"],
        opacity=0.75,
        orientation="h",
        marker=dict(color=[method_color_mapping[method] for method in df["method"]]),
        text=[f"{y:.2f} %" for y in df["yield"] * 100],
    )
)

fig.update_layout(
    title="Reaction yield (2H2 + O2 -> 2H2O, 64 units)",
    xaxis_title="Yield (%)",
    yaxis_title="Method",
)

st.plotly_chart(fig)

# MD runtime speed

fig = go.Figure()

df = df.sort_values("steps_per_second", ascending=True)

fig.add_trace(
    go.Bar(
        x=df["steps_per_second"],
        y=df["method"],
        opacity=0.75,
        orientation="h",
        marker=dict(color=[method_color_mapping[method] for method in df["method"]]),
        text=df["steps_per_second"].round(1)
    )
)

fig.update_layout(
    title="MD runtime speed (on single A100 GPU)",
    xaxis_title="Steps per second",
    yaxis_title="Method",
)

st.plotly_chart(fig)
