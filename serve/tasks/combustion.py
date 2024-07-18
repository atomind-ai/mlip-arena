from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit

from mlip_arena.models import REGISTRY

DATA_DIR = Path("mlip_arena/tasks/combustion")


st.markdown("""
# Combustion
""")

st.markdown("### Methods")
container = st.container(border=True)
valid_models = [model for model, metadata in REGISTRY.items() if Path(__file__).stem in metadata.get("gpu-tasks", [])]

models = container.multiselect("MLIPs", valid_models, ["MACE-MP(M)", "CHGNet", "EquiformerV2(OC22)"])

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
            x=row["timesteps"],
            y=row["nproducts"],
            mode='lines',
            name=method,
            line=dict(color=method_color_mapping[method]),
            showlegend=True,
        ),
    )

fig.update_layout(
    title="Hydrogen Combusiton (2H2 + O2 -> 2H2O, 64 units)",
    xaxis_title="Timesteps",
    yaxis_title="Number of water molecules",
)

st.plotly_chart(fig)

# tempearture 

fig = go.Figure()

for method in df["method"].unique():
    row = df[df["method"] == method].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=row["timesteps"],
            y=row["temperatures"],
            mode='markers',
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
        mode='lines',
        name="Target",
        line=dict(
            dash="dash",
        ),
        showlegend=True,
    ),
)

fig.update_layout(
    title="Hydrogen Combusiton (2H2 + O2 -> 2H2O, 64 units)",
    xaxis_title="Timesteps",
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
