from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mlip_arena.models import REGISTRY

DATA_DIR = Path("mlip_arena/tasks/stability")


st.markdown("# Stability")

st.markdown("### Methods")
container = st.container(border=True)
models = container.multiselect("MLIPs", REGISTRY.keys(), ['MACE-MP(M)', "CHGNet", "EquiformerV2(OC22)"])

st.markdown("### Settings")
vis = st.container(border=True)
# Get all attributes from pcolors.qualitative
all_attributes = dir(pcolors.qualitative)
color_palettes = {attr: getattr(pcolors.qualitative, attr) for attr in all_attributes if isinstance(getattr(pcolors.qualitative, attr), list)}
color_palettes.pop("__all__", None)

palette_names = list(color_palettes.keys())
palette_colors = list(color_palettes.values())

palette_name = vis.selectbox(
    "Color sequence",
    options=palette_names, index=22
)

color_sequence = color_palettes[palette_name]

if not models:
    st.stop()

families = [REGISTRY[str(model)]['family'] for model in models]

dfs = [pd.read_json(DATA_DIR / family.lower() /  "chloride-salts.json") for family in families]
df = pd.concat(dfs, ignore_index=True)
df.drop_duplicates(inplace=True, subset=["material_id", "formula", "method"])

method_color_mapping = {method: color_sequence[i % len(color_sequence)] for i, method in enumerate(df["method"].unique())}



# fig = px.scatter(df, x="natoms", y="seconds_per_step", trendline="ols", trendline_options=dict(log_y=True), log_y=True)
fig = px.scatter(
    df, x="natoms", y="steps_per_second", 
    color="method",
    color_discrete_map=method_color_mapping,
    trendline="ols", trendline_options=dict(log_x=True), log_x=True
)


event = st.plotly_chart(
    fig, 
    key="stability", 
    on_select="rerun"
)

event


