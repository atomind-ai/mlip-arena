from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mlip_arena.models import REGISTRY

DATA_DIR = Path("mlip_arena/tasks/stability")


st.markdown(
    """
# Stability

"""
)

st.markdown("### Methods")
container = st.container(border=True)
models = container.multiselect("MLIPs", REGISTRY.keys(), ["MACE-MP(M)", "CHGNet"])

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
    pd.read_json(DATA_DIR / family.lower() / "chloride-salts.json")
    for family in families
]
df = pd.concat(dfs, ignore_index=True)
df.drop_duplicates(inplace=True, subset=["material_id", "formula", "method"])

method_color_mapping = {
    method: color_sequence[i % len(color_sequence)]
    for i, method in enumerate(df["method"].unique())
}


fig = px.scatter(
    df,
    x="natoms",
    y="steps_per_second",
    color="method",
    size="total_steps",
    hover_data=["material_id", "formula"],
    color_discrete_map=method_color_mapping,
    trendline="ols",
    trendline_options=dict(log_x=True),
    log_x=True,
    title="Inference Speed",
    labels={"steps_per_second": "Steps per second", "natoms": "Number of atoms"},
)
st.plotly_chart(fig)

###

fig = go.Figure()

# Determine the bin edges for the entire dataset to keep them consistent across groups
# bins = np.histogram_bin_edges(df['total_steps'], bins=10)

max_steps = df["total_steps"].max()

bins = np.append(np.arange(0, max_steps - 1, max_steps // 10), max_steps)
bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

num_bins = len(bin_labels)
colormap = px.colors.sequential.Redor
indices = np.linspace(0, len(colormap) - 1, num_bins, dtype=int)
bin_colors = [colormap[i] for i in indices]

# Initialize a dictionary to hold the counts for each method and bin range
counts_per_method = {method: [0] * len(bin_labels) for method in df['method'].unique()}

# Populate the dictionary with counts
for method, group in df.groupby('method'):
    counts, _ = np.histogram(group['total_steps'], bins=bins)
    counts_per_method[method] = counts

# Create a figure
fig = go.Figure()

# Add a bar for each bin range across all methods
for i, bin_label in enumerate(bin_labels):
    for method, counts in counts_per_method.items():
        fig.add_trace(go.Bar(
            # name=method,  # This will be the legend entry
            x=[counts[i]],  # Count for this bin
            y=[method],  # Method as the y-axis category
            # name=bin_label,
            orientation='h',  # Horizontal bars
            marker=dict(
                color=bin_colors[i],
                line=dict(color='rgb(248, 248, 249)', width=1)
            ),
            text=bin_label,
            width=0.5
        ))

# Update the layout to stack the bars
fig.update_layout(
    barmode='stack',  # Stack the bars
    title="Total MD Steps",
    xaxis_title="Count",
    yaxis_title="Method",
    showlegend=False
)

# bins = np.linspace(0, 0.9, 10)

# for method, data in df.groupby("method"):

#     # print(method, data)
#     counts, bins = np.histogram(data['total_steps'])

#     bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]

#     # Create a horizontal bar chart
#     fig = go.Figure(go.Bar(
#         x=[counts[i]],  # Count for this bin
#         y=[method],  # Method as the y-axis category
#         # x=counts,  # Bar lengths
#         # y=bin_labels,  # Bin labels as y-tick labels
#         orientation='h'  # Horizontal bars
#     ))


# # Update layout for clarity
# fig.update_layout(
#     title="Histogram of Total Steps",
#     xaxis_title="Count",
#     yaxis_title="Total Steps Range"
# )

st.plotly_chart(fig)
