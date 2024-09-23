from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mlip_arena.models import REGISTRY as MODELS

DATA_DIR = Path("mlip_arena/tasks/combustion")


st.markdown("""
# Combustion
""")

st.markdown("### Methods")
container = st.container(border=True)
valid_models = [
    model
    for model, metadata in MODELS.items()
    if Path(__file__).stem in metadata.get("gpu-tasks", [])
]

models = container.multiselect(
    "MLIPs",
    valid_models,
    [
        "MACE-MP(M)",
        "CHGNet",
        "M3GNet",
        "SevenNet",
        "ORB",
        "EquiformerV2(OC22)",
        "eSCN(OC20)",
    ],
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
    families = [MODELS[str(model)]["family"] for model in models]
    dfs = [
        pd.read_json(DATA_DIR / family.lower() / "hydrogen.json") for family in families
    ]
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(inplace=True, subset=["formula", "method"])
    return df


df = get_data(models)

# families = [MODELS[str(model)]["family"] for model in models]

# dfs = [pd.read_json(DATA_DIR / family.lower() / "hydrogen.json") for family in families]
# df = pd.concat(dfs, ignore_index=True)
# df.drop_duplicates(inplace=True, subset=["formula", "method"])

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
            line=dict(
                color=method_color_mapping[method],
                # width=1
            ),
            marker=dict(color=method_color_mapping[method], size=3),
            showlegend=True,
        ),
    )

target_steps = df["target_steps"].iloc[0]
fig.add_trace(
    go.Line(
        x=[0, target_steps / 3, target_steps / 3 * 2, target_steps],
        y=[300, 3000, 3000, 300],
        mode="lines",
        name="Target",
        line=dict(dash="dash", color="white"),
        showlegend=True,
    ),
)

fig.update_layout(
    title="Hydrogen Combustion (2H2 + O2 -> 2H2O, 64 units)",
    xaxis_title="Timestep",
    yaxis_title="Temperature (K)",
    # yaxis2=dict(
    #     title="Product Percentage (%)",
    #     overlaying="y",
    #     side="right",
    #     range=[0, 100],
    #     tickmode="sync",
    # ),
    # template="plotly_dark",
)

st.plotly_chart(fig)

# Energy

fig = go.Figure()

for method in df["method"].unique():
    row = df[df["method"] == method].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=row["timestep"],
            y=np.array(row["energies"]) - row["energies"][0],
            mode="lines",
            name=method,
            line=dict(
                color=method_color_mapping[method],
                # width=1
            ),
            marker=dict(color=method_color_mapping[method], size=3),
            showlegend=True,
        ),
    )

fig.update_layout(
    title="Hydrogen Combustion (2H2 + O2 -> 2H2O, 64 units)",
    xaxis_title="Timestep",
    yaxis_title="ΔE (eV)",
    # template="plotly_dark",
)

st.plotly_chart(fig)

# Final reaction rate

fig = go.Figure()

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
        text=df["steps_per_second"].round(1),
    )
)

fig.update_layout(
    title="MD runtime speed (on single A100 GPU)",
    xaxis_title="Steps per second",
    yaxis_title="Method",
)

st.plotly_chart(fig)

# COM drift

st.markdown("""### Center of mass drift

The center of mass (COM) drift is a measure of the stability of the simulation. A well-behaved simulation should have a COM drift close to zero. The COM drift is calculated as the displacement of the COM of the system from the initial position.
""")


@st.cache_data
def get_com_drifts(df):
    df_exploded = df.explode(["timestep", "com_drifts"]).reset_index(drop=True)

    # Convert the 'com_drifts' column (which are arrays) into separate columns for x, y, and z components
    df_exploded[["com_drift_x", "com_drift_y", "com_drift_z"]] = pd.DataFrame(
        df_exploded["com_drifts"].tolist(), index=df_exploded.index
    )

    # Drop the original 'com_drifts' column
    df_flat = df_exploded.drop(columns=["com_drifts"])

    df_flat["total_com_drift"] = np.sqrt(
        df_flat["com_drift_x"] ** 2
        + df_flat["com_drift_y"] ** 2
        + df_flat["com_drift_z"] ** 2
    )

    return df_flat


df_exploded = get_com_drifts(df)

if "play" not in st.session_state:
    st.session_state.play = False


def toggle_playing():
    st.session_state.play = not st.session_state.play


st.button(
    "Play" if not st.session_state.play else "Pause",
    type="primary" if not st.session_state.play else "secondary",
    on_click=toggle_playing,
)

increment = df["target_steps"].max() // 200

if "time_range" not in st.session_state:
    st.session_state.time_range = (0, increment)


@st.experimental_fragment(run_every=1e-3 if st.session_state.play else None)
def draw_com_drifts_plot():
    if st.session_state.play:
        start, end = st.session_state.time_range

        end += increment

        if end > df["target_steps"].max():
            start = 0
            end = 0

        st.session_state.time_range = (start, end)

    start_timestep, end_timestep = st.slider(
        "Timestep",
        min_value=0,
        max_value=df["target_steps"].max(),
        value=st.session_state.time_range,
        key="time_range",
        # on_change=check_range,
    )

    mask = (df_exploded["timestep"] >= start_timestep) & (
        df_exploded["timestep"] <= end_timestep
    )
    df_filtered = df_exploded[mask]
    df_filtered.sort_values(["method", "timestep"], inplace=True)

    fig = px.line_3d(
        data_frame=df_filtered,
        x="com_drift_x",
        y="com_drift_y",
        z="com_drift_z",
        labels={
            "com_drift_x": "Δx (Å)",
            "com_drift_y": "Δy (Å)",
            "com_drift_z": "Δz (Å)",
        },
        category_orders={"method": df_exploded["method"].unique()},
        color_discrete_sequence=[
            method_color_mapping[method] for method in df_exploded["method"].unique()
        ],
        color="method",
        width=800,
        height=800,
    )

    fig.update_layout(
        scene=dict(
            aspectmode="cube",
        ),
        legend=dict(
            orientation="v",
            x=0.95,
            xanchor="right",
            y=1,
            yanchor="top",
            bgcolor="rgba(0, 0, 0, 0)",
        ),
    )
    fig.add_traces(
        [
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                marker=dict(size=3, color="white"),
                name="origin",
            ),
            # add last point of each method and annotate the total drift
            go.Scatter3d(
                # df_filtered.groupby("method")["com_drift_x"].last(),
                x=df_filtered.groupby("method")["com_drift_x"].last(),
                y=df_filtered.groupby("method")["com_drift_y"].last(),
                z=df_filtered.groupby("method")["com_drift_z"].last(),
                mode="markers+text",
                marker=dict(size=3, color="white", opacity=0.5),
                text=df_filtered.groupby("method")["total_com_drift"].last().round(3),
                # size=5,
                name="total drifts",
                textposition="top center",
            ),
        ]
    )

    st.plotly_chart(fig)


draw_com_drifts_plot()
