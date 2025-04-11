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
        "ORBv2",
        "EquiformerV2(OC20)",
        "eSCN(OC20)",
        "MatterSim",
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
    # List comprehension for concise looping and filtering
    dfs = [
        pd.read_json(DATA_DIR / MODELS[str(model)]["family"].lower() / "hydrogen.json")[
            lambda df: df["method"] == model
        ]
        for model in models
    ]
    # Concatenate all filtered DataFrames
    return pd.concat(dfs, ignore_index=True)


df = get_data(models)

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

fig.add_vrect(
    x0=512345.94,
    x1=666667,
    fillcolor="lightblue",
    opacity=0.2,
    layer="below",
    line_width=0,
    annotation_text="Flame Temp. [1]",
    annotation_position="top",
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

fig.add_vrect(
    x0=512345.94,
    x1=666667,
    fillcolor="lightblue",
    opacity=0.2,
    layer="below",
    line_width=0,
    annotation_text="Flame Temp.",
    annotation_position="top",
)

fig.update_layout(
    # title="Hydrogen Combustion (2H2 + O2 -> 2H2O, 64 units)",
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

exp_ref = -68.3078  # kcal/mol
factor = 23.0609
nh2os = 128

fig = go.Figure()

for method in df["method"].unique():
    row = df[df["method"] == method].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=row["timestep"],
            y=(np.array(row["energies"]) - row["energies"][0]) / nh2os * factor,
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

target_steps = df["target_steps"].iloc[0]

fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=0,
        x1=target_steps,
        y0=exp_ref,
        y1=exp_ref,  # y-values for the horizontal line
        line=dict(color="Red", width=2, dash="dash"),
        layer="below",
    )
)

fig.add_annotation(
    go.layout.Annotation(
        x=0.5,
        xref="paper",
        xanchor="center",
        y=exp_ref,
        yanchor="bottom",
        text=f"Experiment: {exp_ref} kcal/mol [2]",
        showarrow=False,
        font=dict(
            color="Red",
        ),
    )
)

fig.add_vrect(
    x0=512345.94,
    x1=666667,
    fillcolor="lightblue",
    opacity=0.2,
    layer="below",
    line_width=0,
    annotation_text="Flame Temp.",
    annotation_position="top",
)


fig.update_layout(
    xaxis_title="Timestep <br> <span style='font-size: 10px;'>[2] Lide, D. R. (Ed.). (2004). CRC handbook of chemistry and physics (Vol. 85). CRC press.</span>",
    yaxis_title="𝚫E (kcal/mol)",
)

st.plotly_chart(fig)

# Reaction energy

fig = go.Figure()


df["reaction_energy"] = (
    df["energies"].apply(lambda x: x[-1] - x[0]) / nh2os * factor
)  # kcal/mol

df["reaction_energy_abs_err"] = np.abs(df["reaction_energy"] - exp_ref)

df.sort_values("reaction_energy_abs_err", inplace=True)

fig.add_traces(
    [
        go.Bar(
            x=df["method"],
            y=df["reaction_energy"],
            marker=dict(
                color=[method_color_mapping[method] for method in df["method"]]
            ),
            text=[f"{y:.2f}" for y in df["reaction_energy"]],
        ),
    ]
)

fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=-0.5,
        x1=len(df["method"]) - 0.5,  # range covering the bars
        y0=exp_ref,
        y1=exp_ref,  # y-values for the horizontal line
        line=dict(color="Red", width=2, dash="dash"),
        layer="below",
    )
)

fig.add_annotation(
    go.layout.Annotation(
        x=0.5,
        xref="paper",
        xanchor="center",
        y=exp_ref,
        yanchor="bottom",
        text=f"Experiment: {exp_ref} kcal/mol [2]",
        showarrow=False,
        font=dict(
            color="Red",
        ),
    )
)

fig.update_layout(
    xaxis_title="Method <br> <span style='font-size: 10px;'>[1] Lide, D. R. (Ed.). (2004). CRC handbook of chemistry and physics (Vol. 85). CRC press.</span>",
    yaxis_title="Reaction energy 𝚫H (kcal/mol)",
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

fig = go.Figure()

for method in df_exploded["method"].unique():
    row = df_exploded[df_exploded["method"] == method]
    fig.add_trace(
        go.Scatter(
            x=row["timestep"],
            y=row["total_com_drift"],
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

fig.update_yaxes(type="log")
fig.update_layout(
    xaxis_title="Timestep",
    yaxis_title="Total COM drift (Å)",
)

st.plotly_chart(fig)

if "play" not in st.session_state:
    st.session_state.play = False


def toggle_playing():
    st.session_state.play = not st.session_state.play


# st.button(
#     "Play" if not st.session_state.play else "Pause",
#     type="primary" if not st.session_state.play else "secondary",
#     on_click=toggle_playing,
# )

increment = df["target_steps"].max() // 200

if "time_range" not in st.session_state:
    st.session_state.time_range = (0, increment)


# @st.experimental_fragment(run_every=1e-3 if st.session_state.play else None)
@st.experimental_fragment()
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
            "com_drift_x": "𝚫x (Å)",
            "com_drift_y": "𝚫y (Å)",
            "com_drift_z": "𝚫z (Å)",
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


st.markdown("""
### References

[1] Hasche, A., Navid, A., Krause, H., & Eckart, S. (2023). Experimental and numerical assessment of the effects of hydrogen admixtures on premixed methane-oxygen flames. Fuel, 352, 128964.
            
[2] Lide, D. R. (Ed.). (2004). CRC handbook of chemistry and physics (Vol. 85). CRC press.
""")
