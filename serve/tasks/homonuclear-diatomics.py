from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from ase.data import chemical_symbols
from plotly.subplots import make_subplots

from mlip_arena.models import REGISTRY

st.markdown(
    """
# Homonuclear Diatomics

Homonuclear diatomics are molecules composed of two atoms of the same element.
The potential energy curves of homonuclear diatomics are the most fundamental interactions between atoms in quantum chemistry.
"""
)

st.markdown("### Methods")
container = st.container(border=True)
valid_models = [
    model
    for model, metadata in REGISTRY.items()
    if Path(__file__).stem in metadata.get("gpu-tasks", [])
]
mlip_methods = container.multiselect(
    "MLIPs",
    valid_models,
    ["MACE-MP(M)", "MatterSim", "SevenNet", "ORBv2", "eqV2(OMat)", "ANI2x", "eSEN"],
)
dft_methods = container.multiselect("DFT Methods", ["PBE"], ["PBE"])

container.info(
    "PBE energies and forces are provided __only__ for reference. Due to the known convergence issue of plane-wave DFT with diatomic molecules and different dataset the models might be trained on, comparing models with PBE is not rigorous and thus these metrics are excluded from rank aggregation.",
    icon=":material/warning:",
)

st.markdown("### Settings")
vis = st.container(border=True)
energy_plot = vis.checkbox("Show energy curves", value=True)
force_plot = vis.checkbox("Show force curves", value=False)
ncols = vis.select_slider("Number of columns", options=[1, 2, 3, 4], value=2)

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

color_sequence = color_palettes[palette_name]  # type: ignore
if not mlip_methods and not dft_methods:
    st.stop()


@st.cache_data
def get_data(mlip_methods, dft_methods):
    DATA_DIR = Path("mlip_arena/tasks/diatomics")

    dfs = [
        pd.read_json(
            DATA_DIR / REGISTRY[method]["family"] / "homonuclear-diatomics.json"
        )
        for method in mlip_methods
    ]
    dfs.extend(
        [
            pd.read_json(DATA_DIR / "vasp" / "homonuclear-diatomics.json")
            # for method in dft_methods
        ]
    )
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(inplace=True, subset=["name", "method"])
    return df


df = get_data(mlip_methods, dft_methods)

method_color_mapping = {
    method: color_sequence[i % len(color_sequence)]
    for i, method in enumerate(df["method"].unique())
}


@st.cache_data
def get_plots(df, energy_plot: bool, force_plot: bool, method_color_mapping: dict):
    figs = []

    for i, symbol in enumerate(chemical_symbols[1:]):
        rows = df[df["name"] == symbol + symbol]

        if rows.empty:
            continue

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        elo, flo = float("inf"), float("inf")

        for j, method in enumerate(rows["method"].unique()):
            if method not in mlip_methods and method not in dft_methods:
                continue
            row = rows[rows["method"] == method].iloc[0]

            rs = np.array(row["R"])
            es = np.array(row["E"])
            fs = np.array(row["F"])

            rs = np.array(rs)
            ind = np.argsort(rs)
            es = np.array(es)
            fs = np.array(fs)

            rs = rs[ind]
            es = es[ind]
            fs = fs[ind]

            # if method not in ["PBE"]:
            es = es - es[-1]

            # if method in ["PBE"]:
            #     xs = np.linspace(rs.min() * 0.99, rs.max() * 1.01, int(5e2))
            # else:
            xs = rs

            if energy_plot:
                # if "GPAW" in method:
                #     cs = CubicSpline(rs, es)
                #     ys = cs(xs)
                # else:
                ys = es

                elo = min(elo, max(ys.min() * 1.2, -15), -1)

                if method in ["PBE"]:
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="markers",
                            line=dict(
                                color=method_color_mapping[method],
                                width=3,
                            ),
                            name=method,
                        ),
                        secondary_y=False,
                    )
                    # xs = np.linspace(rs.min() * 0.99, rs.max() * 1.01, int(5e2))
                    # cs = CubicSpline(rs, es)
                    # ys = cs(xs)
                    # fig.add_trace(
                    #     go.Scatter(
                    #         x=xs,
                    #         y=ys,
                    #         mode="lines",
                    #         line=dict(
                    #             color=method_color_mapping[method],
                    #             width=3,
                    #         ),
                    #         name=method,
                    #         showlegend=False,
                    #     ),
                    #     secondary_y=False,
                    # )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line=dict(
                                color=method_color_mapping[method],
                                width=3,
                            ),
                            name=method,
                        ),
                        secondary_y=False,
                    )

            # if force_plot and method not in ["PBE"]:
            if force_plot:
                ys = fs

                flo = min(flo, max(ys.min() * 1.2, -50))

                if method in ["PBE"]:
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines+markers",
                            line=dict(
                                color=method_color_mapping[method],
                                width=2,
                                dash="dashdot",
                            ),
                            name=method,
                            showlegend=not energy_plot,
                        ),
                        secondary_y=True,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line=dict(
                                color=method_color_mapping[method],
                                width=2,
                                dash="dashdot",
                            ),
                            name=method,
                            showlegend=not energy_plot,
                        ),
                        secondary_y=True,
                    )

        name = f"{symbol}-{symbol}"

        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                x=1.0,
                xanchor="right",
                y=1,
                yanchor="top",
                bgcolor="rgba(0, 0, 0, 0)",
                # traceorder='reversed',
                entrywidth=0.4,
                entrywidthmode="fraction",
            ),
            title_text=f"{name}",
            title_x=0.5,
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Distance [Å]")

        # Set y-axes titles
        if energy_plot:
            fig.update_layout(
                yaxis=dict(
                    title=dict(text="Energy [eV]"),
                    side="left",
                    range=[elo, 2.0 * (abs(elo))],
                )
            )

        if force_plot:
            fig.update_layout(
                yaxis2=dict(
                    title=dict(text="Force [eV/Å]"),
                    side="right",
                    range=[flo, 1.0 * abs(flo)],
                    overlaying="y",
                    tickmode="sync",
                ),
            )

        # cols[i % ncols].plotly_chart(fig, use_container_width=True)

        figs.append(fig)

    return figs
    # fig.write_image(format='svg', file=img_dir / f"{name}.svg")


figs = get_plots(df, energy_plot, force_plot, method_color_mapping)

for i, fig in enumerate(figs):
    if i % ncols == 0:
        cols = st.columns(ncols)
    cols[i % ncols].plotly_chart(fig, use_container_width=True)
