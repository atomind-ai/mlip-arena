from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from ase.data import chemical_symbols
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline

color_sequence = pcolors.qualitative.Plotly



st.markdown("# Homonuclear diatomics")

# button to toggle plots
container = st.container(border=True)
energy_plot = container.checkbox("Show energy curves", value=True)
force_plot = container.checkbox("Show force curves", value=True)

ncols = 2

DATA_DIR = Path("mlip_arena/tasks/diatomics")
mlips = ["MACE-MP", "CHGNet"]

dfs = [pd.read_json(DATA_DIR / mlip.lower() /  "homonuclear-diatomics.json") for mlip in mlips]
df = pd.concat(dfs, ignore_index=True)



df.drop_duplicates(inplace=True, subset=["name", "method"])

for i, symbol in enumerate(chemical_symbols[1:]):

    if i % ncols == 0:
        cols = st.columns(ncols)


    rows = df[df["name"] == symbol + symbol]

    if rows.empty:
        continue

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    elo, flo = float("inf"), float("inf")

    for j, method in enumerate(rows["method"].unique()):
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
        es = es - es[-1]
        fs = fs[ind]

        xs = np.linspace(rs.min()*0.99, rs.max()*1.01, int(5e2))

        if energy_plot:
            cs = CubicSpline(rs, es)
            ys = cs(xs)

            elo = min(elo, ys.min()*1.2, -1)

            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    line=dict(
                        color=color_sequence[j % len(color_sequence)],
                        width=2,
                    ),
                    name=method,
                ),
                secondary_y=False,
            )

        if force_plot:
            cs = CubicSpline(rs, fs)
            ys = cs(xs)

            flo = min(flo, ys.min()*1.2)

            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    line=dict(
                        color=color_sequence[j % len(color_sequence)],
                        width=1,
                        dash="dot",
                    ),
                    name=method,
                    showlegend=False if energy_plot else True,
                ),
                secondary_y=True,
            )


    fig.update_layout(
        showlegend=True,
        title_text=f"{symbol}-{symbol}",
        title_x=0.5,
        # yaxis_range=[ylo, 2*(abs(ylo))],
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Bond length (Å)")

    # Set y-axes titles
    if energy_plot:

        fig.update_layout(
            yaxis=dict(
                title=dict(text="Energy [eV]"),
                side="left",
                range=[elo, 2*(abs(elo))],
            )
        )

        # fig.update_yaxes(title_text="Energy [eV]", secondary_y=False)

    if force_plot:

        fig.update_layout(
            yaxis2=dict(
                title=dict(text="Force [eV/Å]"),
                side="right",
                range=[flo, 2*(abs(flo))],
                overlaying="y",
                tickmode="sync",
            ),
        )

        # fig.update_yaxes(title_text="Force [eV/Å]", secondary_y=True)

    # cols[i % ncols].title(f"{row['name']}")
    cols[i % ncols].plotly_chart(fig, use_container_width=True, height=250)
