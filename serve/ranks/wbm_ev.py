from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("benchmarks/wbm_ev")


table = pd.read_csv(DATA_DIR / "summary.csv")



table = table.rename(
    columns={
        "model": "Model",
        "rank": "Rank",
        "rank-aggregation": "Rank aggr.",
        "energy-diff-flip-times": "Derivative flips",
        "tortuosity": "Tortuosity",
        "spearman-compression-energy": "Spearman's coeff. (compression)",
        "spearman-tension-energy": "Spearman's coeff. (tension)",
        "spearman-compression-derivative": "Spearman's coeff. (compression derivative)",
        "missing": "Missing",
    },
)

table.set_index("Model", inplace=True)

s = (
    table.style.background_gradient(
        cmap="Blues",
        subset=["Rank", "Rank aggr."],
    ).background_gradient(
        cmap="Reds",
        subset=[
            "Spearman's coeff. (compression)",
        ],
    ).background_gradient(
        cmap="Reds_r",
        subset=[
            "Spearman's coeff. (tension)",
            "Spearman's coeff. (compression derivative)",
        ],
    ).background_gradient(
        cmap="RdPu",
        subset=["Tortuosity", "Derivative flips"],
    ).format(
        "{:.5f}",
        subset=[
            "Spearman's coeff. (compression)",
            "Spearman's coeff. (tension)",
            "Spearman's coeff. (compression derivative)",
            "Tortuosity",
            "Derivative flips",
        ],
    )
)

def render():
    st.dataframe(
        s,
        use_container_width=True,
    )
