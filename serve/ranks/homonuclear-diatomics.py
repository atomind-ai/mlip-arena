from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mlip_arena.models import REGISTRY as MODELS

valid_models = [
    model
    for model, metadata in MODELS.items()
    if Path(__file__).stem in metadata.get("gpu-tasks", [])
]

DATA_DIR = Path("mlip_arena/tasks/diatomics")

dfs = [
    pd.read_json(DATA_DIR / MODELS[model].get("family") / "homonuclear-diatomics.json")
    for model in valid_models
]
df = pd.concat(dfs, ignore_index=True)

table = pd.DataFrame()

for model in valid_models:
    rows = df[df["method"] == model]
    metadata = MODELS.get(model, {})

    new_row = {
        "Model": model,
        "Conservation deviation [eV/Å]": rows["conservation-deviation"].mean(),
        "Spearman's coeff. (Energy - repulsion)": rows[
            "spearman-repulsion-energy"
        ].mean(),
        "Spearman's coeff. (Force - descending)": rows[
            "spearman-descending-force"
        ].mean(),
        "Tortuosity": rows["tortuosity"].mean(),
        "Energy jump [eV]": rows["energy-jump"].mean(),
        "Force flips": rows["force-flip-times"].mean(),
        "Spearman's coeff. (Energy - attraction)": rows[
            "spearman-attraction-energy"
        ].mean(),
        "Spearman's coeff. (Force - ascending)": rows[
            "spearman-ascending-force"
        ].mean(),
    }

    table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)

table.set_index("Model", inplace=True)

table.sort_values("Conservation deviation [eV/Å]", ascending=True, inplace=True)
table["Rank"] = np.argsort(table["Conservation deviation [eV/Å]"].to_numpy())

table.sort_values(
    "Spearman's coeff. (Energy - repulsion)", ascending=True, inplace=True
)
table["Rank"] += np.argsort(table["Spearman's coeff. (Energy - repulsion)"].to_numpy())

table.sort_values(
    "Spearman's coeff. (Force - descending)", ascending=True, inplace=True
)
table["Rank"] += np.argsort(table["Spearman's coeff. (Force - descending)"].to_numpy())

table.sort_values("Tortuosity", ascending=True, inplace=True)
table["Rank"] += np.argsort(table["Tortuosity"].to_numpy())

table.sort_values("Energy jump [eV]", ascending=True, inplace=True)
table["Rank"] += np.argsort(table["Energy jump [eV]"].to_numpy())

table.sort_values("Force flips", ascending=True, inplace=True)
table["Rank"] += np.argsort(table["Force flips"].to_numpy())

table.sort_values("Rank", ascending=True, inplace=True)

table["Rank aggr."] = table["Rank"]

table["Rank"] = np.argsort(table["Rank"].to_numpy()) + 1

# table.drop(columns=["rank"], inplace=True)
# table = table.rename(columns={"Rank": "Rank Aggr."})

table = table.reindex(
    columns=[
        "Rank",
        "Rank aggr.",
        "Conservation deviation [eV/Å]",
        "Spearman's coeff. (Energy - repulsion)",
        "Spearman's coeff. (Force - descending)",
        "Tortuosity",
        "Energy jump [eV]",
        "Force flips",
        "Spearman's coeff. (Energy - attraction)",
        "Spearman's coeff. (Force - ascending)",
    ]
)

s = (
    table.style.background_gradient(
        cmap="viridis_r",
        subset=["Conservation deviation [eV/Å]"],
        gmap=np.log(table["Conservation deviation [eV/Å]"].to_numpy()),
    )
    .background_gradient(
        cmap="Reds",
        subset=[
            "Spearman's coeff. (Energy - repulsion)",
            "Spearman's coeff. (Force - descending)",
        ],
        # vmin=-1, vmax=-0.5
    )
    .background_gradient(
        cmap="RdPu",
        subset=["Tortuosity", "Energy jump [eV]", "Force flips"],
    )
    .background_gradient(
        cmap="Blues",
        subset=["Rank", "Rank aggr."],
    )
)


def render():
    st.dataframe(
        s,
        use_container_width=True,
    )
    # return table
