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

DATA_DIR = Path("mlip_arena/tasks/combustion")

@st.cache_data
def get_data(models):
    families = [MODELS[str(model)]["family"] for model in models]
    dfs = [
        pd.read_json(DATA_DIR / family.lower() / "hydrogen.json") for family in families
    ]
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(inplace=True, subset=["formula", "method"])
    return df

df = get_data(valid_models)

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
        df_flat["com_drift_x"] ** 2 + df_flat["com_drift_y"] ** 2 + df_flat["com_drift_z"] ** 2
    )

    return df_flat

df_exploded = get_com_drifts(df)

table = pd.DataFrame()

# def render():
#     st.dataframe(
#         table,
#         use_container_width=True,
#     )