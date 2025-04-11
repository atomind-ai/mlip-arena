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
    df_exploded = df.explode(["timestep", "energies", "com_drifts"]).reset_index(drop=True)

    # Convert the 'com_drifts' column (which are arrays) into separate columns for x, y, and z components
    df_exploded[["com_drift_x", "com_drift_y", "com_drift_z"]] = pd.DataFrame(
        df_exploded["com_drifts"].tolist(), index=df_exploded.index
    )

    # Drop the original 'com_drifts' column
    df_flat = df_exploded.drop(columns=["com_drifts"])

    df_flat["total_com_drift"] = np.sqrt(
        df_flat["com_drift_x"] ** 2 + df_flat["com_drift_y"] ** 2 + df_flat["com_drift_z"] ** 2
    )

    df_flat = df_flat.drop(columns=["com_drift_x", "com_drift_y", "com_drift_z"])

    return df_flat

df_exploded = get_com_drifts(df)

exp_ref = -68.3078 # kcal/mol

for method, row in df_exploded.groupby("method"):
#     # row = df[df["method"] == method].iloc[0]
    energies = np.array(row["energies"])
    df_exploded.loc[df_exploded["method"] == method,"reaction_enthlapy_diff"] = ((energies[-1] - energies[0]) / 128 * 23.) - exp_ref
    df_exploded.loc[df_exploded["method"] == method, "final_com_drift"] = np.array(row["total_com_drift"])[-1]


df_exploded.drop(columns=["temperatures", "pressures", "total_steps", "energies", "kinetic_energies", "timestep", "nproducts", "total_com_drift", "target_steps", "reaction", "formula", "natoms", "seconds_per_step", "seconds_per_step_per_atom", "final_step", "total_time_seconds"], axis=1, inplace=True)

df_exploded.drop_duplicates(inplace=True, subset=["method"])

print(df_exploded.columns)

df_exploded.set_index("method", inplace=True)

df_exploded.rename(columns={
    "method": "Model"
}, inplace=True)


table = pd.DataFrame()

for index, row in df_exploded.iterrows():

    new_row = {
        "Model": index,
        "Reaction enthalpy error [kcal/mol]": row["reaction_enthlapy_diff"],
        "Final COM drift [Å]": row["final_com_drift"],
        "Steps per second": row["steps_per_second"],
        "Yield [%]": row["yield"] * 100,
    }

    table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)

table.set_index("Model", inplace=True)

table.sort_values("Reaction enthalpy error [kcal/mol]", ascending=True, inplace=True)
table["Rank"] = np.argsort(np.abs(table["Reaction enthalpy error [kcal/mol]"].to_numpy()))

table.sort_values("Final COM drift [Å]", ascending=True, inplace=True)
table["Rank"] += np.argsort(table["Final COM drift [Å]"].to_numpy())

table.sort_values("Steps per second", ascending=False, inplace=True)
table["Rank"] += np.argsort(-table["Steps per second"].to_numpy())

table.sort_values("Yield [%]", ascending=False, inplace=True)
table["Rank"] += np.argsort(-table["Yield [%]"].to_numpy())

table["Rank"] += 1

table.sort_values(["Rank"], ascending=True, inplace=True)

table["Rank aggr."] = table["Rank"]
table["Rank"] = table["Rank aggr."].rank(method='min').astype(int)


table = table.reindex(
    columns=[
        "Rank",
        "Rank aggr.",
        "Reaction enthalpy error [kcal/mol]",
        "Final COM drift [Å]",
        "Steps per second",
        "Yield [%]",
    ]
)

s = (
    table.style.background_gradient(
        cmap="Oranges",
        subset=["Reaction enthalpy error [kcal/mol]"],
    )
    .background_gradient(
        cmap="Oranges",
        subset=["Final COM drift [Å]"],
        gmap=np.log10(table["Final COM drift [Å]"].to_numpy() + 1e-10),
    )
    .background_gradient(
        cmap="Oranges_r",
        subset=["Steps per second", "Yield [%]"]
    )
    .background_gradient(
        cmap="Blues",
        subset=["Rank", "Rank aggr."],
    )
    .format(
        "{:.3e}",
        subset=["Final COM drift [Å]"],
    )
)


def render():

    st.dataframe(
        s,
        use_container_width=True,
    )
