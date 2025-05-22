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

# df = df[df["method"].isin([
#     "SevenNet",
#     "ORBv2",
#     "ORB",
#     "MatterSim",
#     "MACE-MPA",
#     "MACE-MP(M)",
#     "M3GNet",
#     "eSEN",
#     "eSCN(OC20)",
#     "eqV2(OMat)",
#     "EquiformerV2(OC22)",
#     "EquiformerV2(OC20)",
#     "CHGNet",
#     "ALIGNN"
# ]
# )]

table = pd.DataFrame()

for model in valid_models:
    rows = df[df["method"] == model]
    metadata = MODELS.get(model, {})

    new_row = {
        "Model": model,
        "Conservation deviation [eV/Å]": rows["conservation-deviation"].mean(),
        "Spearman's coeff. (E: repulsion)": rows["spearman-repulsion-energy"].mean(),
        "Spearman's coeff. (F: descending)": rows["spearman-descending-force"].mean(),
        "Tortuosity": rows["tortuosity"].mean(),
        "Energy jump [eV]": rows["energy-jump"].mean(),
        "Force flips": rows["force-flip-times"].mean(),
        "Spearman's coeff. (E: attraction)": rows["spearman-attraction-energy"].mean(),
        "Spearman's coeff. (F: ascending)": rows["spearman-ascending-force"].mean(),
        "PBE energy MAE [eV]": rows["pbe-energy-mae"].mean(),
        "PBE force MAE [eV/Å]": rows["pbe-force-mae"].mean(),
    }

    table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)

table.set_index("Model", inplace=True)

table.sort_values("Conservation deviation [eV/Å]", ascending=True, inplace=True)
table["Rank"] = np.argsort(table["Conservation deviation [eV/Å]"].to_numpy())

table.sort_values("Spearman's coeff. (E: repulsion)", ascending=True, inplace=True)
table["Rank"] += np.argsort(table["Spearman's coeff. (E: repulsion)"].to_numpy())

table.sort_values("Spearman's coeff. (F: descending)", ascending=True, inplace=True)
table["Rank"] += np.argsort(table["Spearman's coeff. (F: descending)"].to_numpy())

# NOTE: it's not fair to models trained on different level of theory
# table.sort_values("PBE energy MAE [eV]", ascending=True, inplace=True)
# table["Rank"] += np.argsort(table["PBE energy MAE [eV]"].to_numpy())

# table.sort_values("PBE force MAE [eV/Å]", ascending=True, inplace=True)
# table["Rank"] += np.argsort(table["PBE force MAE [eV/Å]"].to_numpy())

table.sort_values("Tortuosity", ascending=True, inplace=True)
table["Rank"] += np.argsort(table["Tortuosity"].to_numpy())

table.sort_values("Energy jump [eV]", ascending=True, inplace=True)
table["Rank"] += np.argsort(table["Energy jump [eV]"].to_numpy())

table.sort_values("Force flips", ascending=True, inplace=True)
table["Rank"] += np.argsort(np.abs(table["Force flips"].to_numpy() - 1))

table["Rank"] += 1

table.sort_values(
    ["Rank", "Conservation deviation [eV/Å]"], ascending=True, inplace=True
)

table["Rank aggr."] = table["Rank"]
table["Rank"] = table["Rank aggr."].rank(method="min").astype(int)

table = table.reindex(
    columns=[
        "Rank",
        "Rank aggr.",
        "Conservation deviation [eV/Å]",
        "Spearman's coeff. (E: repulsion)",
        "Spearman's coeff. (F: descending)",
        "Energy jump [eV]",
        "Force flips",
        "Tortuosity",
        "PBE energy MAE [eV]",
        "PBE force MAE [eV/Å]",
        "Spearman's coeff. (E: attraction)",
        "Spearman's coeff. (F: ascending)",
    ]
)

# cloned = table.copy()
# cloned.drop(columns=[
#     "PBE energy MAE [eV]",
#     "PBE force MAE [eV/Å]",
#     "Spearman's coeff. (E: attraction)",
#     "Spearman's coeff. (F: ascending)",],
#     inplace=True
# )
# cloned.to_latex(
#     DATA_DIR / "homonuclear-diatomics.tex",
#     float_format="%.3f",
#     index=True,
#     column_format="l" + "r" * (len(table.columns) - 1),
# )

s = (
    table.style.background_gradient(
        cmap="viridis_r",
        subset=["Conservation deviation [eV/Å]"],
        gmap=np.log(table["Conservation deviation [eV/Å]"].to_numpy()),
    )
    .background_gradient(
        cmap="Reds",
        subset=[
            "Spearman's coeff. (E: repulsion)",
            "Spearman's coeff. (F: descending)",
        ],
        # vmin=-1, vmax=-0.5
    )
    # .background_gradient(
    #     cmap="Greys",
    #     subset=[
    #         "PBE energy MAE [eV]",
    #         "PBE force MAE [eV/Å]",
    #     ],
    # )
    .background_gradient(
        cmap="RdPu",
        subset=["Tortuosity", "Energy jump [eV]", "Force flips"],
    )
    .background_gradient(
        cmap="Blues",
        subset=["Rank", "Rank aggr."],
    )
    .format(
        "{:.3f}",
        subset=[
            "Conservation deviation [eV/Å]",
            "Spearman's coeff. (E: repulsion)",
            "Spearman's coeff. (F: descending)",
            "Tortuosity",
            "Energy jump [eV]",
            "Force flips",
            "Spearman's coeff. (E: attraction)",
            "Spearman's coeff. (F: ascending)",
            "PBE energy MAE [eV]",
            "PBE force MAE [eV/Å]",
        ],
    )
)


def render():
    st.dataframe(
        s,
        use_container_width=True,
    )
    with st.expander("Explanation", icon=":material/info:"):
        st.caption(
            r"""
            - **Conservation deviation**: The average deviation of force from negative energy gradient along the diatomic curves. 
            
            $$
            \text{Conservation deviation} = \left\langle\left| \mathbf{F}(\mathbf{r})\cdot\frac{\mathbf{r}}{\|\mathbf{r}\|} +  \nabla_rE\right|\right\rangle_{r = \|\mathbf{r}\|}
            $$

            - **Spearman's coeff. (E: repulsion)**: Spearman's correlation coefficient of energy prediction within equilibrium distance $r \in (r_{min}, r_o = \argmin_{r} E(r))$.
            - **Spearman's coeff. (F: descending)**: Spearman's correlation coefficient of force prediction before maximum attraction $r \in (r_{min}, r_a = \argmin_{r} F(r))$.
            - **Tortuosity**: The ratio between total variation in energy and sum of absolute energy differences between $r_{min}$, $r_o$, and $r_{max}$.
            - **Energy jump**: The sum of energy discontinuity between sampled points. 

            $$
            \text{Energy jump} = \sum_{r_i \in [r_\text{min}, r_\text{max}]} \left| \text{sign}{\left[ E(r_{i+1}) - E(r_i)\right]} - \text{sign}{\left[E(r_i) - E(r_{i-1})\right]}\right| \times \\ \left( \left|E(r_{i+1}) - E(r_i)\right| + \left|E(r_i) - E(r_{i-1})\right|\right)
            $$
            - **Force flips**: The number of force direction changes.
            """
        )
        st.info(
            "PBE energies and forces are provided __only__ for reference. Due to the known convergence issue of plane-wave DFT with diatomic molecules and different dataset the models might be trained on, comparing models with PBE is not rigorous and thus these metrics are excluded from rank aggregation.",
            icon=":material/warning:",
        )
