
import pandas as pd
import streamlit as st

from mlip_arena import PKG_DIR

DATA_DIR = PKG_DIR / "tasks" / "thermal-conductivity"

table = pd.read_csv(DATA_DIR / "wte.csv")

table.rename(
    columns={
        "method": "Model",
        "srme": "SRME [1/Å]",
    },
    inplace=True,
)

table.set_index("Model", inplace=True)

table.sort_values(["SRME [1/Å]"], ascending=True, inplace=True)

table["Rank"] = table["SRME [1/Å]"].rank(method='min').astype(int)

table = table.reindex(
    columns=[
        "Rank",
        "SRME [1/Å]",
    ]
)

s = (
    table.style.background_gradient(
        cmap="Reds", subset=["SRME [1/Å]"]
    )
    .background_gradient(
        cmap="Blues",
        subset=["Rank"],
    )
    .format("{:.3f}", subset=["SRME [1/Å]"])
)


def render():

    st.dataframe(
        s,
        use_container_width=True
    )

    with st.expander(":material/info: Explanation"):
        st.caption(
            """
            - **SRME**: symmetric relative mean error of single-phonon conductivity:

            $$
            \\text{SRME}[\\left\lbrace\\mathcal{K}({\\mathbf{q},s)}\\right\\rbrace] = \\frac{2}{N_qV}\\frac{\\sum_{\\mathbf{q}s}|\\mathcal{K}_{\\text{MLIP}}(\\mathbf{q},s) - \\mathcal{K}_{\\text{DFT}}(\\mathbf{q},s)|}{\\kappa_{\\text{MLIP}} + \\kappa_{\\text{DFT}}}
            $$
            """
        )
