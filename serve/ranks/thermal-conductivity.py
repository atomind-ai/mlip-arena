from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("mlip_arena/tasks/thermal-conductivity")

table = pd.read_csv(DATA_DIR / "wte.csv")

table.rename(
    columns={
        "method": "Model",
        "srme": "SRME [W/mK]",
    },
    inplace=True,
)

table.set_index("Model", inplace=True)

table.sort_values(["SRME [W/mK]"], ascending=True, inplace=True)

s = table.style.background_gradient(
    cmap="viridis_r", subset=["SRME [W/mK]"]
)


def render():

    st.dataframe(
        s,
        use_container_width=True
    )
