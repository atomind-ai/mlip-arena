from pathlib import Path

import pandas as pd
import streamlit as st

# from mlip_arena.models.utils import MLIPEnum, REGISTRY
from mlip_arena.models import REGISTRY

DATA_DIR = Path("mlip_arena/tasks/diatomics")
methods = ["MACE-MP", "Equiformer", "CHGNet", "MACE-OFF", "eSCN", "ALIGNN"]
dfs = [pd.read_json(DATA_DIR / method.lower() /  "homonuclear-diatomics.json") for method in methods]
df = pd.concat(dfs, ignore_index=True)


table = pd.DataFrame(columns=[
    "Model",
    "Supported elements",
    # "No. of reversed forces",
    # "Energy-consistent forces",
    "Prediction",
    "NVT",
    "NPT",
    "Code",
    "Paper",
    "Last updated",
    ])

for model in REGISTRY:
    rows = df[df["method"] == model]
    metadata = REGISTRY.get(model, {})
    new_row = {
        "Model": model,
        "Supported elements": len(rows["name"].unique()),
        # "No. of reversed forces": None,  # Replace with actual logic if available
        # "Energy-consistent forces": None,  # Replace with actual logic if available
        "Prediction": metadata.get("prediction", None),
        "NVT": "✅" if metadata.get("nvt", False) else "❌",
        "NPT": "✅" if metadata.get("npt", False) else "❌",
        "Code": metadata.get("github", None) if metadata else None,
        "Paper": metadata.get("doi", None) if metadata else None,
    }
    table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)

table.set_index("Model", inplace=True)


s = table.style.background_gradient(
    cmap="PuRd",
    subset=["Supported elements"],
    vmin=0, vmax=120
)

st.markdown(
"""
<h1 style='text-align: center;'>⚔️ MLIP Arena Leaderboard ⚔️</h1>
""", unsafe_allow_html=True)

# st.image("")

# st.markdown("# MLIP Arena Leaderboard")

st.dataframe(
    s,
    use_container_width=True,
    column_config={
        "Code": st.column_config.LinkColumn(
            # "GitHub",
            # help="The top trending Streamlit apps",
            # validate="^https://[a-z]+\.streamlit\.app$",
            max_chars=100,
            display_text="GitHub",
        ),
        "Paper": st.column_config.LinkColumn(
            # validate="^https://[a-z]+\.streamlit\.app$",
            max_chars=100,
            display_text="arXiv",
        ),
    },
)
