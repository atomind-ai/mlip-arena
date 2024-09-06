from pathlib import Path

import pandas as pd
import streamlit as st

from mlip_arena.models import REGISTRY

DATA_DIR = Path("mlip_arena/tasks/diatomics")

dfs = [pd.read_json(DATA_DIR / REGISTRY[model].get("family") /  "homonuclear-diatomics.json") for model in REGISTRY]
df = pd.concat(dfs, ignore_index=True)


table = pd.DataFrame(columns=[
    "Model",
    "Element Coverage",
    # "No. of reversed forces",
    # "Energy-consistent forces",
    "Prediction",
    "NVT",
    "NPT",
    "Code",
    "Paper",
    "First Release",
    ])

for model in REGISTRY:
    rows = df[df["method"] == model]
    metadata = REGISTRY.get(model, {})
    new_row = {
        "Model": model,
        "Element Coverage": len(rows["name"].unique()),
        # "No. of reversed forces": None,  # Replace with actual logic if available
        # "Energy-consistent forces": None,  # Replace with actual logic if available
        "Prediction": metadata.get("prediction", None),
        "NVT": "✅" if metadata.get("nvt", False) else "❌",
        "NPT": "✅" if metadata.get("npt", False) else "❌",
        "Code": metadata.get("github", None) if metadata else None,
        "Paper": metadata.get("doi", None) if metadata else None,
        "First Release": metadata.get("date", None),
    }
    table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)

table.set_index("Model", inplace=True)


s = table.style.background_gradient(
    cmap="PuRd",
    subset=["Element Coverage"],
    vmin=0, vmax=120
)

st.warning("MLIP Arena is currently in **pre-alpha**. The results are not stable. Please interpret them with care.", icon="⚠️")
st.info("Contributions are welcome. For more information, visit https://github.com/atomind-ai/mlip-arena.", icon="🤗")

st.markdown(
"""
<h1 style='text-align: center;'>⚔️ MLIP Arena Leaderboard ⚔️</h1>

MLIP Arena is a platform for benchmarking foundation machine learning interatomic potentials (MLIPs), mainly for disclosing the learned physics and chemistry of the models and their performance on molecular dynamics (MD) simulations.
The benchmarks are designed to evaluate the readiness and reliability of open-source, open-weight models to reproduce the qualitatively or quantitatively correct physics.
""", unsafe_allow_html=True)

# st.header("Summary", divider=True)

st.dataframe(
    s,
    use_container_width=True,
    column_config={
        "Code": st.column_config.LinkColumn(
            # validate="^https://[a-z]+\.streamlit\.app$",
            width="medium",
            display_text="Link",
        ),
        "Paper": st.column_config.LinkColumn(
            # validate="^https://[a-z]+\.streamlit\.app$",
            width="medium",
            display_text="Link",
        ),
    },
)