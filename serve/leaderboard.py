from pathlib import Path

import pandas as pd
import streamlit as st

from mlip_arena.models import REGISTRY as MODELS
from mlip_arena.tasks import REGISTRY as TASKS

import importlib

DATA_DIR = Path("mlip_arena/tasks/diatomics")

dfs = [pd.read_json(DATA_DIR / MODELS[model].get("family") /  "homonuclear-diatomics.json") for model in MODELS]
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

for model in MODELS:
    rows = df[df["method"] == model]
    metadata = MODELS.get(model, {})
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


for task in TASKS:

    st.header(task, divider=True)

    if TASKS[task]['rank-page'] is None:
        st.write("Rank for this task is not available yet")
        continue

    task_module = importlib.import_module(f"ranks.{TASKS[task]['rank-page']}")

    # task_module = importlib.import_module(f".ranks", TASKS[task]["task-page"])

    #  Call the function from the imported module
    if hasattr(task_module, 'get_rank_page'):
        task_module.get_rank_page()
    else:
        st.write("Results for the task are not available yet.")