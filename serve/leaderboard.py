import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

from mlip_arena.models import REGISTRY as MODELS
from mlip_arena.tasks import REGISTRY as TASKS

DATA_DIR = Path("mlip_arena/tasks/diatomics")

dfs = [
    pd.read_json(DATA_DIR / MODELS[model].get("family") / "homonuclear-diatomics.json")
    for model in MODELS
]
df = pd.concat(dfs, ignore_index=True)


table = pd.DataFrame(
    columns=[
        "Model",
        "Element Coverage",
        # "No. of reversed forces",
        # "Energy-consistent forces",
        "Prediction",
        "NVT",
        "NPT",
        "Training Set",
        "Code",
        "Paper",
        "First Release",
    ]
)

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
        "Training Set": metadata.get("datasets", []),
        "Code": metadata.get("github", None) if metadata else None,
        "Paper": metadata.get("doi", None) if metadata else None,
        "First Release": metadata.get("date", None),
    }
    table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)

table.set_index("Model", inplace=True)

s = table.style.background_gradient(
    cmap="PuRd", subset=["Element Coverage"], vmin=0, vmax=120
)

st.warning(
    "MLIP Arena is currently in **pre-alpha**. The results are not stable. Please interpret them with care.",
    icon="⚠️",
)
st.info(
    "Contributions are welcome. For more information, visit https://github.com/atomind-ai/mlip-arena.",
    icon="🤗",
)

st.markdown(
    """
<h1 style='text-align: center;'>⚔️ MLIP Arena Leaderboard ⚔️</h1>

> MLIP Arena is a platform for evaluating foundation machine learning interatomic potentials (MLIPs) beyond conventional energy and force error metrics. It focuses on revealing the underlying physics and chemistry learned by these models and assessing their performance in molecular dynamics (MD) simulations. The platform's benchmarks are specifically designed to evaluate the readiness and reliability of open-source, open-weight models in accurately reproducing both qualitative and quantitative behaviors of atomic systems.

### :red[Introduction]

Foundation machine learning interatomic potentials (fMLIPs) trained on extensive databases containing millions of density functional theory (DFT) calculations have shown remarkable zero-shot predictive capabilities for complex atomic interactions. These potentials derive quantum mechanical insights with high accuracy, expressivity, and generalizability, significantly outperforming classical empirical force fields while maintaining comparable computational efficiency.

However, MLIPs trained on atomic energy and force labels do not neccessarily capture the correct atomic interactions, even though they usually excel in error-based metrics for bulk systems. To drive further advancements in this field, it is crucial to establish mechanisms that ensure fair and transparent benchmarking practices that go beyond basic regression metrics.

MLIP Arena aims to provide an equitable and transparent platform for benchmarking MLIPs in a collaborative, crowd-sourced environment. Its primary goal is to uncover the learned physics and chemistry of open-source, open-weight MLIPs. The benchmarks are designed to be agnostic to both the underlying architecture and the specific training targets, such as density functionals, ensuring a comprehensive and unbiased evaluation.

""",
    unsafe_allow_html=True,
)


st.subheader(":red[Supported Models]")
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

# st.markdown("<h2 style='text-align: center;'>🏆 Task Ranks 🏆</h2>", unsafe_allow_html=True)

st.subheader(":red[Task Ranks]")

for task in TASKS:
    if TASKS[task]["rank-page"] is None:
        continue

    st.subheader(task, divider=True)

    task_module = importlib.import_module(f"ranks.{TASKS[task]['rank-page']}")

    #  Call the function from the imported module
    if hasattr(task_module, "render"):
        task_module.render()
        # if st.button(f"Go to task page"):
        #     st.switch_page(f"tasks/{TASKS[task]['task-page']}.py")
    else:
        st.write("Rank metrics are not available yet but the task has been implemented. Please see the following task page for more information.")
    
    st.page_link(
        f"tasks/{TASKS[task]['task-page']}.py",
        label="Task page",
        icon=":material/link:",
    )