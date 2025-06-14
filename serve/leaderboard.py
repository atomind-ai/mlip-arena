import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

from mlip_arena import PKG_DIR
from mlip_arena.models import REGISTRY as MODELS
from mlip_arena.tasks import REGISTRY as TASKS

# Read the data
DATA_DIR = PKG_DIR / "tasks" /"diatomics"

dfs = []
for model in MODELS:
    fpath = DATA_DIR / MODELS[model].get("family") / "homonuclear-diatomics.json"
    if fpath.exists():
        dfs.append(pd.read_json(fpath))
df = pd.concat(dfs, ignore_index=True)

# Create a table
table = pd.DataFrame(
    columns=[
        "Model",
        "Element Coverage",
        "Prediction",
        "NVT",
        "NPT",
        "Training Set",
        "Code",
        "Paper",
        "Checkpoint",
        "First Release",
        "License",
    ]
)

for model in MODELS:
    rows = df[df["method"] == model]
    metadata = MODELS.get(model, {})
    new_row = {
        "Model": model,
        "Element Coverage": len(rows["name"].unique()),
        "Prediction": metadata.get("prediction", None),
        "NVT": "✅" if metadata.get("nvt", False) else "❌",
        "NPT": "✅" if metadata.get("npt", False) else "❌",
        "Training Set": metadata.get("datasets", []),
        "Code": metadata.get("github", None) if metadata else None,
        "Paper": metadata.get("doi", None) if metadata else None,
        "Checkpoint": metadata.get("checkpoint", None),
        "First Release": metadata.get("date", None),
        "License": metadata.get("license", None),
    }
    table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)

table.set_index("Model", inplace=True)

s = table.style.background_gradient(
    cmap="PuRd", subset=["Element Coverage"], vmin=0, vmax=120
)

# st.warning(
#     "MLIP Arena is currently in **pre-alpha**. The results are not stable. Please interpret them with care.",
#     icon="⚠️",
# )
st.info(
    "Contributions are welcome. For more information, visit https://github.com/atomind-ai/mlip-arena.",
    icon="🤗",
)

st.markdown(
    """
<h1 style='text-align: center;'>⚔️ MLIP Arena Leaderboard ⚔️</h1>

<div align="center">
    <a href="https://openreview.net/forum?id=ysKfIavYQE#discussion"><img alt="Static Badge" src="https://img.shields.io/badge/ICLR-AI4Mat-blue"></a>
    <a href="https://huggingface.co/spaces/atomind/mlip-arena"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face"></a>
    <a href="https://github.com/atomind-ai/mlip-arena/actions"><img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/atomind-ai/mlip-arena/test.yaml"></a>
    <a href="https://pypi.org/project/mlip-arena/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/mlip-arena"></a>
    <a href="https://zenodo.org/doi/10.5281/zenodo.13704399"><img src="https://zenodo.org/badge/776930320.svg" alt="DOI"></a>
    <!-- <a href="https://discord.gg/W8WvdQtT8T"><img alt="Discord" src="https://img.shields.io/discord/1299613474820984832?logo=discord"> -->
</a>
</div>


> MLIP Arena is a unified platform for evaluating foundation machine learning interatomic potentials (MLIPs) beyond conventional energy and force error metrics. It focuses on revealing the underlying physics and chemistry learned by these models. The platform's benchmarks are specifically designed to evaluate the readiness and reliability of open-source, open-weight models in accurately reproducing both qualitative and quantitative behaviors of atomic systems.

### :red[Introduction]

Foundation machine learning interatomic potentials (MLIPs), trained on extensive databases containing millions of density functional theory (DFT) calculations,have revolutionized molecular and materials modeling, but existing benchmarks suffer from data leakage, limited transferability, and an over-reliance on error-based metrics tied to specific density functional theory (DFT) references.

We introduce MLIP Arena, a unified benchmark platform for evaluating foundation MLIP performance beyond conventional error metrics. It focuses on revealing the physical soundness learned by MLIPs and assessing their utilitarian performance agnostic to underlying model architecture and training dataset. 

***By moving beyond static DFT references and revealing the important failure modes*** of current foundation MLIPs in real-world settings, MLIP Arena provides a reproducible framework to guide the next-generation MLIP development toward improved predictive accuracy and runtime efficiency while maintaining physical consistency.

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

    if TASKS[task]['task-page'] is not None:
        st.page_link(
            f"tasks/{TASKS[task]['task-page']}.py",
            label="Go to the associated task page",
            icon=":material/link:",
        )

    #  Call the function from the imported module
    if hasattr(task_module, "render"):
        task_module.render()
        # if st.button(f"Go to task page"):
        #     st.switch_page(f"tasks/{TASKS[task]['task-page']}.py")
    else:
        st.write(
            "Rank metrics are not available yet but the task has been implemented. Please see the task page for more information."
        )
