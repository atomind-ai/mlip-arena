import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

from mlip_arena import PKG_DIR
from mlip_arena.models import REGISTRY as MODELS
from mlip_arena.tasks import REGISTRY as TASKS

# -----------------------------------------------------------------------------
# Build metadata table
# -----------------------------------------------------------------------------

metadata_table = pd.DataFrame(
    columns=[
        "Model",
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

for model, meta in MODELS.items():
    new_row = {
        "Model": model,
        "Prediction": meta.get("prediction"),
        "NVT": "✅" if meta.get("nvt", False) else "❌",
        "NPT": "✅" if meta.get("npt", False) else "❌",
        "Training Set": meta.get("datasets", []),
        "Code": meta.get("github"),
        "Paper": meta.get("doi"),
        "Checkpoint": meta.get("checkpoint"),
        "First Release": meta.get("date"),
        "License": meta.get("license"),
    }
    metadata_table = pd.concat(
        [metadata_table, pd.DataFrame([new_row])], ignore_index=True
    )

metadata_table.set_index("Model", inplace=True)

# -----------------------------------------------------------------------------
# Collect per-task ranks and compute Arena Rank
# -----------------------------------------------------------------------------

task_ranks = []
for task, meta in TASKS.items():
    rank_page = meta.get("rank-page")
    if rank_page is None:
        continue
    try:
        task_module = importlib.import_module(f"ranks.{rank_page}")
    except ModuleNotFoundError:
        continue

    if hasattr(task_module, "table"):
        t = task_module.table.copy()
        if "Rank" in t.columns:
            t = t[["Rank"]].rename(columns={"Rank": f"{task} Rank"})
            task_ranks.append(t)

rank_df = pd.concat(task_ranks, axis=1) if task_ranks else pd.DataFrame()

leaderboard = metadata_table.join(rank_df, how="left")

# -----------------------------------------------------------------------------
# Compute Arena Rank explicitly without modifying original task ranks
# -----------------------------------------------------------------------------

if not rank_df.empty:
    leaderboard = leaderboard.copy()
    leaderboard["Arena Rank Agg"] = 0

    for col in rank_df.columns:
        ascending = True  # set False if higher value is better for this task
        # compute rank for this column (missing values go to bottom)
        
        leaderboard["Arena Rank Agg"] += leaderboard[col].rank(
            method="min", ascending=ascending, na_option="bottom"
        )
        leaderboard[col] = leaderboard[col].astype("Int64")
        # rename column to indicate rank
        # leaderboard.rename(columns={col: col}, inplace=True)

    # Final Arena Rank
    leaderboard["Arena Rank"] = leaderboard["Arena Rank Agg"].rank(
        method="min", ascending=True, na_option="bottom"
    ).astype("Int64")

    # Reorder columns: Training Set → Arena Rank → task ranks → rest
    rank_cols = [c for c in leaderboard.columns if c.endswith("Rank") and c != "Arena Rank"]
    first_cols = ["Training Set", "Arena Rank"] + rank_cols
    other_cols = [c for c in leaderboard.columns if c not in first_cols and c != "Arena Rank Agg"]
    leaderboard = leaderboard.reindex(columns=first_cols + other_cols)

    # Optional: sort by Arena Rank
    leaderboard.sort_values("Arena Rank", inplace=True)


# -----------------------------------------------------------------------------
# Styling and rendering
# -----------------------------------------------------------------------------

style = leaderboard.drop(columns=["Arena Rank Agg"], errors="ignore").style
if "Arena Rank" in leaderboard.columns:
    style = style.background_gradient(cmap="Blues", subset=["Arena Rank"])

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
</a>
</div>

> MLIP Arena is a unified platform for evaluating foundation machine learning interatomic potentials (MLIPs) beyond conventional energy and force error metrics. It focuses on revealing the underlying physics and chemistry learned by these models. The platform's benchmarks are specifically designed to evaluate the readiness and reliability of open-source, open-weight models in accurately reproducing both qualitative and quantitative behaviors of atomic systems.
""",
    unsafe_allow_html=True,
)

st.subheader(":red[Supported Models + Arena Rank]")
st.dataframe(
    style,
    use_container_width=True,
    column_config={
        "Code": st.column_config.LinkColumn(width="medium", display_text="Link"),
        "Paper": st.column_config.LinkColumn(width="medium", display_text="Link"),
    },
)

st.subheader(":red[Task Ranks]")
for task, meta in TASKS.items():
    if meta["rank-page"] is None:
        continue

    st.subheader(task, divider=True)
    task_module = importlib.import_module(f"ranks.{meta['rank-page']}")

    if meta["task-page"] is not None:
        st.page_link(
            f"tasks/{meta['task-page']}.py",
            label="Go to the associated task page",
            icon=":material/link:",
        )

    if hasattr(task_module, "render"):
        task_module.render()
    else:
        st.write(
            "Rank metrics are not available yet but the task has been implemented."
        )
