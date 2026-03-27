import sys
from pathlib import Path

import pytest

from mlip_arena.models import REGISTRY as MODELS
from mlip_arena.tasks import REGISTRY as TASKS

# Add serve directory to path to import ranks
serve_path = Path(__file__).parents[1] / "serve"
sys.path.append(str(serve_path))


# Get all tasks that have a rank page defined
RANK_TASKS = [
    (meta["rank-page"], meta["rank-page"]) for task_name, meta in TASKS.items() if meta.get("rank-page") is not None
]


@pytest.mark.parametrize("rank_module_name, task_key", RANK_TASKS)
def test_rank_data_loading(rank_module_name, task_key):
    """
    Validate that a rank leaderboard module exposes a non-empty `table` and includes all models registered for the given task.

    Parameters:
        rank_module_name (str): Name of the rank module under the `ranks` package to load.
        task_key (str): Task identifier used to determine which models should appear in the leaderboard (looked up in the `MODELS` registry).
    """
    # Import the rank module
    import importlib

    try:
        module = importlib.import_module(f"ranks.{rank_module_name}")
    except Exception as e:
        pytest.fail(f"Failed to import rank module '{rank_module_name}': {e}")

    # Check if the module has the 'table' attribute and it's not empty
    assert hasattr(module, "table"), f"Module '{rank_module_name}' is missing 'table' attribute"
    assert not module.table.empty, f"Leaderboard table for '{rank_module_name}' is empty"

    # Identify expected models for this task from the registry
    expected_models = [model for model, metadata in MODELS.items() if task_key in metadata.get("gpu-tasks", [])]

    # Verify that all expected models are present in the table
    loaded_models = set(module.table.index)
    missing_models = [m for m in expected_models if m not in loaded_models]

    assert not missing_models, f"Missing data for models in '{rank_module_name}': {missing_models}"