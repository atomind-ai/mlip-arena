from pathlib import Path

import yaml
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from mlip_arena.models import MLIP
from mlip_arena.models import REGISTRY as MODEL_REGISTRY

with open(Path(__file__).parent / "registry.yaml") as f:
    REGISTRY = yaml.safe_load(f)


class Task:
    def __init__(self):
        self.name: str = self.__class__.__name__  # display name on the leaderboard

    def run_local(self, model: MLIP):
        """Run the task using the given model and return the results."""
        raise NotImplementedError

    def run_hf(self, model: MLIP):
        """Run the task using the given model and return the results."""
        raise NotImplementedError

        # Calcualte evaluation metrics and postprocessed data
        api = HfApi()
        api.upload_file(
            path_or_fileobj="results.json",
            path_in_repo=f"{self.__class__.__name__}/{model.__class__.__name__}/results.json",  # Upload to a specific folder
            repo_id="atomind/mlip-arena",
            repo_type="dataset",
        )

    def run_nersc(self, model: MLIP):
        """Run the task using the given model and return the results."""
        raise NotImplementedError

    def get_results(self):
        """Get the results from the task."""
        # fs = HfFileSystem()
        # files = fs.glob(f"datasets/atomind/mlip-arena/{self.__class__.__name__}/*/*.json")

        for model, metadata in MODEL_REGISTRY.items():
            results = hf_hub_download(
                repo_id="atomind/mlip-arena",
                filename="results.json",
                subfolder=f"{self.__class__.__name__}/{model}",
                repo_type="dataset",
                revision=None,
            )

        return results
