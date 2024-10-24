from __future__ import annotations

from pathlib import Path

import yaml
import requests
from deepmd.calculator import DP as DPCalculator

from mlip_arena.models.utils import get_freer_device

with open(Path(__file__).parents[1] / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)

class DeepMD(DPCalculator):
    def __init__(
        self,
        checkpoint=REGISTRY["DeepMD"]["checkpoint"],
        device=None,
        **kwargs,
    ):
        device = device or get_freer_device()

        cache_dir = Path.home() / ".cache" / "deepmd"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / checkpoint

        url = "https://bohrium-api.dp.tech/ds-dl/mlip-arena-tfpk-v1.zip"

        if not model_path.exists():
            import zipfile

            print(f"Downloading DeepMD model from {url} to {model_path}...")
            try:
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                with open(cache_dir/"temp.zip", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download completed.")
                with zipfile.ZipFile(cache_dir/"temp.zip", "r") as zip_ref:
                    zip_ref.extractall(cache_dir)
                print("Unzip completed.")
            except requests.exceptions.RequestException as e:
                raise RuntimeError("Failed to download DeepMD model.") from e

        
        super().__init__(model_path, device=device, **kwargs)