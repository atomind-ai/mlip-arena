from __future__ import annotations

from pathlib import Path

import requests
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from mlip_arena.models.utils import get_freer_device


class ORB(ORBCalculator):
    def __init__(
        self,
        checkpoint="orbff-v1-20240827.ckpt",
        device=None,
        **kwargs,
    ):
        device = device or get_freer_device()

        cache_dir = Path.home() / ".cache" / "orb"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = cache_dir / checkpoint

        url = f"https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/{checkpoint}"

        if not ckpt_path.exists():
            print(f"Downloading ORB model from {url} to {ckpt_path}...")
            try:
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                with open(ckpt_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download completed.")
            except requests.exceptions.RequestException as e:
                raise RuntimeError("Failed to download ORB model.") from e

        orbff = pretrained.orb_v1(weights_path=ckpt_path, device=device)
        super().__init__(orbff, device=device, **kwargs)
