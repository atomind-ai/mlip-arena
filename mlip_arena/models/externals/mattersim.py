from __future__ import annotations

from pathlib import Path

import yaml
from mattersim.forcefield import MatterSimCalculator

from mlip_arena.models.utils import get_freer_device

with open(Path(__file__).parents[1] / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)

class MatterSim(MatterSimCalculator):
    def __init__(
        self,
        checkpoint=REGISTRY["MatterSim"]["checkpoint"],
        device=None,
        **kwargs,
    ):
        super().__init__(
            load_path=checkpoint, 
            device=str(device or get_freer_device()), 
            **kwargs
        )