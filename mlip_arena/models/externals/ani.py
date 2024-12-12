from __future__ import annotations

import yaml
from pathlib import Path

from ase import Atoms

from torchani.ase import Calculator as ANICalculator
from torchani.models import BuiltinEnsemble

from mlip_arena.models.utils import get_freer_device

with open(Path(__file__).parents[1] / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)

class ANI2x(ANICalculator):
    def __init__(
        self,
        checkpoint=REGISTRY["ANI2x"]["checkpoint"],
        device: str | None = None,
        periodic_table_index=False,
        **kwargs,
    ):
        ensemble = BuiltinEnsemble._from_neurochem_resources(checkpoint, periodic_table_index)

        super().__init__(
            ensemble.species,
            ensemble,
            **kwargs
        )

        # self.device = device or str(get_freer_device())
        self.device = device or "cpu" #BUG: cannot use gpu

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = ["energy", "forces", "stress"],
        system_changes: list | None = None,
    ):
        super().calculate(atoms, properties, system_changes)

        
        
