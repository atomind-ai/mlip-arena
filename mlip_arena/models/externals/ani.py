from __future__ import annotations

import yaml
from pathlib import Path

from ase.calculators.calculator import all_changes
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
        self.device = device or str(get_freer_device())
        
        ensemble = BuiltinEnsemble._from_neurochem_resources(
            checkpoint, periodic_table_index
        )
        # TODO: ANICalculator does not offer API to change device
        # ensemble.species.device = self.device
        super().__init__(ensemble.species, ensemble, **kwargs)
    
    
    def calculate(
        self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
