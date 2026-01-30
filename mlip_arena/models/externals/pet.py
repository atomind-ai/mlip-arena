from __future__ import annotations

import yaml
from pathlib import Path

from ase.calculators.calculator import all_changes
from upet.calculator import UPETCalculator

from mlip_arena.models.utils import get_freer_device


class PET_OAM(UPETCalculator):
    def __init__(
        self,
        device: str | None = None,
        **kwargs,
    ):
        self.device = device or str(get_freer_device())
        super().__init__(model="pet-oam-xl", version="1.0.0", device=self.device, **kwargs)
    
    
    def calculate(
        self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)


class PET_MAD(UPETCalculator):
    def __init__(
        self,
        device: str | None = None,
        **kwargs,
    ):
        self.device = device or str(get_freer_device())
        super().__init__(model="pet-mad-s", version="1.0.2", device=self.device, **kwargs)
    
    
    def calculate(
        self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
