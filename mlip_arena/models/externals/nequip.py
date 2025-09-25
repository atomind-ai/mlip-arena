from __future__ import annotations

from ase.calculators.calculator import Calculator, all_changes
from nequip.ase import NequIPCalculator

from mlip_arena.models.utils import get_freer_device


class NequIP_OAM_L(NequIPCalculator):

    nequip_net_id = "nequip.net:mir-group/NequIP-OAM-L:0.1"

    def __init__(
        self,
        device=None,
        **kwargs,
    ):
        device = device or get_freer_device()
        self.calc = super()._from_saved_model(
            self.nequip_net_id, device=device, **kwargs
        )
        # for now, we use the eager model
        # TODO: we can use compiled versions if necessary

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms)
        return self.calc.calculate(atoms, properties, system_changes)
