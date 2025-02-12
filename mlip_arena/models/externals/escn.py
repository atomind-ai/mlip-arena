from __future__ import annotations

from pathlib import Path

import yaml
from ase import Atoms
from fairchem.core import OCPCalculator

with open(Path(__file__).parents[1] / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)

class eSCN(OCPCalculator):
    def __init__(
        self,
        checkpoint=REGISTRY["eSCN(OC20)"]["checkpoint"], # "eSCN-L6-M3-Lay20-S2EF-OC20-All+MD"
        # TODO: cannot assign device
        local_cache="/tmp/ocp/",
        cpu=False,
        seed=0,
        **kwargs,
    ) -> None:
        kwargs.pop("device", None)
        super().__init__(
            model_name=checkpoint,
            local_cache=local_cache,
            cpu=cpu,
            seed=seed,
            **kwargs,
        )

    def calculate(self, atoms: Atoms, properties, system_changes) -> None:
        super().calculate(atoms, properties, system_changes)

        self.results.update(
            force=atoms.get_forces(),
        )
