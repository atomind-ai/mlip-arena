from __future__ import annotations

from pathlib import Path

import yaml
from mattersim.forcefield import MatterSimCalculator

from ase import Atoms
from mlip_arena.models.utils import get_freer_device
# from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms

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
            load_path=checkpoint, device=str(device or get_freer_device()), **kwargs
        )

    def __getstate__(self):
        state = self.__dict__.copy()

        # BUG: remove unpicklizable potential
        state.pop("potential", None)

        return state

    # def calculate(
    #     self,
    #     atoms: Atoms | None = None,
    #     properties: list | None = None,
    #     system_changes: list | None = None,
    # ):
    #     super().calculate(atoms, properties, system_changes)

        # # convert unpicklizable atoms back to picklizable atoms to avoid prefect pickling error
        # if isinstance(self.atoms, MSONAtoms):
        #     atoms = self.atoms.copy()
        #     strucutre = AseAtomsAdaptor().get_structure(atoms)
        #     self.atoms = AseAtomsAdaptor().get_atoms(strucutre, msonable=False)
