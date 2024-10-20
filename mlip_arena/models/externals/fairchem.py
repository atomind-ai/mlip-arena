from __future__ import annotations

from pathlib import Path

import yaml
from ase import Atoms
from fairchem.core import OCPCalculator

with open(Path(__file__).parents[1] / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)

class eqV2(OCPCalculator):
    def __init__(
        self,
        checkpoint="eqV2_86M_omat_mp_salex.pt",  # TODO: import from registry
        # TODO: cannot assign device
        local_cache="/tmp/ocp/",
        cpu=False,
        seed=0,
        **kwargs,
    ) -> None:
        """
        Initialize an eqV2 calculator.

        Parameters
        ----------
        checkpoint : str, default="eqV2_86M_omat_mp_salex.pt"
            The name of the eqV2 checkpoint to use.
        local_cache : str, default="/tmp/ocp/"
            The directory to store the downloaded checkpoint.
        cpu : bool, default=False
            Whether to run the model on CPU or GPU.
        seed : int, default=0
            The random seed for the model.

        Other Parameters
        ----------------
        **kwargs
            Any additional keyword arguments are passed to the superclass.
        """
        super().__init__(
            model_name=checkpoint,
            local_cache=local_cache,
            cpu=cpu,
            seed=seed,
            **kwargs,
        )

class EquiformerV2(OCPCalculator):
    def __init__(
        self,
        checkpoint="EquiformerV2-lE4-lF100-S2EFS-OC22",  # TODO: import from registry
        # TODO: cannot assign device
        local_cache="/tmp/ocp/",
        cpu=False,
        seed=0,
        **kwargs,
    ) -> None:
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


class EquiformerV2OC20(OCPCalculator):
    def __init__(
        self,
        checkpoint="EquiformerV2-31M-S2EF-OC20-All+MD",  # TODO: import from registry
        # TODO: cannot assign device
        local_cache="/tmp/ocp/",
        cpu=False,
        seed=0,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=checkpoint,
            local_cache=local_cache,
            cpu=cpu,
            seed=seed,
            **kwargs,
        )

class eSCN(OCPCalculator):
    def __init__(
        self,
        checkpoint="eSCN-L6-M3-Lay20-S2EF-OC20-All+MD",  # TODO: import from registry
        # TODO: cannot assign device
        local_cache="/tmp/ocp/",
        cpu=False,
        seed=0,
        **kwargs,
    ) -> None:
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
