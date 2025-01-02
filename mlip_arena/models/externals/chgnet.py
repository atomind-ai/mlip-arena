from __future__ import annotations

from typing import Literal

from ase import Atoms
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet as CHGNetModel

from mlip_arena.models.utils import get_freer_device


class CHGNet(CHGNetCalculator):
    def __init__(
        self,
        checkpoint: CHGNetModel | None = None,  # TODO: specifiy version
        device: str | None = None,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        **kwargs,
    ) -> None:
        use_device = str(device or get_freer_device())
        super().__init__(
            model=checkpoint,
            use_device=use_device,
            stress_weight=stress_weight,
            on_isolated_atoms=on_isolated_atoms,
            **kwargs,
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        # for ase.io.write compatibility
        self.results.pop("crystal_fea", None)
