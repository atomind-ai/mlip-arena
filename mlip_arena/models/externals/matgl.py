from __future__ import annotations

import matgl
import torch
from matgl.ext.ase import PESCalculator
from typing import Literal


class M3GNet(PESCalculator):
    def __init__(
        self,
        checkpoint="M3GNet-MP-2021.2.8-PES",
        # TODO: cannot assign device
        state_attr: torch.Tensor | None = None,
        stress_unit: Literal["eV/A3", "GPa"] = "GPa",
        stress_weight: float = 1.0,
        use_voigt: bool = False,
        **kwargs,
    ) -> None:
        potential = matgl.load_model(checkpoint)
        super().__init__(potential, state_attr, stress_unit, stress_weight, use_voigt, **kwargs)
