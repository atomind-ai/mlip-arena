from __future__ import annotations

from pathlib import Path

import yaml
from tensorpotential.calculator.asecalculator import TPCalculator
from tensorpotential.calculator.foundation_models import get_or_download_model

with open(Path(__file__).parents[1] / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)


class GRACE_2L_OAM(TPCalculator):
    def __init__(
        self,
        checkpoint=None,
        device=None,
        **kwargs,
    ):
        model_path = get_or_download_model("GRACE-2L-OAM")

        super().__init__(
            model=model_path,
            pad_neighbors_fraction=0.05,
            pad_atoms_number=10,
            max_number_reduction_recompilation=2,
            min_dist=None,
            device=device,
            **kwargs,
        )
