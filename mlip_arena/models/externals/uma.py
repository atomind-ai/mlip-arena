from __future__ import annotations

from pathlib import Path

import yaml
from fairchem.core import pretrained_mlip, FAIRChemCalculator
 

with open(Path(__file__).parents[1] / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)

class UMA(FAIRChemCalculator):
    def __init__(
        self,
        model_name=REGISTRY["UMA"]["model_name"],
        task_name=REGISTRY["UMA"]["task_name"],
        cpu=False,
        **kwargs,
    ):
        predictor = pretrained_mlip.get_predict_unit(model_name, device="cuda")
        super().__init__(
            predictor=predictor,
            task_name=task_name,
            device="cpu" if cpu else "cuda",
        )