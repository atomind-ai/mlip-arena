"""UMA (FairChem) model integrations."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from fairchem.core import FAIRChemCalculator, pretrained_mlip

from mlip_arena.models.utils import get_freer_device

with open(Path(__file__).parents[1] / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)


class UMA_S_1P1(FAIRChemCalculator):
    def __init__(
        self,
        model_name=REGISTRY["UMA-S-1P1"]["model_name"],
        task_name=REGISTRY["UMA-S-1P1"]["task_name"],
        cpu=False,
        **kwargs,
    ):
        device = get_freer_device() if not cpu else torch.device("cpu")
        predict_unit = pretrained_mlip.get_predict_unit(model_name, device=str(device).split(":")[0])
        super().__init__(
            predict_unit=predict_unit,
            task_name=task_name,
        )
