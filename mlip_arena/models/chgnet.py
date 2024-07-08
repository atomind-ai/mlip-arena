from typing import Optional, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes
from huggingface_hub import hf_hub_download
from torch_geometric.data import Data

from mlip_arena.models import MLIP, MLIPCalculator

# TODO: WIP


class CHGNet(MLIPCalculator):
    def __init__(
        self,
        device: torch.device | None = None,
        restart=None,
        atoms=None,
        directory=".",
        **kwargs,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        super().__init__(
            model=model, restart=restart, atoms=atoms, directory=directory, **kwargs
        )

        self.name: str = self.__class__.__name__
        self.implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self, atoms: Atoms, properties: list[str], system_changes: list = all_changes
    ):
        """Calculate energies and forces for the given Atoms object"""
        super().calculate(atoms, properties, system_changes)

        output = self.forward(atoms)

        self.results = {}
        if "energy" in properties:
            self.results["energy"] = output["energy"].item()
        if "forces" in properties:
            self.results["forces"] = output["forces"].cpu().detach().numpy()
        if "stress" in properties:
            self.results["stress"] = output["stress"].cpu().detach().numpy()

    def forward(self, x: Data | Atoms) -> dict[str, torch.Tensor]:
        """Implement data conversion, graph creation, and model forward pass"""
        # TODO
        raise NotImplementedError
