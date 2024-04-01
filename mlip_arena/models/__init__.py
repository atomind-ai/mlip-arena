import os
from pathlib import Path

import torch
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch_geometric.data import Data

with open(os.path.join(os.path.dirname(__file__), "registry.yaml")) as f:
    REGISTRY = yaml.load(f, Loader=yaml.FullLoader)

# class MLIPEnum(enum.Enum):
#     for model, metadata in REGISTRY.items():
#         model_class = getattr(importlib.import_module(model["module"]), model)
#         self.setattr(model, model_class)


class MLIP(
    nn.Module,
    PyTorchModelHubMixin,
    tags=["atomistic-simulation", "MLIP"],
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class ModuleMLIP(MLIP):
    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_module("model", model)

    def forward(self, x):
        print("Forwarding...")
        out = self.model(x)
        print("Forwarded!")
        return out


class MLIPCalculator(Calculator):
    name: str
    device: torch.device
    model: MLIP
    implemented_properties: list[str] = ["energy", "forces", "stress"]

    def __init__(
        self,
        # PyTorch
        model_path: str | Path,
        device: torch.device | None = None,
        # ASE Calculator
        restart=None,
        atoms=None,
        directory=".",
        **kwargs,
    ):
        super().__init__(restart=restart, atoms=atoms, directory=directory, **kwargs)
        # self.name: str = self.__class__.__name__
        # self.device = device or torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu"
        # )
        # self.model: MLIP = MLIP.from_pretrained(model_path, map_location=self.device)
        # self.implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self, atoms: Atoms, properties: list[str], system_changes: list = all_changes
    ):
        """Calculate energies and forces for the given Atoms object"""
        super().calculate(atoms, properties, system_changes)

        output = self.forward(atoms)

        self.results = {}
        if "energy" in properties:
            self.results["energy"] = output["energy"].squeeze().item()
        if "forces" in properties:
            self.results["forces"] = output["forces"].squeeze().cpu().detach().numpy()
        if "stress" in properties:
            self.results["stress"] = output["stress"].squeeze().cpu().detach().numpy()

    def forward(self, x: Atoms) -> dict[str, torch.Tensor]:
        """Implement data conversion, graph creation, and model forward pass

        Example implementation:
        1. Use `ase.neighborlist.NeighborList` to get neighbor list
        2. Create `torch_geometric.data.Data` object and copy the data
        3. Pass the `Data` object to the model and return the output

        """
        raise NotImplementedError
