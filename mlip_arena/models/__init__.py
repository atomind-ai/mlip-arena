import os

import torch
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from torch import nn
from torch_geometric.data import Data

with open(os.path.join(os.path.dirname(__file__), "registry.yaml")) as f:
    REGISTRY = yaml.load(f, Loader=yaml.FullLoader)

class MLIP(Calculator):
    def __init__(self, 
                 model_path: str = None,
                 device: torch.device = None):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.version: str = None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = torch.load(model_path, map_location=self.device)
        self.implemented_properties = ["energy", "forces"]

    def calculate(self, atoms: Atoms, properties: list[str], system_changes: dict = all_changes):
        """Calculate energies and forces for the given Atoms object"""
        super().calculate(atoms, properties, system_changes)

        output = self.forward(atoms)

        self.results = {}
        if "energy" in properties:
            self.results["energy"] = output["energy"].item()
        if "forces" in properties:
            self.results["forces"] = output["forces"].cpu().detach().numpy()

    def forward(self, x: Data | Atoms) -> dict[str, torch.Tensor]:
        """Implement data conversion, graph creation, and model forward pass
        
        Example implementation:
        1. Use `ase.neighborlist.NeighborList` to get neighbor list
        2. Create `torch_geometric.data.Data` object and copy the data
        3. Pass the `Data` object to the model and return the output

        """
        raise NotImplementedError
