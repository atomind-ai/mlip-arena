import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from torch import nn
from torch_geometric.data import Data

REGISTRY_FILE = 'registry.yaml'


class MLIP(Calculator):
    def __init__(self):
        super().__init__()
        self.name: str = "MLIP"
        self.version: str = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = None
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
