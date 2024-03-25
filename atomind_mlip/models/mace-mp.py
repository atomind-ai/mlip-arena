import torch
from huggingface_hub import hf_hub_download
from torch_geometric.data import Data

from ase import Atoms
from ase.calculators.calculator import all_changes
from atomind_mlip.models import MLIP


class MACE_MP_Medium(MLIP):
    def __init__(self):
        super().__init__()
        self.name = "MACE-MP-0 (medium)"
        self.version = "1.0.0"

        fpath = hf_hub_download(repo_id="cyrusyc/mace-universal", subfolder="pretrained", filename="2023-12-12-mace-128-L1_epoch-199.model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(fpath, map_location="cpu")
        self.model.to(self.device)
        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
        ]

    def calculate(self, atoms: Atoms, properties: list[str], system_changes: dict = all_changes):
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
