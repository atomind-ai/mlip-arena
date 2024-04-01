import torch
from ase import Atoms
from ase.calculators.calculator import all_changes
from huggingface_hub import hf_hub_download
from torch_geometric.data import Data

from mlip_arena.models import MLIPCalculator


class MACE_MP_Medium(MLIPCalculator):
    def __init__(self, device: torch.device | None = None):
        fpath = hf_hub_download(
            repo_id="cyrusyc/mace-universal",
            subfolder="pretrained",
            filename="2023-12-12-mace-128-L1_epoch-199.model",
            revision=None,  # TODO: Add revision
        )
        super().__init__(model_path=fpath, device=device)

        self.name = "MACE-MP-0 (medium)"
        self.version = "1.0.0"
        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
        ]

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
