import torch
from ase import Atoms
from ase.calculators.calculator import all_changes
from huggingface_hub import hf_hub_download
from torch_geometric.data import Data

from mlip_arena.models import MLIP, MLIPCalculator, ModuleMLIP


class MACE_MP_Medium(MLIPCalculator):
    def __init__(
        self,
        device: torch.device | None = None,
        restart=None,
        atoms=None,
        directory=".",
        **kwargs,
    ):
        # Download the pytorch model from huggingface to local and load it
        # NOTE: this is not the ideal way to load the model, but it is the simplest
        # way to do it for now. Ideally, if the model is the subclass of PyTorchModelHubMixin,
        # we should be able to load it directly from the hub or local using MLIP class.
        fpath = hf_hub_download(
            repo_id="cyrusyc/mace-universal",
            subfolder="pretrained",
            filename="2023-12-12-mace-128-L1_epoch-199.model",
            revision=None,  # TODO: Add revision
        )
        # module = ModuleMLIP(torch.load(fpath, map_location="cpu"))
        print(torch.load(fpath, map_location="cpu"))
        repo_id = f"atomind/{self.__class__.__name__}".replace("_", "-")
        # module.save_pretrained(
        #     save_directory=self.__class__.__name__,
        #     repo_id=repo_id,
        #     push_to_hub=True,
        # )

        super().__init__(
            model_path=repo_id,
            device=device,
            restart=restart,
            atoms=atoms,
            directory=directory,
            **kwargs,
        )

        # self.name: str = self.__class__.__name__
        # self.device = device or torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu"
        # )
        # self.model: MLIP = ModuleMLIP.from_pretrained(repo_id, map_location=self.device)
        # self.implemented_properties = ["energy", "forces", "stress"]

        self.display = "MACE-MP-0 (medium)"
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
