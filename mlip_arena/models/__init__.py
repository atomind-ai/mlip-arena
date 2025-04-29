from __future__ import annotations

import importlib
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Type, TypeVar, Union

T = TypeVar("T", bound="MLIP")

import torch
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from typing_extensions import Self

try:
    from mlip_arena.data.collate import collate_fn
except ImportError:
    # Fallback to a dummy function if the import fails
    def collate_fn(batch: list[Atoms], cutoff: float) -> None:
        raise ImportError(
            "collate_fn import failed. Please install the required dependencies."
        )

try:
    from prefect.logging import get_run_logger

    logger = get_run_logger()
except (ImportError, RuntimeError):
    from loguru import logger

with open(Path(__file__).parent / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)

MLIPMap = {}

for model, metadata in REGISTRY.items():
    try:
        module = importlib.import_module(
            f"{__package__}.{metadata['module']}.{metadata['family']}"
        )
        MLIPMap[model] = getattr(module, metadata["class"])
    except (ModuleNotFoundError, AttributeError, ValueError, ImportError) as e:
        logger.warning(e)
        continue

MLIPEnum = Enum("MLIPEnum", MLIPMap)


class MLIP(
    nn.Module,
    PyTorchModelHubMixin,
    tags=["atomistic-simulation", "MLIP"],
):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # https://github.com/pytorch/pytorch/blob/3cbc8c54fd37eb590e2a9206aecf3ab568b3e63c/torch/_dynamo/config.py#L534
        # torch._dynamo.config.compiled_autograd = True
        # self.model = torch.compile(model)
        self.model = model

    def _save_pretrained(self, save_directory: Path) -> None:
        return super()._save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **model_kwargs,
    ) -> Self:
        return super().from_pretrained(
            pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            **model_kwargs,
        )

    def forward(self, x):
        return self.model(x)


class MLIPCalculator(MLIP, Calculator):
    name: str
    implemented_properties: list[str] = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        cutoff: float = 6.0,
        # ASE Calculator
        restart=None,
        atoms=None,
        directory=".",
        calculator_kwargs: dict = {},
    ):
        MLIP.__init__(self, model=model)  # Initialize MLIP part
        Calculator.__init__(
            self, restart=restart, atoms=atoms, directory=directory, **calculator_kwargs
        )  # Initialize ASE Calculator part
        # Additional initialization if needed
        # self.name: str = self.__class__.__name__
        from mlip_arena.models.utils import get_freer_device

        self.device = device or get_freer_device()
        self.cutoff = cutoff
        self.model.to(self.device)
        # self.device = device or torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu"
        # )
        # self.model: MLIP = MLIP.from_pretrained(model_path, map_location=self.device)
        # self.implemented_properties = ["energy", "forces", "stress"]

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state["_modules"]["model"] = state["_modules"]["model"]._orig_mod
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     self.model = torch.compile(state["_modules"]["model"])

    def calculate(
        self,
        atoms: Atoms,
        properties: list[str],
        system_changes: list = all_changes,
    ):
        """Calculate energies and forces for the given Atoms object"""
        super().calculate(atoms, properties, system_changes)

        # TODO: move collate_fn to here in MLIPCalculator
        data = collate_fn([atoms], cutoff=self.cutoff).to(self.device)
        output = self.forward(data)

        # TODO: decollate_fn

        self.results = {}
        if "energy" in properties:
            self.results["energy"] = output["energy"].squeeze().item()
        if "forces" in properties:
            self.results["forces"] = output["forces"].squeeze().cpu().detach().numpy()
        if "stress" in properties:
            self.results["stress"] = output["stress"].squeeze().cpu().detach().numpy()

    # def forward(self, x: Atoms) -> dict[str, torch.Tensor]:
    #     """Implement data conversion, graph creation, and model forward pass

    #     Example implementation:
    #     1. Use `ase.neighborlist.NeighborList` to get neighbor list
    #     2. Create `torch_geometric.data.Data` object and copy the data
    #     3. Pass the `Data` object to the model and return the output

    #     """

    #     raise NotImplementedError
