from __future__ import annotations

import importlib
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import torch
import yaml
from ase.calculators.calculator import Calculator, all_changes
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from typing_extensions import Self

if TYPE_CHECKING:
    from ase import Atoms

# ASE 3.23 moved full_3x3_to_voigt_6_stress from constraints to stress,
# breaking mattersim and potentially other models.
import ase.constraints

if not hasattr(ase.constraints, "full_3x3_to_voigt_6_stress"):
    try:
        import ase.stress

        ase.constraints.full_3x3_to_voigt_6_stress = ase.stress.full_3x3_to_voigt_6_stress
    except ImportError:
        pass

try:
    from mlip_arena.data.collate import collate_fn
except ImportError:
    # Fallback to a dummy function if the import fails
    def collate_fn(batch: list[Atoms], cutoff: float) -> None:
        raise ImportError("collate_fn import failed. Please install the required dependencies.")


try:
    from prefect.logging import get_run_logger

    logger = get_run_logger()
except (ImportError, RuntimeError):
    from loguru import logger


_PACKAGE = __package__  # "mlip_arena.models"

with open(Path(__file__).parent / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)


def _make_mlip_enum() -> Enum:
    """Build MLIPEnum from registry.yaml without importing any model packages.

    Each member's *value* is the metadata dict straight from the registry.
    Actual model classes are imported lazily on first ``.load()`` call.
    Deprecated models (``deprecated: true`` in registry.yaml) are excluded.
    """
    members: dict[str, dict] = {}
    deprecated: list[str] = []

    for model_name, meta in REGISTRY.items():
        if meta.get("deprecated", False):
            deprecated.append(model_name)
            continue
        members[model_name] = meta

    if deprecated:
        logger.info(f"Skipping deprecated models: {deprecated}")

    logger.info(f"Registered models: {list(members.keys())}")
    return Enum("MLIPEnum", members)


MLIPEnum = _make_mlip_enum()

# ── Patch enum members with convenience methods ────────────────────────────────


def _metadata(self) -> dict:
    """Return the raw registry metadata dict for this model."""
    return self.value


def _load(self, **kwargs):
    """Import the model class and return an instantiated calculator.

    Parameters
    ----------
    **kwargs
        Forwarded to the calculator's ``__init__``.

    Returns
    -------
    An ASE-compatible calculator instance.
    """
    meta = self.value
    try:
        module = importlib.import_module(f"{_PACKAGE}.{meta['module']}.{meta['family']}")
        cls = getattr(module, meta["class"])
    except (ModuleNotFoundError, AttributeError, ImportError) as e:
        raise ImportError(
            f"Cannot load model '{self.name}': {e}. "
            f"Make sure '{meta.get('package', 'the required package')}' is installed."
        ) from e
    return cls(**kwargs)


def _call(self, **kwargs):
    """Shorthand: ``MLIPEnum['MACE-MP(M)']()`` → calls ``.load()``."""
    return self.load(**kwargs)


# Attach methods to every member of the dynamically created enum
MLIPEnum.metadata = property(_metadata)  # type: ignore[attr-defined]
MLIPEnum.load = _load  # type: ignore[attr-defined]
MLIPEnum.__call__ = _call  # type: ignore[attr-defined]


# ── Core model base classes (unchanged) ───────────────────────────────────────


class MLIP(
    nn.Module,
    PyTorchModelHubMixin,
    tags=["atomistic-simulation", "MLIP"],
):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
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
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        cutoff: float = 6.0,
        # ASE Calculator
        restart=None,
        atoms=None,
        directory=".",
        calculator_kwargs: dict | None = None,
    ):
        MLIP.__init__(self, model=model)
        Calculator.__init__(
            self,
            restart=restart,
            atoms=atoms,
            directory=directory,
            **(calculator_kwargs or {}),
        )
        from mlip_arena.models.utils import get_freer_device

        self.device = device or get_freer_device()
        self.cutoff = cutoff
        self.model.to(self.device)

    def calculate(
        self,
        atoms: Atoms,
        properties: list[str],
        system_changes: list = all_changes,
    ):
        """Calculate energies and forces for the given Atoms object"""
        super().calculate(atoms, properties, system_changes)

        data = collate_fn([atoms], cutoff=self.cutoff).to(self.device)
        output = self.forward(data)

        self.results = {}
        if "energy" in properties:
            self.results["energy"] = output["energy"].squeeze().item()
        if "forces" in properties:
            self.results["forces"] = output["forces"].squeeze().cpu().detach().numpy()
        if "stress" in properties:
            self.results["stress"] = output["stress"].squeeze().cpu().detach().numpy()
