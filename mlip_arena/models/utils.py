"""Utility functions for MLIP models."""

import importlib
from enum import Enum
from typing import Any

import torch
from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

from mlip_arena.models import REGISTRY

MLIPMap = {
    model: getattr(
        importlib.import_module(f"{__package__}.{metadata['module']}"), model
    )
    for model, metadata in REGISTRY.items()
}


class EXTMLIPEnum(Enum):
    """
    Enumeration class for EXTMLIP models.

    Attributes:
        M3GNet (str): M3GNet model.
        CHGNet (str): CHGNet model.
        MACE (str): MACE model.
    """

    M3GNet = "M3GNet"
    CHGNet = "CHGNet"
    MACE = "MACE"


def get_freer_device() -> torch.device:
    """Get the GPU with the most free memory.

    Returns:
        torch.device: The selected GPU device.

    Raises:
        ValueError: If no GPU is available.
    """
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("No GPU available. Using CPU.")
        return torch.device("cpu")

    mem_free = [
        torch.cuda.get_device_properties(i).total_memory
        - torch.cuda.memory_allocated(i)
        for i in range(device_count)
    ]

    free_gpu_index = mem_free.index(max(mem_free))

    print(
        f"Selected GPU {free_gpu_index} with {mem_free[free_gpu_index] / 1024**2:.2f} MB free memory from {device_count} GPUs"
    )

    return torch.device(f"cuda:{free_gpu_index}")


def external_ase_calculator(name: EXTMLIPEnum, **kwargs: Any) -> Calculator:
    """Construct an ASE calculator from an external third-party MLIP packages"""

    calculator = None
    device = get_freer_device()

    if name == EXTMLIPEnum.MACE:
        from mace.calculators import mace_mp

        calculator = mace_mp(device=str(device), **kwargs)

    elif name == EXTMLIPEnum.CHGNet:
        from chgnet.model.dynamics import CHGNetCalculator

        calculator = CHGNetCalculator(use_device=str(device), **kwargs)

    elif name == EXTMLIPEnum.M3GNet:
        import matgl
        from matgl.ext.ase import PESCalculator

        potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        calculator = PESCalculator(potential, **kwargs)

    calculator.__setattr__("name", name.value)

    return calculator
