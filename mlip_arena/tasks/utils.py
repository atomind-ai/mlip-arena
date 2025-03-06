"""Utility functions for MLIP models."""

from __future__ import annotations

from pprint import pformat

import torch
from ase import units
from ase.calculators.calculator import BaseCalculator
from ase.calculators.mixing import SumCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

from mlip_arena.models import MLIPEnum

try:
    from prefect.logging import get_run_logger

    logger = get_run_logger()
except (ImportError, RuntimeError):
    from loguru import logger


def get_freer_device() -> torch.device:
    """Get the GPU with the most free memory, or use MPS if available.
    s
        Returns:
            torch.device: The selected GPU device or MPS.

        Raises:
            ValueError: If no GPU or MPS is available.
    """
    device_count = torch.cuda.device_count()
    if device_count > 0:
        # If CUDA GPUs are available, select the one with the most free memory
        mem_free = [
            torch.cuda.get_device_properties(i).total_memory
            - torch.cuda.memory_allocated(i)
            for i in range(device_count)
        ]
        free_gpu_index = mem_free.index(max(mem_free))
        device = torch.device(f"cuda:{free_gpu_index}")
        logger.info(
            f"Selected GPU {device} with {mem_free[free_gpu_index] / 1024**2:.2f} MB free memory from {device_count} GPUs"
        )
    elif torch.backends.mps.is_available():
        # If no CUDA GPUs are available but MPS is, use MPS
        logger.info("No GPU available. Using MPS.")
        device = torch.device("mps")
    else:
        # Fallback to CPU if neither CUDA GPUs nor MPS are available
        logger.info("No GPU or MPS available. Using CPU.")
        device = torch.device("cpu")

    return device


def get_calculator(
    calculator_name: str | MLIPEnum | BaseCalculator,
    calculator_kwargs: dict | None = None,
    dispersion: bool = False,
    dispersion_kwargs: dict | None = None,
    device: str | None = None,
) -> BaseCalculator:
    """Get a calculator with optional dispersion correction."""

    device = device or str(get_freer_device())

    calculator_kwargs = calculator_kwargs or {}
    calculator_kwargs.update({"device": device})

    logger.info(f"Using device: {device}")

    if isinstance(calculator_name, MLIPEnum) and calculator_name in MLIPEnum:
        calc = calculator_name.value(**calculator_kwargs)
        calc.__str__ = lambda: calculator_name.name
    elif isinstance(calculator_name, str) and hasattr(MLIPEnum, calculator_name):
        calc = MLIPEnum[calculator_name].value(**calculator_kwargs)
        calc.__str__ = lambda: calculator_name
    elif isinstance(calculator_name, type) and issubclass(
        calculator_name, BaseCalculator
    ):
        logger.warning(f"Using custom calculator class: {calculator_name}")
        calc = calculator_name(**calculator_kwargs)
        calc.__str__ = lambda: f"{calc.__class__.__name__}"
    elif isinstance(calculator_name, BaseCalculator):
        logger.warning(
            f"Using custom calculator object (kwargs are ignored): {calculator_name}"
        )
        calc = calculator_name
        calc.__str__ = lambda: f"{calc.__class__.__name__}"
    else:
        raise ValueError(f"Invalid calculator: {calculator_name}")

    logger.info(f"Using calculator: {calc}")
    if calculator_kwargs:
        logger.info(pformat(calculator_kwargs))

    dispersion_kwargs = dispersion_kwargs or dict(
        damping="bj", xc="pbe", cutoff=40.0 * units.Bohr
    )

    dispersion_kwargs.update({"device": device})

    if dispersion:
        disp_calc = TorchDFTD3Calculator(
            **dispersion_kwargs,
        )
        calc = SumCalculator([calc, disp_calc])
        # TODO: rename the SumCalculator

        logger.info(f"Using dispersion: {disp_calc}")
        if dispersion_kwargs:
            logger.info(pformat(dispersion_kwargs))

    assert isinstance(calc, BaseCalculator)    
    return calc
