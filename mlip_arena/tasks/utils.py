"""Utility functions for MLIP models."""

from __future__ import annotations

from pprint import pformat

import torch
from ase import units
from ase.calculators.calculator import BaseCalculator
from ase.calculators.mixing import SumCalculator

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
    calculator: str | MLIPEnum | BaseCalculator,
    calculator_kwargs: dict | None = None,
    dispersion: bool = False,
    dispersion_kwargs: dict | None = None,
    device: str | None = None,
) -> BaseCalculator:
    """
    Constructs an ASE BaseCalculator configured for the specified MLIP model and device, optionally combined with a dispersion correction.
    
    Parameters:
        calculator (str | MLIPEnum | BaseCalculator): The calculator specifier — an MLIPEnum member, the name of an MLIPEnum entry, a subclass of ASE BaseCalculator, or an instance of BaseCalculator.
        calculator_kwargs (dict | None): Keyword arguments passed to the calculator constructor. The resolved device is injected into this dict.
        dispersion (bool): If True, attach a DFT-D3-like dispersion calculator and combine it with the main calculator.
        dispersion_kwargs (dict | None): Keyword arguments for the dispersion calculator. If not provided, defaults to {'damping': 'bj', 'xc': 'pbe', 'cutoff': 40.0 * units.Bohr}. The resolved device is injected into this dict.
        device (str | None): Device identifier to use (e.g., 'cuda:0', 'mps', 'cpu'). If None, the function selects a device via get_freer_device().
    
    Returns:
        BaseCalculator: The constructed calculator. If dispersion is requested, returns a SumCalculator that combines the main calculator with the dispersion calculator.
    
    Raises:
        ValueError: If `calculator` is not a supported type or value.
        ImportError: If `dispersion` is True but the required torch_dftd dependency is not installed.
    """

    device = device or str(get_freer_device())

    calculator_kwargs = calculator_kwargs or {}
    calculator_kwargs.update({"device": device})

    logger.info(f"Using device: {device}")

    if isinstance(calculator, MLIPEnum) and calculator in MLIPEnum:
        calc = calculator.value(**calculator_kwargs)
        calc.__str__ = lambda: calculator.name
    elif isinstance(calculator, str) and hasattr(MLIPEnum, calculator):
        calc = MLIPEnum[calculator].value(**calculator_kwargs)
        calc.__str__ = lambda: calculator
    elif isinstance(calculator, type) and issubclass(
        calculator, BaseCalculator
    ):
        logger.warning(f"Using custom calculator class: {calculator}")
        calc = calculator(**calculator_kwargs)
        calc.__str__ = lambda: f"{calc.__class__.__name__}"
    elif isinstance(calculator, BaseCalculator):
        logger.warning(
            f"Using custom calculator object (kwargs are ignored): {calculator}"
        )
        calc = calculator
        calc.__str__ = lambda: f"{calc.__class__.__name__}"
    else:
        raise ValueError(f"Invalid calculator: {calculator}")

    logger.info(f"Using calculator: {calc}")
    if calculator_kwargs:
        logger.info(pformat(calculator_kwargs))

    dispersion_kwargs = dispersion_kwargs or dict(
        damping="bj", xc="pbe", cutoff=40.0 * units.Bohr
    )

    dispersion_kwargs.update({"device": device})

    if dispersion:
        try:
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        except ImportError as e:
            raise ImportError(
                "torch_dftd is required for dispersion but is not installed."
            ) from e

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