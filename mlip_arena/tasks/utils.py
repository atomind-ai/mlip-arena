"""Utility functions for MLIP models."""

from __future__ import annotations

from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

from ase import units
from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator
from mlip_arena.models import MLIPEnum
from mlip_arena.models.utils import get_freer_device

try:
    from prefect.logging import get_run_logger

    logger = get_run_logger()
except (ImportError, RuntimeError):
    from loguru import logger

from pprint import pformat


def get_calculator(
    calculator_name: str | MLIPEnum,
    calculator_kwargs: dict | None,
    dispersion: bool = False,
    dispersion_kwargs: dict | None = None,
    device: str | None = None,
) -> Calculator | SumCalculator:
    """Get a calculator with optional dispersion correction."""
    device = device or str(get_freer_device())

    logger.info(f"Using device: {device}")

    calculator_kwargs = calculator_kwargs or {}

    if isinstance(calculator_name, MLIPEnum) and calculator_name in MLIPEnum:
        assert issubclass(calculator_name.value, Calculator)
        calc = calculator_name.value(**calculator_kwargs)
    elif isinstance(calculator_name, str) and hasattr(MLIPEnum, calculator_name):
        calc = MLIPEnum[calculator_name].value(**calculator_kwargs)
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

        logger.info(f"Using dispersion: {disp_calc}")
        if dispersion_kwargs:
            logger.info(pformat(dispersion_kwargs))

    assert isinstance(calc, Calculator) or isinstance(calc, SumCalculator)
    return calc
