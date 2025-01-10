"""Utility functions for MLIP models."""

from __future__ import annotations

from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator
from ase.filters import *  # type: ignore
from ase.optimize import *  # type: ignore
from mlip_arena.models import MLIPEnum
from mlip_arena.models.utils import get_freer_device


def get_calculator(
    calculator_name: str | MLIPEnum,
    calculator_kwargs: dict | None,
    dispersion: str | None = None,
    dispersion_kwargs: dict | None = None,
    device: str | None = None,
) -> Calculator:
    device = device or str(get_freer_device())

    print(f"Using device: {device}")

    calculator_kwargs = calculator_kwargs or {}

    if isinstance(calculator_name, MLIPEnum) and calculator_name in MLIPEnum:
        assert issubclass(calculator_name.value, Calculator)
        calc = calculator_name.value(**calculator_kwargs)
    elif isinstance(calculator_name, str) and hasattr(MLIPEnum, calculator_name):
        calc = MLIPEnum[calculator_name].value(**calculator_kwargs)
    else:
        raise ValueError(f"Invalid calculator: {calculator_name}")

    print(f"Using calculator: {calc}")

    dispersion_kwargs = dispersion_kwargs or {}

    dispersion_kwargs.update({"device": device})

    if dispersion is not None:
        disp_calc = TorchDFTD3Calculator(
            **dispersion_kwargs,
        )
        calc = SumCalculator([calc, disp_calc])

        print(f"Using dispersion: {dispersion}")

    assert isinstance(calc, Calculator)
    return calc
