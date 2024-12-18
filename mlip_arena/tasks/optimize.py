"""
Define structure optimization tasks.
"""

from __future__ import annotations

from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import task_run
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixSymmetry
from ase.filters import *  # type: ignore
from ase.filters import Filter
from ase.optimize import *  # type: ignore
from ase.optimize.optimize import Optimizer
from mlip_arena.models import MLIPEnum
from mlip_arena.models.utils import get_freer_device

_valid_filters: dict[str, Filter] = {
    "Filter": Filter,
    "UnitCell": UnitCellFilter,
    "ExpCell": ExpCellFilter,
    "Strain": StrainFilter,
    "FrechetCell": FrechetCellFilter,
}  # type: ignore

_valid_optimizers: dict[str, Optimizer] = {
    "MDMin": MDMin,
    "FIRE": FIRE,
    "FIRE2": FIRE2,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "BFGS": BFGS,
    "BFGSLineSearch": BFGSLineSearch,
    "QuasiNewton": QuasiNewton,
    "GPMin": GPMin,
    "CellAwareBFGS": CellAwareBFGS,
    "ODE12r": ODE12r,
}  # type: ignore


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters

    atoms = parameters["atoms"]
    calculator_name = parameters["calculator_name"]

    return f"{task_name}: {atoms.get_chemical_formula()} - {calculator_name}"


@task(
    name="OPT",
    task_run_name=_generate_task_run_name,
    cache_policy=TASK_SOURCE + INPUTS
    # cache_key_fn=task_input_hash,
    # cache_expiration=timedelta(days=1)
)
def run(
    atoms: Atoms,
    calculator_name: str | MLIPEnum,
    calculator_kwargs: dict | None = None,
    dispersion: str | None = None,
    dispersion_kwargs: dict | None = None,
    device: str | None = None,
    optimizer: Optimizer | str = BFGSLineSearch,
    optimizer_kwargs: dict | None = None,
    filter: Filter | str | None = None,
    filter_kwargs: dict | None = None,
    criterion: dict | None = None,
    symmetry: bool = False,
):
    device = device or str(get_freer_device())

    print(f"Using device: {device}")

    calculator_kwargs = calculator_kwargs or {}

    if isinstance(calculator_name, MLIPEnum) and calculator_name in MLIPEnum:
        assert issubclass(calculator_name.value, Calculator)
        calc = calculator_name.value(**calculator_kwargs)
    elif (
        isinstance(calculator_name, str) and calculator_name in MLIPEnum._member_names_
    ):
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

    atoms.calc = calc

    if isinstance(filter, str):
        if filter not in _valid_filters:
            raise ValueError(f"Invalid filter: {filter}")
        filter = _valid_filters[filter]

    if isinstance(optimizer, str):
        if optimizer not in _valid_optimizers:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        optimizer = _valid_optimizers[optimizer]

    filter_kwargs = filter_kwargs or {}
    optimizer_kwargs = optimizer_kwargs or {}
    criterion = criterion or {}

    if symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    if isinstance(filter, type) and issubclass(filter, Filter):
        filter_instance = filter(atoms, **filter_kwargs)
        print(f"Using filter: {filter_instance}")

        optimizer_instance = optimizer(atoms, **optimizer_kwargs)
        print(f"Using optimizer: {optimizer_instance}")

        optimizer_instance.run(**criterion)

    elif filter is None:
        optimizer_instance = optimizer(atoms, **optimizer_kwargs)
        print(f"Using optimizer: {optimizer_instance}")
        optimizer_instance.run(**criterion)

    return {
        "atoms": atoms,
    }
